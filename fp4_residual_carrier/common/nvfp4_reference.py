"""Kernel-faithful numerical reference for Blackwell NVFP4 quantization.

The repository CUDA path uses E2M1 values, one unsigned E4M3 scale for every
16 consecutive values on the reduction (last) dimension, and an external
tensor-wise FP32 scale.  This module mirrors those numerical rules while also
making the otherwise packed metadata observable.

``permutation`` follows the CUDA ``reorder_index`` convention: entry ``p`` is
the original channel placed at packed position ``p``.  Public logical outputs
are restored to the input channel order; ``packed_*`` fields expose the exact
physical order (including right-zero padding) used to form scale groups.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

import torch
import torch.nn.functional as F


E2M1_MAX = 6.0
E2M1_POSITIVE_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
UE4M3_MIN_SUBNORMAL = 2.0**-9
UE4M3_MAX = 448.0
NVFP4_GROUP_SIZE = 16
NVFP4_GLOBAL_DENOMINATOR = E2M1_MAX * UE4M3_MAX

LocalScaleMode = Literal["ue4m3", "exact"]


@dataclass(frozen=True)
class NVFP4Result:
    """Observable result of :func:`quantize_nvfp4`.

    ``dequantized`` and ``codes`` have the original input shape and logical
    channel order.  ``packed_dequantized`` and ``packed_codes`` retain the
    permuted, padded physical layout.  Local-scale tensors have shape
    ``input.shape[:-1] + (ceil(K / group_size),)``.

    In ``local_scale_mode="exact"``, ``local_scale_codes`` still records the
    nearest UE4M3 code for diagnostic comparisons, but ``local_scales`` holds
    and applies the unclipped-FP32 value after the hardware range clamp.
    """

    dequantized: torch.Tensor
    codes: torch.Tensor
    packed_dequantized: torch.Tensor
    packed_codes: torch.Tensor
    local_scales: torch.Tensor
    local_scale_codes: torch.Tensor
    raw_local_scales: torch.Tensor
    clamped_local_scales: torch.Tensor
    global_scale: torch.Tensor
    permutation: torch.Tensor
    inverse_permutation: torch.Tensor
    metadata: dict[str, Any]


def _float_tensor(value: torch.Tensor | float | Sequence[float]) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(dtype=torch.float32)
    return torch.as_tensor(value, dtype=torch.float32)


def _reject_nan(value: torch.Tensor, name: str) -> None:
    if bool(torch.isnan(value).any().item()):
        raise ValueError(f"{name} contains NaN")


def e2m1_codebook(
    *, device: torch.device | str | None = None, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Return decoded E2M1 values indexed by their four-bit storage code."""

    positive = torch.tensor(E2M1_POSITIVE_VALUES, device=device, dtype=dtype)
    negative = -positive
    return torch.cat((positive, negative))


def decode_e2m1(
    codes: torch.Tensor | Sequence[int], *, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Decode four-bit E2M1 storage codes (including signed zero)."""

    code_tensor = torch.as_tensor(codes)
    if bool(((code_tensor < 0) | (code_tensor > 15)).any().item()):
        raise ValueError("E2M1 codes must be in [0, 15]")
    table = e2m1_codebook(device=code_tensor.device, dtype=dtype)
    return table[code_tensor.to(torch.long)]


def encode_e2m1(value: torch.Tensor | float | Sequence[float]) -> torch.Tensor:
    """Encode FP32 values as E2M1 using satfinite round-to-nearest-even."""

    x = _float_tensor(value)
    _reject_nan(x, "E2M1 input")
    magnitude = x.abs().clamp(max=E2M1_MAX)
    levels = torch.tensor(E2M1_POSITIVE_VALUES, device=x.device, dtype=x.dtype)
    boundaries = (levels[:-1] + levels[1:]) * 0.5

    # bucketize(right=False) selects the lower code at an exact midpoint.  RNE
    # instead selects the upper code when the lower significand/code is odd.
    magnitude_code = torch.bucketize(magnitude.contiguous(), boundaries, right=False)
    boundary_index = magnitude_code.clamp(max=boundaries.numel() - 1)
    is_midpoint = (magnitude_code < boundaries.numel()) & (
        magnitude == boundaries[boundary_index]
    )
    magnitude_code = magnitude_code + (
        is_midpoint & magnitude_code.remainder(2).eq(1)
    ).to(magnitude_code.dtype)

    sign_code = torch.signbit(x).to(torch.long) << 3
    return (magnitude_code.to(torch.long) | sign_code).to(torch.uint8)


def quantize_e2m1(
    value: torch.Tensor | float | Sequence[float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(decoded_value, storage_code)`` for E2M1."""

    codes = encode_e2m1(value)
    return decode_e2m1(codes), codes


def ue4m3_codebook(
    *,
    include_reserved_nan: bool = True,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return UE4M3 values in storage-code order.

    Codes 0--126 are finite.  Code 127 is the canonical reserved NaN; it is
    included by default so the tensor index is identical to the byte code.
    """

    limit = 128 if include_reserved_nan else 127
    codes = torch.arange(limit, device=device, dtype=torch.long)
    exponent = codes >> 3
    mantissa = codes & 0x7
    values = torch.where(
        exponent == 0,
        mantissa.to(dtype) * UE4M3_MIN_SUBNORMAL,
        (1.0 + mantissa.to(dtype) / 8.0)
        * torch.pow(
            torch.tensor(2.0, device=device, dtype=dtype),
            exponent.to(dtype) - 7.0,
        ),
    )
    if include_reserved_nan:
        values[-1] = float("nan")
    return values


def decode_ue4m3(
    codes: torch.Tensor | Sequence[int], *, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Decode unsigned E4M3 storage bytes; code 127 decodes to NaN."""

    code_tensor = torch.as_tensor(codes)
    if bool(((code_tensor < 0) | (code_tensor > 127)).any().item()):
        raise ValueError("UE4M3 codes must be in [0, 127]")
    table = ue4m3_codebook(device=code_tensor.device, dtype=dtype)
    return table[code_tensor.to(torch.long)]


def encode_ue4m3(value: torch.Tensor | float | Sequence[float]) -> torch.Tensor:
    """Encode non-negative FP32 values as finite UE4M3 with RNE saturation."""

    x = _float_tensor(value)
    _reject_nan(x, "UE4M3 input")
    if bool((x < 0).any().item()):
        raise ValueError("UE4M3 input must be non-negative")
    x = x.clamp(max=UE4M3_MAX)
    levels = ue4m3_codebook(
        include_reserved_nan=False, device=x.device, dtype=x.dtype
    )
    boundaries = (levels[:-1] + levels[1:]) * 0.5
    codes = torch.bucketize(x.contiguous(), boundaries, right=False)
    boundary_index = codes.clamp(max=boundaries.numel() - 1)
    is_midpoint = (codes < boundaries.numel()) & (x == boundaries[boundary_index])
    codes = codes + (is_midpoint & codes.remainder(2).eq(1)).to(codes.dtype)
    return codes.to(torch.uint8)


def quantize_ue4m3(
    value: torch.Tensor | float | Sequence[float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(decoded_value, storage_code)`` for finite UE4M3."""

    codes = encode_ue4m3(value)
    return decode_ue4m3(codes), codes


def compute_global_scale(tensor: torch.Tensor) -> torch.Tensor:
    """Compute the repository tensor-wise NVFP4 scale with a zero-safe rule."""

    x = _float_tensor(tensor)
    _reject_nan(x, "NVFP4 input")
    if bool(torch.isinf(x).any().item()):
        raise ValueError("NVFP4 input must be finite")
    maximum = x.abs().amax()
    scale = maximum / NVFP4_GLOBAL_DENOMINATOR
    # The legacy wrapper produces zero for an all-zero tensor and subsequently
    # divides by it.  A neutral 1.0 is the explicit reference convention.
    return torch.where(scale > 0, scale, torch.ones_like(scale))


def _validate_permutation(
    permutation: torch.Tensor | Sequence[int] | None,
    width: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if permutation is None:
        packed_to_logical = torch.arange(width, device=device, dtype=torch.long)
    else:
        packed_to_logical = torch.as_tensor(
            permutation, device=device, dtype=torch.long
        )
        if packed_to_logical.ndim != 1 or packed_to_logical.numel() != width:
            raise ValueError(f"permutation must be one-dimensional with length {width}")
        if not torch.equal(
            torch.sort(packed_to_logical).values,
            torch.arange(width, device=device, dtype=torch.long),
        ):
            raise ValueError("permutation must contain each last-dimension index once")
    logical_to_packed = torch.empty_like(packed_to_logical)
    logical_to_packed[packed_to_logical] = torch.arange(
        width, device=device, dtype=torch.long
    )
    return packed_to_logical, logical_to_packed


@torch.no_grad()
def quantize_nvfp4(
    tensor: torch.Tensor,
    *,
    group_size: int = NVFP4_GROUP_SIZE,
    global_scale: torch.Tensor | float | None = None,
    permutation: torch.Tensor | Sequence[int] | None = None,
    local_scale_mode: LocalScaleMode = "ue4m3",
) -> NVFP4Result:
    """Quantize a tensor with observable NVFP4 codes and scale metadata.

    Quantization groups are always consecutive chunks of the packed last
    dimension.  A non-aligned tail is padded on the right with numerical zero.
    The CUDA kernels in this repository require aligned widths; the explicit
    padding here defines the numerical extension used by CPU experiments.
    """

    if not isinstance(tensor, torch.Tensor):
        raise TypeError("tensor must be a torch.Tensor")
    if tensor.ndim < 1 or tensor.shape[-1] <= 0:
        raise ValueError("tensor must have a non-empty last dimension")
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if local_scale_mode not in {"ue4m3", "exact"}:
        raise ValueError("local_scale_mode must be 'ue4m3' or 'exact'")

    x = tensor.to(dtype=torch.float32)
    if bool((~torch.isfinite(x)).any().item()):
        raise ValueError("NVFP4 input must be finite")
    width = int(x.shape[-1])
    packed_to_logical, logical_to_packed = _validate_permutation(
        permutation, width, x.device
    )

    if global_scale is None:
        scale_g = compute_global_scale(x).to(device=x.device)
        global_scale_source = "computed_absmax_over_2688"
    else:
        scale_g = torch.as_tensor(global_scale, device=x.device, dtype=torch.float32)
        if scale_g.numel() != 1:
            raise ValueError("global_scale must be a tensor-global scalar")
        scale_g = scale_g.reshape(())
        if not bool(torch.isfinite(scale_g).item()) or not bool((scale_g > 0).item()):
            raise ValueError("global_scale must be finite and positive")
        global_scale_source = "provided"

    padding = (-width) % group_size
    packed = x.index_select(-1, packed_to_logical) / scale_g
    if padding:
        packed = F.pad(packed, (0, padding), mode="constant", value=0.0)
    padded_width = int(packed.shape[-1])
    groups_per_row = padded_width // group_size
    blocks = packed.reshape(*packed.shape[:-1], groups_per_row, group_size)

    raw_local = blocks.abs().amax(dim=-1) / E2M1_MAX
    clamped_local = raw_local.clamp(
        min=UE4M3_MIN_SUBNORMAL, max=UE4M3_MAX
    )
    rounded_local, local_codes = quantize_ue4m3(clamped_local)
    applied_local = rounded_local if local_scale_mode == "ue4m3" else clamped_local

    normalized = (blocks / applied_local.unsqueeze(-1)).clamp(
        min=-E2M1_MAX, max=E2M1_MAX
    )
    value_codes = encode_e2m1(normalized)
    decoded = decode_e2m1(value_codes) * applied_local.unsqueeze(-1) * scale_g
    packed_dequantized = decoded.reshape(*packed.shape[:-1], padded_width)
    packed_codes = value_codes.reshape(*packed.shape[:-1], padded_width)

    unpacked_dequantized = packed_dequantized[..., :width].index_select(
        -1, logical_to_packed
    )
    unpacked_codes = packed_codes[..., :width].index_select(-1, logical_to_packed)

    logical_zero = (unpacked_codes.to(torch.int16) & 0x7) == 0
    packed_zero = (packed_codes.to(torch.int16) & 0x7) == 0
    input_nonzero = x != 0
    underflow = input_nonzero & logical_zero
    magnitude_code = unpacked_codes.to(torch.int16) & 0x7
    block_channels = packed_to_logical.detach().cpu().tolist() + [-1] * padding
    block_channels = [
        block_channels[start : start + group_size]
        for start in range(0, padded_width, group_size)
    ]
    metadata: dict[str, Any] = {
        "format": "NVFP4_E2M1_UE4M3",
        "rounding": "round_to_nearest_even",
        "saturation": "satfinite",
        "group_axis": -1,
        "group_size": group_size,
        "original_shape": list(x.shape),
        "padded_shape": list(packed_dequantized.shape),
        "padding": padding,
        "padding_policy": "right_zero",
        "permutation_semantics": "packed_position_to_original_channel",
        "permutation": packed_to_logical.detach().cpu().tolist(),
        "inverse_permutation": logical_to_packed.detach().cpu().tolist(),
        "block_channel_indices": block_channels,
        "global_scale": float(scale_g.item()),
        "global_scale_source": global_scale_source,
        "local_scale_mode": local_scale_mode,
        "local_scale_min": UE4M3_MIN_SUBNORMAL,
        "local_scale_max": UE4M3_MAX,
        "zero_rate": float(logical_zero.float().mean().item()),
        "padded_zero_rate": float(packed_zero.float().mean().item()),
        "underflow_rate_nonzero_input": float(underflow.float().mean().item()),
        "e2m1_saturation_rate": float((magnitude_code == 7).float().mean().item()),
    }

    local_shape = (*x.shape[:-1], groups_per_row)
    return NVFP4Result(
        dequantized=unpacked_dequantized.reshape(x.shape),
        codes=unpacked_codes.reshape(x.shape),
        packed_dequantized=packed_dequantized,
        packed_codes=packed_codes,
        local_scales=applied_local.reshape(local_shape),
        local_scale_codes=local_codes.reshape(local_shape),
        raw_local_scales=raw_local.reshape(local_shape),
        clamped_local_scales=clamped_local.reshape(local_shape),
        global_scale=scale_g,
        permutation=packed_to_logical,
        inverse_permutation=logical_to_packed,
        metadata=metadata,
    )


# Short alias retained for experiments that use the noun-first spelling.
nvfp4_quantize = quantize_nvfp4


__all__ = [
    "E2M1_MAX",
    "E2M1_POSITIVE_VALUES",
    "LocalScaleMode",
    "NVFP4_GLOBAL_DENOMINATOR",
    "NVFP4_GROUP_SIZE",
    "NVFP4Result",
    "UE4M3_MAX",
    "UE4M3_MIN_SUBNORMAL",
    "compute_global_scale",
    "decode_e2m1",
    "decode_ue4m3",
    "e2m1_codebook",
    "encode_e2m1",
    "encode_ue4m3",
    "nvfp4_quantize",
    "quantize_e2m1",
    "quantize_nvfp4",
    "quantize_ue4m3",
    "ue4m3_codebook",
]
