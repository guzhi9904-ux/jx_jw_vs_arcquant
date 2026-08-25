"""Thin wrappers around the single audited NVFP4 numerical reference."""

from __future__ import annotations

from typing import Any

import torch

from fp4_residual_carrier.common.nvfp4_reference import (
    compute_global_scale,
    quantize_nvfp4,
)


@torch.no_grad()
def quantize_matrix_chunked(
    value: torch.Tensor,
    *,
    global_scale: torch.Tensor,
    row_chunk: int = 256,
    output_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Apply the audited reference with one fixed tensor-global scale."""

    if value.ndim != 2 or value.shape[0] <= 0 or value.shape[1] <= 0:
        raise ValueError("expected a non-empty matrix")
    if row_chunk <= 0:
        raise ValueError("row_chunk must be positive")
    scale = torch.as_tensor(global_scale, device=value.device, dtype=torch.float32).reshape(())
    if not bool(torch.isfinite(scale).item()) or not bool((scale > 0).item()):
        raise ValueError("global_scale must be finite and positive")
    output = torch.empty(value.shape, device=value.device, dtype=output_dtype)
    underflow = 0
    local_min = float("inf")
    local_max = float("-inf")
    for start in range(0, value.shape[0], row_chunk):
        end = min(start + row_chunk, value.shape[0])
        source = value[start:end].float()
        result = quantize_nvfp4(
            source,
            group_size=16,
            global_scale=scale,
            local_scale_mode="ue4m3",
        )
        chunk = result.dequantized
        output[start:end].copy_(chunk.to(output_dtype))
        underflow += int(((chunk == 0) & (source != 0)).sum().item())
        local_min = min(local_min, float(result.local_scales.min().item()))
        local_max = max(local_max, float(result.local_scales.max().item()))
    return output, {
        "global_scale": float(scale.item()),
        "global_scale_source": "provided_tensor_global_absmax",
        "group_size": 16,
        "local_scale_mode": "ue4m3",
        "local_scale_min": local_min,
        "local_scale_max": local_max,
        "underflow_rate": underflow / value.numel(),
        "row_chunk": row_chunk,
    }


__all__ = ["compute_global_scale", "quantize_matrix_chunked"]
