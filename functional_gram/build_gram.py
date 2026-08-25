"""Construct K x K activation- and weight-side functional Gram matrices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass(frozen=True)
class FunctionalGrams:
    """Functional Grams using the repository ``weight[out_features, K]`` layout."""

    gx: torch.Tensor
    gw: torch.Tensor
    jx: torch.Tensor
    jw: torch.Tensor
    token_count: int


@torch.no_grad()
def build_functional_gram(
    activation_batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    weight: torch.Tensor,
    qweight: torch.Tensor,
    *,
    source: str,
    accumulation_dtype: torch.dtype = torch.float64,
    normalize_by_tokens: bool = False,
) -> tuple[torch.Tensor, int]:
    """Memory-bounded construction of one source Gram."""

    if source not in {"X", "W"}:
        raise ValueError("source must be 'X' or 'W'")
    weight = _matrix(weight, "weight")
    qweight = _matrix(qweight, "qweight")
    if weight.shape != qweight.shape:
        raise ValueError("weight and qweight shapes differ")
    if accumulation_dtype not in {torch.float32, torch.float64}:
        raise ValueError("accumulation_dtype must be float32 or float64")
    k = int(weight.shape[1])
    activation_covariance = torch.zeros(
        (k, k), dtype=accumulation_dtype, device=weight.device
    )
    token_count = 0
    for x, qx in activation_batches:
        x = _matrix(x, "x")
        qx = _matrix(qx, "qx")
        if x.shape != qx.shape or int(x.shape[1]) != k:
            raise ValueError("activation/quantized activation shape mismatch")
        token_count += int(x.shape[0])
        if source == "X":
            value = x.to(device=weight.device, dtype=accumulation_dtype) - qx.to(
                device=weight.device, dtype=accumulation_dtype
            )
        else:
            value = x.to(device=weight.device, dtype=accumulation_dtype)
        activation_covariance.add_(value.T.matmul(value))
    if token_count == 0:
        raise ValueError("at least one activation batch is required")

    if source == "X":
        factor = weight.to(dtype=accumulation_dtype)
    else:
        factor = weight.to(dtype=accumulation_dtype) - qweight.to(
            device=weight.device, dtype=accumulation_dtype
        )
    gram = activation_covariance.mul(factor.T.matmul(factor))
    gram = (gram + gram.T).mul_(0.5)
    if normalize_by_tokens:
        gram.div_(token_count)
    return gram, token_count


def _matrix(value: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.numel() == 0:
        raise ValueError(f"{name} must be a non-empty rank-two tensor")
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} contains non-finite values")
    return value


def _covariance(value: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    work = value.to(dtype=dtype)
    return work.T.matmul(work)


@torch.no_grad()
def build_functional_grams(
    activation_batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    weight: torch.Tensor,
    qweight: torch.Tensor,
    *,
    accumulation_dtype: torch.dtype = torch.float64,
    normalize_by_tokens: bool = False,
) -> FunctionalGrams:
    """Accumulate covariance first, then form both Hadamard-product Grams.

    Each activation batch is ``(X, qX)`` with shape ``[tokens, K]``.  Weight
    tensors retain the PyTorch Linear layout ``[out_features, K]``; therefore
    mathematical ``W`` in the experiment specification is ``weight.T``.
    """

    weight = _matrix(weight, "weight")
    qweight = _matrix(qweight, "qweight")
    if weight.shape != qweight.shape:
        raise ValueError("weight and qweight shapes differ")
    if accumulation_dtype not in {torch.float32, torch.float64}:
        raise ValueError("accumulation_dtype must be float32 or float64")

    k = int(weight.shape[1])
    sx = torch.zeros((k, k), dtype=accumulation_dtype, device=weight.device)
    sex = torch.zeros_like(sx)
    token_count = 0
    saw_batch = False
    for x, qx in activation_batches:
        x = _matrix(x, "x")
        qx = _matrix(qx, "qx")
        if x.shape != qx.shape or int(x.shape[1]) != k:
            raise ValueError("activation/quantized activation shape mismatch")
        saw_batch = True
        token_count += int(x.shape[0])
        x_work = x.to(device=weight.device, dtype=accumulation_dtype)
        ex_work = x_work - qx.to(device=weight.device, dtype=accumulation_dtype)
        sx.add_(x_work.T.matmul(x_work))
        sex.add_(ex_work.T.matmul(ex_work))
    if not saw_batch:
        raise ValueError("at least one activation batch is required")

    weight_work = weight.to(dtype=accumulation_dtype)
    ew_work = weight_work - qweight.to(device=weight.device, dtype=accumulation_dtype)
    ww = weight_work.T.matmul(weight_work)
    ewew = ew_work.T.matmul(ew_work)
    gx = sex.mul(ww)
    gw = sx.mul(ewew)
    gx = (gx + gx.T).mul_(0.5)
    gw = (gw + gw.T).mul_(0.5)
    if normalize_by_tokens:
        gx.div_(token_count)
        gw.div_(token_count)
    return FunctionalGrams(
        gx=gx,
        gw=gw,
        jx=torch.diagonal(gx).clone(),
        jw=torch.diagonal(gw).clone(),
        token_count=token_count,
    )


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    numerator = float((actual - expected).abs().max().item())
    denominator = max(float(expected.abs().max().item()), 1e-30)
    return numerator / denominator


@torch.no_grad()
def sanity_check_grams(
    grams: FunctionalGrams,
    x: torch.Tensor,
    qx: torch.Tensor,
    weight: torch.Tensor,
    qweight: torch.Tensor,
    *,
    channels: Iterable[int] | None = None,
    rtol: float = 2e-8,
) -> dict[str, float | list[int] | str]:
    """Audit the diagonal and full functional-norm identities in FP64."""

    x64 = _matrix(x, "x").double()
    qx64 = _matrix(qx, "qx").double()
    w64 = _matrix(weight, "weight").double()
    qw64 = _matrix(qweight, "qweight").double()
    if x64.shape != qx64.shape or w64.shape != qw64.shape:
        raise ValueError("original/quantized operand shape mismatch")
    k = int(x64.shape[1])
    if int(w64.shape[1]) != k or grams.gx.shape != (k, k) or grams.gw.shape != (k, k):
        raise ValueError("incompatible functional-Gram dimensions")
    if channels is None:
        channels = sorted({0, k // 4, k // 2, (3 * k) // 4, k - 1})
    indices = torch.as_tensor(list(channels), dtype=torch.long, device=x64.device)
    if indices.numel() == 0 or int(indices.min()) < 0 or int(indices.max()) >= k:
        raise ValueError("sanity-check channels are invalid")

    ex = x64 - qx64
    ew = w64 - qw64
    diag_x = ex.square().sum(0).index_select(0, indices) * w64.square().sum(0).index_select(0, indices)
    diag_w = x64.square().sum(0).index_select(0, indices) * ew.square().sum(0).index_select(0, indices)
    gram_diag_x = torch.diagonal(grams.gx).index_select(0, indices)
    gram_diag_w = torch.diagonal(grams.gw).index_select(0, indices)
    gx_one = grams.gx.sum()
    gw_one = grams.gw.sum()
    direct_x = ex.matmul(w64.T).square().sum()
    direct_w = x64.matmul(ew.T).square().sum()
    checks = {
        "diag_x_relative_error": _relative_error(gram_diag_x, diag_x),
        "diag_w_relative_error": _relative_error(gram_diag_w, diag_w),
        "functional_norm_x_relative_error": _relative_error(gx_one, direct_x),
        "functional_norm_w_relative_error": _relative_error(gw_one, direct_w),
        "sampled_channels": indices.detach().cpu().tolist(),
    }
    maximum = max(float(value) for key, value in checks.items() if key.endswith("relative_error"))
    checks["max_relative_error"] = maximum
    checks["status"] = "passed" if maximum <= rtol else "failed"
    if maximum > rtol:
        raise AssertionError(f"functional-Gram identity check failed: {checks}")
    return checks
