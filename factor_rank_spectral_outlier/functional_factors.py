"""Exact factor and Hadamard Functional-Gram construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class FunctionalFactorGrams:
    source: str
    factor1_name: str
    factor2_name: str
    factor1: torch.Tensor
    factor2: torch.Tensor
    covariance1: torch.Tensor | None
    covariance2: torch.Tensor | None
    gram: torch.Tensor
    diagnostics: dict[str, Any]


def _matrix(value: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.numel() == 0:
        raise ValueError(f"{name} must be a non-empty matrix")
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} contains non-finite values")
    return value


def _relative(actual: torch.Tensor | float, expected: torch.Tensor | float) -> float:
    actual_value = torch.as_tensor(actual, dtype=torch.float64)
    expected_value = torch.as_tensor(expected, dtype=torch.float64)
    return float((actual_value - expected_value).abs().max().item()) / max(
        float(expected_value.abs().max().item()), 1e-30
    )


@torch.no_grad()
def build_functional_factor_grams(
    x: torch.Tensor,
    qx: torch.Tensor,
    weight: torch.Tensor,
    qweight: torch.Tensor,
    *,
    source: str,
    accumulation_dtype: torch.dtype = torch.float64,
    run_sanity: bool = True,
    retain_covariances: bool = True,
) -> FunctionalFactorGrams:
    """Build the two PSD factors and their Hadamard product.

    PyTorch weights are ``[M, K]``. For X-source the factors are ``E_X`` and
    ``W``; for W-source they are exactly ``qX`` and ``E_W`` as required by
    ``XW-qXqW = E_X W + qX E_W``.
    """

    x = _matrix(x, "x")
    qx = _matrix(qx, "qx")
    weight = _matrix(weight, "weight")
    qweight = _matrix(qweight, "qweight")
    if x.shape != qx.shape or weight.shape != qweight.shape:
        raise ValueError("original and quantized operands have different shapes")
    if x.shape[1] != weight.shape[1]:
        raise ValueError("activation and weight K differ")
    if source not in ("X", "W"):
        raise ValueError("source must be X or W")
    if accumulation_dtype not in (torch.float32, torch.float64):
        raise ValueError("accumulation_dtype must be float32 or float64")

    xw = x.to(dtype=accumulation_dtype)
    qxw = qx.to(device=x.device, dtype=accumulation_dtype)
    ww = weight.to(device=x.device, dtype=accumulation_dtype)
    qww = qweight.to(device=x.device, dtype=accumulation_dtype)
    if source == "X":
        factor1 = xw - qxw
        factor2 = ww
        names = ("R_EX", "R_W")
    else:
        factor1 = qxw
        factor2 = ww - qww
        names = ("R_qX", "R_EW")
    covariance1 = factor1.T.matmul(factor1)
    covariance2 = factor2.T.matmul(factor2)
    gram = covariance1.mul(covariance2)
    gram = (gram + gram.T) * 0.5

    diagnostics: dict[str, Any] = {}
    if run_sanity:
        hadamard = covariance1 * covariance2
        hadamard_error = _relative(gram, hadamard)
        direct = factor1.matmul(factor2.T)
        direct_norm = direct.double().square().sum()
        functional_norm_error = _relative(gram.double().sum(), direct_norm)
        diagonal_expected = factor1.double().square().sum(0) * factor2.double().square().sum(0)
        diagonal_error = _relative(gram.diagonal(), diagonal_expected)
        tolerance = 2e-8 if accumulation_dtype == torch.float64 else 3e-4
        maximum = max(hadamard_error, functional_norm_error, diagonal_error)
        diagnostics = {
            "hadamard_identity_relative_error": hadamard_error,
            "functional_norm_relative_error": functional_norm_error,
            "diagonal_identity_relative_error": diagonal_error,
            "max_relative_error": maximum,
            "tolerance": tolerance,
            "status": "passed" if maximum <= tolerance else "failed",
        }
        if maximum > tolerance:
            raise AssertionError(f"functional factor sanity failed: {diagnostics}")

    return FunctionalFactorGrams(
        source=source,
        factor1_name=names[0],
        factor2_name=names[1],
        factor1=factor1,
        factor2=factor2,
        covariance1=covariance1 if retain_covariances else None,
        covariance2=covariance2 if retain_covariances else None,
        gram=gram,
        diagnostics=diagnostics,
    )


__all__ = ["FunctionalFactorGrams", "build_functional_factor_grams"]
