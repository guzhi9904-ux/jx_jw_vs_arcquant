"""Exact dual-source decomposition and functional Gram construction.

PyTorch Linear weights use ``[out_features, K]``.  Thus the mathematical
matrix W in ``X W`` is represented by ``weight.T`` in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class DualSourceGrams:
    gx: torch.Tensor
    gw: torch.Tensor
    h: torch.Tensor
    gp: torch.Tensor
    l_x: float
    l_w: float
    cross_term: float
    total_error: float
    kappa: float
    gamma: float
    diagnostics: dict[str, Any]

    def add(self, other: "DualSourceGrams") -> "DualSourceGrams":
        """Add independent token splits while retaining exact energy sums."""

        gx = self.gx + other.gx
        gw = self.gw + other.gw
        h = self.h + other.h
        gp = _symmetrize(gx + gw + h + h.T)
        l_x = self.l_x + other.l_x
        l_w = self.l_w + other.l_w
        cross = self.cross_term + other.cross_term
        denominator = max(l_x + l_w, 1e-30)
        total = l_x + l_w + cross
        return DualSourceGrams(
            gx=gx,
            gw=gw,
            h=h,
            gp=gp,
            l_x=l_x,
            l_w=l_w,
            cross_term=cross,
            total_error=total,
            kappa=cross / denominator,
            gamma=total / denominator,
            diagnostics={"composition": "sum_of_disjoint_splits"},
        )


def _matrix(value: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.numel() == 0:
        raise ValueError(f"{name} must be a non-empty matrix")
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} contains non-finite values")
    return value


def _symmetrize(value: torch.Tensor) -> torch.Tensor:
    return (value + value.T) * 0.5


def _relative_max(actual: torch.Tensor, expected: torch.Tensor) -> float:
    numerator = float((actual - expected).abs().max().item())
    denominator = max(float(expected.abs().max().item()), 1e-30)
    return numerator / denominator


def _sample_indices(k: int, device: torch.device) -> list[tuple[int, int]]:
    candidates = ((0, k - 1), (k // 3, (2 * k) // 3), (k // 2, k // 4))
    return [(min(i, k - 1), min(j, k - 1)) for i, j in candidates]


@torch.no_grad()
def build_dual_source_grams(
    x: torch.Tensor,
    qx: torch.Tensor,
    weight: torch.Tensor,
    qweight: torch.Tensor,
    *,
    accumulation_dtype: torch.dtype = torch.float32,
    run_sanity: bool = True,
) -> DualSourceGrams:
    """Build G_X, exact-source G_W, H, and G_P.

    The W source is always ``qX @ (W-qW).T``.  Inputs may live on CPU or CUDA;
    returned matrices remain on the same device.
    """

    x = _matrix(x, "x")
    qx = _matrix(qx, "qx")
    weight = _matrix(weight, "weight")
    qweight = _matrix(qweight, "qweight")
    if x.shape != qx.shape or weight.shape != qweight.shape:
        raise ValueError("original and quantized operand shapes differ")
    if x.shape[1] != weight.shape[1]:
        raise ValueError("activation and weight K dimensions differ")
    if accumulation_dtype not in (torch.float32, torch.float64):
        raise ValueError("accumulation_dtype must be float32 or float64")

    xw = x.to(dtype=accumulation_dtype)
    qxw = qx.to(device=x.device, dtype=accumulation_dtype)
    ww = weight.to(device=x.device, dtype=accumulation_dtype)
    qww = qweight.to(device=x.device, dtype=accumulation_dtype)
    ex = xw - qxw
    ew = ww - qww

    left = ex.T.matmul(ex)
    right = ww.T.matmul(ww)
    gx = _symmetrize(left.mul_(right))
    del left, right

    left = qxw.T.matmul(qxw)
    right = ew.T.matmul(ew)
    gw = _symmetrize(left.mul_(right))
    del left, right

    left = ex.T.matmul(qxw)
    right = ww.T.matmul(ew)
    h = left.mul_(right)
    del left, right
    gp = _symmetrize(gx + gw + h + h.T)

    # Output-space energies are evaluated directly, rather than inferred from
    # a very large signed sum of Gram entries.
    dx = ex.matmul(ww.T)
    dw = qxw.matmul(ew.T)
    l_x_tensor = dx.double().square().sum()
    l_w_tensor = dw.double().square().sum()
    cross_tensor = 2.0 * (dx.double() * dw.double()).sum()
    l_x = float(l_x_tensor.item())
    l_w = float(l_w_tensor.item())
    cross = float(cross_tensor.item())
    denominator = max(l_x + l_w, 1e-30)
    total = l_x + l_w + cross

    diagnostics: dict[str, Any] = {}
    if run_sanity:
        # Check the algebra in FP64 on a deterministic submatrix.  Forming the
        # two large FP32 products and subtracting them is itself ill-conditioned
        # because the quantization error is much smaller than either product.
        # The sampled FP64 check audits the identity rather than that avoidable
        # cancellation artifact.
        sanity_rows = min(32, xw.shape[0])
        sanity_outputs = min(64, ww.shape[0])
        x64 = xw[:sanity_rows].double()
        qx64 = qxw[:sanity_rows].double()
        w64 = ww[:sanity_outputs].double()
        qw64 = qww[:sanity_outputs].double()
        direct = x64.matmul(w64.T) - qx64.matmul(qw64.T)
        decomposed = (x64 - qx64).matmul(w64.T) + qx64.matmul((w64 - qw64).T)
        exact_decomposition_error = _relative_max(decomposed, direct)
        pair_identity_error = _relative_max(gp, gx + gw + h + h.T)
        gram_total = float(gp.double().sum().item())
        total_norm_error = abs(gram_total - total) / max(abs(total), 1e-30)
        cross_errors: list[float] = []
        for i, j in _sample_indices(int(x.shape[1]), x.device):
            atom_x = ex[:, i : i + 1].double() * ww[:, i].double().reshape(1, -1)
            atom_w = qxw[:, j : j + 1].double() * ew[:, j].double().reshape(1, -1)
            expected = (atom_x * atom_w).sum()
            actual = (ex[:, i].double() @ qxw[:, j].double()) * (
                ww[:, i].double() @ ew[:, j].double()
            )
            # Normalize by Cauchy-Schwarz scale so nearly orthogonal pairs do
            # not turn roundoff on a near-zero inner product into a large and
            # misleading relative error.
            scale = max(
                float(atom_x.square().sum().sqrt().item())
                * float(atom_w.square().sum().sqrt().item()),
                1e-30,
            )
            cross_errors.append(
                abs(float((actual - expected).item())) / scale
            )
        diagnostics = {
            "exact_decomposition_max_relative_error": exact_decomposition_error,
            "cross_gram_max_sampled_relative_error": max(cross_errors, default=0.0),
            "pair_gram_max_relative_error": pair_identity_error,
            "pair_total_norm_relative_error": total_norm_error,
            "joint_total_norm_relative_error": total_norm_error,
            "sampled_cross_pairs": _sample_indices(int(x.shape[1]), x.device),
        }

    return DualSourceGrams(
        gx=gx,
        gw=gw,
        h=h,
        gp=gp,
        l_x=l_x,
        l_w=l_w,
        cross_term=cross,
        total_error=total,
        kappa=cross / denominator,
        gamma=total / denominator,
        diagnostics=diagnostics,
    )


def materialize_joint(grams: DualSourceGrams) -> torch.Tensor:
    """Materialize the 2K x 2K joint Gram for moderate K."""

    return torch.cat(
        (
            torch.cat((grams.gx, grams.h), dim=1),
            torch.cat((grams.h.T, grams.gw), dim=1),
        ),
        dim=0,
    )


__all__ = ["DualSourceGrams", "build_dual_source_grams", "materialize_joint"]
