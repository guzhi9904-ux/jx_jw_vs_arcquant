"""Channel-level spectral-tail scores and score-comparison diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from scipy.stats import rankdata, spearmanr

from .factor_spectrum import SpectrumAnalysis


@dataclass(frozen=True)
class SpectralOutlierScores:
    diag_j: torch.Tensor
    tail_energy_h: torch.Tensor
    tail_ratio_h: torch.Tensor
    rank_j: torch.Tensor
    rank_h: torch.Tensor
    rank_tail_ratio: torch.Tensor
    order_j: torch.Tensor
    order_h: torch.Tensor
    order_tail_ratio: torch.Tensor
    diagnostics: dict[str, Any]


def _descending_rank(value: torch.Tensor) -> torch.Tensor:
    ranked = rankdata(-value.double().cpu().numpy(), method="average")
    return torch.from_numpy(np.asarray(ranked, dtype=np.float64))


@torch.no_grad()
def compute_spectral_outlier_scores(
    gram: torch.Tensor,
    analysis: SpectrumAnalysis,
    *,
    bulk_rank: int = 256,
    epsilon: float = 1e-30,
) -> SpectralOutlierScores:
    """Compute absolute and normalized energy outside the top-r bulk."""

    if gram.ndim != 2 or gram.shape[0] != gram.shape[1]:
        raise ValueError("gram must be square")
    usable = min(bulk_rank, analysis.eigenvalues.numel(), analysis.eigenvectors.shape[1])
    values = analysis.eigenvalues[:usable].double()
    vectors = analysis.eigenvectors[:, :usable].double()
    diag = gram.diagonal().detach().double().cpu()
    represented = vectors.square().matmul(values)
    scale = max(float(diag.max().item()), 1e-30)
    raw_tail = diag - represented
    minimum = float(raw_tail.min().item())
    if minimum < -2e-6 * scale:
        raise AssertionError(f"tail energy is materially negative: {minimum:.6e}")
    tail = raw_tail.clamp_min(0)
    ratio = tail / diag.clamp_min(epsilon)
    reconstructed = represented + tail
    identity_error = float((reconstructed - diag).abs().max().item()) / scale
    order_j = torch.argsort(diag, descending=True, stable=True)
    order_h = torch.argsort(tail, descending=True, stable=True)
    order_ratio = torch.argsort(ratio, descending=True, stable=True)

    j_np = diag.numpy()
    h_np = tail.numpy()
    pearson = float(np.corrcoef(j_np, h_np)[0, 1]) if np.std(h_np) > 0 else float("nan")
    spearman = float(spearmanr(j_np, h_np).statistic) if np.std(h_np) > 0 else float("nan")
    overlaps: dict[str, float] = {}
    for budget in (32, 64, 128, 256):
        use = min(budget, diag.numel())
        overlap = len(set(order_j[:use].tolist()) & set(order_h[:use].tolist())) / use
        overlaps[str(budget)] = overlap
    return SpectralOutlierScores(
        diag_j=diag,
        tail_energy_h=tail,
        tail_ratio_h=ratio,
        rank_j=_descending_rank(diag),
        rank_h=_descending_rank(tail),
        rank_tail_ratio=_descending_rank(ratio),
        order_j=order_j,
        order_h=order_h,
        order_tail_ratio=order_ratio,
        diagnostics={
            "bulk_rank": usable,
            "tail_energy_identity_relative_error": identity_error,
            "raw_minimum_tail_energy": minimum,
            "pearson_J_h": pearson,
            "spearman_J_h": spearman,
            "top_k_overlap_J_h": overlaps,
        },
    )


def support_jaccard(first_order: torch.Tensor, second_order: torch.Tensor, budget: int) -> float:
    use = min(int(budget), first_order.numel(), second_order.numel())
    first = set(first_order[:use].tolist())
    second = set(second_order[:use].tolist())
    union = first | second
    return len(first & second) / len(union) if union else 1.0


__all__ = [
    "SpectralOutlierScores",
    "compute_spectral_outlier_scores",
    "support_jaccard",
]
