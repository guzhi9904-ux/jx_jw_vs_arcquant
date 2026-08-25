"""Calibration split-to-split subspace stability."""

from __future__ import annotations

import torch


@torch.no_grad()
def projector_overlap_curve(
    eigenvectors_a: torch.Tensor,
    eigenvectors_b: torch.Tensor,
    *,
    max_rank: int | None = None,
) -> torch.Tensor:
    """Return overlap(r) for every r from 1 through ``max_rank``."""

    if eigenvectors_a.ndim != 2 or eigenvectors_b.ndim != 2:
        raise ValueError("eigenvector inputs must be matrices")
    if eigenvectors_a.shape[0] != eigenvectors_b.shape[0]:
        raise ValueError("eigenvector ambient dimensions differ")
    limit = min(eigenvectors_a.shape[1], eigenvectors_b.shape[1])
    if max_rank is not None:
        limit = min(limit, max_rank)
    if limit <= 0:
        raise ValueError("max_rank must be positive")
    a = eigenvectors_a[:, :limit]
    b = eigenvectors_b[:, :limit]
    cross_sq = a.T.matmul(b).square()
    block_sums = cross_sq.cumsum(0).cumsum(1).diagonal()
    ranks = torch.arange(1, limit + 1, device=block_sums.device, dtype=block_sums.dtype)
    overlap = block_sums / ranks
    tolerance = 1e-8 if overlap.dtype == torch.float64 else 2e-5
    if bool(((overlap < -tolerance) | (overlap > 1 + tolerance)).any().item()):
        raise AssertionError("projector overlap left [0, 1]")
    return overlap.clamp_(0, 1)
