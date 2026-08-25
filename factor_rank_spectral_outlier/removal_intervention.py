"""Principal-submatrix interventions for spectral-outlier channels."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .factor_spectrum import SpectrumAnalysis, analyze_psd, coverage_at


@dataclass(frozen=True)
class RemovalResult:
    remove_budget: int
    remaining_indices: torch.Tensor
    analysis: SpectrumAnalysis
    remaining_trace: float
    remaining_total_functional_error: float
    rho_struct_64: float
    rho_struct_128: float
    rho_struct_256: float
    rho_func_64: float
    rho_func_128: float
    rho_func_256: float


@torch.no_grad()
def run_removal(
    gram: torch.Tensor,
    order: torch.Tensor,
    *,
    remove_budget: int,
    exact_max_dimension: int,
    seed: int,
    oversample: int,
    power_iterations: int,
    slq_probes: int,
    slq_steps: int,
    raw_analysis: SpectrumAnalysis | None = None,
) -> RemovalResult:
    dimension = int(gram.shape[0])
    budget = min(max(int(remove_budget), 0), max(dimension - 2, 0))
    mask = torch.ones(dimension, dtype=torch.bool)
    if budget:
        mask[order[:budget].cpu().long()] = False
    indices = torch.nonzero(mask, as_tuple=False).flatten()
    if budget == 0 and raw_analysis is not None:
        analysis = raw_analysis
    else:
        work_indices = indices.to(gram.device)
        remaining = gram.index_select(0, work_indices).index_select(1, work_indices)
        analysis = analyze_psd(
            remaining,
            max_rank=min(256, remaining.shape[0]),
            exact=remaining.shape[0] <= exact_max_dimension,
            seed=seed,
            oversample=oversample,
            power_iterations=power_iterations,
            slq_probes=slq_probes,
            slq_steps=slq_steps,
        )
        del remaining
    return RemovalResult(
        remove_budget=budget,
        remaining_indices=indices,
        analysis=analysis,
        remaining_trace=analysis.trace,
        remaining_total_functional_error=float(analysis.one_g_one or 0.0),
        rho_struct_64=coverage_at(analysis, 64),
        rho_struct_128=coverage_at(analysis, 128),
        rho_struct_256=coverage_at(analysis, 256),
        rho_func_64=coverage_at(analysis, 64, functional=True),
        rho_func_128=coverage_at(analysis, 128, functional=True),
        rho_func_256=coverage_at(analysis, 256, functional=True),
    )


@torch.no_grad()
def removal_norm_audit(
    factor1: torch.Tensor,
    factor2: torch.Tensor,
    gram: torch.Tensor,
    remaining_indices: torch.Tensor,
) -> float:
    indices = remaining_indices.to(factor1.device)
    direct = factor1.index_select(1, indices).matmul(factor2.index_select(1, indices).T)
    work_indices = remaining_indices.to(gram.device)
    gram_total = gram.index_select(0, work_indices).index_select(1, work_indices).double().sum()
    direct_total = direct.double().square().sum()
    return float((gram_total - direct_total).abs().item()) / max(
        float(direct_total.abs().item()), 1e-30
    )


__all__ = ["RemovalResult", "removal_norm_audit", "run_removal"]
