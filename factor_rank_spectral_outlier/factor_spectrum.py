"""PSD spectrum utilities for exact and scalable Stage 1.6 analysis."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class SpectrumAnalysis:
    eigenvalues: torch.Tensor
    eigenvectors: torch.Tensor
    dimension: int
    trace: float
    rho_struct_curve: torch.Tensor
    rho_func_curve: torch.Tensor | None
    entropy_effective_rank: float
    entropy_effective_rank_ratio: float
    participation_rank: float
    participation_rank_ratio: float
    one_g_one: float | None
    raw_min_eigenvalue: float | None
    clamped_eigenvalue_count: int
    method: str
    entropy_method: str
    relative_residual_max: float


def _validate_matrix(value: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.numel() == 0:
        raise ValueError(f"{name} must be a non-empty matrix")
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} contains non-finite values")
    return value


def _ranks_from_eigenvalues(eigenvalues: torch.Tensor, dimension: int) -> tuple[float, float]:
    values = eigenvalues.double().clamp_min(0)
    total = values.sum()
    if not bool((total > 0).item()):
        raise ValueError("PSD matrix has zero trace")
    probabilities = values[values > 0] / total
    entropy_rank = float(torch.exp(-(probabilities * probabilities.log()).sum()).item())
    participation = float((total.square() / values.square().sum()).item())
    return entropy_rank, participation


@torch.no_grad()
def analyze_factor(
    factor: torch.Tensor,
    *,
    max_vectors: int = 256,
    compute_dtype: torch.dtype = torch.float64,
) -> SpectrumAnalysis:
    """Analyze ``factor.T @ factor`` via its thinner SVD.

    This avoids materializing a large K x K factor covariance for wide MLP
    projections while retaining its complete non-zero spectrum.
    """

    factor = _validate_matrix(factor, "factor")
    if compute_dtype not in (torch.float32, torch.float64):
        raise ValueError("compute_dtype must be float32 or float64")
    work = factor.to(dtype=compute_dtype)
    dimension = int(factor.shape[1])
    if factor.shape[0] < factor.shape[1]:
        covariance = work.matmul(work.T)
        values, left_vectors = torch.linalg.eigh((covariance + covariance.T) * 0.5)
        values = values.clamp_min(0).flip(0).contiguous()
        left_vectors = left_vectors.flip(1).contiguous()
        positive = values > max(float(values[0].item()) * 1e-14, 1e-30)
        eigenvalues = values[positive].double()
        usable = min(max_vectors, int(positive.sum().item()))
        singular = values[:usable].sqrt()
        vectors = work.T.matmul(left_vectors[:, :usable]) / singular.reshape(1, -1)
        vectors = torch.linalg.qr(vectors, mode="reduced")[0]
        method = f"dual_covariance_eigh_{str(compute_dtype).split('.')[-1]}"
    else:
        covariance = work.T.matmul(work)
        values, vectors = torch.linalg.eigh((covariance + covariance.T) * 0.5)
        values = values.clamp_min(0).flip(0).contiguous()
        vectors = vectors.flip(1)[:, : min(max_vectors, dimension)].contiguous()
        eigenvalues = values.double()
        method = f"covariance_eigh_{str(compute_dtype).split('.')[-1]}"
    trace = float(eigenvalues.sum().item())
    entropy_rank, participation = _ranks_from_eigenvalues(eigenvalues, dimension)
    rho = eigenvalues.cumsum(0) / trace
    return SpectrumAnalysis(
        eigenvalues=eigenvalues.cpu(),
        eigenvectors=vectors.cpu().contiguous(),
        dimension=dimension,
        trace=trace,
        rho_struct_curve=rho.cpu(),
        rho_func_curve=None,
        entropy_effective_rank=entropy_rank,
        entropy_effective_rank_ratio=entropy_rank / dimension,
        participation_rank=participation,
        participation_rank_ratio=participation / dimension,
        one_g_one=None,
        raw_min_eigenvalue=0.0 if factor.shape[0] < factor.shape[1] else None,
        clamped_eigenvalue_count=0,
        method=method,
        entropy_method="complete_nonzero_spectrum",
        relative_residual_max=0.0,
    )


def _functional_curve(
    eigenvalues: torch.Tensor,
    eigenvectors: torch.Tensor,
    one_g_one: float,
) -> torch.Tensor:
    modal = eigenvalues.double() * eigenvectors.double().sum(dim=0).square()
    return (modal.cumsum(0) / max(one_g_one, 1e-30)).cpu()


def _slq_entropy_rank(
    matrix: torch.Tensor,
    *,
    trace: float,
    probes: int,
    steps: int,
    seed: int,
) -> float:
    """Estimate entropy effective rank with stochastic Lanczos quadrature."""

    dimension = int(matrix.shape[0])
    device = matrix.device
    work = matrix.float() / max(trace, 1e-30)
    generator = torch.Generator(device="cpu").manual_seed(int(seed) % (2**63 - 1))
    estimates: list[float] = []
    for _ in range(probes):
        z = torch.randint(0, 2, (dimension,), generator=generator, dtype=torch.int8)
        z = z.to(device=device, dtype=torch.float32).mul_(2).sub_(1)
        norm2 = float(z.square().sum().item())
        q = z / math.sqrt(norm2)
        previous = torch.zeros_like(q)
        beta_previous = torch.zeros((), device=device)
        alphas: list[torch.Tensor] = []
        betas: list[torch.Tensor] = []
        basis: list[torch.Tensor] = []
        for iteration in range(min(steps, dimension)):
            basis.append(q)
            value = work @ q
            if iteration:
                value = value - beta_previous * previous
            alpha = q @ value
            value = value - alpha * q
            # Full reorthogonalization keeps the short SLQ recurrence stable.
            for old in basis:
                value = value - (old @ value) * old
            beta = value.norm()
            alphas.append(alpha.double().cpu())
            if iteration + 1 == min(steps, dimension) or float(beta.item()) <= 1e-8:
                break
            betas.append(beta.double().cpu())
            previous, q = q, value / beta
            beta_previous = beta
        diagonal = torch.stack(alphas)
        tridiagonal = torch.diag(diagonal)
        if betas:
            off = torch.stack(betas)
            tridiagonal += torch.diag(off, diagonal=1) + torch.diag(off, diagonal=-1)
        values, vectors = torch.linalg.eigh(tridiagonal)
        values = values.clamp_min(0)
        fvalues = torch.where(values > 0, values * values.log(), torch.zeros_like(values))
        estimates.append(norm2 * float((vectors[0].square() * fvalues).sum().item()))
    trace_p_log_p = sum(estimates) / len(estimates)
    return float(math.exp(-trace_p_log_p))


@torch.no_grad()
def analyze_psd(
    matrix: torch.Tensor,
    *,
    max_rank: int,
    exact: bool,
    seed: int = 0,
    oversample: int = 16,
    power_iterations: int = 1,
    slq_probes: int = 8,
    slq_steps: int = 32,
) -> SpectrumAnalysis:
    """Analyze a materialized PSD matrix.

    Exact mode returns the complete spectrum. Scalable mode returns a
    deterministic randomized top-r spectrum, exact trace/participation rank,
    and an explicitly labelled SLQ estimate of entropy effective rank.
    """

    matrix = _validate_matrix(matrix, "matrix")
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
    symmetric = (matrix + matrix.T) * 0.5
    dimension = int(matrix.shape[0])
    limit = min(max_rank, dimension)
    trace = float(torch.diagonal(symmetric).double().sum().item())
    one_g_one = float(symmetric.double().sum().item())
    if trace <= 0 or one_g_one <= 0:
        raise ValueError("PSD trace and functional energy must be positive")
    frobenius_sq = float(symmetric.double().square().sum().item())
    participation = trace * trace / max(frobenius_sq, 1e-30)

    if exact:
        values, vectors = torch.linalg.eigh(symmetric.double())
        raw_min = float(values[0].item())
        scale = max(float(values.abs().max().item()), 1e-30)
        if raw_min < -2e-8 * scale:
            raise ValueError(f"matrix is materially non-PSD: {raw_min:.6e}")
        clamped = int((values < 0).sum().item())
        values = values.clamp_min(0).flip(0).contiguous()
        vectors = vectors.flip(1).contiguous()
        entropy_rank, participation_spectrum = _ranks_from_eigenvalues(values, dimension)
        if abs(participation - participation_spectrum) > max(1e-6 * participation, 1e-6):
            raise AssertionError("participation rank identities disagree")
        rho = values.cumsum(0) / trace
        rho_func = _functional_curve(values, vectors, one_g_one)
        return SpectrumAnalysis(
            eigenvalues=values.cpu(),
            eigenvectors=vectors.cpu(),
            dimension=dimension,
            trace=trace,
            rho_struct_curve=rho.cpu(),
            rho_func_curve=rho_func,
            entropy_effective_rank=entropy_rank,
            entropy_effective_rank_ratio=entropy_rank / dimension,
            participation_rank=participation,
            participation_rank_ratio=participation / dimension,
            one_g_one=one_g_one,
            raw_min_eigenvalue=raw_min,
            clamped_eigenvalue_count=clamped,
            method="exact_torch_eigh_fp64",
            entropy_method="complete_spectrum",
            relative_residual_max=0.0,
        )

    width = min(dimension, limit + max(0, oversample))
    generator = torch.Generator(device="cpu").manual_seed(int(seed) % (2**63 - 1))
    q = torch.randn((dimension, width), generator=generator, dtype=torch.float32).to(matrix.device)
    q, _ = torch.linalg.qr(q, mode="reduced")
    work = symmetric.float()
    for _ in range(power_iterations + 1):
        q, _ = torch.linalg.qr(work @ q, mode="reduced")
    aq = work @ q
    core = (q.T @ aq + aq.T @ q) * 0.5
    values, rotations = torch.linalg.eigh(core.double())
    values = values.flip(0)[:limit].clamp_min(0)
    rotations = rotations.flip(1)[:, :limit].to(q.dtype)
    vectors = q @ rotations
    residual = aq @ rotations - vectors * values.float().reshape(1, -1)
    scale = max(float(values[0].item()), 1e-30)
    relative = residual.norm(dim=0).double() / values.abs().clamp_min(scale * 1e-6)
    entropy_rank = (
        _slq_entropy_rank(
            work, trace=trace, probes=slq_probes, steps=slq_steps, seed=seed + 97_531
        )
        if slq_probes > 0
        else float("nan")
    )
    return SpectrumAnalysis(
        eigenvalues=values.cpu(),
        eigenvectors=vectors.cpu(),
        dimension=dimension,
        trace=trace,
        rho_struct_curve=(values.cumsum(0) / trace).cpu(),
        rho_func_curve=_functional_curve(values, vectors, one_g_one),
        entropy_effective_rank=entropy_rank,
        entropy_effective_rank_ratio=entropy_rank / dimension,
        participation_rank=participation,
        participation_rank_ratio=participation / dimension,
        one_g_one=one_g_one,
        raw_min_eigenvalue=None,
        clamped_eigenvalue_count=int((values < 0).sum().item()),
        method=f"randomized_subspace_fp32_o{oversample}_p{power_iterations}",
        entropy_method=(
            f"slq_rademacher_p{slq_probes}_s{slq_steps}"
            if slq_probes > 0
            else "not_requested"
        ),
        relative_residual_max=float(relative.max().item()),
    )


def coverage_at(analysis: SpectrumAnalysis, rank: int, *, functional: bool = False) -> float:
    curve = analysis.rho_func_curve if functional else analysis.rho_struct_curve
    if curve is None:
        raise ValueError("functional coverage is unavailable")
    index = min(max(int(rank), 1), curve.numel()) - 1
    return float(curve[index].item())


def projector_overlap(first: SpectrumAnalysis, second: SpectrumAnalysis, rank: int) -> float:
    usable = min(rank, first.eigenvectors.shape[1], second.eigenvectors.shape[1])
    if usable <= 0:
        return float("nan")
    left = first.eigenvectors[:, :usable].double()
    right = second.eigenvectors[:, :usable].double()
    return float(left.T.matmul(right).square().sum().item() / usable)


def lightweight_payload(analysis: SpectrumAnalysis) -> dict[str, Any]:
    return {
        "eigenvalues": analysis.eigenvalues,
        "eigenvectors": analysis.eigenvectors,
        "dimension": analysis.dimension,
        "trace": analysis.trace,
        "rho_struct_curve": analysis.rho_struct_curve,
        "rho_func_curve": analysis.rho_func_curve,
        "entropy_effective_rank": analysis.entropy_effective_rank,
        "entropy_effective_rank_ratio": analysis.entropy_effective_rank_ratio,
        "participation_rank": analysis.participation_rank,
        "participation_rank_ratio": analysis.participation_rank_ratio,
        "one_g_one": analysis.one_g_one,
        "raw_min_eigenvalue": analysis.raw_min_eigenvalue,
        "clamped_eigenvalue_count": analysis.clamped_eigenvalue_count,
        "method": analysis.method,
        "entropy_method": analysis.entropy_method,
        "relative_residual_max": analysis.relative_residual_max,
    }


__all__ = [
    "SpectrumAnalysis",
    "analyze_factor",
    "analyze_psd",
    "coverage_at",
    "lightweight_payload",
    "projector_overlap",
]
