"""PSD eigendecomposition and Stage-1 coverage metrics."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class EigenAnalysis:
    eigenvalues: torch.Tensor
    eigenvectors: torch.Tensor
    rho_struct_curve: torch.Tensor
    rho_func_curve: torch.Tensor
    effective_rank: float
    effective_rank_ratio: float
    trace_g: float
    one_g_one: float
    raw_min_eigenvalue: float | None
    clamped_eigenvalue_count: int
    method: str = "exact_torch_eigh_fp64"
    relative_residual_max: float = 0.0


@torch.no_grad()
def analyze_gram(
    gram: torch.Tensor,
    *,
    negative_relative_tolerance: float = 1e-9,
    eigen_dtype: torch.dtype = torch.float64,
) -> EigenAnalysis:
    """Analyze a symmetric PSD Gram using ``torch.linalg.eigh``."""

    if gram.ndim != 2 or gram.shape[0] != gram.shape[1] or gram.shape[0] == 0:
        raise ValueError("gram must be a non-empty square matrix")
    symmetric = ((gram + gram.T) * 0.5).to(dtype=eigen_dtype)
    eigenvalues, eigenvectors = torch.linalg.eigh(symmetric)
    raw_min = float(eigenvalues[0].item())
    spectral_scale = max(float(eigenvalues.abs().max().item()), 1e-30)
    if raw_min < -negative_relative_tolerance * spectral_scale:
        raise ValueError(
            f"Gram is materially non-PSD: lambda_min={raw_min:.6e}, scale={spectral_scale:.6e}"
        )
    clamped = int((eigenvalues < 0).sum().item())
    eigenvalues = eigenvalues.clamp_min_(0).flip(0).contiguous()
    eigenvectors = eigenvectors.flip(1).contiguous()

    trace = eigenvalues.sum()
    if not bool((trace > 0).item()):
        raise ValueError("Gram has zero trace")
    rho_struct = eigenvalues.cumsum(0) / trace
    projection = eigenvectors.sum(dim=0)
    modal_energy = eigenvalues * projection.square()
    one_g_one = symmetric.sum()
    modal_total = modal_energy.sum()
    tolerance = max(abs(float(one_g_one.item())) * 1e-8, spectral_scale * 1e-10, 1e-24)
    if abs(float((modal_total - one_g_one).item())) > tolerance:
        raise AssertionError("eigenbasis does not reconstruct one^T G one")
    if not bool((one_g_one > 0).item()):
        raise ValueError("summed functional error has zero energy")
    rho_func = modal_energy.cumsum(0) / one_g_one

    probabilities = eigenvalues[eigenvalues > 0] / trace
    effective_rank = float(torch.exp(-(probabilities * probabilities.log()).sum()).item())
    return EigenAnalysis(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        rho_struct_curve=rho_struct,
        rho_func_curve=rho_func,
        effective_rank=effective_rank,
        effective_rank_ratio=effective_rank / gram.shape[0],
        trace_g=float(trace.item()),
        one_g_one=float(one_g_one.item()),
        raw_min_eigenvalue=raw_min,
        clamped_eigenvalue_count=clamped,
        method="exact_torch_eigh_fp64",
        relative_residual_max=0.0,
    )


@torch.no_grad()
def analyze_gram_randomized(
    gram: torch.Tensor,
    *,
    max_rank: int,
    oversample: int = 16,
    power_iterations: int = 1,
    seed: int = 0,
) -> EigenAnalysis:
    """Deterministic top-r subspace iteration for large dense Grams.

    Trace and ``1^T G 1`` remain exact reductions of the materialized Gram.
    Entropy effective rank and the minimum eigenvalue require the full spectrum
    and are therefore reported as unavailable for this scalable path.
    """

    if gram.ndim != 2 or gram.shape[0] != gram.shape[1] or gram.shape[0] == 0:
        raise ValueError("gram must be a non-empty square matrix")
    if max_rank <= 0:
        raise ValueError("max_rank must be positive")
    dimension = int(gram.shape[0])
    limit = min(max_rank, dimension)
    width = min(dimension, limit + max(0, oversample))
    generator = torch.Generator(device="cpu").manual_seed(int(seed) % (2**63 - 1))
    q = torch.randn((dimension, width), generator=generator, dtype=torch.float32).to(
        gram.device
    )
    q, _ = torch.linalg.qr(q, mode="reduced")
    work = ((gram + gram.T) * 0.5).float()
    for _ in range(power_iterations + 1):
        q, _ = torch.linalg.qr(work @ q, mode="reduced")
    aq = work @ q
    core = (q.T @ aq + aq.T @ q) * 0.5
    values, rotations = torch.linalg.eigh(core.double())
    values = values.flip(0)
    rotations = rotations.flip(1)
    negative = int((values < 0).sum().item())
    values = values[:limit].clamp_min_(0)
    rotations = rotations[:, :limit].to(q.dtype)
    vectors = q @ rotations
    trace = float(torch.diagonal(work).double().sum().item())
    one_g_one = float(work.double().sum().item())
    if trace <= 0 or one_g_one <= 0:
        raise ValueError("Gram trace and functional energy must be positive")
    modal = values * vectors.double().sum(dim=0).square()
    rho_struct = values.cumsum(0) / trace
    rho_func = modal.cumsum(0) / one_g_one
    residual = aq @ rotations - vectors * values.float().reshape(1, -1)
    spectral_scale = max(float(values[0].item()), 1e-30)
    # Near-null Ritz values otherwise turn harmless FP32 roundoff into an
    # arbitrarily large relative residual.  Use a scale-relative floor while
    # retaining the ordinary |Av-lambda v|/|lambda| diagnostic for meaningful
    # modes.
    denominator = values.abs().clamp_min(spectral_scale * 1e-6)
    relative = residual.norm(dim=0).double() / denominator
    return EigenAnalysis(
        eigenvalues=values,
        eigenvectors=vectors,
        rho_struct_curve=rho_struct,
        rho_func_curve=rho_func,
        effective_rank=float("nan"),
        effective_rank_ratio=float("nan"),
        trace_g=trace,
        one_g_one=one_g_one,
        raw_min_eigenvalue=None,
        clamped_eigenvalue_count=negative,
        method=f"randomized_subspace_fp32_oversample{oversample}_power{power_iterations}",
        relative_residual_max=float(relative.max().item()),
    )
