"""Exact and matrix-free randomized spectral analysis for Stage 1.5."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from .exact_dual_source import DualSourceGrams, materialize_joint


Matmul = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class Spectrum:
    eigenvalues: torch.Tensor
    top_eigenvectors: torch.Tensor
    rho_struct_curve: torch.Tensor
    rho_func_curve: torch.Tensor
    trace: float
    functional_energy: float
    raw_min_eigenvalue: float | None
    method: str
    relative_residual_max: float
    negative_ritz_count: int

    def value(self, rank: int, *, functional: bool = True) -> float:
        curve = self.rho_func_curve if functional else self.rho_struct_curve
        if rank <= 0 or rank > curve.numel():
            raise ValueError(f"rank {rank} is unavailable")
        return float(curve[rank - 1].item())


def _seeded_random(
    rows: int, columns: int, *, seed: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    # Generate on CPU so results do not depend on CUDA RNG implementation.
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) % (2**63 - 1))
    return torch.randn((rows, columns), generator=generator, dtype=dtype).to(device)


@torch.no_grad()
def _exact_spectrum(
    matrix: torch.Tensor,
    *,
    functional_vector: torch.Tensor,
    max_rank: int,
    device: torch.device,
) -> Spectrum:
    work = ((matrix + matrix.T) * 0.5).to(device=device, dtype=torch.float64)
    eigenvalues, eigenvectors = torch.linalg.eigh(work)
    raw_min = float(eigenvalues[0].item())
    spectral_scale = max(float(eigenvalues.abs().max().item()), 1e-30)
    # FP32 Gram construction may leave cancellation-level negative modes.  No
    # materially negative mode is silently clamped.
    if raw_min < -2e-5 * spectral_scale:
        raise ValueError(
            f"materially non-PSD Gram: lambda_min={raw_min:.6e}, scale={spectral_scale:.6e}"
        )
    eigenvalues = eigenvalues.flip(0).clamp_min_(0)
    eigenvectors = eigenvectors.flip(1)
    limit = min(max_rank, matrix.shape[0])
    values = eigenvalues[:limit].contiguous()
    vectors = eigenvectors[:, :limit].contiguous()
    trace = float(eigenvalues.sum().item())
    vector = functional_vector.to(device=device, dtype=torch.float64)
    functional_energy = float((vector @ work @ vector).item())
    modal = values * (vectors.T @ vector).square()
    rho_struct = values.cumsum(0) / max(trace, 1e-30)
    rho_func = modal.cumsum(0) / max(functional_energy, 1e-30)
    residual = work @ vectors - vectors * values.reshape(1, -1)
    relative = residual.norm(dim=0) / (values.abs() + spectral_scale * 1e-12)
    return Spectrum(
        eigenvalues=values.cpu(),
        top_eigenvectors=vectors.float().cpu(),
        rho_struct_curve=rho_struct.cpu(),
        rho_func_curve=rho_func.cpu(),
        trace=trace,
        functional_energy=functional_energy,
        raw_min_eigenvalue=raw_min,
        method=f"exact_torch_eigh_{device.type}_fp64",
        relative_residual_max=float(relative.max().item()),
        negative_ritz_count=0,
    )


@torch.no_grad()
def _randomized_spectrum(
    *,
    dimension: int,
    matmul: Matmul,
    trace: float,
    functional_energy: float,
    functional_vector: torch.Tensor,
    max_rank: int,
    oversample: int,
    power_iterations: int,
    seed: int,
    device: torch.device,
) -> Spectrum:
    limit = min(max_rank, dimension)
    width = min(dimension, limit + max(0, oversample))
    q = _seeded_random(dimension, width, seed=seed, device=device, dtype=torch.float32)
    q, _ = torch.linalg.qr(q, mode="reduced")
    # One range-forming product is always performed; power_iterations adds
    # conventional subspace iterations.
    for _ in range(power_iterations + 1):
        q, _ = torch.linalg.qr(matmul(q), mode="reduced")
    aq = matmul(q)
    core = (q.T @ aq + aq.T @ q) * 0.5
    values, rotations = torch.linalg.eigh(core.double())
    values = values.flip(0)
    rotations = rotations.flip(1)
    negative_count = int((values < 0).sum().item())
    values = values[:limit].clamp_min_(0)
    rotations = rotations[:, :limit].to(dtype=q.dtype)
    vectors = q @ rotations
    vector = functional_vector.to(device=device, dtype=vectors.dtype)
    modal = values * (vectors.T @ vector).double().square()
    rho_struct = values.cumsum(0) / max(float(trace), 1e-30)
    rho_func = modal.cumsum(0) / max(float(functional_energy), 1e-30)
    residual = aq @ rotations - vectors * values.to(vectors.dtype).reshape(1, -1)
    scale = max(float(values[0].item()), 1e-30)
    relative = residual.norm(dim=0).double() / (values.abs() + scale * 1e-8)
    return Spectrum(
        eigenvalues=values.cpu(),
        top_eigenvectors=vectors.cpu(),
        rho_struct_curve=rho_struct.cpu(),
        rho_func_curve=rho_func.cpu(),
        trace=float(trace),
        functional_energy=float(functional_energy),
        raw_min_eigenvalue=None,
        method=(
            f"randomized_subspace_fp32_oversample{oversample}_power{power_iterations}"
        ),
        relative_residual_max=float(relative.max().item()),
        negative_ritz_count=negative_count,
    )


def _matrix_functional_energy(matrix: torch.Tensor) -> float:
    return float(matrix.double().sum().item())


@torch.no_grad()
def analyze_matrix(
    matrix: torch.Tensor,
    *,
    max_rank: int,
    exact: bool,
    device: torch.device,
    oversample: int,
    power_iterations: int,
    seed: int,
) -> Spectrum:
    dimension = int(matrix.shape[0])
    if matrix.ndim != 2 or matrix.shape[1] != dimension:
        raise ValueError("matrix must be square")
    vector = torch.ones(dimension)
    if exact:
        return _exact_spectrum(
            matrix,
            functional_vector=vector,
            max_rank=max_rank,
            device=device,
        )
    resident = matrix.to(device=device, dtype=torch.float32)
    trace = float(torch.diagonal(matrix).double().sum().item())
    energy = _matrix_functional_energy(matrix)
    return _randomized_spectrum(
        dimension=dimension,
        matmul=lambda value: resident @ value,
        trace=trace,
        functional_energy=energy,
        functional_vector=vector,
        max_rank=max_rank,
        oversample=oversample,
        power_iterations=power_iterations,
        seed=seed,
        device=device,
    )


@torch.no_grad()
def analyze_joint(
    grams: DualSourceGrams,
    *,
    max_rank: int,
    exact: bool,
    device: torch.device,
    oversample: int,
    power_iterations: int,
    seed: int,
) -> Spectrum:
    k = int(grams.gx.shape[0])
    vector = torch.ones(2 * k)
    if exact:
        return _exact_spectrum(
            materialize_joint(grams),
            functional_vector=vector,
            max_rank=max_rank,
            device=device,
        )
    gx = grams.gx.to(device=device, dtype=torch.float32)
    gw = grams.gw.to(device=device, dtype=torch.float32)
    h = grams.h.to(device=device, dtype=torch.float32)

    def joint_matmul(value: torch.Tensor) -> torch.Tensor:
        top, bottom = value[:k], value[k:]
        return torch.cat((gx @ top + h @ bottom, h.T @ top + gw @ bottom), dim=0)

    trace = float(torch.diagonal(grams.gx).double().sum().item()) + float(
        torch.diagonal(grams.gw).double().sum().item()
    )
    return _randomized_spectrum(
        dimension=2 * k,
        matmul=joint_matmul,
        trace=trace,
        functional_energy=float(grams.total_error),
        functional_vector=vector,
        max_rank=max_rank,
        oversample=oversample,
        power_iterations=power_iterations,
        seed=seed,
        device=device,
    )


@torch.no_grad()
def analyze_all(
    grams: DualSourceGrams,
    *,
    max_rank: int,
    exact: bool,
    device: torch.device,
    oversample: int,
    power_iterations: int,
    seed: int,
) -> dict[str, Spectrum]:
    result: dict[str, Spectrum] = {}
    for offset, (name, matrix) in enumerate(
        (("x", grams.gx), ("w", grams.gw), ("pair", grams.gp))
    ):
        result[name] = analyze_matrix(
            matrix,
            max_rank=max_rank,
            exact=exact,
            device=device,
            oversample=oversample,
            power_iterations=power_iterations,
            seed=seed + offset * 1009,
        )
    result["joint"] = analyze_joint(
        grams,
        max_rank=max_rank,
        exact=exact,
        device=device,
        oversample=oversample,
        power_iterations=power_iterations,
        seed=seed + 4001,
    )
    return result


@torch.no_grad()
def projector_overlap(left: torch.Tensor, right: torch.Tensor, rank: int) -> float:
    if left.shape[0] != right.shape[0] or rank > min(left.shape[1], right.shape[1]):
        raise ValueError("incompatible eigenspaces or rank")
    cross = left[:, :rank].double().T @ right[:, :rank].double()
    return float(cross.square().sum().item() / rank)


def spectrum_payload(spectrum: Spectrum, *, include_vectors: bool = False) -> dict[str, object]:
    result: dict[str, object] = {
        "eigenvalues": spectrum.eigenvalues,
        "rho_struct_curve": spectrum.rho_struct_curve,
        "rho_func_curve": spectrum.rho_func_curve,
        "trace": spectrum.trace,
        "functional_energy": spectrum.functional_energy,
        "raw_min_eigenvalue": spectrum.raw_min_eigenvalue,
        "method": spectrum.method,
        "relative_residual_max": spectrum.relative_residual_max,
        "negative_ritz_count": spectrum.negative_ritz_count,
    }
    if include_vectors:
        result["top_eigenvectors"] = spectrum.top_eigenvectors
    return result


__all__ = [
    "Spectrum",
    "analyze_all",
    "analyze_joint",
    "analyze_matrix",
    "projector_overlap",
    "spectrum_payload",
]
