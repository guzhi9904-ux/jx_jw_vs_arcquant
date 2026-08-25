"""Bounded SmoothQuant-style function-preserving diagonal scaling."""

from __future__ import annotations

import torch


@torch.no_grad()
def smoothquant_scale(
    x_split_a: torch.Tensor,
    weight: torch.Tensor,
    alpha: float,
    *,
    epsilon: float = 1e-8,
    minimum: float = 0.25,
    maximum: float = 4.0,
) -> torch.Tensor:
    if x_split_a.ndim != 2 or weight.ndim != 2:
        raise ValueError("x_split_a and weight must be matrices")
    if x_split_a.shape[1] != weight.shape[1]:
        raise ValueError("activation and weight K dimensions differ")
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if not 0 < minimum <= maximum:
        raise ValueError("invalid scale bounds")
    a = x_split_a.float().abs().amax(dim=0)
    b = weight.float().abs().amax(dim=0)
    log_d = float(alpha) * torch.log(a + epsilon) - (1.0 - float(alpha)) * torch.log(
        b + epsilon
    )
    log_d = log_d - log_d.mean()
    return torch.exp(log_d).clamp_(minimum, maximum)


@torch.no_grad()
def apply_reparameterization(
    x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if scale.ndim != 1 or x.shape[1] != scale.numel() or weight.shape[1] != scale.numel():
        raise ValueError("scale must have one value per input channel")
    d = scale.to(device=x.device, dtype=torch.float32)
    return x.float() / d, weight.to(device=x.device, dtype=torch.float32) * d


__all__ = ["apply_reparameterization", "smoothquant_scale"]
