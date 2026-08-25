"""Audited NVFP4 requantization wrapper used by every shaping candidate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from functional_gram.quantization import (
    compute_global_scale,
    quantize_matrix_chunked,
)


@dataclass
class RequantizationAudit:
    calls: int = 0
    nonidentity_calls: int = 0

    def record(self, *, identity: bool) -> None:
        self.calls += 1
        if not identity:
            self.nonidentity_calls += 1


@torch.no_grad()
def requantize_matrix(
    value: torch.Tensor,
    *,
    row_chunk: int,
    audit: RequantizationAudit,
    identity: bool,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Recompute the tensor-global scale and call the repository reference."""

    scale = compute_global_scale(value)
    quantized, diagnostics = quantize_matrix_chunked(
        value,
        global_scale=scale,
        row_chunk=row_chunk,
        output_dtype=torch.float32,
    )
    audit.record(identity=identity)
    diagnostics = dict(diagnostics)
    diagnostics["stage1_5_requantized"] = True
    diagnostics["identity_candidate"] = identity
    return quantized, diagnostics


__all__ = ["RequantizationAudit", "requantize_matrix"]
