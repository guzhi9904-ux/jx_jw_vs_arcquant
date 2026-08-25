from __future__ import annotations

import torch

from functional_gram.quantization import compute_global_scale, quantize_matrix_chunked


def make_operands() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(17)
    x = torch.randn(11, 19, generator=generator)
    weight = torch.randn(13, 19, generator=generator)
    qx, _ = quantize_matrix_chunked(x, global_scale=compute_global_scale(x), row_chunk=4)
    qw, _ = quantize_matrix_chunked(
        weight, global_scale=compute_global_scale(weight), row_chunk=4
    )
    return x, qx, weight, qw
