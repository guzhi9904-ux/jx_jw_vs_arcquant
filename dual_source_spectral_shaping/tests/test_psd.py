from __future__ import annotations

import unittest

import torch

from dual_source_spectral_shaping.exact_dual_source import (
    build_dual_source_grams,
    materialize_joint,
)
from dual_source_spectral_shaping.tests.common import make_operands


class PsdTest(unittest.TestCase):
    def test_all_grams_are_psd(self) -> None:
        grams = build_dual_source_grams(*make_operands(), accumulation_dtype=torch.float64)
        for matrix in (grams.gx, grams.gw, grams.gp, materialize_joint(grams)):
            minimum = torch.linalg.eigvalsh((matrix + matrix.T) * 0.5).min()
            scale = torch.linalg.matrix_norm(matrix, ord=2)
            self.assertGreaterEqual(float(minimum), -1e-10 * max(float(scale), 1.0))
