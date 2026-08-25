from __future__ import annotations

import unittest

import torch

from dual_source_spectral_shaping.exact_dual_source import build_dual_source_grams
from dual_source_spectral_shaping.tests.common import make_operands


class CrossGramTest(unittest.TestCase):
    def test_cross_gram_atoms(self) -> None:
        x, qx, weight, qw = make_operands()
        grams = build_dual_source_grams(
            x, qx, weight, qw, accumulation_dtype=torch.float64
        )
        i, j = 3, 14
        cx = (x[:, i] - qx[:, i]).double().reshape(-1, 1) * weight[:, i].double()
        cw = qx[:, j].double().reshape(-1, 1) * (weight[:, j] - qw[:, j]).double()
        self.assertTrue(
            torch.allclose(grams.h[i, j], (cx * cw).sum(), rtol=1e-11, atol=1e-11)
        )
