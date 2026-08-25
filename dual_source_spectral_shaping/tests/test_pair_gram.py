from __future__ import annotations

import unittest

import torch

from dual_source_spectral_shaping.exact_dual_source import build_dual_source_grams
from dual_source_spectral_shaping.tests.common import make_operands


class PairGramTest(unittest.TestCase):
    def test_pair_gram_expansion(self) -> None:
        grams = build_dual_source_grams(*make_operands(), accumulation_dtype=torch.float64)
        expected = grams.gx + grams.gw + grams.h + grams.h.T
        self.assertTrue(torch.allclose(grams.gp, expected, rtol=1e-11, atol=1e-11))
