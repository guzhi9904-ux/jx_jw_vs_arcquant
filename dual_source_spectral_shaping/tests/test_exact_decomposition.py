from __future__ import annotations

import unittest

import torch

from dual_source_spectral_shaping.tests.common import make_operands


class ExactDecompositionTest(unittest.TestCase):
    def test_exact_decomposition(self) -> None:
        x, qx, weight, qw = (value.double() for value in make_operands())
        direct = x @ weight.T - qx @ qw.T
        decomposed = (x - qx) @ weight.T + qx @ (weight - qw).T
        self.assertTrue(torch.allclose(direct, decomposed, rtol=1e-11, atol=1e-11))
