from __future__ import annotations

import unittest

import torch

from dual_source_spectral_shaping.exact_dual_source import (
    build_dual_source_grams,
    materialize_joint,
)
from dual_source_spectral_shaping.tests.common import make_operands


class TotalNormTest(unittest.TestCase):
    def test_pair_and_joint_total_norm(self) -> None:
        x, qx, weight, qw = make_operands()
        grams = build_dual_source_grams(
            x, qx, weight, qw, accumulation_dtype=torch.float64
        )
        direct = x.double() @ weight.double().T - qx.double() @ qw.double().T
        expected = direct.square().sum()
        self.assertTrue(torch.allclose(grams.gp.sum(), expected, rtol=1e-10, atol=1e-10))
        self.assertTrue(
            torch.allclose(materialize_joint(grams).sum(), expected, rtol=1e-10, atol=1e-10)
        )
