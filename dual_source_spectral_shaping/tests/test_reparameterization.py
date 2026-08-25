from __future__ import annotations

import unittest

import torch

from dual_source_spectral_shaping.smooth_scaling import (
    apply_reparameterization,
    smoothquant_scale,
)
from dual_source_spectral_shaping.tests.common import make_operands


class ReparameterizationTest(unittest.TestCase):
    def test_reparameterization_preserves_function(self) -> None:
        x, _, weight, _ = make_operands()
        scale = smoothquant_scale(x, weight, 0.5)
        scaled_x, scaled_weight = apply_reparameterization(x, weight, scale)
        self.assertGreaterEqual(float(scale.min()), 0.25)
        self.assertLessEqual(float(scale.max()), 4.0)
        self.assertTrue(
            torch.allclose(x @ weight.T, scaled_x @ scaled_weight.T, rtol=2e-6, atol=2e-6)
        )
