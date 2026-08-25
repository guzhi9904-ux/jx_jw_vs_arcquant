from __future__ import annotations

import unittest

import torch

from factor_rank_spectral_outlier.functional_factors import build_functional_factor_grams


class HadamardIdentityTest(unittest.TestCase):
    def setUp(self) -> None:
        generator = torch.Generator().manual_seed(17)
        self.x = torch.randn((9, 6), generator=generator, dtype=torch.float64)
        self.qx = self.x + 0.05 * torch.randn((9, 6), generator=generator, dtype=torch.float64)
        self.weight = torch.randn((7, 6), generator=generator, dtype=torch.float64)
        self.qweight = self.weight + 0.04 * torch.randn((7, 6), generator=generator, dtype=torch.float64)

    def test_x_source(self) -> None:
        built = build_functional_factor_grams(self.x, self.qx, self.weight, self.qweight, source="X")
        expected = ((self.x - self.qx).T @ (self.x - self.qx)) * (self.weight.T @ self.weight)
        self.assertTrue(torch.allclose(built.gram, expected, atol=1e-12, rtol=1e-12))

    def test_w_source_uses_quantized_activation(self) -> None:
        built = build_functional_factor_grams(self.x, self.qx, self.weight, self.qweight, source="W")
        ew = self.weight - self.qweight
        expected = (self.qx.T @ self.qx) * (ew.T @ ew)
        wrong = (self.x.T @ self.x) * (ew.T @ ew)
        self.assertTrue(torch.allclose(built.gram, expected, atol=1e-12, rtol=1e-12))
        self.assertFalse(torch.allclose(built.gram, wrong, atol=1e-12, rtol=1e-12))


if __name__ == "__main__":
    unittest.main()
