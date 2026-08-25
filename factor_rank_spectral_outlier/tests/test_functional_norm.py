from __future__ import annotations

import unittest

import torch

from factor_rank_spectral_outlier.functional_factors import build_functional_factor_grams


class FunctionalNormTest(unittest.TestCase):
    def test_both_sources_match_output_space_norm(self) -> None:
        generator = torch.Generator().manual_seed(23)
        x = torch.randn((11, 8), generator=generator, dtype=torch.float64)
        qx = x + 0.07 * torch.randn((11, 8), generator=generator, dtype=torch.float64)
        weight = torch.randn((5, 8), generator=generator, dtype=torch.float64)
        qweight = weight + 0.03 * torch.randn((5, 8), generator=generator, dtype=torch.float64)
        for source in ("X", "W"):
            built = build_functional_factor_grams(x, qx, weight, qweight, source=source)
            if source == "X":
                direct = (x - qx) @ weight.T
            else:
                direct = qx @ (weight - qweight).T
            self.assertAlmostEqual(float(built.gram.sum()), float(direct.square().sum()), places=10)
            self.assertEqual(built.diagnostics["status"], "passed")


if __name__ == "__main__":
    unittest.main()
