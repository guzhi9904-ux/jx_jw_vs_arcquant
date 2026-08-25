from __future__ import annotations

import unittest

import torch

from functional_gram.eig_analysis import analyze_gram, analyze_gram_randomized


class RandomizedEigenTests(unittest.TestCase):
    def test_top_rank_coverage_tracks_exact_solver(self) -> None:
        generator = torch.Generator().manual_seed(101)
        factors = torch.randn(96, 24, generator=generator, dtype=torch.float64)
        factors *= torch.linspace(2.0, 0.2, 24, dtype=torch.float64)
        gram = factors @ factors.T + torch.eye(96, dtype=torch.float64) * 1e-3
        exact = analyze_gram(gram)
        approximate = analyze_gram_randomized(
            gram.float(), max_rank=32, oversample=16, power_iterations=2, seed=7
        )
        self.assertIn("randomized_subspace", approximate.method)
        self.assertLess(approximate.relative_residual_max, 0.05)
        self.assertAlmostEqual(
            float(approximate.rho_struct_curve[15]),
            float(exact.rho_struct_curve[15]),
            delta=0.015,
        )
        self.assertAlmostEqual(
            float(approximate.rho_func_curve[15]),
            float(exact.rho_func_curve[15]),
            delta=0.025,
        )


if __name__ == "__main__":
    unittest.main()
