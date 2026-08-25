from __future__ import annotations

import unittest

import torch

from factor_rank_spectral_outlier.factor_spectrum import analyze_factor, analyze_psd


class PsdTest(unittest.TestCase):
    def test_exact_spectrum_and_effective_ranks(self) -> None:
        gram = torch.diag(torch.tensor([4.0, 1.0, 0.0], dtype=torch.float64))
        analysis = analyze_psd(gram, max_rank=3, exact=True)
        self.assertTrue(torch.allclose(analysis.eigenvalues, torch.tensor([4.0, 1.0, 0.0], dtype=torch.float64)))
        self.assertAlmostEqual(analysis.participation_rank, 25.0 / 17.0, places=12)
        self.assertAlmostEqual(float(analysis.rho_struct_curve[0]), 0.8, places=12)

    def test_material_negative_eigenvalue_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "materially non-PSD"):
            analyze_psd(torch.diag(torch.tensor([1.0, -0.1])), max_rank=2, exact=True)

    def test_wide_factor_uses_complete_nonzero_spectrum(self) -> None:
        factor = torch.tensor([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        analysis = analyze_factor(factor, max_vectors=2)
        self.assertEqual(analysis.dimension, 3)
        self.assertTrue(torch.allclose(analysis.eigenvalues, torch.tensor([4.0, 1.0], dtype=torch.float64)))
        self.assertAlmostEqual(float(analysis.rho_struct_curve[-1]), 1.0, places=12)


if __name__ == "__main__":
    unittest.main()
