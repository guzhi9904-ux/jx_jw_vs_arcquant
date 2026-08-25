from __future__ import annotations

import unittest

import torch

from factor_rank_spectral_outlier.factor_spectrum import analyze_psd
from factor_rank_spectral_outlier.spectral_outlier_score import compute_spectral_outlier_scores


class TailEnergyTest(unittest.TestCase):
    def test_tail_energy_identity(self) -> None:
        generator = torch.Generator().manual_seed(29)
        atoms = torch.randn((7, 5), generator=generator, dtype=torch.float64)
        gram = atoms @ atoms.T
        analysis = analyze_psd(gram, max_rank=7, exact=True)
        scores = compute_spectral_outlier_scores(gram, analysis, bulk_rank=2)
        represented = (
            analysis.eigenvectors[:, :2].double().square()
            @ analysis.eigenvalues[:2].double()
        )
        self.assertTrue(torch.allclose(scores.diag_j, represented + scores.tail_energy_h, atol=1e-10))
        self.assertLess(scores.diagnostics["tail_energy_identity_relative_error"], 1e-10)
        self.assertTrue(bool((scores.tail_ratio_h >= 0).all()))
        self.assertTrue(bool((scores.tail_ratio_h <= 1 + 1e-10).all()))


if __name__ == "__main__":
    unittest.main()
