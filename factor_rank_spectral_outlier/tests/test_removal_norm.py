from __future__ import annotations

import unittest

import torch

from factor_rank_spectral_outlier.factor_spectrum import analyze_psd
from factor_rank_spectral_outlier.functional_factors import build_functional_factor_grams
from factor_rank_spectral_outlier.removal_intervention import removal_norm_audit, run_removal
from factor_rank_spectral_outlier.spectral_outlier_score import compute_spectral_outlier_scores


class RemovalNormTest(unittest.TestCase):
    def test_principal_submatrix_matches_remaining_atoms(self) -> None:
        generator = torch.Generator().manual_seed(31)
        x = torch.randn((10, 9), generator=generator, dtype=torch.float64)
        qx = x + 0.1 * torch.randn((10, 9), generator=generator, dtype=torch.float64)
        weight = torch.randn((6, 9), generator=generator, dtype=torch.float64)
        qweight = weight + 0.1 * torch.randn((6, 9), generator=generator, dtype=torch.float64)
        built = build_functional_factor_grams(x, qx, weight, qweight, source="X")
        raw = analyze_psd(built.gram, max_rank=9, exact=True)
        scores = compute_spectral_outlier_scores(built.gram, raw, bulk_rank=3)
        result = run_removal(
            built.gram,
            scores.order_h,
            remove_budget=2,
            exact_max_dimension=32,
            seed=0,
            oversample=2,
            power_iterations=0,
            slq_probes=0,
            slq_steps=8,
            raw_analysis=raw,
        )
        error = removal_norm_audit(
            built.factor1, built.factor2, built.gram, result.remaining_indices
        )
        self.assertLess(error, 1e-12)
        self.assertAlmostEqual(
            result.remaining_total_functional_error,
            float(
                built.gram.index_select(0, result.remaining_indices)
                .index_select(1, result.remaining_indices)
                .sum()
            ),
            places=10,
        )


if __name__ == "__main__":
    unittest.main()
