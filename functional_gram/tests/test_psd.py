from __future__ import annotations

import unittest

import torch

from functional_gram.build_gram import build_functional_grams
from functional_gram.eig_analysis import analyze_gram
from functional_gram.split_stability import projector_overlap_curve


class PSDTests(unittest.TestCase):
    def test_both_hadamard_grams_are_psd_and_metrics_are_bounded(self) -> None:
        generator = torch.Generator().manual_seed(37)
        x = torch.randn(20, 16, generator=generator)
        qx = x + 0.08 * torch.randn(20, 16, generator=generator)
        weight = torch.randn(12, 16, generator=generator)
        qweight = weight + 0.06 * torch.randn(12, 16, generator=generator)
        grams = build_functional_grams([(x, qx)], weight, qweight)
        for gram in (grams.gx, grams.gw):
            analysis = analyze_gram(gram)
            self.assertGreaterEqual(analysis.raw_min_eigenvalue, -1e-10)
            self.assertTrue(bool((analysis.eigenvalues >= 0).all().item()))
            self.assertTrue(bool(((analysis.rho_struct_curve >= 0) & (analysis.rho_struct_curve <= 1 + 1e-10)).all().item()))
            self.assertTrue(bool(((analysis.rho_func_curve >= 0) & (analysis.rho_func_curve <= 1 + 1e-10)).all().item()))
            overlap = projector_overlap_curve(analysis.eigenvectors, analysis.eigenvectors)
            torch.testing.assert_close(overlap, torch.ones_like(overlap), rtol=1e-10, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
