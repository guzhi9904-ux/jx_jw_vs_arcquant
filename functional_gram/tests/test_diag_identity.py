from __future__ import annotations

import unittest

import torch

from functional_gram.build_gram import build_functional_grams


class DiagonalIdentityTests(unittest.TestCase):
    def test_both_source_diagonals_match_channel_formula(self) -> None:
        generator = torch.Generator().manual_seed(11)
        x = torch.randn(13, 16, generator=generator)
        qx = x + 0.07 * torch.randn(13, 16, generator=generator)
        weight = torch.randn(9, 16, generator=generator)
        qweight = weight + 0.05 * torch.randn(9, 16, generator=generator)
        grams = build_functional_grams([(x, qx)], weight, qweight)
        ex = x.double() - qx.double()
        ew = weight.double() - qweight.double()
        expected_x = ex.square().sum(0) * weight.double().square().sum(0)
        expected_w = x.double().square().sum(0) * ew.square().sum(0)
        torch.testing.assert_close(grams.jx, expected_x, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(grams.jw, expected_w, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
