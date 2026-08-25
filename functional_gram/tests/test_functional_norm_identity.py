from __future__ import annotations

import unittest

import torch

from functional_gram.build_gram import build_functional_grams, sanity_check_grams


class FunctionalNormIdentityTests(unittest.TestCase):
    def test_one_g_one_matches_direct_functional_norm(self) -> None:
        generator = torch.Generator().manual_seed(23)
        xa = torch.randn(7, 16, generator=generator)
        xb = torch.randn(5, 16, generator=generator)
        qxa = xa + 0.04 * torch.randn(7, 16, generator=generator)
        qxb = xb + 0.04 * torch.randn(5, 16, generator=generator)
        weight = torch.randn(8, 16, generator=generator)
        qweight = weight + 0.03 * torch.randn(8, 16, generator=generator)
        grams = build_functional_grams([(xa, qxa), (xb, qxb)], weight, qweight)
        x, qx = torch.cat((xa, xb)), torch.cat((qxa, qxb))
        result = sanity_check_grams(grams, x, qx, weight, qweight, rtol=1e-11)
        self.assertEqual(result["status"], "passed")
        self.assertLess(float(result["functional_norm_x_relative_error"]), 1e-12)
        self.assertLess(float(result["functional_norm_w_relative_error"]), 1e-12)


if __name__ == "__main__":
    unittest.main()
