from __future__ import annotations

import unittest

import torch

from dual_source_spectral_shaping.requantize_fp4 import (
    RequantizationAudit,
    requantize_matrix,
)
from dual_source_spectral_shaping.tests.common import make_operands


class RequantizationTest(unittest.TestCase):
    def test_nonidentity_calls_real_quantizer(self) -> None:
        x, _, _, _ = make_operands()
        audit = RequantizationAudit()
        scaled = x * torch.linspace(0.5, 1.5, x.shape[1])
        result, diagnostics = requantize_matrix(
            scaled, row_chunk=4, audit=audit, identity=False
        )
        self.assertEqual(result.shape, x.shape)
        self.assertIs(diagnostics["stage1_5_requantized"], True)
        self.assertEqual(audit.calls, 1)
        self.assertEqual(audit.nonidentity_calls, 1)
