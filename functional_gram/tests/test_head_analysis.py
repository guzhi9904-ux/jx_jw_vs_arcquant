from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import torch

from functional_gram.head_analysis import (
    functional_gram_from_covariance,
    head_row_slices,
    run_head_analysis,
)
from functional_gram.eig_analysis import analyze_gram
from functional_gram.run_stage1 import (
    equal_fraction_rank,
    parse_args,
    summary_ranks_for_k,
)


class HeadAnalysisTests(unittest.TestCase):
    def test_head_blocks_sum_to_full_functional_gram(self) -> None:
        generator = torch.Generator().manual_seed(23)
        activation = torch.randn(19, 12, generator=generator, dtype=torch.float64)
        factor = torch.randn(8, 12, generator=generator, dtype=torch.float64)
        covariance = activation.T @ activation
        full = functional_gram_from_covariance(covariance, factor)
        parts = sum(
            (
                functional_gram_from_covariance(covariance, factor[row_slice])
                for row_slice in head_row_slices(8, 4, 2)
            ),
            torch.zeros_like(full),
        )
        self.assertTrue(torch.allclose(parts, full, rtol=1e-12, atol=1e-12))

    def test_invalid_head_layout_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            head_row_slices(10, 4, 2)

    def test_equal_fraction_rank_plan(self) -> None:
        self.assertEqual(equal_fraction_rank(4096), 256)
        self.assertEqual(equal_fraction_rank(14336), 896)
        self.assertEqual(summary_ranks_for_k(4096)[-1], 512)
        self.assertEqual(summary_ranks_for_k(14336)[-2:], (896, 1024))

    def test_depth_control_cli_preset_is_preserved(self) -> None:
        self.assertEqual(parse_args(["--modules", "depth-control"]).modules, "depth-control")

    def test_end_to_end_head_analysis_writes_valid_outputs(self) -> None:
        generator = torch.Generator().manual_seed(29)
        x_a = torch.randn(10, 128, generator=generator)
        x_b = torch.randn(10, 128, generator=generator)
        qx_a = x_a + 0.03 * torch.randn(10, 128, generator=generator)
        qx_b = x_b + 0.03 * torch.randn(10, 128, generator=generator)
        weight = torch.randn(4, 128, generator=generator)
        qweight = weight + 0.03 * torch.randn(4, 128, generator=generator)
        operands = {
            "module": "layers.0.self_attn.q_proj",
            "module_type": "q_proj",
            "layer": 0,
            "K": 128,
            "out_features": 4,
            "attention_heads": 2,
            "head_dim": 2,
            "split_a": {"x": x_a, "qx": qx_a},
            "split_b": {"x": x_b, "qx": qx_b},
            "weight": weight,
            "qweight": qweight,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            operand_path = root / "operand.pt"
            artifacts = root / "artifacts"
            artifacts.mkdir()
            torch.save(operands, operand_path)
            x = torch.cat((x_a, x_b))
            qx = torch.cat((qx_a, qx_b))
            for source in ("X", "W"):
                activation = x - qx if source == "X" else x
                factor = weight if source == "X" else weight - qweight
                activation64 = activation.double()
                factor64 = factor.double()
                gram = functional_gram_from_covariance(
                    activation64.T @ activation64, factor64
                )
                analysis = analyze_gram(gram)
                torch.save(
                    {
                        "combined": {
                            "top_eigenvectors": analysis.eigenvectors.float(),
                            "rho_func_curve": analysis.rho_func_curve.float(),
                        }
                    },
                    artifacts / f"layers__0__self_attn__q_proj__source_{source}.pt",
                )
            result = run_head_analysis(
                [operand_path],
                artifact_dir=artifacts,
                output_dir=root / "head_analysis",
                device="cpu",
                randomized_min_k=1,
                oversample=4,
                power_iterations=1,
                overlap_rank=16,
            )
            self.assertEqual(result["status"], "passed", msg=result)
            for path in result["outputs"].values():
                self.assertTrue(Path(path).is_file())


if __name__ == "__main__":
    unittest.main()
