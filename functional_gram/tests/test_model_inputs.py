from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from functional_gram.collect_stats import initial_cost_control_modules
from functional_gram.model_inputs import model_weight_files


class ModelInputTests(unittest.TestCase):
    def test_sharded_safetensors_are_discovered_in_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model-00002-of-00002.safetensors").write_bytes(b"b")
            (root / "model-00001-of-00002.safetensors").write_bytes(b"a")
            self.assertEqual(
                [path.name for path in model_weight_files(root)],
                ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"],
            )

    def test_llama_depth_maps_to_early_middle_late_targets(self) -> None:
        modules = initial_cost_control_modules(32)
        self.assertEqual(len(modules), 7)
        self.assertIn("layers.0.self_attn.q_proj", modules)
        self.assertIn("layers.16.mlp.gate_proj", modules)
        self.assertIn("layers.31.mlp.down_proj", modules)


if __name__ == "__main__":
    unittest.main()
