from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from functional_gram.plot_stage1 import _equal_fraction_heatmap, _heatmap_at_rank


class Stage1PlotTests(unittest.TestCase):
    def test_rank512_and_equal_fraction_heatmaps_render(self) -> None:
        records = []
        summary = []
        for layer in (0, 16, 31):
            for module_type in ("q_proj", "k_proj", "v_proj", "o_proj", "down_proj"):
                module = f"layers.{layer}.test.{module_type}"
                k = 14336 if module_type == "down_proj" else 4096
                for source in ("X", "W"):
                    records.append(
                        {
                            "layer": layer,
                            "module": module,
                            "module_type": module_type,
                            "source": source,
                        }
                    )
                    for rank in ({256, 512, 896} if k == 14336 else {256, 512}):
                        summary.append(
                            {
                                "layer": layer,
                                "module": module,
                                "module_type": module_type,
                                "source": source,
                                "rank": rank,
                                "rho_func": 0.4,
                                "rho_struct": 0.5,
                                "split_overlap": 0.7,
                                "equal_fraction_reference": rank == (896 if k == 14336 else 256),
                            }
                        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rank_path = root / "rank512.png"
            equal_path = root / "equal.png"
            _heatmap_at_rank(
                records,
                summary,
                rank_path,
                rank=512,
                module_order=("q_proj", "k_proj", "v_proj", "o_proj"),
                figure_title="test",
            )
            _equal_fraction_heatmap(records, summary, equal_path)
            self.assertGreater(rank_path.stat().st_size, 10_000)
            self.assertGreater(equal_path.stat().st_size, 10_000)


if __name__ == "__main__":
    unittest.main()
