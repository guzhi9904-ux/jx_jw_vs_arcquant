"""Architecture-independent integrity checks for one server comparison seed."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch


STRATEGIES = {
    "paper_reorder_arc",
    "activation_energy_train",
    "weight_energy_train",
    "proxy_fixed_half_train",
    "proxy_shared_train",
    "random_fixed_half",
    "oracle_independent_fixed_half_train",
    "oracle_shared_train",
    "oracle_independent_fixed_half_holdout_leaked",
    "oracle_shared_holdout_leaked",
}
LEAKED = {name for name in STRATEGIES if name.endswith("_holdout_leaked")}
STORED = STRATEGIES - LEAKED


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", required=True)
    return parser.parse_args()


def close(left: float, right: float, tolerance: float = 2e-10) -> bool:
    return math.isclose(
        float(left), float(right), rel_tol=tolerance, abs_tol=tolerance
    )


def main() -> None:
    analysis_dir = Path(parse_args().analysis_dir)
    modules = pd.read_csv(analysis_dir / "module_metrics.csv")
    summary = pd.read_csv(analysis_dir / "strategy_summary.csv")
    metadata = json.loads(
        (analysis_dir / "metadata.json").read_text(encoding="utf-8")
    )
    selections = torch.load(
        analysis_dir / "selection_indices.pt", map_location="cpu"
    )

    module_names = set(modules.module.unique())
    module_count = len(module_names)
    strategy_count = len(STRATEGIES)
    checks: dict[str, bool] = {
        "experiment_label": metadata.get("experiment") == "server-comparison",
        "strategy_set": set(modules.strategy.unique()) == STRATEGIES,
        "summary_strategy_set": set(summary.strategy) == STRATEGIES,
        "row_count": len(modules) == module_count * strategy_count,
        "strategies_per_module": bool(
            (modules.groupby("module").strategy.nunique() == strategy_count).all()
        ),
        "metadata_module_count": int(metadata.get("module_rows", -1))
        == module_count,
        "all_modules_stored": set(selections) == module_names,
    }

    numeric = modules.select_dtypes(include=[np.number]).to_numpy(np.float64)
    checks["all_numeric_finite"] = bool(np.isfinite(numeric).all())
    checks["positive_split_width"] = bool(
        (modules.selection_rows > 0).all() and (modules.holdout_rows > 0).all()
    )
    checks["same_split_width_per_module"] = bool(
        (modules.groupby("module").selection_rows.nunique() == 1).all()
        and (modules.groupby("module").holdout_rows.nunique() == 1).all()
    )
    checks["budget_exact"] = bool(
        (
            modules.activation_branch_num
            + modules.weight_branch_num
            == modules.select_num
        ).all()
    )
    checks["paper_arc_budget"] = bool(
        (
            modules.loc[
                modules.strategy == "paper_reorder_arc",
                "activation_branch_num",
            ]
            == modules.loc[
                modules.strategy == "paper_reorder_arc", "select_num"
            ]
        ).all()
    )

    def branch_fraction(strategy: str, numerator: int, denominator: int) -> bool:
        rows = modules[modules.strategy == strategy]
        return bool(
            (
                rows.activation_branch_num * denominator
                == rows.select_num * numerator
            ).all()
        )

    checks["activation_only_budget"] = branch_fraction(
        "activation_energy_train", 1, 1
    )
    checks["weight_only_budget"] = branch_fraction("weight_energy_train", 0, 1)
    for strategy in (
        "proxy_fixed_half_train",
        "proxy_shared_train",
        "random_fixed_half",
        "oracle_independent_fixed_half_train",
        "oracle_shared_train",
        "oracle_independent_fixed_half_holdout_leaked",
        "oracle_shared_holdout_leaked",
    ):
        checks[f"{strategy}_half_half"] = branch_fraction(strategy, 1, 2)

    leaked_rows = modules[modules.strategy.isin(LEAKED)]
    clean_rows = modules[~modules.strategy.isin(LEAKED)]
    checks["leakage_labels"] = bool(
        (leaked_rows.selection_split == "holdout_leaked").all()
        and (clean_rows.selection_split == "train_or_paper").all()
    )

    baseline_consistency = modules.groupby("module").agg(
        train_min=("train_baseline_sse", "min"),
        train_max=("train_baseline_sse", "max"),
        holdout_min=("holdout_baseline_sse", "min"),
        holdout_max=("holdout_baseline_sse", "max"),
    )
    checks["same_rtn_baseline"] = bool(
        np.allclose(
            baseline_consistency.train_min,
            baseline_consistency.train_max,
            rtol=1e-12,
            atol=1e-7,
        )
        and np.allclose(
            baseline_consistency.holdout_min,
            baseline_consistency.holdout_max,
            rtol=1e-12,
            atol=1e-7,
        )
    )

    summary_index = summary.set_index("strategy")
    summary_checks: list[bool] = []
    for strategy in STRATEGIES:
        rows = modules[modules.strategy == strategy]
        train_gain = 1.0 - (
            rows.train_corrected_sse.sum() / rows.train_baseline_sse.sum()
        )
        holdout_gain = 1.0 - (
            rows.holdout_corrected_sse.sum() / rows.holdout_baseline_sse.sum()
        )
        summary_checks.extend(
            [
                close(summary_index.loc[strategy, "train_gain"], train_gain),
                close(summary_index.loc[strategy, "holdout_gain"], holdout_gain),
                close(
                    summary_index.loc[strategy, "generalization_gap"],
                    train_gain - holdout_gain,
                ),
            ]
        )
    checks["summaries_recompute"] = all(summary_checks)

    selection_checks: list[bool] = []
    for payload in selections.values():
        select_num = int(payload["select_num"])
        selection_checks.append(set(payload) == {"select_num", *STORED})
        for strategy in STORED:
            selected = payload[strategy]
            activation = selected["activation"].long()
            weight = selected["weight"].long()
            selection_checks.extend(
                [
                    activation.numel() + weight.numel() == select_num,
                    activation.unique().numel() == activation.numel(),
                    weight.unique().numel() == weight.numel(),
                ]
            )
            if strategy in {"proxy_shared_train", "oracle_shared_train"}:
                selection_checks.append(torch.equal(activation, weight))
    checks["stored_indices_valid"] = all(selection_checks)

    failed = [name for name, passed in checks.items() if not passed]
    result = {
        "status": "passed" if not failed else "failed",
        "analysis_dir": str(analysis_dir.resolve()),
        "module_count": module_count,
        "layer_count": int(modules.layer.nunique()),
        "module_type_count": int(modules.module_type.nunique()),
        "checks": checks,
        "failed": failed,
    }
    (analysis_dir / "server_validation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if failed:
        raise AssertionError(f"Server seed validation failed: {failed}")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
