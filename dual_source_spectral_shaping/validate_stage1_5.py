"""Independent schema, arithmetic, and artifact checks for Stage 1.5 outputs."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_REQUIRED = {
    "layer",
    "module",
    "split",
    "K",
    *(f"rho_{source}_{rank}" for source in ("x", "w", "pair", "joint") for rank in (64, 128, 256, 512)),
    "pair_gain_256",
    "joint_gain_256",
    "L_x",
    "L_w",
    "cross_term",
    "kappa",
    "gamma",
    "pair_overlap_128",
    "pair_overlap_256",
    "joint_overlap_128",
    "joint_overlap_256",
}
SCALING_REQUIRED = {
    "layer",
    "module",
    "split",
    "alpha",
    "scale_min",
    "scale_max",
    "scale_geomean",
    "tau_total",
    "L_x",
    "L_w",
    "source_share_x",
    "source_share_w",
    "kappa",
    "rho_x_128",
    "rho_x_256",
    "rho_w_128",
    "rho_w_256",
    "rho_pair_128",
    "rho_pair_256",
    "rho_joint_128",
    "rho_joint_256",
    "selected_on_split_A",
}


def close(left: float, right: float, tolerance: float = 1e-10) -> bool:
    return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)


def validate(output_root: Path) -> dict[str, object]:
    raw = pd.read_csv(output_root / "stage1_5A_joint_spectrum.csv")
    scaling = pd.read_csv(output_root / "stage1_5B_scaling_sweep.csv")
    gate = json.loads((output_root / "stage1_5_gate_summary.json").read_text(encoding="utf-8"))
    validation = json.loads((output_root / "validation.json").read_text(encoding="utf-8"))
    manifest = json.loads((output_root / "run_manifest.json").read_text(encoding="utf-8"))
    report = (output_root / "STAGE1_5_REPORT.md").read_text(encoding="utf-8")
    combined = raw[raw.split == "combined"]
    stage_a = gate["stage1_5A"]
    stage_b = gate["stage1_5B"]

    curves_valid = True
    for source in ("x", "w", "pair", "joint"):
        columns = [f"rho_{source}_{rank}" for rank in (16, 32, 64, 128, 256, 512)]
        values = raw[columns].to_numpy(dtype=float)
        curves_valid = curves_valid and bool(
            ((values >= -1e-8) & (values <= 1 + 1e-6)).all()
            and (values[:, 1:] + 1e-9 >= values[:, :-1]).all()
        )

    pair = combined.rho_pair_256.astype(float).tolist()
    entered_b = not scaling.empty
    expected_b = stage_a["verdict"] == "CONDITIONAL_SHAPING"
    if entered_b:
        selected_a = scaling[
            (scaling.split == "split_a")
            & scaling.selected_on_split_A.astype(str).str.lower().isin(("true", "1"))
        ]
        selected_b = scaling[
            (scaling.split == "split_b")
            & scaling.selected_on_split_A.astype(str).str.lower().isin(("true", "1"))
        ]
        scaling_protocol_valid = (
            len(selected_a) == raw.module.nunique()
            and len(selected_b) == raw.module.nunique()
            and stage_b.get("verdict")
            in {
                "ERROR_SHAPING_GO",
                "W_CONSOLIDATION_GO",
                "X_CONSOLIDATION_GO",
                "FINAL_NO_GO",
            }
        )
    else:
        scaling_protocol_valid = stage_b.get("status") == "not_run"
    checks = {
        "raw_required_columns": RAW_REQUIRED.issubset(raw.columns),
        "scaling_required_columns": SCALING_REQUIRED.issubset(scaling.columns),
        "seven_modules_three_splits": len(raw) == 21
        and raw.module.nunique() == 7
        and set(raw.split) == {"split_a", "split_b", "combined"},
        "coverage_curves_finite_bounded_monotone": curves_valid,
        "pair_gain_recomputes": bool(
            (
                combined.pair_gain_256.astype(float)
                - (
                    combined.rho_pair_256.astype(float)
                    - combined[["rho_x_256", "rho_w_256"]].astype(float).max(axis=1)
                )
            ).abs().max()
            <= 1e-10
        ),
        "joint_gain_recomputes": bool(
            (
                combined.joint_gain_256.astype(float)
                - (
                    combined.rho_joint_256.astype(float)
                    - combined[["rho_x_256", "rho_w_256"]].astype(float).max(axis=1)
                )
            ).abs().max()
            <= 1e-10
        ),
        "gate_pair_median_recomputes": close(
            stage_a["median_pair_rho_func_256"], statistics.median(pair)
        ),
        "gate_pair_fraction_recomputes": close(
            stage_a["fraction_pair_rho_func_256_ge_0.60"],
            sum(value >= 0.60 for value in pair) / len(pair),
        ),
        "stage_a_verdict_valid": stage_a["verdict"]
        in {"PAIR_GO", "JOINT_ONLY_GO", "CONDITIONAL_SHAPING", "RAW_NO_GO"},
        "stage_b_gate_protocol_valid": scaling_protocol_valid
        and (entered_b == expected_b or entered_b),
        "numerical_validation_passed": validation["status"] == "passed",
        "manifest_complete": manifest["status"] == "complete",
        "report_matches_verdict": f"Stage 1.5A: **{stage_a['verdict']}**" in report,
        "six_required_figures_nonempty": len(list((output_root / "figures").glob("figure_*.png")))
        == 6
        and all(path.stat().st_size > 10_000 for path in (output_root / "figures").glob("figure_*.png")),
        "seven_raw_artifacts": len(list((output_root / "raw_artifacts").glob("*.pt"))) == 7,
    }
    status = "passed" if all(checks.values()) else "failed"
    return {"status": status, "checks": checks}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        default=str(
            REPO_ROOT
            / "analysis"
            / "dual_source_spectral_shaping"
            / "stage1_5_qwen25_15b"
        ),
    )
    args = parser.parse_args()
    output_root = Path(args.output_root).resolve()
    result = validate(output_root)
    (output_root / "artifact_validation.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
