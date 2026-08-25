"""Run Stage 1.5A and conditionally Stage 1.5B on saved Stage-1 operands."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dual_source_spectral_shaping.exact_dual_source import (  # noqa: E402
    DualSourceGrams,
    build_dual_source_grams,
)
from dual_source_spectral_shaping.plot_stage1_5 import render_figures  # noqa: E402
from dual_source_spectral_shaping.requantize_fp4 import (  # noqa: E402
    RequantizationAudit,
    requantize_matrix,
)
from dual_source_spectral_shaping.smooth_scaling import (  # noqa: E402
    apply_reparameterization,
    smoothquant_scale,
)
from dual_source_spectral_shaping.spectral_analysis import (  # noqa: E402
    Spectrum,
    analyze_all,
    projector_overlap,
    spectrum_payload,
)
from fp4_residual_carrier.common.reproducibility import (  # noqa: E402
    environment_metadata,
    git_metadata,
    write_json,
)


RANKS = (16, 32, 64, 128, 256, 512)
SCALING_RANKS = (64, 128, 256, 512)
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=str(REPO_ROOT.parent / "modelzoo" / "Qwen" / "Qwen2.5-1.5B-Instruct"),
    )
    parser.add_argument(
        "--operands-dir",
        default=None,
        help="Stage-1 operands; defaults to analysis/functional_gram/<model-name>/operands",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Output directory; defaults to analysis/dual_source_spectral_shaping/<model-name>",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--exact-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--exact-max-dimension", type=int, default=4096)
    parser.add_argument("--oversample", type=int, default=16)
    parser.add_argument("--raw-power-iterations", type=int, default=1)
    parser.add_argument("--scaling-power-iterations", type=int, default=1)
    parser.add_argument("--quant-row-chunk", type=int, default=128)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--force-stage1-5b", action="store_true")
    parser.add_argument("--skip-stage1-5b", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args(None if argv is None else list(argv))
    if args.cpu_threads > 0:
        torch.set_num_threads(args.cpu_threads)
    if args.output_root is None:
        args.output_root = str(
            REPO_ROOT
            / "analysis"
            / "dual_source_spectral_shaping"
            / Path(args.model).name
        )
    if args.operands_dir is None:
        args.operands_dir = str(
            REPO_ROOT / "analysis" / "functional_gram" / Path(args.model).name / "operands"
        )
    if args.force_stage1_5b and args.skip_stage1_5b:
        parser.error("--force-stage1-5b and --skip-stage1-5b are mutually exclusive")
    return args


def _stable_seed(*values: object) -> int:
    digest = hashlib.sha256("|".join(map(str, values)).encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**31 - 1)


def _cpu_grams(grams: DualSourceGrams) -> DualSourceGrams:
    return DualSourceGrams(
        gx=grams.gx.detach().cpu(),
        gw=grams.gw.detach().cpu(),
        h=grams.h.detach().cpu(),
        gp=grams.gp.detach().cpu(),
        l_x=grams.l_x,
        l_w=grams.l_w,
        cross_term=grams.cross_term,
        total_error=grams.total_error,
        kappa=grams.kappa,
        gamma=grams.gamma,
        diagnostics=grams.diagnostics,
    )


def _spectrum_device(args: argparse.Namespace, *, exact: bool) -> torch.device:
    if exact:
        if args.exact_device == "cuda":
            return torch.device(args.device)
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("randomized large-Gram analysis requires CUDA")
    return torch.device(args.device)


def _analyze(
    grams: DualSourceGrams,
    *,
    args: argparse.Namespace,
    label: str,
    scaling: bool,
) -> dict[str, Spectrum]:
    dimension = 2 * int(grams.gx.shape[0])
    exact = (not scaling) and dimension <= args.exact_max_dimension
    print(
        f"spectral {label}: K={grams.gx.shape[0]}, method={'exact' if exact else 'randomized'}",
        flush=True,
    )
    return analyze_all(
        grams,
        max_rank=min(max(RANKS), int(grams.gx.shape[0])),
        exact=exact,
        device=_spectrum_device(args, exact=exact),
        oversample=args.oversample,
        power_iterations=(
            args.scaling_power_iterations if scaling else args.raw_power_iterations
        ),
        seed=_stable_seed(label),
    )


def _build(
    x: torch.Tensor,
    qx: torch.Tensor,
    weight: torch.Tensor,
    qweight: torch.Tensor,
    *,
    device: torch.device,
) -> DualSourceGrams:
    grams = build_dual_source_grams(
        x.to(device),
        qx.to(device),
        weight.to(device),
        qweight.to(device),
        accumulation_dtype=torch.float32,
        run_sanity=True,
    )
    result = _cpu_grams(grams)
    del grams
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _rank_values(spectra: dict[str, Spectrum], ranks: Iterable[int]) -> dict[str, float]:
    result: dict[str, float] = {}
    for source in ("x", "w", "pair", "joint"):
        for rank in ranks:
            if rank <= spectra[source].rho_func_curve.numel():
                result[f"rho_{source}_{rank}"] = spectra[source].value(rank)
                result[f"rho_{source}_struct_{rank}"] = spectra[source].value(
                    rank, functional=False
                )
    return result


def _raw_row(
    *,
    operands: dict[str, Any],
    split: str,
    grams: DualSourceGrams,
    spectra: dict[str, Spectrum],
    overlaps: dict[str, float],
) -> dict[str, Any]:
    values = _rank_values(spectra, RANKS)
    rho_max = max(values["rho_x_256"], values["rho_w_256"])
    return {
        "layer": int(operands["layer"]),
        "module": operands["module"],
        "module_type": operands["module_type"],
        "split": split,
        "K": int(operands["K"]),
        **values,
        "pair_gain_256": values["rho_pair_256"] - rho_max,
        "joint_gain_256": values["rho_joint_256"] - rho_max,
        "L_x": grams.l_x,
        "L_w": grams.l_w,
        "cross_term": grams.cross_term,
        "kappa": grams.kappa,
        "gamma": grams.gamma,
        "pair_overlap_128": overlaps["pair_128"],
        "pair_overlap_256": overlaps["pair_256"],
        "joint_overlap_128": overlaps["joint_128"],
        "joint_overlap_256": overlaps["joint_256"],
        "min_eigenvalue_x": spectra["x"].raw_min_eigenvalue,
        "min_eigenvalue_w": spectra["w"].raw_min_eigenvalue,
        "min_eigenvalue_pair": spectra["pair"].raw_min_eigenvalue,
        "min_eigenvalue_joint": spectra["joint"].raw_min_eigenvalue,
        "spectral_method_x": spectra["x"].method,
        "spectral_method_w": spectra["w"].method,
        "spectral_method_pair": spectra["pair"].method,
        "spectral_method_joint": spectra["joint"].method,
        "spectral_residual_max": max(
            spectrum.relative_residual_max for spectrum in spectra.values()
        ),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_empty_scaling_csv(path: Path) -> None:
    """Emit the required Stage-1.5B schema even when its gate is not entered."""

    fieldnames = [
        "layer",
        "module",
        "module_type",
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
        "rho_x_64",
        "rho_x_128",
        "rho_x_256",
        "rho_x_512",
        "rho_w_64",
        "rho_w_128",
        "rho_w_256",
        "rho_w_512",
        "rho_pair_64",
        "rho_pair_128",
        "rho_pair_256",
        "rho_pair_512",
        "rho_joint_64",
        "rho_joint_128",
        "rho_joint_256",
        "rho_joint_512",
        "selected_on_split_A",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        csv.DictWriter(handle, fieldnames=fieldnames).writeheader()


def _stage_a_gate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected = [row for row in rows if row["split"] == "combined"]
    if not selected:
        raise ValueError("combined Stage 1.5A rows are missing")
    pair = [float(row["rho_pair_256"]) for row in selected]
    pair_gain = [float(row["pair_gain_256"]) for row in selected]
    pair_overlap = [float(row["pair_overlap_256"]) for row in selected]
    joint = [float(row["rho_joint_256"]) for row in selected]
    joint_gain = [float(row["joint_gain_256"]) for row in selected]
    joint_overlap = [float(row["joint_overlap_256"]) for row in selected]
    w_minus_x = [float(row["rho_w_256"] - row["rho_x_256"]) for row in selected]
    x_share = [float(row["L_x"] / max(row["L_x"] + row["L_w"], 1e-30)) for row in selected]
    values: dict[str, Any] = {
        "module_count": len(selected),
        "median_pair_rho_func_256": statistics.median(pair),
        "fraction_pair_rho_func_256_ge_0.60": sum(value >= 0.60 for value in pair)
        / len(pair),
        "median_pair_gain_256": statistics.median(pair_gain),
        "median_pair_overlap_256": statistics.median(pair_overlap),
        "median_joint_rho_func_256": statistics.median(joint),
        "median_joint_gain_256": statistics.median(joint_gain),
        "median_joint_overlap_256": statistics.median(joint_overlap),
        "median_w_minus_x_rho_func_256": statistics.median(w_minus_x),
        "median_x_source_share": statistics.median(x_share),
    }
    pair_conditions = {
        "median_pair_rho_func_256_ge_0.70": values["median_pair_rho_func_256"] >= 0.70,
        "fraction_pair_rho_func_256_ge_0.60_at_least_0.60": values[
            "fraction_pair_rho_func_256_ge_0.60"
        ]
        >= 0.60,
        "median_pair_gain_256_ge_0.08": values["median_pair_gain_256"] >= 0.08,
        "median_pair_overlap_256_ge_0.50": values["median_pair_overlap_256"] >= 0.50,
    }
    joint_conditions = {
        "median_joint_rho_func_256_ge_0.70": values["median_joint_rho_func_256"] >= 0.70,
        "median_joint_gain_256_ge_0.08": values["median_joint_gain_256"] >= 0.08,
        "median_joint_overlap_256_ge_0.50": values["median_joint_overlap_256"] >= 0.50,
    }
    shaping_conditions = {
        "median_w_minus_x_rho_func_256_ge_0.08": values[
            "median_w_minus_x_rho_func_256"
        ]
        >= 0.08,
        "median_x_source_share_ge_0.20": values["median_x_source_share"] >= 0.20,
    }
    if all(pair_conditions.values()):
        verdict = "PAIR_GO"
    elif all(joint_conditions.values()):
        verdict = "JOINT_ONLY_GO"
    elif all(shaping_conditions.values()):
        verdict = "CONDITIONAL_SHAPING"
    else:
        verdict = "RAW_NO_GO"
    values.update(
        {
            "pair_conditions": pair_conditions,
            "joint_conditions": joint_conditions,
            "conditional_shaping_conditions": shaping_conditions,
            "verdict": verdict,
        }
    )
    return values


def _raw_analysis(
    operands_paths: list[Path],
    *,
    args: argparse.Namespace,
    output_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, dict[str, Any]]]:
    raw_rows: list[dict[str, Any]] = []
    identity_lookup: dict[str, dict[str, Any]] = {}
    artifact_dir = output_root / "raw_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    build_device = torch.device(args.device)

    for operand_path in operands_paths:
        operands = torch.load(operand_path, map_location="cpu", weights_only=True)
        module = operands["module"]
        print(f"\nraw dual-source: {module}", flush=True)
        weight = operands["weight"].float()
        qweight = operands["qweight"].float()
        grams_by_split: dict[str, DualSourceGrams] = {}
        spectra_by_split: dict[str, dict[str, Spectrum]] = {}
        for split in ("split_a", "split_b"):
            grams_by_split[split] = _build(
                operands[split]["x"].float(),
                operands[split]["qx"].float(),
                weight,
                qweight,
                device=build_device,
            )
            spectra_by_split[split] = _analyze(
                grams_by_split[split], args=args, label=f"raw|{module}|{split}", scaling=False
            )
        grams_by_split["combined"] = grams_by_split["split_a"].add(
            grams_by_split["split_b"]
        )
        spectra_by_split["combined"] = _analyze(
            grams_by_split["combined"],
            args=args,
            label=f"raw|{module}|combined",
            scaling=False,
        )
        overlaps = {
            f"{source}_{rank}": projector_overlap(
                spectra_by_split["split_a"][source].top_eigenvectors,
                spectra_by_split["split_b"][source].top_eigenvectors,
                rank,
            )
            for source in ("pair", "joint")
            for rank in (128, 256)
        }
        module_rows = [
            _raw_row(
                operands=operands,
                split=split,
                grams=grams_by_split[split],
                spectra=spectra_by_split[split],
                overlaps=overlaps,
            )
            for split in ("split_a", "split_b", "combined")
        ]
        raw_rows.extend(module_rows)
        identity_lookup[module] = {row["split"]: row for row in module_rows}
        artifact = {
            "schema_version": 1,
            "module": module,
            "K": int(operands["K"]),
            "overlaps": overlaps,
            "grams": {
                split: {
                    "L_x": grams_by_split[split].l_x,
                    "L_w": grams_by_split[split].l_w,
                    "cross_term": grams_by_split[split].cross_term,
                    "total_error": grams_by_split[split].total_error,
                    "diagnostics": grams_by_split[split].diagnostics,
                }
                for split in grams_by_split
            },
            "spectra": {
                split: {
                    source: spectrum_payload(spectrum)
                    for source, spectrum in spectra_by_split[split].items()
                }
                for split in spectra_by_split
            },
        }
        torch.save(artifact, artifact_dir / f"{module.replace('.', '__')}.pt")
        del operands, weight, qweight, grams_by_split, spectra_by_split, artifact
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    raw_rows.sort(key=lambda row: (row["layer"], row["module"], row["split"]))
    _write_csv(output_root / "stage1_5A_joint_spectrum.csv", raw_rows)
    gate = _stage_a_gate(raw_rows)
    return raw_rows, gate, identity_lookup


def _scaling_identity_row(raw: dict[str, Any], *, split: str) -> dict[str, Any]:
    row = {
        "layer": raw["layer"],
        "module": raw["module"],
        "module_type": raw["module_type"],
        "split": split,
        "alpha": "identity",
        "scale_min": 1.0,
        "scale_max": 1.0,
        "scale_geomean": 1.0,
        "tau_total": 1.0,
        "L_x": raw["L_x"],
        "L_w": raw["L_w"],
        "source_share_x": raw["L_x"] / max(raw["L_x"] + raw["L_w"], 1e-30),
        "kappa": raw["kappa"],
        "selected_on_split_A": False,
        "requantized": False,
        "reparameterization_relative_error": 0.0,
        "exact_decomposition_relative_error": 0.0,
        "pair_total_norm_relative_error": 0.0,
        "spectral_residual_max": raw["spectral_residual_max"],
    }
    row["source_share_w"] = 1.0 - row["source_share_x"]
    for source in ("x", "w", "pair", "joint"):
        for rank in (128, 256):
            row[f"rho_{source}_{rank}"] = raw[f"rho_{source}_{rank}"]
        for rank in SCALING_RANKS:
            if f"rho_{source}_{rank}" in raw:
                row[f"rho_{source}_{rank}"] = raw[f"rho_{source}_{rank}"]
    return row


def _reparameterization_error(
    x: torch.Tensor,
    weight: torch.Tensor,
    scaled_x: torch.Tensor,
    scaled_weight: torch.Tensor,
) -> float:
    rows = min(32, x.shape[0])
    outputs = min(64, weight.shape[0])
    expected = x[:rows].float() @ weight[:outputs].float().T
    actual = scaled_x[:rows].float() @ scaled_weight[:outputs].float().T
    return float((actual - expected).abs().max().item()) / max(
        float(expected.abs().max().item()), 1e-30
    )


def _candidate_row(
    *,
    operands: dict[str, Any],
    split: str,
    alpha: float,
    scale: torch.Tensor,
    grams: DualSourceGrams,
    spectra: dict[str, Spectrum],
    identity_total: float,
    reparam_error: float,
) -> dict[str, Any]:
    values = _rank_values(spectra, SCALING_RANKS)
    source_denominator = max(grams.l_x + grams.l_w, 1e-30)
    diagnostics = grams.diagnostics
    return {
        "layer": int(operands["layer"]),
        "module": operands["module"],
        "module_type": operands["module_type"],
        "split": split,
        "alpha": alpha,
        "scale_min": float(scale.min().item()),
        "scale_max": float(scale.max().item()),
        "scale_geomean": float(torch.exp(torch.log(scale.double()).mean()).item()),
        "tau_total": grams.total_error / max(identity_total, 1e-30),
        "L_x": grams.l_x,
        "L_w": grams.l_w,
        "source_share_x": grams.l_x / source_denominator,
        "source_share_w": grams.l_w / source_denominator,
        "kappa": grams.kappa,
        **values,
        "selected_on_split_A": False,
        "requantized": True,
        "reparameterization_relative_error": reparam_error,
        "exact_decomposition_relative_error": diagnostics.get(
            "exact_decomposition_max_relative_error", math.nan
        ),
        "pair_total_norm_relative_error": diagnostics.get(
            "pair_total_norm_relative_error", math.nan
        ),
        "spectral_residual_max": max(
            spectrum.relative_residual_max for spectrum in spectra.values()
        ),
    }


def _evaluate_candidate(
    *,
    operands: dict[str, Any],
    split: str,
    alpha: float,
    scale: torch.Tensor,
    identity_total: float,
    args: argparse.Namespace,
    audit: RequantizationAudit,
) -> tuple[dict[str, Any], dict[str, Any]]:
    device = torch.device(args.device)
    x = operands[split]["x"].float().to(device)
    weight = operands["weight"].float().to(device)
    d = scale.to(device)
    scaled_x, scaled_weight = apply_reparameterization(x, weight, d)
    reparam_error = _reparameterization_error(x, weight, scaled_x, scaled_weight)
    qweight, weight_diag = requantize_matrix(
        scaled_weight,
        row_chunk=args.quant_row_chunk,
        audit=audit,
        identity=False,
    )
    qx, activation_diag = requantize_matrix(
        scaled_x,
        row_chunk=args.quant_row_chunk,
        audit=audit,
        identity=False,
    )
    grams = _cpu_grams(
        build_dual_source_grams(
            scaled_x,
            qx,
            scaled_weight,
            qweight,
            accumulation_dtype=torch.float32,
            run_sanity=True,
        )
    )
    del x, weight, d, scaled_x, scaled_weight, qweight, qx
    torch.cuda.empty_cache()
    spectra = _analyze(
        grams,
        args=args,
        label=f"scaling|{operands['module']}|{split}|{alpha:.2f}",
        scaling=True,
    )
    row = _candidate_row(
        operands=operands,
        split=split,
        alpha=alpha,
        scale=scale,
        grams=grams,
        spectra=spectra,
        identity_total=identity_total,
        reparam_error=reparam_error,
    )
    diagnostics = {
        "alpha": alpha,
        "split": split,
        "weight_quantizer": weight_diag,
        "activation_quantizer": activation_diag,
        "gram_sanity": grams.diagnostics,
        "spectra": {
            source: spectrum_payload(spectrum) for source, spectrum in spectra.items()
        },
    }
    del grams, spectra
    gc.collect()
    torch.cuda.empty_cache()
    return row, diagnostics


def _select_candidate(rows_a: list[dict[str, Any]]) -> tuple[dict[str, Any], str]:
    identity = next(row for row in rows_a if row["alpha"] == "identity")
    nonidentity = [row for row in rows_a if row["alpha"] != "identity"]
    safe_nonidentity = [row for row in nonidentity if float(row["tau_total"]) <= 1.05]
    if not safe_nonidentity:
        return identity, "NO_SAFE_SCALING"
    safe = [identity, *safe_nonidentity]
    # Stable preregistered tie breaks.  Rounded comparisons avoid allowing
    # sub-ulp noise to bypass the specified secondary criteria.
    chosen = min(
        safe,
        key=lambda row: (
            -round(float(row["rho_pair_256"]), 12),
            round(float(row["tau_total"]), 12),
            abs(float(row["alpha"]) - 0.5) if row["alpha"] != "identity" else math.inf,
        ),
    )
    return chosen, "SELECTED_SAFE_CANDIDATE"


def _stage_b_gate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected_b = [
        row for row in rows if row["split"] == "split_b" and row["selected_on_split_A"]
    ]
    identity_b = {
        row["module"]: row
        for row in rows
        if row["split"] == "split_b" and row["alpha"] == "identity"
    }
    if not selected_b:
        raise ValueError("selected held-out scaling rows are missing")
    pair = [float(row["rho_pair_256"]) for row in selected_b]
    tau = [float(row["tau_total"]) for row in selected_b]
    pair_delta = [
        float(row["rho_pair_256"] - identity_b[row["module"]]["rho_pair_256"])
        for row in selected_b
    ]
    share_x = [float(row["source_share_x"]) for row in selected_b]
    share_w = [float(row["source_share_w"]) for row in selected_b]
    rho_w = [float(row["rho_w_256"]) for row in selected_b]
    rho_x = [float(row["rho_x_256"]) for row in selected_b]
    result: dict[str, Any] = {
        "module_count": len(selected_b),
        "median_holdout_pair_rho_func_256": statistics.median(pair),
        "fraction_holdout_pair_rho_func_256_ge_0.60": sum(value >= 0.60 for value in pair)
        / len(pair),
        "median_holdout_total_error_ratio": statistics.median(tau),
        "median_pair_gain_vs_identity": statistics.median(pair_delta),
        "median_source_share_x": statistics.median(share_x),
        "median_source_share_w": statistics.median(share_w),
        "median_w_rho_func_256": statistics.median(rho_w),
        "median_x_rho_func_256": statistics.median(rho_x),
        "fraction_w_consolidated_modules": sum(
            row["source_share_x"] <= 0.25 and row["rho_w_256"] >= 0.70
            for row in selected_b
        )
        / len(selected_b),
        "fraction_x_consolidated_modules": sum(
            row["source_share_w"] <= 0.25 and row["rho_x_256"] >= 0.70
            for row in selected_b
        )
        / len(selected_b),
    }
    pair_conditions = {
        "median_pair_rho_ge_0.70": result["median_holdout_pair_rho_func_256"] >= 0.70,
        "fraction_pair_rho_ge_0.60_at_least_0.60": result[
            "fraction_holdout_pair_rho_func_256_ge_0.60"
        ]
        >= 0.60,
        "median_tau_le_1.05": result["median_holdout_total_error_ratio"] <= 1.05,
        "median_pair_gain_vs_identity_ge_0.08": result["median_pair_gain_vs_identity"] >= 0.08,
    }
    w_conditions = {
        "median_source_share_x_le_0.20": result["median_source_share_x"] <= 0.20,
        "median_w_rho_ge_0.75": result["median_w_rho_func_256"] >= 0.75,
        "median_tau_le_1.05": result["median_holdout_total_error_ratio"] <= 1.05,
        "fraction_joint_conditions_at_least_0.60": result[
            "fraction_w_consolidated_modules"
        ]
        >= 0.60,
    }
    x_conditions = {
        "median_source_share_w_le_0.20": result["median_source_share_w"] <= 0.20,
        "median_x_rho_ge_0.75": result["median_x_rho_func_256"] >= 0.75,
        "median_tau_le_1.05": result["median_holdout_total_error_ratio"] <= 1.05,
        "fraction_joint_conditions_at_least_0.60": result[
            "fraction_x_consolidated_modules"
        ]
        >= 0.60,
    }
    if all(pair_conditions.values()):
        verdict = "ERROR_SHAPING_GO"
    elif all(w_conditions.values()):
        verdict = "W_CONSOLIDATION_GO"
    elif all(x_conditions.values()):
        verdict = "X_CONSOLIDATION_GO"
    else:
        verdict = "FINAL_NO_GO"
    result.update(
        {
            "error_shaping_conditions": pair_conditions,
            "w_consolidation_conditions": w_conditions,
            "x_consolidation_conditions": x_conditions,
            "verdict": verdict,
        }
    )
    return result


def _scaling_analysis(
    operand_paths: list[Path],
    *,
    args: argparse.Namespace,
    output_root: Path,
    identity_lookup: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    audit = RequantizationAudit()
    selections: dict[str, Any] = {"schema_version": 1, "modules": {}}
    diagnostics_dir = output_root / "scaling_artifacts"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    for operand_path in operand_paths:
        operands = torch.load(operand_path, map_location="cpu", weights_only=True)
        module = operands["module"]
        print(f"\nscaling sweep: {module}", flush=True)
        raw_a = identity_lookup[module]["split_a"]
        raw_b = identity_lookup[module]["split_b"]
        identity_a = _scaling_identity_row(raw_a, split="split_a")
        identity_b = _scaling_identity_row(raw_b, split="split_b")
        module_rows_a = [identity_a]
        module_diagnostics: list[dict[str, Any]] = []
        scales: dict[float, torch.Tensor] = {}
        for alpha in ALPHAS:
            scale = smoothquant_scale(operands["split_a"]["x"], operands["weight"], alpha)
            scales[alpha] = scale.cpu()
            row, diagnostics = _evaluate_candidate(
                operands=operands,
                split="split_a",
                alpha=alpha,
                scale=scale,
                identity_total=float(raw_a["L_x"] + raw_a["L_w"] + raw_a["cross_term"]),
                args=args,
                audit=audit,
            )
            module_rows_a.append(row)
            module_diagnostics.append(diagnostics)
        selected, selection_status = _select_candidate(module_rows_a)
        selected["selected_on_split_A"] = True
        identity_b["selected_on_split_A"] = selected["alpha"] == "identity"
        rows.extend(module_rows_a)
        rows.append(identity_b)
        if selected["alpha"] == "identity":
            selected_b = identity_b
            selected_scale = torch.ones(int(operands["K"]), dtype=torch.float32)
        else:
            selected_alpha = float(selected["alpha"])
            selected_scale = scales[selected_alpha]
            selected_b, diagnostics_b = _evaluate_candidate(
                operands=operands,
                split="split_b",
                alpha=selected_alpha,
                scale=selected_scale,
                identity_total=float(raw_b["L_x"] + raw_b["L_w"] + raw_b["cross_term"]),
                args=args,
                audit=audit,
            )
            selected_b["selected_on_split_A"] = True
            rows.append(selected_b)
            module_diagnostics.append(diagnostics_b)
        selections["modules"][module] = {
            "alpha": selected["alpha"],
            "selection_status": selection_status,
            "scale": selected_scale,
            "split_a_pair_rho_256": selected["rho_pair_256"],
            "split_a_tau": selected["tau_total"],
        }
        torch.save(
            {
                "module": module,
                "selection_status": selection_status,
                "candidates": module_diagnostics,
            },
            diagnostics_dir / f"{module.replace('.', '__')}.pt",
        )
        del operands, module_rows_a, module_diagnostics, scales, selected_scale
        gc.collect()
        torch.cuda.empty_cache()

    rows.sort(
        key=lambda row: (
            row["layer"],
            row["module"],
            row["split"],
            -1 if row["alpha"] == "identity" else float(row["alpha"]),
        )
    )
    _write_csv(output_root / "stage1_5B_scaling_sweep.csv", rows)
    torch.save(selections, output_root / "selected_scales.pt")
    gate = _stage_b_gate(rows)
    audit_payload = {
        "quantizer_calls": audit.calls,
        "nonidentity_quantizer_calls": audit.nonidentity_calls,
        "all_nonidentity_candidates_requantized": all(
            bool(row["requantized"])
            for row in rows
            if row["alpha"] != "identity"
        ),
        "no_safe_scaling_modules": [
            module
            for module, item in selections["modules"].items()
            if item["selection_status"] == "NO_SAFE_SCALING"
        ],
    }
    return rows, gate, audit_payload


def _validation(
    *,
    output_root: Path,
    raw_rows: list[dict[str, Any]],
    scaling_rows: list[dict[str, Any]],
    audit: dict[str, Any] | None,
    split_disjoint: bool,
) -> dict[str, Any]:
    raw_artifacts = [
        torch.load(path, map_location="cpu", weights_only=True)
        for path in sorted((output_root / "raw_artifacts").glob("*.pt"))
    ]
    gram_checks = [
        {
            "module": artifact["module"],
            "split": split,
            **values["diagnostics"],
        }
        for artifact in raw_artifacts
        for split, values in artifact["grams"].items()
        if split != "combined"
    ]
    exact_max = max(
        float(item["exact_decomposition_max_relative_error"]) for item in gram_checks
    )
    cross_max = max(float(item["cross_gram_max_sampled_relative_error"]) for item in gram_checks)
    pair_max = max(float(item["pair_gram_max_relative_error"]) for item in gram_checks)
    norm_max = max(float(item["pair_total_norm_relative_error"]) for item in gram_checks)
    scaling_reparam = max(
        [float(row["reparameterization_relative_error"]) for row in scaling_rows]
        or [0.0]
    )
    scaling_decomposition = max(
        [
            float(row["exact_decomposition_relative_error"])
            for row in scaling_rows
            if row["alpha"] != "identity"
        ]
        or [0.0]
    )
    raw_spectral_residual = max(float(row["spectral_residual_max"]) for row in raw_rows)
    scaling_spectral_residual = max(
        [float(row["spectral_residual_max"]) for row in scaling_rows] or [0.0]
    )
    exact_minima = [
        float(row[key])
        for row in raw_rows
        for key in (
            "min_eigenvalue_x",
            "min_eigenvalue_w",
            "min_eigenvalue_pair",
            "min_eigenvalue_joint",
        )
        if row[key] not in (None, "") and not math.isnan(float(row[key]))
    ]
    checks = {
        "exact_decomposition_max_relative_error": exact_max,
        "cross_gram_max_sampled_relative_error": cross_max,
        "pair_gram_max_relative_error": pair_max,
        "total_norm_max_relative_error": norm_max,
        "reparameterization_max_relative_error": scaling_reparam,
        "scaling_exact_decomposition_max_relative_error": scaling_decomposition,
        "raw_spectral_relative_residual_max": raw_spectral_residual,
        "scaling_spectral_relative_residual_max": scaling_spectral_residual,
        "spectral_relative_residual_tolerance": 0.5,
        "minimum_explicit_eigenvalue": min(exact_minima) if exact_minima else None,
        "large_gram_psd_check": (
            "algebraic Gram certificate plus nonnegative randomized Ritz spectrum; "
            "minimum explicit eigenvalue is reported whenever the exact solver is used"
        ),
        "requantization_audit": audit,
        "split_a_b_interval_disjoint": split_disjoint,
    }
    passed = (
        exact_max <= 2e-5
        and cross_max <= 2e-4
        and pair_max <= 2e-6
        and norm_max <= 2e-4
        and scaling_reparam <= 2e-5
        and scaling_decomposition <= 2e-5
        and raw_spectral_residual <= 0.5
        and scaling_spectral_residual <= 0.5
        and (audit is None or audit["all_nonidentity_candidates_requantized"])
        and split_disjoint
    )
    return {"status": "passed" if passed else "failed", "checks": checks}


def _write_report(
    path: Path,
    *,
    stage_a: dict[str, Any],
    stage_b: dict[str, Any],
    validation: dict[str, Any],
    figures: dict[str, str],
    raw_rows: list[dict[str, Any]],
) -> None:
    k_values = sorted({int(row["K"]) for row in raw_rows})
    methods = sorted(
        {
            str(row[f"spectral_method_{source}"])
            for row in raw_rows
            if row["split"] == "combined"
            for source in ("x", "w", "pair", "joint")
        }
    )
    lines = [
        "# Stage 1.5 — Dual-Source Functional Spectrum 与 FP4 Error Consolidation",
        "",
        "## 结论",
        "",
        f"- Stage 1.5A: **{stage_a['verdict']}**",
        f"- Stage 1.5B: **{stage_b.get('verdict', stage_b.get('status', 'NOT_RUN'))}**",
        "",
        "所有 gate 均严格使用预注册的 rank-256 阈值；Split B 未用于 scaling 选择。",
        "",
        "## Stage 1.5A raw dual-source",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| median pair rho_func@256 | {stage_a['median_pair_rho_func_256']:.4f} |",
        f"| pair fraction >= 0.60 | {stage_a['fraction_pair_rho_func_256_ge_0.60']:.4f} |",
        f"| median pair gain@256 | {stage_a['median_pair_gain_256']:.4f} |",
        f"| median pair overlap@256 | {stage_a['median_pair_overlap_256']:.4f} |",
        f"| median joint rho_func@256 | {stage_a['median_joint_rho_func_256']:.4f} |",
        f"| median joint gain@256 | {stage_a['median_joint_gain_256']:.4f} |",
        f"| median joint overlap@256 | {stage_a['median_joint_overlap_256']:.4f} |",
        f"| median W-X rho_func@256 | {stage_a['median_w_minus_x_rho_func_256']:.4f} |",
        f"| median X source share | {stage_a['median_x_source_share']:.4f} |",
        "",
        "## Stage 1.5B held-out shaping",
        "",
    ]
    if stage_b.get("status") == "not_run":
        lines.append(f"未运行：{stage_b['reason']}")
    else:
        lines.extend(
            [
                "| metric | Split B value |",
                "|---|---:|",
                f"| median pair rho_func@256 | {stage_b['median_holdout_pair_rho_func_256']:.4f} |",
                f"| pair fraction >= 0.60 | {stage_b['fraction_holdout_pair_rho_func_256_ge_0.60']:.4f} |",
                f"| median total-error ratio tau | {stage_b['median_holdout_total_error_ratio']:.4f} |",
                f"| median pair gain vs identity | {stage_b['median_pair_gain_vs_identity']:.4f} |",
                f"| median X/W source share | {stage_b['median_source_share_x']:.4f} / {stage_b['median_source_share_w']:.4f} |",
                f"| median W rho_func@256 | {stage_b['median_w_rho_func_256']:.4f} |",
            ]
        )
    lines.extend(
        [
            "",
            "## 正确性与数值方法",
            "",
            f"Validation: **{validation['status']}**。G_X/G_W 使用 exact decomposition 中的 "
            "`E_X W` 与 `qX E_W`；H、G_P、总范数与重参数化恒等式均有独立数值审计。",
            "",
            f"本轮 K 取值为 {k_values}。实际 solver 为 {methods}；CSV/artifact 逐模块记录求解方法和 "
            "Ritz residual。大型 Gram 的 PSD 由其显式 Gram 构造给出代数证书，使用 exact solver 的"
            "矩阵另报告显式最小特征值。",
            "",
            "## 图表",
            "",
        ]
    )
    for name, figure in figures.items():
        relative = Path(figure).resolve().relative_to(path.parent.resolve()).as_posix()
        lines.append(f"- [{name}]({relative})")
    lines.extend(
        [
            "",
            "## Runtime caveat",
            "",
            "本轮允许 per-module D_s 作为 mechanism diagnostic。即使 gate 为 GO，也不能自动推出 "
            "q/k/v shared activation site 可部署、D_s 可免费折叠、或满足真实 K+S kernel 约束；这些需在 "
            "Stage 2/runtime claim 前单独验证。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    started = time.time()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    operands_dir = Path(args.operands_dir).resolve()
    operand_paths = sorted(operands_dir.glob("*.pt"))
    if len(operand_paths) != 7:
        raise ValueError(
            f"Stage-1 preregistered target set must contain exactly 7 operands, found {len(operand_paths)}"
        )
    if not Path(args.model).resolve().is_dir():
        raise FileNotFoundError(args.model)
    operand_headers = [
        torch.load(path, map_location="cpu", weights_only=True) for path in operand_paths
    ]
    expected_model = str(Path(args.model).resolve()).casefold()
    if any(
        str(Path(item["provenance"]["model"]).resolve()).casefold() != expected_model
        for item in operand_headers
    ):
        raise ValueError("saved operands do not belong to the requested model")
    split_disjoint = all(
        bool(item["provenance"]["split_manifest"]["selection_holdout_exact_disjoint"])
        and bool(item["provenance"]["split_manifest"]["selection_holdout_interval_disjoint"])
        for item in operand_headers
    )
    if not torch.cuda.is_available():
        raise RuntimeError("this full experiment requires CUDA for large functional-Gram operators")
    torch.backends.cuda.matmul.allow_tf32 = True

    raw_csv = output_root / "stage1_5A_joint_spectrum.csv"
    # Raw spectral vectors are intentionally not serialized.  Resume therefore
    # reruns Stage 1.5A to recompute split overlap rather than trusting a partial CSV.
    raw_rows, stage_a, identity_lookup = _raw_analysis(
        operand_paths, args=args, output_root=output_root
    )
    run_b = (stage_a["verdict"] == "CONDITIONAL_SHAPING" or args.force_stage1_5b) and not args.skip_stage1_5b
    if run_b:
        scaling_rows, stage_b, audit = _scaling_analysis(
            operand_paths,
            args=args,
            output_root=output_root,
            identity_lookup=identity_lookup,
        )
    else:
        scaling_rows = []
        audit = None
        _write_empty_scaling_csv(output_root / "stage1_5B_scaling_sweep.csv")
        stage_b = {
            "status": "not_run",
            "reason": (
                "explicit --skip-stage1-5b"
                if args.skip_stage1_5b
                else f"Stage 1.5A verdict was {stage_a['verdict']}"
            ),
        }
    gate = {"stage1_5A": stage_a, "stage1_5B": stage_b}
    write_json(output_root / "stage1_5_gate_summary.json", gate)
    validation = _validation(
        output_root=output_root,
        raw_rows=raw_rows,
        scaling_rows=scaling_rows,
        audit=audit,
        split_disjoint=split_disjoint,
    )
    write_json(output_root / "validation.json", validation)
    figures = render_figures(
        raw_csv,
        output_root / "stage1_5B_scaling_sweep.csv" if scaling_rows else None,
        output_root / "figures",
    )
    _write_report(
        output_root / "STAGE1_5_REPORT.md",
        stage_a=stage_a,
        stage_b=stage_b,
        validation=validation,
        figures=figures,
        raw_rows=raw_rows,
    )
    manifest = {
        "status": "complete" if validation["status"] == "passed" else "validation_failed",
        "config": vars(args),
        "protocol": {
            "model": str(Path(args.model).resolve()),
            "operands": str(operands_dir),
            "modules": [item["module"] for item in operand_headers],
            "split_a": "4 x 2048 WikiText2 tokens, 512 deterministic rows",
            "split_b": "disjoint 4 x 2048 WikiText2 tokens, 512 deterministic rows",
            "ranks": list(RANKS),
            "alphas": list(ALPHAS),
            "scale_bounds": [0.25, 4.0],
        },
        "environment": environment_metadata(),
        "git": git_metadata(REPO_ROOT),
        "elapsed_seconds": time.time() - started,
        "outputs": {
            "stage1_5A_csv": str(raw_csv),
            "stage1_5B_csv": str(output_root / "stage1_5B_scaling_sweep.csv"),
            "gate": str(output_root / "stage1_5_gate_summary.json"),
            "validation": str(output_root / "validation.json"),
            "report": str(output_root / "STAGE1_5_REPORT.md"),
            "figures": figures,
        },
    }
    write_json(output_root / "run_manifest.json", manifest)
    print(json.dumps(gate, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
