"""End-to-end Stage 1.6 factor-rank and spectral-outlier experiment."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from factor_rank_spectral_outlier.factor_spectrum import (  # noqa: E402
    SpectrumAnalysis,
    analyze_factor,
    analyze_psd,
    coverage_at,
    projector_overlap,
)
from factor_rank_spectral_outlier.functional_factors import (  # noqa: E402
    FunctionalFactorGrams,
    build_functional_factor_grams,
)
from factor_rank_spectral_outlier.plot_stage1_6 import (  # noqa: E402
    render_stage1_6_figures,
)
from factor_rank_spectral_outlier.removal_intervention import (  # noqa: E402
    RemovalResult,
    removal_norm_audit,
    run_removal,
)
from factor_rank_spectral_outlier.spectral_outlier_score import (  # noqa: E402
    SpectralOutlierScores,
    compute_spectral_outlier_scores,
    support_jaccard,
)
from fp4_residual_carrier.common.reproducibility import (  # noqa: E402
    environment_metadata,
    git_metadata,
    write_json,
)
from functional_gram.collect_stats import collect_decoder_operands  # noqa: E402


RANKS = (16, 32, 64, 128, 256, 512)
OVERLAP_RANKS = (64, 128, 256)
SUPPORT_BUDGETS = (32, 64, 128, 256)
DEFAULT_REMOVAL_BUDGETS = (16, 32, 64, 128, 256)
RUN_SCHEMA_VERSION = 1


def _default_wikitext_cache() -> Path:
    configured = os.environ.get("ARCQUANT_WIKITEXT_CACHE_DIR")
    if configured:
        return Path(configured)
    root = Path.home() / ".cache" / "huggingface" / "datasets" / "wikitext" / "wikitext-2-raw-v1"
    for candidate in reversed(sorted(root.glob("*/*"))):
        if (candidate / "wikitext-train.arrow").is_file():
            return candidate
    return root


def _integer_tuple(value: str) -> tuple[int, ...]:
    result = tuple(sorted({int(item.strip()) for item in value.split(",") if item.strip()}))
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return result


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=str(REPO_ROOT.parent / "modelzoo" / "Qwen" / "Qwen2.5-1.5B-Instruct"),
    )
    parser.add_argument("--wikitext-cache-dir", default=str(_default_wikitext_cache()))
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--operands-dir", default=None)
    parser.add_argument(
        "--modules",
        default="auto",
        help="auto, depth-control, or comma-separated layers.N.* module names",
    )
    parser.add_argument("--selection-samples", type=int, default=4)
    parser.add_argument("--holdout-samples", type=int, default=4)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--rows-per-split", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--analysis-device", default="auto", help="auto, cpu, or a CUDA device")
    parser.add_argument("--quant-row-chunk", type=int, default=128)
    parser.add_argument("--skip-collection", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--exact-max-dimension", type=int, default=4096)
    parser.add_argument("--removal-exact-max-dimension", type=int, default=2048)
    parser.add_argument("--bulk-rank", type=int, default=256)
    parser.add_argument("--removal-budgets", type=_integer_tuple, default=DEFAULT_REMOVAL_BUDGETS)
    parser.add_argument("--random-seeds", type=int, default=10)
    parser.add_argument("--randomized-oversample", type=int, default=16)
    parser.add_argument("--randomized-power-iterations", type=int, default=1)
    parser.add_argument("--slq-probes", type=int, default=8)
    parser.add_argument("--slq-steps", type=int, default=32)
    parser.add_argument("--cpu-threads", type=int, default=0)
    args = parser.parse_args(None if argv is None else list(argv))
    if args.modules == "auto":
        args.modules = None
    elif args.modules != "depth-control":
        args.modules = tuple(item.strip() for item in args.modules.split(",") if item.strip())
    if args.random_seeds <= 0 or args.bulk_rank <= 0:
        parser.error("random-seeds and bulk-rank must be positive")
    if args.cpu_threads > 0:
        torch.set_num_threads(args.cpu_threads)
    model_name = Path(args.model).name.replace("-Instruct", "").lower()
    if args.output_root is None:
        args.output_root = str(REPO_ROOT / "analysis" / "factor_rank_spectral_outlier" / model_name)
    if args.operands_dir is None:
        args.operands_dir = str(Path(args.output_root) / "operands")
    return args


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...] | None = None) -> None:
    if fields is None:
        fields = tuple(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _analysis_device(args: argparse.Namespace) -> torch.device:
    if args.analysis_device == "auto":
        return torch.device(args.device if torch.cuda.is_available() else "cpu")
    device = torch.device(args.analysis_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA analysis requested but unavailable")
    return device


def _coverage(analysis: SpectrumAnalysis, rank: int, *, functional: bool = False) -> float:
    return coverage_at(analysis, rank, functional=functional)


def _factor_row(
    header: dict[str, Any],
    source: str,
    split: str,
    name: str,
    role: str,
    analysis: SpectrumAnalysis,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "layer": int(header["layer"]),
        "module": header["module"],
        "module_type": header["module_type"],
        "source": source,
        "split": split,
        "K": int(header["K"]),
        "factor_name": name,
        "factor_role": role,
    }
    row.update({f"rho_{rank}": _coverage(analysis, rank) for rank in RANKS})
    row.update(
        {
            "entropy_effective_rank": analysis.entropy_effective_rank,
            "entropy_effective_rank_ratio": analysis.entropy_effective_rank_ratio,
            "participation_rank": analysis.participation_rank,
            "participation_rank_ratio": analysis.participation_rank_ratio,
            "spectral_method": analysis.method,
            "entropy_method": analysis.entropy_method,
            "spectral_residual_max": analysis.relative_residual_max,
        }
    )
    return row


def _mechanism_class(
    source: str,
    factor1: SpectrumAnalysis,
    factor2: SpectrumAnalysis,
    gram: SpectrumAnalysis,
) -> str:
    rho1 = _coverage(factor1, 256)
    rho2 = _coverage(factor2, 256)
    rho_g = _coverage(gram, 256)
    normalized_expansion = gram.entropy_effective_rank_ratio > 1.5 * max(
        factor1.entropy_effective_rank_ratio, factor2.entropy_effective_rank_ratio
    )
    if (rho1 >= 0.70 and rho2 >= 0.70 and rho_g <= 0.60) or normalized_expansion:
        return "RANK_EXPANSION"
    error_rho, mapping_rho = (rho1, rho2) if source == "X" else (rho2, rho1)
    if mapping_rho - error_rho >= 0.10 and abs(rho_g - error_rho) <= 0.10:
        return "ERROR_FACTOR"
    if error_rho - mapping_rho >= 0.10 and abs(rho_g - mapping_rho) <= 0.10:
        return "MAPPING_FACTOR"
    return "MIXED"


def _expansion_row(
    header: dict[str, Any],
    source: str,
    split: str,
    factor1: SpectrumAnalysis,
    factor2: SpectrumAnalysis,
    gram: SpectrumAnalysis,
) -> dict[str, Any]:
    denominator = max(factor1.entropy_effective_rank, factor2.entropy_effective_rank, 1e-30)
    product = max(factor1.entropy_effective_rank * factor2.entropy_effective_rank, 1e-30)
    return {
        "layer": int(header["layer"]),
        "module": header["module"],
        "module_type": header["module_type"],
        "source": source,
        "split": split,
        "K": int(header["K"]),
        "factor1_reff": factor1.entropy_effective_rank,
        "factor2_reff": factor2.entropy_effective_rank,
        "gram_reff": gram.entropy_effective_rank,
        "rank_inflation_ratio": gram.entropy_effective_rank / denominator,
        "product_fraction": gram.entropy_effective_rank / product,
        "factor_overlap_64": projector_overlap(factor1, factor2, 64),
        "factor_overlap_128": projector_overlap(factor1, factor2, 128),
        "factor_overlap_256": projector_overlap(factor1, factor2, 256),
        "mechanism_class": _mechanism_class(source, factor1, factor2, gram),
        "gram_spectral_method": gram.method,
        "gram_entropy_method": gram.entropy_method,
    }


def _channel_rows(
    header: dict[str, Any], source: str, split: str, scores: SpectralOutlierScores
) -> list[dict[str, Any]]:
    return [
        {
            "layer": int(header["layer"]),
            "module": header["module"],
            "module_type": header["module_type"],
            "source": source,
            "split": split,
            "channel": channel,
            "diag_J": float(scores.diag_j[channel].item()),
            "tail_energy_h": float(scores.tail_energy_h[channel].item()),
            "tail_ratio_h": float(scores.tail_ratio_h[channel].item()),
            "rank_J": float(scores.rank_j[channel].item()),
            "rank_h": float(scores.rank_h[channel].item()),
            "rank_tail_ratio": float(scores.rank_tail_ratio[channel].item()),
        }
        for channel in range(scores.diag_j.numel())
    ]


def _collapse_row(
    header: dict[str, Any],
    source: str,
    split: str,
    method: str,
    random_seed: str | int,
    raw: SpectrumAnalysis,
    result: RemovalResult,
) -> dict[str, Any]:
    raw_func = _coverage(raw, 256, functional=True)
    raw_struct = _coverage(raw, 256)
    return {
        "layer": int(header["layer"]),
        "module": header["module"],
        "module_type": header["module_type"],
        "source": source,
        "split": split,
        "removal_method": method,
        "random_seed": random_seed,
        "remove_budget": result.remove_budget,
        "raw_rho_func_256": raw_func,
        "remain_rho_func_64": result.rho_func_64,
        "remain_rho_func_128": result.rho_func_128,
        "remain_rho_func_256": result.rho_func_256,
        "delta_func_256": result.rho_func_256 - raw_func,
        "raw_rho_struct_256": raw_struct,
        "remain_rho_struct_64": result.rho_struct_64,
        "remain_rho_struct_128": result.rho_struct_128,
        "remain_rho_struct_256": result.rho_struct_256,
        "delta_struct_256": result.rho_struct_256 - raw_struct,
        "remaining_trace": result.remaining_trace,
        "remaining_total_functional_error": result.remaining_total_functional_error,
        "random_rho_func_256_std": 0.0,
        "random_rho_struct_256_std": 0.0,
        "spectral_method": result.analysis.method,
        "spectral_residual_max": result.analysis.relative_residual_max,
    }


def _aggregate_random_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["layer"], row["module"], row["source"], row["split"], row["remove_budget"])
        groups.setdefault(key, []).append(row)
    result: list[dict[str, Any]] = []
    mean_fields = (
        "remain_rho_func_64", "remain_rho_func_128", "remain_rho_func_256",
        "delta_func_256", "remain_rho_struct_64", "remain_rho_struct_128",
        "remain_rho_struct_256", "delta_struct_256", "remaining_trace",
        "remaining_total_functional_error",
    )
    for group in groups.values():
        row = dict(group[0])
        row["random_seed"] = "aggregate"
        for field in mean_fields:
            row[field] = statistics.fmean(float(item[field]) for item in group)
        row["random_rho_func_256_std"] = statistics.pstdev(
            float(item["remain_rho_func_256"]) for item in group
        )
        row["random_rho_struct_256_std"] = statistics.pstdev(
            float(item["remain_rho_struct_256"]) for item in group
        )
        row["spectral_method"] = "mean_of_random_seed_interventions"
        row["spectral_residual_max"] = max(float(item["spectral_residual_max"]) for item in group)
        result.append(row)
    return result


def _curve_rows(
    header: dict[str, Any],
    source: str,
    split: str,
    curve_group: str,
    curve_name: str,
    analysis: SpectrumAnalysis,
    *,
    remove_budget: int = 0,
    maximum_rank: int | None = None,
) -> list[dict[str, Any]]:
    limit = analysis.rho_struct_curve.numel()
    if maximum_rank is not None:
        limit = min(limit, maximum_rank)
    return [
        {
            "layer": int(header["layer"]),
            "module": header["module"],
            "module_type": header["module_type"],
            "source": source,
            "split": split,
            "curve_group": curve_group,
            "curve_name": curve_name,
            "remove_budget": remove_budget,
            "rank": rank,
            "rho": float(analysis.rho_struct_curve[rank - 1].item()),
        }
        for rank in range(1, limit + 1)
    ]


def _removal_call(
    args: argparse.Namespace,
    gram: torch.Tensor,
    order: torch.Tensor,
    budget: int,
    seed: int,
    raw: SpectrumAnalysis,
) -> RemovalResult:
    return run_removal(
        gram,
        order,
        remove_budget=budget,
        exact_max_dimension=args.removal_exact_max_dimension,
        seed=seed,
        oversample=args.randomized_oversample,
        power_iterations=args.randomized_power_iterations,
        slq_probes=0,
        slq_steps=args.slq_steps,
        raw_analysis=raw,
    )


def _process_module(
    operand_path: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, list[dict[str, Any]]]:
    operands = torch.load(operand_path, map_location="cpu", weights_only=True)
    header = {
        "layer": int(operands["layer"]),
        "module": operands["module"],
        "module_type": operands["module_type"],
        "K": int(operands["K"]),
    }
    k = int(operands["K"])
    accumulation_dtype = torch.float64 if k <= args.exact_max_dimension else torch.float32
    print(
        f"Stage 1.6: {header['module']} (K={k}, {device}, {accumulation_dtype})",
        flush=True,
    )
    output: dict[str, list[dict[str, Any]]] = {
        "factor_rows": [], "expansion_rows": [], "channel_rows": [],
        "collapse_rows": [], "curve_rows": [], "stability_rows": [],
        "validation_rows": [],
    }
    x_by_split = {
        "split_a": operands["split_a"]["x"],
        "split_b": operands["split_b"]["x"],
        "combined": torch.cat((operands["split_a"]["x"], operands["split_b"]["x"]), dim=0),
    }
    qx_by_split = {
        "split_a": operands["split_a"]["qx"],
        "split_b": operands["split_b"]["qx"],
        "combined": torch.cat((operands["split_a"]["qx"], operands["split_b"]["qx"]), dim=0),
    }
    weight = operands["weight"].to(device)
    qweight = operands["qweight"].to(device)

    for source_index, source in enumerate(("X", "W")):
        split_data: dict[str, dict[str, Any]] = {}
        invariant_factor2: SpectrumAnalysis | None = None
        for split_index, split in enumerate(("split_a", "split_b", "combined")):
            built = build_functional_factor_grams(
                x_by_split[split].to(device), qx_by_split[split].to(device), weight, qweight,
                source=source, accumulation_dtype=accumulation_dtype, run_sanity=True,
                retain_covariances=False,
            )
            factor_dtype = torch.float64 if k <= args.exact_max_dimension else torch.float32
            factor1 = analyze_factor(built.factor1, max_vectors=max(OVERLAP_RANKS), compute_dtype=factor_dtype)
            if invariant_factor2 is None:
                invariant_factor2 = analyze_factor(
                    built.factor2, max_vectors=max(OVERLAP_RANKS), compute_dtype=factor_dtype
                )
            factor2 = invariant_factor2
            gram_analysis = analyze_psd(
                built.gram,
                max_rank=max(max(RANKS), args.bulk_rank),
                exact=k <= args.exact_max_dimension,
                seed=args.seed + int(header["layer"]) * 100_003 + source_index * 10_007 + split_index,
                oversample=args.randomized_oversample,
                power_iterations=args.randomized_power_iterations,
                slq_probes=args.slq_probes,
                slq_steps=args.slq_steps,
            )
            output["factor_rows"].extend(
                (
                    _factor_row(header, source, split, built.factor1_name, "factor1", factor1),
                    _factor_row(header, source, split, built.factor2_name, "factor2", factor2),
                    _factor_row(header, source, split, f"G_{source}", "gram", gram_analysis),
                )
            )
            output["expansion_rows"].append(
                _expansion_row(header, source, split, factor1, factor2, gram_analysis)
            )
            if split == "combined":
                output["curve_rows"].extend(
                    _curve_rows(header, source, split, "factor", "factor1", factor1, maximum_rank=512)
                    + _curve_rows(header, source, split, "factor", "factor2", factor2, maximum_rank=512)
                    + _curve_rows(header, source, split, "factor", "gram", gram_analysis, maximum_rank=512)
                )
            if split in ("split_a", "split_b"):
                scores = compute_spectral_outlier_scores(
                    built.gram, gram_analysis, bulk_rank=args.bulk_rank
                )
                output["channel_rows"].extend(_channel_rows(header, source, split, scores))
                output["validation_rows"].append(
                    {
                        "module": header["module"], "source": source, "split": split,
                        **built.diagnostics, **scores.diagnostics,
                        "raw_min_eigenvalue": gram_analysis.raw_min_eigenvalue,
                        "spectral_method": gram_analysis.method,
                        "spectral_residual_max": gram_analysis.relative_residual_max,
                    }
                )
                split_data[split] = {
                    "gram": built.gram.detach(), "factor1": built.factor1.detach(),
                    "factor2": built.factor2.detach(), "analysis": gram_analysis, "scores": scores,
                }
            else:
                del built
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        for budget in SUPPORT_BUDGETS:
            output["stability_rows"].append(
                {
                    **header, "source": source, "budget": min(budget, k),
                    "jaccard": support_jaccard(
                        split_data["split_a"]["scores"].order_h,
                        split_data["split_b"]["scores"].order_h,
                        budget,
                    ),
                }
            )

        for split_index, split in enumerate(("split_a", "split_b")):
            data = split_data[split]
            raw = data["analysis"]
            scores = data["scores"]
            method_orders = {
                "top_h": scores.order_h,
                "top_tail_ratio": scores.order_tail_ratio,
                "top_J": scores.order_j,
            }
            for method_index, (method, order) in enumerate(method_orders.items()):
                for budget in (0, *args.removal_budgets):
                    result = _removal_call(
                        args, data["gram"], order, budget,
                        args.seed + int(header["layer"]) * 1_000_003 + source_index * 100_003
                        + split_index * 10_007 + method_index * 1_009 + budget,
                        raw,
                    )
                    output["collapse_rows"].append(
                        _collapse_row(header, source, split, method, "none", raw, result)
                    )
                    if method == "top_h" and budget in (0, 64, 128):
                        output["curve_rows"].extend(
                            _curve_rows(
                                header, source, split, "remaining", "top_h", result.analysis,
                                remove_budget=budget,
                            )
                        )
                    if method == "top_h" and budget == min(args.removal_budgets):
                        audit = removal_norm_audit(
                            data["factor1"], data["factor2"], data["gram"], result.remaining_indices
                        )
                        output["validation_rows"].append(
                            {
                                "module": header["module"], "source": source, "split": split,
                                "removal_norm_relative_error": audit,
                                "removal_budget": budget,
                            }
                        )

            random_rows: list[dict[str, Any]] = []
            for random_seed in range(args.random_seeds):
                generator = torch.Generator(device="cpu").manual_seed(
                    args.seed + int(header["layer"]) * 1_000_003 + source_index * 100_003
                    + split_index * 10_007 + random_seed
                )
                order = torch.randperm(k, generator=generator)
                for budget in (0, *args.removal_budgets):
                    removal = _removal_call(
                        args, data["gram"], order, budget,
                        args.seed + 50_000_003 + random_seed * 1_009 + budget,
                        raw,
                    )
                    random_rows.append(
                        _collapse_row(header, source, split, "random", random_seed, raw, removal)
                    )
            output["collapse_rows"].extend(random_rows)
            output["collapse_rows"].extend(_aggregate_random_rows(random_rows))

        # Split-A support applied without reselection on Split B.
        data_b = split_data["split_b"]
        for budget in (0, *args.removal_budgets):
            result = _removal_call(
                args, data_b["gram"], split_data["split_a"]["scores"].order_h, budget,
                args.seed + 90_000_019 + int(header["layer"]) * 1_009 + source_index * 97 + budget,
                data_b["analysis"],
            )
            output["collapse_rows"].append(
                _collapse_row(
                    header, source, "split_b", "top_h_selected_split_a", "none",
                    data_b["analysis"], result,
                )
            )
        del split_data
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    del operands, weight, qweight, x_by_split, qx_by_split
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return output


def _configuration_signature(args: argparse.Namespace, operand_path: Path) -> str:
    payload = {
        "run_schema_version": RUN_SCHEMA_VERSION,
        "operand": str(operand_path.resolve()),
        "operand_bytes": operand_path.stat().st_size,
        "operand_mtime_ns": operand_path.stat().st_mtime_ns,
        "exact_max_dimension": args.exact_max_dimension,
        "removal_exact_max_dimension": args.removal_exact_max_dimension,
        "bulk_rank": args.bulk_rank,
        "removal_budgets": args.removal_budgets,
        "random_seeds": args.random_seeds,
        "oversample": args.randomized_oversample,
        "power_iterations": args.randomized_power_iterations,
        "slq_probes": args.slq_probes,
        "slq_steps": args.slq_steps,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _module_results(
    paths: list[Path], args: argparse.Namespace, output_root: Path, device: torch.device
) -> list[dict[str, list[dict[str, Any]]]]:
    artifacts = output_root / "module_artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, list[dict[str, Any]]]] = []
    for path in paths:
        target = artifacts / path.name
        signature = _configuration_signature(args, path)
        if target.is_file() and not args.no_resume:
            cached = torch.load(target, map_location="cpu", weights_only=False)
            if cached.get("configuration_signature") == signature:
                print(f"resuming {path.stem}", flush=True)
                results.append(cached["result"])
                continue
        result = _process_module(path, args, device)
        torch.save({"configuration_signature": signature, "result": result}, target)
        results.append(result)
    return results


def _canonical_protocol(args: argparse.Namespace, module_count: int) -> bool:
    return (
        tuple(args.removal_budgets) == DEFAULT_REMOVAL_BUDGETS
        and args.random_seeds == 10
        and args.bulk_rank == 256
        and module_count == 7
    )


def _gate(
    expansion_rows: list[dict[str, Any]],
    collapse_rows: list[dict[str, Any]],
    *,
    canonical: bool,
) -> dict[str, Any]:
    factor_diagnosis: dict[str, Any] = {}
    label_map = {
        "ERROR_FACTOR": "ERROR_FACTOR",
        "MAPPING_FACTOR": "MAPPING_FACTOR",
        "RANK_EXPANSION": "RANK_EXPANSION",
        "MIXED": "MIXED",
    }
    combined = [row for row in expansion_rows if row["split"] == "combined"]
    for source in ("X", "W"):
        rows = [row for row in combined if row["source"] == source]
        counts = Counter(row["mechanism_class"] for row in rows)
        dominant = counts.most_common(1)[0][0] if counts else "MIXED"
        factor_diagnosis["x_source" if source == "X" else "w_source"] = {
            "dominant_mechanism": label_map[dominant],
            "mechanism_counts": dict(counts),
            "median_rank_inflation_ratio": statistics.median(
                float(row["rank_inflation_ratio"]) for row in rows
            ) if rows else float("nan"),
        }

    candidate_budgets = sorted(
        {
            int(row["remove_budget"]) for row in collapse_rows
            if row["removal_method"] == "top_h" and 0 < int(row["remove_budget"]) <= 128
        }
    )
    diagnostics: list[dict[str, Any]] = []
    for budget in candidate_budgets:
        top_h = [
            row for row in collapse_rows
            if row["split"] == "split_a" and row["removal_method"] == "top_h"
            and row["random_seed"] == "none" and int(row["remove_budget"]) == budget
        ]
        top_j_lookup = {
            (row["module"], row["source"]): row for row in collapse_rows
            if row["split"] == "split_a" and row["removal_method"] == "top_J"
            and row["random_seed"] == "none" and int(row["remove_budget"]) == budget
        }
        holdout_lookup = {
            (row["module"], row["source"]): row for row in collapse_rows
            if row["split"] == "split_b" and row["removal_method"] == "top_h_selected_split_a"
            and int(row["remove_budget"]) == budget
        }
        gains = [
            float(row["remain_rho_func_256"]) - float(top_j_lookup[(row["module"], row["source"])]["remain_rho_func_256"])
            for row in top_h if (row["module"], row["source"]) in top_j_lookup
        ]
        holdout = [
            float(holdout_lookup[(row["module"], row["source"])]["remain_rho_func_256"])
            for row in top_h if (row["module"], row["source"]) in holdout_lookup
        ]
        diagnostic = {
                "budget": budget,
                "median_delta_collapse": statistics.median(float(row["delta_func_256"]) for row in top_h),
                "median_remaining_rho_func_256": statistics.median(float(row["remain_rho_func_256"]) for row in top_h),
                "median_gain_vs_top_J": statistics.median(gains),
                "median_holdout_remaining_rho_func_256": statistics.median(holdout),
            }
        diagnostic["conditions"] = {
            "median_delta_collapse_ge_0.15": diagnostic["median_delta_collapse"] >= 0.15,
            "median_remaining_rho_func_256_ge_0.75": diagnostic[
                "median_remaining_rho_func_256"
            ] >= 0.75,
            "median_gain_vs_top_J_ge_0.08": diagnostic["median_gain_vs_top_J"] >= 0.08,
            "median_holdout_remaining_rho_func_256_ge_0.70": diagnostic[
                "median_holdout_remaining_rho_func_256"
            ] >= 0.70,
        }
        diagnostics.append(diagnostic)
    qualifying = [item for item in diagnostics if all(item["conditions"].values())]
    candidates = qualifying or diagnostics
    best = max(candidates, key=lambda item: item["median_delta_collapse"], default={
        "budget": 0, "median_delta_collapse": float("nan"),
        "median_remaining_rho_func_256": float("nan"), "median_gain_vs_top_J": float("nan"),
        "median_holdout_remaining_rho_func_256": float("nan"),
        "conditions": {
            "median_delta_collapse_ge_0.15": False,
            "median_remaining_rho_func_256_ge_0.75": False,
            "median_gain_vs_top_J_ge_0.08": False,
            "median_holdout_remaining_rho_func_256_ge_0.70": False,
        },
    })
    conditions = best["conditions"]
    if all(conditions.values()):
        provisional = "LOW_RANK_BULK_GO"
    elif (
        0.05 <= best["median_delta_collapse"] < 0.15
        or 0.65 <= best["median_remaining_rho_func_256"] < 0.75
    ):
        provisional = "WEAK_SPARSE_TAIL"
    else:
        provisional = "GLOBAL_HIGH_RANK"
    spectral = {
        "best_budget": best["budget"],
        "median_delta_collapse": best["median_delta_collapse"],
        "median_remaining_rho_func_256": best["median_remaining_rho_func_256"],
        "median_gain_vs_top_J": best["median_gain_vs_top_J"],
        "median_holdout_remaining_rho_func_256": best[
            "median_holdout_remaining_rho_func_256"
        ],
        "conditions": conditions,
        "budget_diagnostics": diagnostics,
        "verdict": provisional if canonical else "SMOKE_ONLY",
    }
    if not canonical:
        spectral["provisional_verdict"] = provisional
        spectral["noncanonical_reason"] = (
            "gate requires seven preregistered modules, budgets 16/32/64/128/256, "
            "bulk rank 256, and ten random seeds"
        )
    return {"factor_diagnosis": factor_diagnosis, "spectral_outlier": spectral}


def _validation(
    validation_rows: list[dict[str, Any]], operand_headers: list[dict[str, Any]]
) -> dict[str, Any]:
    identity = [row for row in validation_rows if "max_relative_error" in row]
    removal = [row for row in validation_rows if "removal_norm_relative_error" in row]
    max_factor = max((float(row["max_relative_error"]) for row in identity), default=0.0)
    max_tail = max(
        (float(row["tail_energy_identity_relative_error"]) for row in identity), default=0.0
    )
    max_removal = max(
        (float(row["removal_norm_relative_error"]) for row in removal), default=0.0
    )
    max_residual = max(
        (float(row.get("spectral_residual_max", 0.0)) for row in identity), default=0.0
    )
    split_disjoint = all(
        bool(item["provenance"]["split_manifest"]["selection_holdout_exact_disjoint"])
        and bool(item["provenance"]["split_manifest"]["selection_holdout_interval_disjoint"])
        for item in operand_headers
    )
    passed = (
        max_factor <= 3e-4 and max_tail <= 3e-6 and max_removal <= 3e-4
        and max_residual <= 0.5 and split_disjoint
    )
    return {
        "status": "passed" if passed else "failed",
        "checks": {
            "factor_hadamard_functional_norm_max_relative_error": max_factor,
            "tail_energy_identity_max_relative_error": max_tail,
            "removal_norm_max_relative_error": max_removal,
            "randomized_spectral_residual_max": max_residual,
            "split_a_b_interval_disjoint": split_disjoint,
            "psd_certificate": (
                "factor covariances and Functional Grams are explicit Gram/Hadamard products; "
                "exact paths additionally reject material negative eigenvalues"
            ),
        },
        "details": validation_rows,
    }


def _write_report(
    path: Path,
    gate: dict[str, Any],
    validation: dict[str, Any],
    figures: dict[str, str],
    canonical: bool,
) -> None:
    spectral = gate["spectral_outlier"]
    lines = [
        "# Stage 1.6 — Factor Spectrum 与 Spectral-Outlier Channel Analysis",
        "",
        "## 结论",
        "",
        f"- Protocol: **{'canonical' if canonical else 'smoke/noncanonical'}**",
        f"- Spectral-outlier verdict: **{spectral['verdict']}**",
    ]
    if "provisional_verdict" in spectral:
        lines.append(f"- Provisional diagnostic only: **{spectral['provisional_verdict']}**")
    lines.extend(
        [
            f"- Validation: **{validation['status']}**",
            "",
            "删除 channel 仅为 mechanism intervention，不等价于 runtime K+S correction。",
            "",
            "## Gate metrics",
            "",
            "| metric | value |",
            "|---|---:|",
            f"| best S | {spectral['best_budget']} |",
            f"| median collapse gain | {spectral['median_delta_collapse']:.4f} |",
            f"| median remaining functional rho@256 | {spectral['median_remaining_rho_func_256']:.4f} |",
            f"| median gain vs top-J | {spectral['median_gain_vs_top_J']:.4f} |",
            f"| median Split-A support on Split-B rho@256 | {spectral['median_holdout_remaining_rho_func_256']:.4f} |",
            "",
            "## Figures",
            "",
        ]
    )
    for name, figure in figures.items():
        relative = Path(figure).resolve().relative_to(path.parent.resolve()).as_posix()
        lines.append(f"- [{name}]({relative})")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    started = time.time()
    output_root = Path(args.output_root).resolve()
    operands_dir = Path(args.operands_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    model_path = Path(args.model).resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)
    if not args.skip_collection:
        collect_decoder_operands(
            model_path=model_path,
            wikitext_cache_dir=args.wikitext_cache_dir,
            output_dir=operands_dir,
            modules=args.modules,
            selection_samples=args.selection_samples,
            holdout_samples=args.holdout_samples,
            seqlen=args.seqlen,
            rows_per_split=args.rows_per_split,
            seed=args.seed,
            device=args.device,
            quant_row_chunk=args.quant_row_chunk,
            resume=not args.no_resume,
        )
    if not operands_dir.is_dir():
        raise FileNotFoundError(operands_dir)
    operand_paths = sorted(operands_dir.glob("*.pt"))
    if args.modules not in (None, "depth-control"):
        wanted = set(args.modules)
        operand_paths = [
            path for path in operand_paths
            if torch.load(path, map_location="cpu", weights_only=True)["module"] in wanted
        ]
    if not operand_paths:
        raise ValueError("no operand artifacts selected")
    operand_headers = []
    for path in operand_paths:
        item = torch.load(path, map_location="cpu", weights_only=True)
        operand_headers.append(
            {
                "module": item["module"],
                "provenance": item["provenance"],
            }
        )
        del item
    expected_model = str(model_path).casefold()
    if any(str(Path(item["provenance"]["model"]).resolve()).casefold() != expected_model for item in operand_headers):
        raise ValueError("operand provenance model does not match --model")
    device = _analysis_device(args)
    results = _module_results(operand_paths, args, output_root, device)
    keys = results[0].keys()
    merged = {key: [row for result in results for row in result[key]] for key in keys}
    _write_csv(output_root / "factor_spectrum_summary.csv", merged["factor_rows"])
    _write_csv(output_root / "rank_expansion_summary.csv", merged["expansion_rows"])
    _write_csv(output_root / "spectral_outlier_channels.csv", merged["channel_rows"])
    _write_csv(output_root / "spectral_collapse_summary.csv", merged["collapse_rows"])
    _write_csv(output_root / "spectrum_curves.csv", merged["curve_rows"])
    _write_csv(output_root / "support_stability_summary.csv", merged["stability_rows"])
    canonical = _canonical_protocol(args, len(operand_paths))
    gate = _gate(merged["expansion_rows"], merged["collapse_rows"], canonical=canonical)
    validation = _validation(merged["validation_rows"], operand_headers)
    write_json(output_root / "stage1_6_gate_summary.json", gate)
    write_json(output_root / "validation.json", validation)
    figures = render_stage1_6_figures(output_root)
    _write_report(output_root / "STAGE1_6_REPORT.md", gate, validation, figures, canonical)
    manifest = {
        "status": "complete" if validation["status"] == "passed" else "validation_failed",
        "canonical_protocol": canonical,
        "config": vars(args),
        "protocol": {
            "model": str(model_path),
            "modules": [item["module"] for item in operand_headers],
            "sources": ["X: (X-qX)W", "W: qX(W-qW)"],
            "factor_ranks": list(RANKS),
            "bulk_rank": args.bulk_rank,
            "removal_budgets": list(args.removal_budgets),
            "random_seeds": args.random_seeds,
            "split_a_b_disjoint": True,
        },
        "environment": environment_metadata(),
        "git": git_metadata(REPO_ROOT),
        "elapsed_seconds": time.time() - started,
        "outputs": {
            "factor_spectrum_summary": str(output_root / "factor_spectrum_summary.csv"),
            "rank_expansion_summary": str(output_root / "rank_expansion_summary.csv"),
            "spectral_outlier_channels": str(output_root / "spectral_outlier_channels.csv"),
            "spectral_collapse_summary": str(output_root / "spectral_collapse_summary.csv"),
            "gate": str(output_root / "stage1_6_gate_summary.json"),
            "validation": str(output_root / "validation.json"),
            "report": str(output_root / "STAGE1_6_REPORT.md"),
            "figures": figures,
        },
    }
    # Path objects are not JSON serializable.
    manifest["config"] = {
        key: str(value) if isinstance(value, Path) else list(value) if isinstance(value, tuple) else value
        for key, value in manifest["config"].items()
    }
    write_json(output_root / "run_manifest.json", manifest)
    print(json.dumps(gate, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
