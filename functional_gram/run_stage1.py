"""End-to-end Stage-1 Functional Gram experiment runner."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fp4_residual_carrier.common.reproducibility import (  # noqa: E402
    environment_metadata,
    git_metadata,
    write_json,
)
from functional_gram.build_gram import build_functional_gram  # noqa: E402
from functional_gram.collect_stats import collect_decoder_operands  # noqa: E402
from functional_gram.eig_analysis import (  # noqa: E402
    EigenAnalysis,
    analyze_gram,
    analyze_gram_randomized,
)
from functional_gram.head_analysis import run_head_analysis  # noqa: E402
from functional_gram.plot_stage1 import render_stage1_figures  # noqa: E402
from functional_gram.split_stability import projector_overlap_curve  # noqa: E402


BASE_RANKS = (16, 32, 64, 128, 256, 512)
LARGE_K_RANKS = (896, 1024)


def equal_fraction_rank(k: int) -> int:
    """Rank matching the preregistered 256/4096 = 6.25% budget."""

    if k <= 0:
        raise ValueError("K must be positive")
    return max(1, min(k, int(round(k / 16))))


def summary_ranks_for_k(k: int) -> tuple[int, ...]:
    ranks = {rank for rank in BASE_RANKS if rank <= k}
    ranks.add(equal_fraction_rank(k))
    if k > 4096:
        ranks.update(rank for rank in LARGE_K_RANKS if rank <= k)
    return tuple(sorted(ranks))


def _default_wikitext_cache() -> Path:
    configured = os.environ.get("ARCQUANT_WIKITEXT_CACHE_DIR")
    if configured:
        return Path(configured)
    candidates = sorted(
        (Path.home() / ".cache" / "huggingface" / "datasets" / "wikitext" / "wikitext-2-raw-v1").glob("*/*")
    )
    for candidate in reversed(candidates):
        if (candidate / "wikitext-train.arrow").is_file():
            return candidate
    return Path.home() / ".cache" / "huggingface" / "datasets" / "wikitext"


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=str(REPO_ROOT.parent / "modelzoo" / "Qwen" / "Qwen2.5-1.5B-Instruct"),
    )
    parser.add_argument("--wikitext-cache-dir", default=str(_default_wikitext_cache()))
    parser.add_argument(
        "--output-root",
        default=None,
        help="Output directory; defaults to analysis/functional_gram/<model-name>",
    )
    parser.add_argument(
        "--modules",
        default="auto",
        help=(
            "auto, depth-control, attention-all, down-depth-scan, or "
            "comma-separated layers.N.* names"
        ),
    )
    parser.add_argument("--selection-samples", type=int, default=4)
    parser.add_argument("--holdout-samples", type=int, default=4)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--rows-per-split", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--quant-row-chunk", type=int, default=128)
    parser.add_argument("--eigh-device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--gpu-max-k", type=int, default=4096)
    parser.add_argument(
        "--randomized-min-k",
        type=int,
        default=4097,
        help="Use deterministic top-512 subspace iteration when K is at least this value",
    )
    parser.add_argument("--randomized-oversample", type=int, default=16)
    parser.add_argument("--randomized-power-iterations", type=int, default=1)
    parser.add_argument(
        "--gram-dtype", choices=("auto", "float32", "float64"), default="auto"
    )
    parser.add_argument(
        "--omit-full-gram",
        action="store_true",
        help="Do not serialize the combined KxK Gram (recommended for 8B models)",
    )
    parser.add_argument("--cpu-threads", type=int, default=0)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--skip-collection", action="store_true")
    parser.add_argument(
        "--head-analysis",
        action="store_true",
        help="Analyze every exact head_dim-wide q/k/v head on the combined split",
    )
    parser.add_argument(
        "--head-analysis-layers",
        default="all",
        help="all or a comma-separated layer list limiting the expensive per-head pass",
    )
    parser.add_argument("--head-oversample", type=int, default=16)
    parser.add_argument("--head-power-iterations", type=int, default=1)
    parser.add_argument("--head-overlap-rank", type=int, default=32)
    args = parser.parse_args(None if argv is None else list(argv))
    args.module_scope = args.modules
    if args.modules == "auto":
        args.modules = None
    elif args.modules not in {"depth-control", "attention-all", "down-depth-scan"}:
        args.modules = tuple(
            item.strip() for item in args.modules.split(",") if item.strip()
        )
        args.module_scope = "explicit"
    if args.head_analysis_layers == "all":
        args.head_analysis_layers = None
    else:
        try:
            args.head_analysis_layers = tuple(
                sorted(
                    {
                        int(item.strip())
                        for item in args.head_analysis_layers.split(",")
                        if item.strip()
                    }
                )
            )
        except ValueError as error:
            parser.error(f"invalid --head-analysis-layers: {error}")
        if not args.head_analysis_layers or min(args.head_analysis_layers) < 0:
            parser.error("--head-analysis-layers must contain nonnegative layers")
        if not args.head_analysis:
            parser.error("--head-analysis-layers requires --head-analysis")
    if args.cpu_threads > 0:
        torch.set_num_threads(args.cpu_threads)
    if args.output_root is None:
        args.output_root = str(REPO_ROOT / "analysis" / "functional_gram" / Path(args.model).name)
    return args


def _artifact_name(module: str, source: str) -> str:
    return f"{module.replace('.', '__')}__source_{source}.pt"


def _device_for_k(args: argparse.Namespace, k: int) -> torch.device:
    if args.eigh_device == "cpu":
        return torch.device("cpu")
    if args.eigh_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA eigendecomposition requested but unavailable")
        return torch.device(args.device)
    if torch.cuda.is_available() and (k <= args.gpu_max_k or k >= args.randomized_min_k):
        return torch.device(args.device)
    return torch.device("cpu")


def _analysis_payload(analysis: EigenAnalysis, max_rank: int) -> dict[str, Any]:
    return {
        "eigenvalues": analysis.eigenvalues.detach().cpu(),
        # On CPU, a narrow column slice otherwise keeps the full KxK storage.
        # Materialize only the requested top-r block before torch.save.
        "top_eigenvectors": analysis.eigenvectors[:, :max_rank].detach().cpu().contiguous(),
        "rho_struct_curve": analysis.rho_struct_curve.detach().cpu(),
        "rho_func_curve": analysis.rho_func_curve.detach().cpu(),
        "effective_rank": analysis.effective_rank,
        "effective_rank_ratio": analysis.effective_rank_ratio,
        "trace_G": analysis.trace_g,
        "one_G_one": analysis.one_g_one,
        "raw_min_eigenvalue": analysis.raw_min_eigenvalue,
        "clamped_eigenvalue_count": analysis.clamped_eigenvalue_count,
        "method": analysis.method,
        "relative_residual_max": analysis.relative_residual_max,
    }


def _gram_dtype(args: argparse.Namespace, *, randomized: bool) -> torch.dtype:
    if args.gram_dtype == "float32":
        return torch.float32
    if args.gram_dtype == "float64":
        return torch.float64
    return torch.float32 if randomized else torch.float64


def _spectral_analysis(
    gram: torch.Tensor,
    *,
    args: argparse.Namespace,
    max_rank: int,
    randomized: bool,
    seed: int,
) -> EigenAnalysis:
    if randomized:
        return analyze_gram_randomized(
            gram,
            max_rank=max_rank,
            oversample=args.randomized_oversample,
            power_iterations=args.randomized_power_iterations,
            seed=seed,
        )
    return analyze_gram(gram, eigen_dtype=torch.float64)


@torch.no_grad()
def _source_sanity(
    gram: torch.Tensor,
    source: str,
    activation_batches: list[tuple[torch.Tensor, torch.Tensor]],
    weight: torch.Tensor,
    qweight: torch.Tensor,
) -> dict[str, Any]:
    device = gram.device
    sanity_dtype = torch.float64 if gram.dtype == torch.float64 else torch.float32
    x = torch.cat([pair[0] for pair in activation_batches]).to(device=device, dtype=sanity_dtype)
    qx = torch.cat([pair[1] for pair in activation_batches]).to(device=device, dtype=sanity_dtype)
    weight_work = weight.to(device=device, dtype=sanity_dtype)
    qweight_work = qweight.to(device=device, dtype=sanity_dtype)
    k = int(x.shape[1])
    channels = torch.as_tensor(sorted({0, k // 4, k // 2, 3 * k // 4, k - 1}), device=device)
    if source == "X":
        left = x - qx
        right = weight_work
    else:
        left = x
        right = weight_work - qweight_work
    expected_diag = left.square().sum(0).index_select(0, channels) * right.square().sum(0).index_select(0, channels)
    actual_diag = gram.diagonal().index_select(0, channels)
    direct_norm = left.matmul(right.T).double().square().sum()
    gram_norm = gram.double().sum()

    def relative(actual: torch.Tensor, expected: torch.Tensor) -> float:
        return float((actual - expected).abs().max().item()) / max(float(expected.abs().max().item()), 1e-30)

    diagonal_error = relative(actual_diag, expected_diag)
    norm_error = relative(gram_norm, direct_norm)
    maximum = max(diagonal_error, norm_error)
    tolerance = 2e-8 if sanity_dtype == torch.float64 else 2e-4
    result = {
        "status": "passed" if maximum <= tolerance else "failed",
        "source": source,
        "sampled_channels": channels.cpu().tolist(),
        "diagonal_identity_relative_error": diagonal_error,
        "functional_norm_identity_relative_error": norm_error,
        "max_relative_error": maximum,
        "tolerance": tolerance,
        "dtype": str(sanity_dtype),
    }
    if maximum > tolerance:
        raise AssertionError(f"source sanity check failed: {result}")
    return result


def _analyze_one_source(
    operand_path: Path,
    source: str,
    output_path: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if output_path.is_file() and not args.no_resume:
        try:
            return torch.load(output_path, map_location="cpu", weights_only=True, mmap=True)
        except TypeError:
            return torch.load(output_path, map_location="cpu")
    operands = torch.load(operand_path, map_location="cpu", weights_only=True)
    k = int(operands["K"])
    randomized = k >= args.randomized_min_k
    device = _device_for_k(args, k)
    accumulation_dtype = _gram_dtype(args, randomized=randomized)
    print(
        f"analyzing {operands['module']} / {source} on {device} "
        f"(K={k}, {'randomized' if randomized else 'exact'}, {accumulation_dtype})",
        flush=True,
    )
    weight = operands["weight"]
    qweight = operands["qweight"]
    split_batches = {
        "split_a": [(operands["split_a"]["x"], operands["split_a"]["qx"])],
        "split_b": [(operands["split_b"]["x"], operands["split_b"]["qx"])],
    }
    requested_ranks = summary_ranks_for_k(k)
    max_rank = max(requested_ranks)
    split_payloads: dict[str, dict[str, Any]] = {}
    top_vectors: dict[str, torch.Tensor] = {}
    for split_index, split in enumerate(("split_a", "split_b")):
        gram, _ = build_functional_gram(
            [(x.to(device), qx.to(device)) for x, qx in split_batches[split]],
            weight.to(device),
            qweight.to(device),
            source=source,
            accumulation_dtype=accumulation_dtype,
        )
        analysis = _spectral_analysis(
            gram,
            args=args,
            max_rank=max_rank,
            randomized=randomized,
            seed=int(operands["layer"]) * 100_003
            + (0 if source == "X" else 10_007)
            + split_index,
        )
        split_payloads[split] = _analysis_payload(analysis, max_rank)
        top_vectors[split] = split_payloads[split]["top_eigenvectors"]
        del gram, analysis
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    overlap = projector_overlap_curve(
        top_vectors["split_a"].double(), top_vectors["split_b"].double(), max_rank=max_rank
    )

    combined_batches = [*split_batches["split_a"], *split_batches["split_b"]]
    combined_gram, token_count = build_functional_gram(
        [(x.to(device), qx.to(device)) for x, qx in combined_batches],
        weight.to(device),
        qweight.to(device),
        source=source,
        accumulation_dtype=accumulation_dtype,
    )
    sanity = _source_sanity(combined_gram, source, combined_batches, weight, qweight)
    combined_analysis = _spectral_analysis(
        combined_gram,
        args=args,
        max_rank=max_rank,
        randomized=randomized,
        seed=int(operands["layer"]) * 100_003
        + (0 if source == "X" else 10_007)
        + 2,
    )
    payload = {
        "schema_version": 1,
        "module": operands["module"],
        "module_type": operands["module_type"],
        "layer": int(operands["layer"]),
        "source": source,
        "K": k,
        "out_features": int(operands["out_features"]),
        "token_count": token_count,
        # diagonal() is a strided view; clone it so its serialized storage is O(K),
        # not the full O(K^2) FP64 Gram backing storage.
        "diagonal": combined_gram.diagonal().detach().cpu().clone(),
        "combined": _analysis_payload(combined_analysis, max_rank),
        "split_a": split_payloads["split_a"],
        "split_b": split_payloads["split_b"],
        "split_overlap_curve": overlap.cpu(),
        "sanity": sanity,
        "eigh_device": str(device),
        "gram_accumulation_dtype": str(accumulation_dtype),
        "provenance": operands["provenance"],
        "quantizer": operands["quantizer"],
    }
    if not args.omit_full_gram:
        payload["gram"] = combined_gram.detach().cpu().float()
    diagonal = payload["diagonal"]
    if diagonal.untyped_storage().nbytes() > diagonal.numel() * diagonal.element_size():
        raise AssertionError("diagonal serialization still retains oversized backing storage")
    torch.save(payload, output_path)
    expected_tensor_bytes = (
        (0 if args.omit_full_gram else 4 * k * k)  # optional saved FP32 Gram
        + 3 * 8 * k * max_rank  # top-r vectors for A, B, and combined
        + 12 * 8 * k  # eigenvalues and coverage curves with generous allowance
    )
    maximum_artifact_bytes = int(expected_tensor_bytes * 1.25 + 5 * 1024 * 1024)
    if output_path.stat().st_size > maximum_artifact_bytes:
        raise AssertionError(
            f"serialized source artifact is unexpectedly large: {output_path.stat().st_size} > {maximum_artifact_bytes}"
        )
    print(f"saved {output_path.name}", flush=True)
    del operands, combined_gram, combined_analysis, split_payloads, top_vectors, overlap
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return payload


def _summary_rows(artifacts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for artifact in artifacts:
        k = int(artifact["K"])
        combined = artifact["combined"]
        overlap = artifact["split_overlap_curve"]
        for rank in summary_ranks_for_k(k):
            if rank > k or rank > overlap.numel():
                continue
            rows.append(
                {
                    "layer": int(artifact["layer"]),
                    "module": artifact["module"],
                    "module_type": artifact["module_type"],
                    "source": artifact["source"],
                    "K": k,
                    "rank": rank,
                    "rank_fraction": rank / k,
                    "equal_fraction_reference": rank == equal_fraction_rank(k),
                    "rho_struct": float(combined["rho_struct_curve"][rank - 1].item()),
                    "rho_func": float(combined["rho_func_curve"][rank - 1].item()),
                    "rho_struct_split_a": float(artifact["split_a"]["rho_struct_curve"][rank - 1].item()),
                    "rho_struct_split_b": float(artifact["split_b"]["rho_struct_curve"][rank - 1].item()),
                    "rho_func_split_a": float(artifact["split_a"]["rho_func_curve"][rank - 1].item()),
                    "rho_func_split_b": float(artifact["split_b"]["rho_func_curve"][rank - 1].item()),
                    "effective_rank": float(combined["effective_rank"]),
                    "effective_rank_ratio": float(combined["effective_rank_ratio"]),
                    "split_overlap": float(overlap[rank - 1].item()),
                    "trace_G": float(combined["trace_G"]),
                    "one_G_one": float(combined["one_G_one"]),
                    "spectral_method": combined.get("method", "exact_torch_eigh_fp64"),
                    "spectral_residual_max": float(combined.get("relative_residual_max", 0.0)),
                }
            )
    return sorted(rows, key=lambda row: (row["source"], row["layer"], row["module"], row["rank"]))


def _lightweight_artifact(payload: dict[str, Any]) -> dict[str, Any]:
    """Keep only fields needed by CSV/gate/validation after the source file is saved."""

    return {
        "module": payload["module"],
        "module_type": payload["module_type"],
        "layer": payload["layer"],
        "source": payload["source"],
        "K": payload["K"],
        "combined": {
            key: value
            for key, value in payload["combined"].items()
            if key != "top_eigenvectors"
        },
        "split_a": {
            key: value
            for key, value in payload["split_a"].items()
            if key not in {"top_eigenvectors", "eigenvalues"}
        },
        "split_b": {
            key: value
            for key, value in payload["split_b"].items()
            if key not in {"top_eigenvectors", "eigenvalues"}
        },
        "split_overlap_curve": payload["split_overlap_curve"],
        "sanity": payload["sanity"],
    }


def _gate_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for source in ("X", "W"):
        selected = [row for row in rows if row["source"] == source and int(row["rank"]) == 256]
        if not selected:
            raise ValueError(f"no rank-256 rows for source {source}")
        rho_func = [float(row["rho_func"]) for row in selected]
        rho_struct = [float(row["rho_struct"]) for row in selected]
        overlap = [float(row["split_overlap"]) for row in selected]
        values = {
            "module_count": len(selected),
            "median_rho_func_256": statistics.median(rho_func),
            "fraction_rho_func_256_ge_0.60": sum(value >= 0.60 for value in rho_func) / len(rho_func),
            "median_rho_struct_256": statistics.median(rho_struct),
            "median_overlap_256": statistics.median(overlap),
        }
        conditions = {
            "functional_median_ge_0.70": values["median_rho_func_256"] >= 0.70,
            "functional_fraction_ge_0.60_at_least_0.60": values["fraction_rho_func_256_ge_0.60"] >= 0.60,
            "structural_median_ge_0.50": values["median_rho_struct_256"] >= 0.50,
            "stability_median_ge_0.50": values["median_overlap_256"] >= 0.50,
        }
        values["conditions"] = conditions
        values["verdict"] = "GO" if all(conditions.values()) else "NO-GO"
        result[source] = values
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "layer", "module", "module_type", "source", "K", "rank", "rank_fraction",
        "equal_fraction_reference", "rho_struct",
        "rho_func", "effective_rank", "effective_rank_ratio", "split_overlap", "trace_G", "one_G_one",
        "rho_struct_split_a", "rho_struct_split_b", "rho_func_split_a", "rho_func_split_b",
        "spectral_method", "spectral_residual_max",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown_report(
    path: Path,
    gate: dict[str, Any],
    rows: list[dict[str, Any]],
    validation: dict[str, Any],
    figures: dict[str, str],
    model_label: str,
    head_analysis: dict[str, Any] | None,
    module_scope: str,
) -> None:
    rank256 = [row for row in rows if int(row["rank"]) == 256]
    by_source = {
        source: [row for row in rank256 if row["source"] == source]
        for source in ("X", "W")
    }
    frozen_scope = module_scope == "auto"
    protocol_statement = (
        "判定严格使用预注册 rank-256 门槛；没有使用 PPL、下游任务或 Stage 2 预期收益改写阈值。"
        if frozen_scope
        else f"本轮 `{module_scope}` 是冻结 Stage 1/1.5 之后的诊断扩展；rank-256 gate 仅作描述，不能替代七模块预注册结论。"
    )
    lines = [
        f"# Stage 1 — Functional Gram 低秩结构验证（{model_label}）",
        "",
        "## 技术结论",
        "",
        f"- **X source: {gate['X']['verdict']}**",
        f"- **W source: {gate['W']['verdict']}**",
        "",
        protocol_statement,
        "",
        "## Rank-256 核心指标",
        "",
        "| Source | median rho_func | fraction rho_func >= 0.60 | median rho_struct | median overlap | Verdict |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for source in ("X", "W"):
        item = gate[source]
        lines.append(
            f"| {source} | {item['median_rho_func_256']:.4f} | {item['fraction_rho_func_256_ge_0.60']:.4f} | "
            f"{item['median_rho_struct_256']:.4f} | {item['median_overlap_256']:.4f} | {item['verdict']} |"
        )
    x_struct = [float(row["rho_struct"]) for row in by_source["X"]]
    w_struct = [float(row["rho_struct"]) for row in by_source["W"]]
    lines.extend(
        [
            "",
            "## 附件要求的四个问题",
            "",
            f"1. **G_X 的 spectrum 是否明显衰减？** 部分模块衰减，但不形成跨模块一致的强低秩结构。"
            f"rank-256 structural coverage 中位数为 {gate['X']['median_rho_struct_256']:.4f}，模块范围为 {min(x_struct):.4f}–{max(x_struct):.4f}。",
            f"2. **G_W 的 spectrum 是否明显衰减？** 相比 X 更集中，但仍有明显模块异质性。"
            f"rank-256 structural coverage 中位数为 {gate['W']['median_rho_struct_256']:.4f}，模块范围为 {min(w_struct):.4f}–{max(w_struct):.4f}。",
            f"3. **rank-256 覆盖多少？** X: structural {gate['X']['median_rho_struct_256']:.4f}、actual functional {gate['X']['median_rho_func_256']:.4f}；"
            f"W: structural {gate['W']['median_rho_struct_256']:.4f}、actual functional {gate['W']['median_rho_func_256']:.4f}。",
            f"4. **不同 calibration split 是否稳定？** 稳定性本身较高：X/W 的 median overlap@256 分别为 "
            f"{gate['X']['median_overlap_256']:.4f}/{gate['W']['median_overlap_256']:.4f}；但稳定不等于 functional coverage 达标。",
        ]
    )
    lines.extend(["", "## 模块级证据", "", "| Layer | Module | Source | rho_struct@256 | rho_func@256 | overlap@256 | effective rank / K |", "|---:|---|---|---:|---:|---:|---:|"])
    for row in rank256:
        lines.append(
            f"| {row['layer']} | {row['module_type']} | {row['source']} | {float(row['rho_struct']):.4f} | "
            f"{float(row['rho_func']):.4f} | {float(row['split_overlap']):.4f} | {float(row['effective_rank_ratio']):.4f} |"
        )
    rank512_qkvo = [
        row
        for row in rows
        if int(row["rank"]) == 512
        and row["module_type"] in {"q_proj", "k_proj", "v_proj", "o_proj"}
    ]
    lines.extend(
        [
            "",
            "## 同层 q/k/v/o 的 rank-512 对照",
            "",
            "| Layer | Module | Source | rho_struct@512 | rho_func@512 | overlap@512 |",
            "|---:|---|---|---:|---:|---:|",
        ]
    )
    for row in rank512_qkvo:
        lines.append(
            f"| {row['layer']} | {row['module_type']} | {row['source']} | "
            f"{float(row['rho_struct']):.4f} | {float(row['rho_func']):.4f} | "
            f"{float(row['split_overlap']):.4f} |"
        )
    equal_fraction = [row for row in rows if bool(row["equal_fraction_reference"])]
    lines.extend(
        [
            "",
            "## 等 rank 比例（6.25% of K）对照",
            "",
            "普通 K=4096 模块使用 rank 256；K=14336 的 down_proj 使用 rank 896。",
            "",
            "| Layer | Module | Source | K | Rank | rho_struct | rho_func | overlap |",
            "|---:|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in equal_fraction:
        lines.append(
            f"| {row['layer']} | {row['module_type']} | {row['source']} | {row['K']} | "
            f"{row['rank']} | {float(row['rho_struct']):.4f} | "
            f"{float(row['rho_func']):.4f} | {float(row['split_overlap']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## 范围、方法与正确性",
            "",
            f"本轮覆盖 {len({row['module'] for row in rank256})} 个 {model_label} Linear module；"
            "校准使用两个互斥 WikiText2 split（各 4 个 2048-token window，每 split 固定抽取 512 行）。"
            "激活与权重均调用仓库已审计的 NVFP4 E2M1/UE4M3 group-16 reference。"
            "CSV 逐模块记录 exact eigendecomposition 或 randomized top-r solver 及其 Ritz residual。",
            "",
            f"Sanity/PSD 审计状态：**{validation['status']}**。逐模块检查 diagonal identity、`1^T G 1` functional norm identity 和显著负特征值。",
        ]
    )
    if head_analysis is not None:
        lines.extend(
            [
                "",
                "## Per-head 诊断",
                "",
                f"q/k/v 按连续 `head_dim` 行拆分；head Gram 求和恒等式与 randomized solver 审计状态："
                f"**{head_analysis['status']}**。详细结果见 `head_analysis/head_spectrum_summary.csv` 与 "
                "`head_analysis/head_module_summary.csv`。",
            ]
        )
    lines.extend(["", "## 图表", ""])
    for name, figure in figures.items():
        relative = Path(figure).resolve().relative_to(path.parent.resolve()).as_posix()
        lines.append(f"- [{name}]({relative})")
    lines.extend(
        [
            "",
            "## 局限与下一步",
            "",
            "`auto` 范围采用预注册的 7-module early/middle/late 设计；其他 module preset 均属于机制诊断，"
            "用于消除深度混杂或定位候选层，不能与冻结 gate 混写。",
            "",
            (
                "两个 source 均为 NO-GO，因此按预注册规则，本方向不进入 Stage 2 的 SVD-based support selection。"
                if frozen_scope
                else "本轮结果用于绘制深度轨迹与选择后续机制验证层，不对冻结 Stage 1/1.5 verdict 作任何改写。"
            ),
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    started = time.time()
    output_root = Path(args.output_root).resolve()
    operands_dir = output_root / "operands"
    artifact_dir = output_root / "gram_artifacts"
    figures_dir = output_root / "figures"
    for directory in (output_root, operands_dir, artifact_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)

    if args.skip_collection:
        operand_paths = sorted(operands_dir.glob("*.pt"))
        if not operand_paths:
            raise FileNotFoundError("--skip-collection was set but no operands exist")
        operand_headers = [
            torch.load(path, map_location="cpu", weights_only=True)
            for path in operand_paths
        ]
        collection = {
            "status": "reused_existing",
            "modules": [item["module"] for item in operand_headers],
            "artifacts": {
                item["module"]: str(path)
                for item, path in zip(operand_headers, operand_paths)
            },
            "provenance": operand_headers[0]["provenance"],
        }
        del operand_headers
    else:
        collection = collect_decoder_operands(
            model_path=args.model,
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
        operand_paths = [Path(path) for path in collection["artifacts"].values()]
    write_json(output_root / "collection_manifest.json", collection)

    source_artifacts: list[dict[str, Any]] = []
    for operand_path in sorted(operand_paths):
        metadata = torch.load(operand_path, map_location="cpu", weights_only=True)
        module = metadata["module"]
        del metadata
        for source in ("X", "W"):
            output_path = artifact_dir / _artifact_name(module, source)
            full_artifact = _analyze_one_source(operand_path, source, output_path, args)
            source_artifacts.append(_lightweight_artifact(full_artifact))
            del full_artifact
            gc.collect()

    head_result: dict[str, Any] | None = None
    if args.head_analysis:
        head_result = run_head_analysis(
            operand_paths,
            artifact_dir=artifact_dir,
            output_dir=output_root / "head_analysis",
            device=args.device,
            randomized_min_k=args.randomized_min_k,
            oversample=args.head_oversample,
            power_iterations=args.head_power_iterations,
            overlap_rank=args.head_overlap_rank,
            layers=args.head_analysis_layers,
        )

    rows = _summary_rows(source_artifacts)
    _write_csv(output_root / "stage1_summary.csv", rows)
    gate = _gate_summary(rows)
    write_json(output_root / "stage1_gate_summary.json", gate)
    maximum_solver_residual = max(
        float(item[split].get("relative_residual_max", 0.0))
        for item in source_artifacts
        for split in ("split_a", "split_b", "combined")
    )
    validation = {
        "status": "passed"
        if all(item["sanity"]["status"] == "passed" for item in source_artifacts)
        and maximum_solver_residual <= 0.5
        and (head_result is None or head_result["status"] == "passed")
        else "failed",
        "source_artifacts": len(source_artifacts),
        "maximum_spectral_relative_residual": maximum_solver_residual,
        "spectral_relative_residual_tolerance": 0.5,
        "head_analysis": head_result,
        "sanity_checks": [
            {
                "module": item["module"],
                "source": item["source"],
                **item["sanity"],
                "spectral_method": item["combined"].get("method", "exact_torch_eigh_fp64"),
                "spectral_relative_residual": item["combined"].get("relative_residual_max", 0.0),
                "combined_raw_min_eigenvalue": item["combined"]["raw_min_eigenvalue"],
                "split_a_raw_min_eigenvalue": item["split_a"]["raw_min_eigenvalue"],
                "split_b_raw_min_eigenvalue": item["split_b"]["raw_min_eigenvalue"],
            }
            for item in source_artifacts
        ],
    }
    write_json(output_root / "validation.json", validation)
    figures = render_stage1_figures(artifact_dir, rows, figures_dir)
    if head_result is not None:
        figures["figure_H_per_head_diagnostics"] = head_result["outputs"]["figure"]
    _write_markdown_report(
        output_root / "STAGE1_REPORT.md",
        gate,
        rows,
        validation,
        figures,
        Path(args.model).name,
        head_result,
        args.module_scope,
    )
    manifest = {
        "status": "complete" if validation["status"] == "passed" else "validation_failed",
        "config": vars(args),
        "environment": environment_metadata(),
        "git": git_metadata(REPO_ROOT),
        "elapsed_seconds": time.time() - started,
        "outputs": {
            "summary": str(output_root / "stage1_summary.csv"),
            "gate": str(output_root / "stage1_gate_summary.json"),
            "validation": str(output_root / "validation.json"),
            "report": str(output_root / "STAGE1_REPORT.md"),
            "figures": figures,
            "head_analysis": None if head_result is None else head_result["outputs"],
        },
    }
    # argparse fields are JSON-safe except a tuple/None, both handled by json.
    write_json(output_root / "run_manifest.json", manifest)
    print(json.dumps(gate, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
