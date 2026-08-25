"""Per-head Functional Gram analysis for q/k/v projection row blocks."""

from __future__ import annotations

import csv
import gc
import json
import statistics
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from functional_gram.eig_analysis import analyze_gram, analyze_gram_randomized


HEAD_RANKS = (16, 32, 64, 128)
ATTENTION_MODULES = ("q_proj", "k_proj", "v_proj")


def head_row_slices(
    out_features: int, head_count: int, head_dim: int
) -> tuple[slice, ...]:
    """Return the exact contiguous output-row block belonging to each head."""

    if min(out_features, head_count, head_dim) <= 0:
        raise ValueError("head dimensions must be positive")
    if head_count * head_dim != out_features:
        raise ValueError(
            f"head layout mismatch: {head_count} x {head_dim} != {out_features}"
        )
    return tuple(
        slice(head * head_dim, (head + 1) * head_dim)
        for head in range(head_count)
    )


def functional_gram_from_covariance(
    activation_covariance: torch.Tensor, factor: torch.Tensor
) -> torch.Tensor:
    """Build ``S activation Hadamard (factor.T @ factor)`` for one head."""

    if activation_covariance.ndim != 2 or activation_covariance.shape[0] != activation_covariance.shape[1]:
        raise ValueError("activation covariance must be square")
    if factor.ndim != 2 or factor.shape[1] != activation_covariance.shape[0]:
        raise ValueError("factor has an incompatible K dimension")
    gram = activation_covariance.mul(factor.T.matmul(factor))
    return (gram + gram.T).mul_(0.5)


def _load(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _overlap(left: torch.Tensor, right: torch.Tensor, rank: int) -> float:
    limit = min(rank, left.shape[1], right.shape[1])
    if limit <= 0:
        raise ValueError("overlap rank must be positive")
    cross = left[:, :limit].T.matmul(right[:, :limit])
    return float(cross.square().sum().item() / limit)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write an empty table: {path}")
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _render_summary_heatmap(
    module_rows: list[dict[str, Any]], output: Path, overlap_rank: int
) -> None:
    layers = sorted({int(row["layer"]) for row in module_rows})
    modules = ATTENTION_MODULES
    metrics = (
        ("median_head_rho_func_128", "median head rho_func @128"),
        (
            "median_pairwise_head_overlap",
            f"median pairwise overlap @{overlap_rank}",
        ),
        (
            "median_aggregate_subspace_capture_256",
            "median aggregate capture @256",
        ),
    )
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
    cmap = plt.cm.Blues.copy()
    cmap.set_bad("#ECEFF3")
    image = None
    for source_index, source in enumerate(("X", "W")):
        for metric_index, (metric, title) in enumerate(metrics):
            matrix = np.full((len(layers), len(modules)), np.nan)
            for row in module_rows:
                if row["source"] != source:
                    continue
                matrix[layers.index(int(row["layer"])), modules.index(row["module_type"])] = float(
                    row[metric]
                )
            ax = axes[source_index, metric_index]
            image = ax.imshow(matrix, vmin=0, vmax=1, cmap=cmap, aspect="auto")
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    if np.isfinite(matrix[i, j]):
                        ax.text(
                            j,
                            i,
                            f"{matrix[i, j]:.2f}",
                            ha="center",
                            va="center",
                            fontsize=8,
                            color="white" if matrix[i, j] >= 0.65 else "#22252A",
                        )
            ax.set_xticks(range(len(modules)), [name.replace("_proj", "") for name in modules])
            ax.set_yticks(range(len(layers)), layers)
            ax.set_xlabel("Projection")
            ax.set_ylabel("Layer")
            ax.set_title(f"{source} source — {title}", fontsize=9, fontweight="bold")
    assert image is not None
    fig.colorbar(image, ax=axes, shrink=0.82, label="Coverage / overlap")
    fig.suptitle("Per-head Functional Gram diagnostics", fontsize=15, fontweight="bold")
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


@torch.no_grad()
def run_head_analysis(
    operand_paths: Iterable[Path],
    *,
    artifact_dir: Path,
    output_dir: Path,
    device: str,
    randomized_min_k: int,
    oversample: int,
    power_iterations: int,
    overlap_rank: int,
) -> dict[str, Any]:
    """Analyze every q/k/v row block as one attention head on the combined split."""

    output_dir.mkdir(parents=True, exist_ok=True)
    target_paths: list[Path] = []
    for path in operand_paths:
        header = _load(path)
        if header["module_type"] in ATTENTION_MODULES:
            target_paths.append(path)
        del header
    if not target_paths:
        raise ValueError("head analysis requires q/k/v operands")

    work_device = torch.device(device)
    detail_rows: list[dict[str, Any]] = []
    module_rows: list[dict[str, Any]] = []
    identity_errors: list[float] = []
    residuals: list[float] = []

    for operand_path in sorted(target_paths):
        operands = _load(operand_path)
        module = str(operands["module"])
        module_type = str(operands["module_type"])
        layer = int(operands["layer"])
        k = int(operands["K"])
        head_count = int(operands["attention_heads"])
        head_dim = int(operands["head_dim"])
        slices = head_row_slices(int(operands["out_features"]), head_count, head_dim)
        x = torch.cat((operands["split_a"]["x"], operands["split_b"]["x"])).to(
            work_device, dtype=torch.float32
        )
        qx = torch.cat((operands["split_a"]["qx"], operands["split_b"]["qx"])).to(
            work_device, dtype=torch.float32
        )
        weight = operands["weight"].to(work_device, dtype=torch.float32)
        qweight = operands["qweight"].to(work_device, dtype=torch.float32)

        for source in ("X", "W"):
            activation = x - qx if source == "X" else x
            covariance = activation.T.matmul(activation)
            factor = weight if source == "X" else weight - qweight
            aggregate = functional_gram_from_covariance(covariance, factor)
            aggregate_scale = max(float(aggregate.abs().max().item()), 1e-30)
            head_sum = torch.zeros_like(aggregate)
            artifact_path = artifact_dir / f"{module.replace('.', '__')}__source_{source}.pt"
            aggregate_artifact = _load(artifact_path)
            aggregate_vectors = aggregate_artifact["combined"]["top_eigenvectors"].to(
                work_device, dtype=torch.float32
            )
            aggregate_rank = min(256, aggregate_vectors.shape[1])
            aggregate_basis = aggregate_vectors[:, :aggregate_rank]
            head_vectors: list[torch.Tensor] = []
            module_detail: list[dict[str, Any]] = []

            for head_index, row_slice in enumerate(slices):
                gram = functional_gram_from_covariance(covariance, factor[row_slice])
                head_sum.add_(gram)
                max_rank = min(max(HEAD_RANKS), k)
                if k >= randomized_min_k:
                    analysis = analyze_gram_randomized(
                        gram,
                        max_rank=max_rank,
                        oversample=oversample,
                        power_iterations=power_iterations,
                        seed=layer * 1_000_003
                        + ATTENTION_MODULES.index(module_type) * 10_007
                        + (0 if source == "X" else 503)
                        + head_index,
                    )
                else:
                    analysis = analyze_gram(
                        gram,
                        eigen_dtype=torch.float64,
                        negative_relative_tolerance=2e-6,
                    )
                residuals.append(float(analysis.relative_residual_max))
                basis_product = gram.matmul(aggregate_basis)
                shared_capture = float(
                    (aggregate_basis * basis_product).sum().double().item()
                    / max(analysis.trace_g, 1e-30)
                )
                shared_capture = min(1.0, max(0.0, shared_capture))
                head_vectors.append(
                    analysis.eigenvectors[:, : min(overlap_rank, analysis.eigenvectors.shape[1])]
                    .detach()
                    .cpu()
                    .float()
                    .contiguous()
                )
                for rank in HEAD_RANKS:
                    if rank > analysis.rho_func_curve.numel():
                        continue
                    row = {
                        "layer": layer,
                        "module": module,
                        "module_type": module_type,
                        "source": source,
                        "head_index": head_index,
                        "head_count": head_count,
                        "head_dim": head_dim,
                        "K": k,
                        "rank": rank,
                        "rho_struct": float(analysis.rho_struct_curve[rank - 1].item()),
                        "rho_func": float(analysis.rho_func_curve[rank - 1].item()),
                        "aggregate_subspace_capture_256": shared_capture,
                        "spectral_method": analysis.method,
                        "spectral_residual_max": float(analysis.relative_residual_max),
                    }
                    detail_rows.append(row)
                    module_detail.append(row)
                del gram, analysis, basis_product
                gc.collect()
                if work_device.type == "cuda":
                    torch.cuda.empty_cache()

            identity_error = float((head_sum - aggregate).abs().max().item()) / aggregate_scale
            identity_errors.append(identity_error)
            overlaps = [
                _overlap(head_vectors[left], head_vectors[right], overlap_rank)
                for left in range(len(head_vectors))
                for right in range(left + 1, len(head_vectors))
            ]
            rank128 = [row for row in module_detail if int(row["rank"]) == 128]
            full_curve = aggregate_artifact["combined"]["rho_func_curve"]
            module_rows.append(
                {
                    "layer": layer,
                    "module": module,
                    "module_type": module_type,
                    "source": source,
                    "head_count": head_count,
                    "head_dim": head_dim,
                    "K": k,
                    "median_head_rho_struct_128": statistics.median(
                        float(row["rho_struct"]) for row in rank128
                    ),
                    "median_head_rho_func_128": statistics.median(
                        float(row["rho_func"]) for row in rank128
                    ),
                    "full_module_rho_func_128": float(full_curve[127].item()),
                    "median_pairwise_head_overlap": statistics.median(overlaps)
                    if overlaps
                    else 1.0,
                    "pairwise_overlap_rank": overlap_rank,
                    "median_aggregate_subspace_capture_256": statistics.median(
                        float(row["aggregate_subspace_capture_256"]) for row in rank128
                    ),
                    "head_sum_identity_relative_error": identity_error,
                    "maximum_spectral_relative_residual": max(
                        float(row["spectral_residual_max"]) for row in rank128
                    ),
                }
            )
            del covariance, factor, aggregate, head_sum, aggregate_artifact
            del aggregate_vectors, aggregate_basis, head_vectors, module_detail
            gc.collect()
            if work_device.type == "cuda":
                torch.cuda.empty_cache()

        del operands, x, qx, weight, qweight
        gc.collect()
        if work_device.type == "cuda":
            torch.cuda.empty_cache()

    detail_rows.sort(
        key=lambda row: (
            row["source"],
            row["layer"],
            row["module"],
            row["head_index"],
            row["rank"],
        )
    )
    module_rows.sort(key=lambda row: (row["source"], row["layer"], row["module"]))
    detail_csv = output_dir / "head_spectrum_summary.csv"
    module_csv = output_dir / "head_module_summary.csv"
    figure = output_dir / "figure_H_per_head_diagnostics.png"
    _write_csv(detail_csv, detail_rows)
    _write_csv(module_csv, module_rows)
    _render_summary_heatmap(module_rows, figure, overlap_rank)
    maximum_identity_error = max(identity_errors)
    maximum_residual = max(residuals)
    result = {
        "status": "passed"
        if maximum_identity_error <= 2e-4 and maximum_residual <= 0.5
        else "failed",
        "module_source_pairs": len(module_rows),
        "head_rows": len(detail_rows),
        "head_sum_identity_relative_error_max": maximum_identity_error,
        "head_sum_identity_tolerance": 2e-4,
        "spectral_relative_residual_max": maximum_residual,
        "spectral_relative_residual_tolerance": 0.5,
        "outputs": {
            "head_spectrum_summary": str(detail_csv),
            "head_module_summary": str(module_csv),
            "figure": str(figure),
        },
    }
    summary_path = output_dir / "head_analysis_summary.json"
    result["outputs"]["summary"] = str(summary_path)
    summary_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


__all__ = [
    "ATTENTION_MODULES",
    "HEAD_RANKS",
    "functional_gram_from_covariance",
    "head_row_slices",
    "run_head_analysis",
]
