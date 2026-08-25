"""Static Figure A--E renderer for Stage 1."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


MODULE_ORDER = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
BLUE = "#2F6BFF"
ORANGE = "#E07A2D"
INK = "#22252A"
GRID = "#D9DEE7"


def _style_axes(ax: Any) -> None:
    ax.set_facecolor("#FFFFFF")
    ax.grid(True, color=GRID, linewidth=0.7, alpha=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors=INK, labelsize=8)


def _load_curves(artifact_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(artifact_dir.glob("*__source_*.pt")):
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
        rows.append(
            {
                "module": payload["module"],
                "module_type": payload["module_type"],
                "layer": int(payload["layer"]),
                "source": payload["source"],
                "eigenvalues": payload["combined"]["eigenvalues"].numpy().copy(),
                "trace_G": float(payload["combined"]["trace_G"]),
                "rho_struct": payload["combined"]["rho_struct_curve"].numpy().copy(),
                "rho_func": payload["combined"]["rho_func_curve"].numpy().copy(),
                "overlap": payload["split_overlap_curve"].numpy().copy(),
            }
        )
        del payload
    if not rows:
        raise FileNotFoundError(f"no source artifacts found in {artifact_dir}")
    return rows


def _line_figure(
    records: list[dict[str, Any]],
    *,
    source: str,
    field: str,
    title: str,
    subtitle: str,
    ylabel: str,
    output: Path,
    log_y: bool = False,
    reference_ranks: tuple[int, ...] = (),
) -> None:
    subset = [row for row in records if row["source"] == source]
    fig, ax = plt.subplots(figsize=(10.5, 6.2), constrained_layout=True)
    colors = plt.cm.Blues(np.linspace(0.4, 0.95, max(len(subset), 2)))
    line_styles = ("-", "--", "-.", ":")
    for index, row in enumerate(subset):
        values = np.asarray(row[field], dtype=np.float64)
        if field == "eigenvalues":
            total = float(row["trace_G"])
            values = values / total if total > 0 else values
        ranks = np.arange(1, values.size + 1)
        ax.plot(
            ranks,
            values,
            color=colors[index],
            linestyle=line_styles[index % len(line_styles)],
            linewidth=1.4,
            label=f"L{row['layer']} {row['module_type']}",
        )
    for rank in reference_ranks:
        ax.axvline(rank, color="#70757D", linewidth=0.9, linestyle="--")
        ax.text(rank, 0.02, str(rank), rotation=90, va="bottom", ha="right", fontsize=7, color="#555B65", transform=ax.get_xaxis_transform())
    ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    else:
        ax.set_ylim(0, 1.02)
    ax.set_xlabel("Rank / eigenvalue index", color=INK)
    ax.set_ylabel(ylabel, color=INK)
    ax.set_title(title, loc="left", fontsize=15, fontweight="bold", color=INK, pad=20)
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9, color="#555B65", va="bottom")
    ax.legend(loc="best", fontsize=8, frameon=False, ncol=2)
    _style_axes(ax)
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def _heatmap(
    records: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    output: Path,
) -> None:
    layers = sorted({int(row["layer"]) for row in records})
    lookup: dict[tuple[int, str, str, str], float] = {}
    for row in summary_rows:
        if int(row["rank"]) == 256:
            for metric in ("rho_func", "rho_struct", "split_overlap"):
                lookup[(int(row["layer"]), row["module"], row["source"], metric)] = float(row[metric])
    metrics = (
        ("rho_func", "Functional coverage at rank 256"),
        ("rho_struct", "Structural coverage at rank 256"),
        ("split_overlap", "Split overlap at rank 256"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(16, 7.5), constrained_layout=True, squeeze=False)
    cmap = plt.cm.Blues.copy()
    cmap.set_bad("#ECEFF3")
    image = None
    for source_index, source in enumerate(("X", "W")):
        for metric_index, (metric, label) in enumerate(metrics):
            ax = axes[source_index, metric_index]
            matrix = np.full((len(layers), len(MODULE_ORDER)), np.nan)
            for i, layer in enumerate(layers):
                for j, module_type in enumerate(MODULE_ORDER):
                    module_name = next(
                        (
                            row["module"]
                            for row in records
                            if row["layer"] == layer and row["module_type"] == module_type
                        ),
                        None,
                    )
                    if module_name is not None:
                        matrix[i, j] = lookup.get((layer, module_name, source, metric), np.nan)
            image = ax.imshow(matrix, vmin=0, vmax=1, cmap=cmap, aspect="auto")
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    if np.isfinite(matrix[i, j]):
                        ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8, color=INK if matrix[i, j] < 0.65 else "white")
            ax.set_xticks(range(len(MODULE_ORDER)), [name.replace("_proj", "") for name in MODULE_ORDER], rotation=35, ha="right")
            ax.set_yticks(range(len(layers)), layers)
            ax.set_xlabel("Linear module type")
            ax.set_ylabel("Layer")
            ax.set_title(f"{source} source — {label}", fontsize=11, fontweight="bold", color=INK)
            ax.tick_params(labelsize=8)
    assert image is not None
    fig.colorbar(image, ax=axes, location="right", shrink=0.82, label="Coverage / overlap")
    fig.suptitle("Stage 1 layer/module heatmaps", fontsize=16, fontweight="bold", color=INK)
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def render_stage1_figures(
    artifact_dir: str | Path,
    summary_rows: list[dict[str, Any]],
    output_dir: str | Path,
) -> dict[str, str]:
    artifact_dir, output_dir = Path(artifact_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = _load_curves(artifact_dir)
    outputs: dict[str, str] = {}
    definitions = (
        ("eigenvalues", "figure_A_eigenspectrum", "Normalized eigenvalue spectrum", "Eigenvalue share of total functional-Gram trace; log-log view", r"$\lambda_i / \sum_j \lambda_j$", True, ()),
        ("rho_struct", "figure_B_structural_coverage", "Cumulative structural coverage", "Share of functional-atom energy represented by the top-r eigenspace", r"$\rho_{struct}(r)$", False, (64, 128, 256)),
        ("rho_func", "figure_C_functional_coverage", "Cumulative summed functional-error coverage", "Share of the actual summed output error represented by the top-r modes", r"$\rho_{func}(r)$", False, (64, 128, 256)),
        ("overlap", "figure_D_split_stability", "Calibration split subspace stability", "Projector overlap between disjoint WikiText2 split A and split B", "Overlap(r)", False, (128, 256)),
    )
    for source in ("X", "W"):
        for field, stem, title, subtitle, ylabel, log_y, reference_ranks in definitions:
            path = output_dir / f"{stem}_{source}.png"
            _line_figure(
                records,
                source=source,
                field=field,
                title=f"{title} — {source} source",
                subtitle=subtitle,
                ylabel=ylabel,
                output=path,
                log_y=log_y,
                reference_ranks=reference_ranks,
            )
            outputs[f"{stem}_{source}"] = str(path)
    heatmap_path = output_dir / "figure_E_layer_module_heatmaps.png"
    _heatmap(records, summary_rows, heatmap_path)
    outputs["figure_E_layer_module_heatmaps"] = str(heatmap_path)
    chart_map = {
        "surface": "static_matplotlib_png",
        "palette_policy": "single blue root plus neutral references",
        "charts": outputs,
        "qa_surface": "exported PNG files",
    }
    (output_dir / "chart_map.json").write_text(json.dumps(chart_map, ensure_ascii=False, indent=2), encoding="utf-8")
    return outputs
