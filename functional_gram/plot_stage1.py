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
ATTENTION_ORDER = ("q_proj", "k_proj", "v_proj", "o_proj")
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
    if len(subset) > 24:
        representative: list[dict[str, Any]] = []
        for module_type in MODULE_ORDER:
            group = sorted(
                (row for row in subset if row["module_type"] == module_type),
                key=lambda row: int(row["layer"]),
            )
            if not group:
                continue
            for index in dict.fromkeys((0, len(group) // 2, len(group) - 1)):
                representative.append(group[index])
        subset = representative
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


def _heatmap_at_rank(
    records: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    output: Path,
    *,
    rank: int,
    module_order: tuple[str, ...],
    figure_title: str,
) -> None:
    layers = sorted({int(row["layer"]) for row in records})
    lookup: dict[tuple[int, str, str, str], float] = {}
    for row in summary_rows:
        if int(row["rank"]) == rank:
            for metric in ("rho_func", "rho_struct", "split_overlap"):
                lookup[(int(row["layer"]), row["module"], row["source"], metric)] = float(row[metric])
    metrics = (
        ("rho_func", f"Functional coverage at rank {rank}"),
        ("rho_struct", f"Structural coverage at rank {rank}"),
        ("split_overlap", f"Split overlap at rank {rank}"),
    )
    figure_height = max(7.5, 2.5 + 0.4 * len(layers))
    annotation_size = 8 if len(layers) <= 12 else 5.5
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(16, figure_height),
        constrained_layout=True,
        squeeze=False,
    )
    cmap = plt.cm.Blues.copy()
    cmap.set_bad("#ECEFF3")
    image = None
    for source_index, source in enumerate(("X", "W")):
        for metric_index, (metric, label) in enumerate(metrics):
            ax = axes[source_index, metric_index]
            matrix = np.full((len(layers), len(module_order)), np.nan)
            for i, layer in enumerate(layers):
                for j, module_type in enumerate(module_order):
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
                        ax.text(
                            j,
                            i,
                            f"{matrix[i, j]:.2f}",
                            ha="center",
                            va="center",
                            fontsize=annotation_size,
                            color=INK if matrix[i, j] < 0.65 else "white",
                        )
            ax.set_xticks(range(len(module_order)), [name.replace("_proj", "") for name in module_order], rotation=35, ha="right")
            ax.set_yticks(range(len(layers)), layers)
            ax.set_xlabel("Linear module type")
            ax.set_ylabel("Layer")
            ax.set_title(f"{source} source — {label}", fontsize=11, fontweight="bold", color=INK)
            ax.tick_params(labelsize=8)
    assert image is not None
    fig.colorbar(image, ax=axes, location="right", shrink=0.82, label="Coverage / overlap")
    fig.suptitle(figure_title, fontsize=16, fontweight="bold", color=INK)
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def _equal_fraction_heatmap(
    records: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    output: Path,
) -> None:
    """Compare every module at 6.25% of K (256/4096; 896/14336)."""

    layers = sorted({int(row["layer"]) for row in records})
    module_order = tuple(
        module
        for module in MODULE_ORDER
        if any(row["module_type"] == module for row in records)
    )
    selected = [row for row in summary_rows if bool(row["equal_fraction_reference"])]
    lookup = {
        (int(row["layer"]), row["module"], row["source"]): row
        for row in selected
    }
    metrics = (
        ("rho_func", "Functional coverage at 6.25% of K"),
        ("rho_struct", "Structural coverage at 6.25% of K"),
        ("split_overlap", "Split overlap at 6.25% of K"),
    )
    figure_height = max(7.5, 2.5 + 0.4 * len(layers))
    annotation_size = 7 if len(layers) <= 12 else 5
    fig, axes = plt.subplots(
        2, 3, figsize=(16, figure_height), constrained_layout=True
    )
    cmap = plt.cm.Blues.copy()
    cmap.set_bad("#ECEFF3")
    image = None
    for source_index, source in enumerate(("X", "W")):
        for metric_index, (metric, title) in enumerate(metrics):
            matrix = np.full((len(layers), len(module_order)), np.nan)
            ranks = np.full_like(matrix, np.nan)
            for i, layer in enumerate(layers):
                for j, module_type in enumerate(module_order):
                    module_name = next(
                        (
                            row["module"]
                            for row in records
                            if row["layer"] == layer and row["module_type"] == module_type
                        ),
                        None,
                    )
                    row = lookup.get((layer, module_name, source)) if module_name else None
                    if row is not None:
                        matrix[i, j] = float(row[metric])
                        ranks[i, j] = int(row["rank"])
            ax = axes[source_index, metric_index]
            image = ax.imshow(matrix, vmin=0, vmax=1, cmap=cmap, aspect="auto")
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    if np.isfinite(matrix[i, j]):
                        ax.text(
                            j,
                            i,
                            f"{matrix[i, j]:.2f}\nr{int(ranks[i, j])}",
                            ha="center",
                            va="center",
                            fontsize=annotation_size,
                            color=INK if matrix[i, j] < 0.65 else "white",
                        )
            ax.set_xticks(
                range(len(module_order)),
                [name.replace("_proj", "") for name in module_order],
                rotation=35,
                ha="right",
            )
            ax.set_yticks(range(len(layers)), layers)
            ax.set_xlabel("Linear module type")
            ax.set_ylabel("Layer")
            ax.set_title(f"{source} source — {title}", fontsize=10, fontweight="bold")
    assert image is not None
    fig.colorbar(image, ax=axes, location="right", shrink=0.82, label="Coverage / overlap")
    fig.suptitle(
        "Stage 1 equal-rank-fraction heatmaps",
        fontsize=16,
        fontweight="bold",
        color=INK,
    )
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def _attention_depth_profiles(
    summary_rows: list[dict[str, Any]], output: Path
) -> None:
    """Render full-depth attention trajectories without overcrowded cell labels."""

    attention_rows = [
        row for row in summary_rows if row["module_type"] in ATTENTION_ORDER
    ]
    if not attention_rows:
        raise ValueError("attention depth profiles require q/k/v/o rows")
    layers = sorted({int(row["layer"]) for row in attention_rows})
    metrics = (
        ("rho_func", 256, "Functional coverage @256"),
        ("rho_func", 512, "Functional coverage @512"),
        ("split_overlap", 512, "Split overlap @512"),
    )
    colors = {
        module: color
        for module, color in zip(
            ATTENTION_ORDER, plt.cm.Blues(np.linspace(0.42, 0.95, len(ATTENTION_ORDER)))
        )
    }
    line_styles = dict(zip(ATTENTION_ORDER, ("-", "--", "-.", ":")))
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5), constrained_layout=True)
    for source_index, source in enumerate(("X", "W")):
        for metric_index, (metric, rank, title) in enumerate(metrics):
            ax = axes[source_index, metric_index]
            for module_type in ATTENTION_ORDER:
                selected = sorted(
                    (
                        row
                        for row in attention_rows
                        if row["source"] == source
                        and row["module_type"] == module_type
                        and int(row["rank"]) == rank
                    ),
                    key=lambda row: int(row["layer"]),
                )
                if not selected:
                    continue
                ax.plot(
                    [int(row["layer"]) for row in selected],
                    [float(row[metric]) for row in selected],
                    marker="o",
                    markersize=2.8,
                    linewidth=1.6,
                    linestyle=line_styles[module_type],
                    color=colors[module_type],
                    label=module_type.replace("_proj", ""),
                )
            ax.set_ylim(0, 1.02)
            ax.set_xlim(min(layers), max(layers))
            tick_step = max(1, len(layers) // 8)
            ticks = layers[::tick_step]
            if ticks[-1] != layers[-1]:
                ticks.append(layers[-1])
            ax.set_xticks(ticks)
            ax.set_xlabel("Decoder layer")
            ax.set_ylabel("Coverage / overlap")
            ax.set_title(f"{source} source — {title}", fontsize=11, fontweight="bold")
            ax.legend(frameon=False, ncol=2, fontsize=8)
            _style_axes(ax)
    fig.suptitle(
        "Full-depth attention Functional Gram profiles",
        fontsize=16,
        fontweight="bold",
        color=INK,
    )
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def _down_depth_profiles(
    summary_rows: list[dict[str, Any]], output: Path
) -> None:
    """Render down-projection depth trajectories at fixed and fair ranks."""

    down_rows = [row for row in summary_rows if row["module_type"] == "down_proj"]
    if not down_rows:
        raise ValueError("down depth profiles require down_proj rows")
    layers = sorted({int(row["layer"]) for row in down_rows})
    metrics = (
        ("rho_func", "Functional coverage"),
        ("rho_struct", "Structural coverage"),
        ("split_overlap", "Split overlap"),
    )
    ranks = (256, 896, 1024)
    colors = dict(zip(ranks, plt.cm.Blues(np.linspace(0.45, 0.95, len(ranks)))))
    line_styles = dict(zip(ranks, ("--", "-", ":")))
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5), constrained_layout=True)
    for source_index, source in enumerate(("X", "W")):
        for metric_index, (metric, title) in enumerate(metrics):
            ax = axes[source_index, metric_index]
            for rank in ranks:
                selected = sorted(
                    (
                        row
                        for row in down_rows
                        if row["source"] == source and int(row["rank"]) == rank
                    ),
                    key=lambda row: int(row["layer"]),
                )
                if not selected:
                    continue
                ax.plot(
                    [int(row["layer"]) for row in selected],
                    [float(row[metric]) for row in selected],
                    marker="o",
                    markersize=3.2,
                    linewidth=1.7,
                    linestyle=line_styles[rank],
                    color=colors[rank],
                    label=f"rank {rank}",
                )
            ax.set_ylim(0, 1.02)
            ax.set_xlim(min(layers), max(layers))
            ax.set_xticks(layers)
            ax.set_xlabel("Decoder layer")
            ax.set_ylabel("Coverage / overlap")
            ax.set_title(f"{source} source — {title}", fontsize=11, fontweight="bold")
            ax.legend(frameon=False, fontsize=8)
            _style_axes(ax)
    fig.suptitle(
        "Down-projection depth profiles",
        fontsize=16,
        fontweight="bold",
        color=INK,
    )
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
    available_module_order = tuple(
        module
        for module in MODULE_ORDER
        if any(row["module_type"] == module for row in records)
    )
    definitions = (
        ("eigenvalues", "figure_A_eigenspectrum", "Normalized eigenvalue spectrum", "Eigenvalue share of total functional-Gram trace; log-log view", r"$\lambda_i / \sum_j \lambda_j$", True, ()),
        ("rho_struct", "figure_B_structural_coverage", "Cumulative structural coverage", "Share of functional-atom energy represented by the top-r eigenspace", r"$\rho_{struct}(r)$", False, (64, 128, 256, 512, 896)),
        ("rho_func", "figure_C_functional_coverage", "Cumulative summed functional-error coverage", "Share of the actual summed output error represented by the top-r modes", r"$\rho_{func}(r)$", False, (64, 128, 256, 512, 896)),
        ("overlap", "figure_D_split_stability", "Calibration split subspace stability", "Projector overlap between disjoint WikiText2 split A and split B", "Overlap(r)", False, (128, 256, 512, 896)),
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
    _heatmap_at_rank(
        records,
        summary_rows,
        heatmap_path,
        rank=256,
        module_order=available_module_order,
        figure_title="Stage 1 layer/module heatmaps at rank 256",
    )
    outputs["figure_E_layer_module_heatmaps"] = str(heatmap_path)
    has_attention_rank512 = any(
        row["module_type"] in ATTENTION_ORDER and int(row["rank"]) == 512
        for row in summary_rows
    )
    if has_attention_rank512:
        qkvo_path = output_dir / "figure_F_qkvo_depth_heatmaps_rank512.png"
        _heatmap_at_rank(
            records,
            summary_rows,
            qkvo_path,
            rank=512,
            module_order=ATTENTION_ORDER,
            figure_title="Matched-depth q/k/v/o heatmaps at rank 512",
        )
        outputs["figure_F_qkvo_depth_heatmaps_rank512"] = str(qkvo_path)
    equal_fraction_path = output_dir / "figure_G_equal_fraction_heatmaps.png"
    _equal_fraction_heatmap(records, summary_rows, equal_fraction_path)
    outputs["figure_G_equal_fraction_heatmaps"] = str(equal_fraction_path)
    if has_attention_rank512:
        depth_profile_path = output_dir / "figure_I_attention_depth_profiles.png"
        _attention_depth_profiles(summary_rows, depth_profile_path)
        outputs["figure_I_attention_depth_profiles"] = str(depth_profile_path)
    if any(row["module_type"] == "down_proj" for row in summary_rows):
        down_profile_path = output_dir / "figure_J_down_depth_profiles.png"
        _down_depth_profiles(summary_rows, down_profile_path)
        outputs["figure_J_down_depth_profiles"] = str(down_profile_path)
    chart_map = {
        "surface": "static_matplotlib_png",
        "palette_policy": "single blue root plus neutral references",
        "charts": outputs,
        "qa_surface": "exported PNG files",
    }
    (output_dir / "chart_map.json").write_text(json.dumps(chart_map, ensure_ascii=False, indent=2), encoding="utf-8")
    return outputs
