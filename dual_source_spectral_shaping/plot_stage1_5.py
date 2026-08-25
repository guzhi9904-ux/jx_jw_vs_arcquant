"""Required Stage 1.5 Figure A--F rendering."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INK = "#14213d"
BLUE = "#2f6690"
ORANGE = "#e07a5f"
GREEN = "#4f772d"
PURPLE = "#6d597a"
GRID = "#d9e2ec"
RANKS = (16, 32, 64, 128, 256, 512)


def _finish(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _heatmap(
    frame: pd.DataFrame,
    *,
    value: str,
    title: str,
    path: Path,
    cmap: str,
) -> None:
    modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    layers = [0, 14, 27]
    values = np.full((len(layers), len(modules)), np.nan)
    for _, row in frame.iterrows():
        if row.module_type in modules and int(row.layer) in layers:
            values[layers.index(int(row.layer)), modules.index(row.module_type)] = float(row[value])
    masked = np.ma.masked_invalid(values)
    fig, ax = plt.subplots(figsize=(11, 3.8))
    image = ax.imshow(masked, aspect="auto", cmap=cmap)
    ax.set_xticks(range(len(modules)), modules, rotation=25, ha="right")
    ax.set_yticks(range(len(layers)), layers)
    ax.set_xlabel("module")
    ax.set_ylabel("layer")
    ax.set_title(title, loc="left", color=INK, fontweight="bold")
    for i in range(len(layers)):
        for j in range(len(modules)):
            if np.isfinite(values[i, j]):
                ax.text(j, i, f"{values[i, j]:.3f}", ha="center", va="center", fontsize=9)
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.03)
    _finish(fig, path)


def _placeholder(path: Path, title: str, reason: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.axis("off")
    ax.text(0.03, 0.85, title, transform=ax.transAxes, fontsize=16, fontweight="bold", color=INK)
    ax.text(0.03, 0.55, reason, transform=ax.transAxes, fontsize=12, color="#52616b")
    _finish(fig, path)


def _alpha_position(value: Any) -> float:
    return -0.25 if str(value) == "identity" else float(value)


def render_figures(
    raw_csv: str | Path,
    scaling_csv: str | Path | None,
    output_dir: str | Path,
) -> dict[str, str]:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(raw_csv)
    combined = raw[raw.split == "combined"].copy()
    result: dict[str, str] = {}

    # Figure A
    path = output_dir / "figure_A_raw_dual_source_coverage.png"
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    for source, label, color in (
        ("x", "X source", BLUE),
        ("w", "W source", ORANGE),
        ("pair", "Pair", GREEN),
        ("joint", "Joint", PURPLE),
    ):
        medians = [combined[f"rho_{source}_{rank}"].median() for rank in RANKS]
        ax.plot(RANKS, medians, marker="o", linewidth=2.2, label=label, color=color)
    for rank in (64, 128, 256):
        ax.axvline(rank, color=GRID, linewidth=1, zorder=0)
    ax.axhline(0.70, color="#a23e48", linestyle="--", linewidth=1, label="GO median 0.70")
    ax.set(xlabel="rank", ylabel="median functional coverage", ylim=(0, 1.02))
    ax.set_title("Figure A — Raw dual-source coverage", loc="left", color=INK, fontweight="bold")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.legend(ncol=3, frameon=False)
    _finish(fig, path)
    result["figure_A_raw_dual_source_coverage"] = str(path)

    path = output_dir / "figure_B_pair_gain_heatmap.png"
    _heatmap(
        combined,
        value="pair_gain_256",
        title="Figure B — Pair gain @ rank 256",
        path=path,
        cmap="RdBu_r",
    )
    result["figure_B_pair_gain_heatmap"] = str(path)

    path = output_dir / "figure_C_cross_source_interaction_heatmap.png"
    _heatmap(
        combined,
        value="kappa",
        title="Figure C — Cross-source interaction κ",
        path=path,
        cmap="PuOr_r",
    )
    result["figure_C_cross_source_interaction_heatmap"] = str(path)

    paths = {
        "figure_D_scaling_trajectory": output_dir / "figure_D_scaling_trajectory.png",
        "figure_E_error_consolidation_path": output_dir / "figure_E_error_consolidation_path.png",
        "figure_F_pair_coverage_before_after": output_dir / "figure_F_pair_coverage_before_after.png",
    }
    if scaling_csv is None or not Path(scaling_csv).is_file():
        for name, figure_path in paths.items():
            _placeholder(
                figure_path,
                name.replace("_", " ").title(),
                "Stage 1.5B was not entered by the preregistered Stage 1.5A gate.",
            )
            result[name] = str(figure_path)
        return result

    scaling = pd.read_csv(scaling_csv, dtype={"alpha": str})
    split_a = scaling[scaling.split == "split_a"].copy()
    split_a["alpha_position"] = split_a.alpha.map(_alpha_position)
    positions = sorted(split_a.alpha_position.unique())

    # Figure D
    path = paths["figure_D_scaling_trajectory"]
    metrics = (
        ("tau_total", "τ total", "#a23e48"),
        ("rho_x_256", "ρX@256", BLUE),
        ("rho_w_256", "ρW@256", ORANGE),
        ("rho_pair_256", "ρPair@256", GREEN),
        ("source_share_x", "sX", PURPLE),
        ("source_share_w", "sW", "#bc6c25"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), sharex=True)
    for ax, (metric, label, color) in zip(axes.flat, metrics):
        medians = [split_a.loc[split_a.alpha_position == position, metric].median() for position in positions]
        q25 = [split_a.loc[split_a.alpha_position == position, metric].quantile(0.25) for position in positions]
        q75 = [split_a.loc[split_a.alpha_position == position, metric].quantile(0.75) for position in positions]
        ax.plot(positions, medians, marker="o", color=color, linewidth=2)
        ax.fill_between(positions, q25, q75, color=color, alpha=0.14)
        ax.set_title(label, loc="left", fontweight="bold", color=INK)
        ax.grid(axis="y", color=GRID, linewidth=0.7)
        if metric == "tau_total":
            ax.axhline(1.05, color="#a23e48", linestyle="--", linewidth=1)
    for ax in axes[-1]:
        ax.set_xticks(positions, ["I" if p < 0 else f"{p:.2f}" for p in positions])
        ax.set_xlabel("alpha (I = identity)")
    fig.suptitle("Figure D — Scaling trajectory (Split A median and IQR)", x=0.06, ha="left", color=INK, fontweight="bold")
    _finish(fig, path)
    result["figure_D_scaling_trajectory"] = str(path)

    # Figure E
    path = paths["figure_E_error_consolidation_path"]
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    cmap = plt.get_cmap("viridis")
    numeric = split_a[split_a.alpha != "identity"].copy()
    for module, frame in numeric.groupby("module"):
        frame = frame.sort_values("alpha_position")
        ax.plot(frame.source_share_w, frame.rho_w_256, color="#aab7b8", linewidth=0.8, alpha=0.7)
        points = ax.scatter(
            frame.source_share_w,
            frame.rho_w_256,
            c=frame.alpha_position,
            cmap=cmap,
            vmin=0,
            vmax=1,
            s=38,
            alpha=0.82,
        )
    identity = split_a[split_a.alpha == "identity"]
    ax.scatter(identity.source_share_w, identity.rho_w_256, marker="X", color="#1f2933", s=65, label="identity")
    ax.set(xlabel="W source share", ylabel="W functional coverage @256", xlim=(0, 1), ylim=(0, 1.02))
    ax.set_title("Figure E — Error consolidation path (Split A)", loc="left", color=INK, fontweight="bold")
    ax.grid(color=GRID, linewidth=0.7)
    ax.legend(frameon=False)
    fig.colorbar(points, ax=ax, label="alpha")
    _finish(fig, path)
    result["figure_E_error_consolidation_path"] = str(path)

    # Figure F
    path = paths["figure_F_pair_coverage_before_after"]
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    modules = list(dict.fromkeys(scaling.module_type.tolist()))
    xloc = np.arange(len(modules))
    for ax, split, title in zip(axes, ("split_a", "split_b"), ("Split A", "Split B held-out")):
        frame = scaling[scaling.split == split]
        before = frame[frame.alpha == "identity"].set_index("module_type").reindex(modules)
        after = frame[frame.selected_on_split_A.astype(str).str.lower().isin(("true", "1"))]
        after = after.set_index("module_type").reindex(modules)
        ax.bar(xloc - 0.18, before.rho_pair_256, 0.36, color="#aab7b8", label="identity")
        ax.bar(xloc + 0.18, after.rho_pair_256, 0.36, color=GREEN, label="selected on A")
        ax.axhline(0.70, color="#a23e48", linestyle="--", linewidth=1)
        ax.set_ylabel("ρPair@256")
        ax.set_title(title, loc="left", fontweight="bold", color=INK)
        ax.grid(axis="y", color=GRID, linewidth=0.7)
        ax.legend(frameon=False, ncol=2)
    axes[-1].set_xticks(xloc, modules, rotation=25, ha="right")
    fig.suptitle("Figure F — Pair coverage before / after shaping", x=0.06, ha="left", color=INK, fontweight="bold")
    _finish(fig, path)
    result["figure_F_pair_coverage_before_after"] = str(path)
    return result


__all__ = ["render_figures"]
