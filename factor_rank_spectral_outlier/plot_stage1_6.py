"""Render the required Stage 1.6 Figure A--G diagnostics."""

from __future__ import annotations

from pathlib import Path

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
RED = "#a23e48"
GRID = "#d9e2ec"


def _finish(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_stage1_6_figures(output_root: str | Path) -> dict[str, str]:
    root = Path(output_root).resolve()
    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    factors = pd.read_csv(root / "factor_spectrum_summary.csv")
    expansion = pd.read_csv(root / "rank_expansion_summary.csv")
    channels = pd.read_csv(root / "spectral_outlier_channels.csv")
    collapse = pd.read_csv(root / "spectral_collapse_summary.csv")
    curves = pd.read_csv(root / "spectrum_curves.csv")
    stability = pd.read_csv(root / "support_stability_summary.csv")
    result: dict[str, str] = {}

    # Figure A -- Factor spectra.
    path = figures / "figure_A_factor_spectra.png"
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    palette = {"factor1": BLUE, "factor2": ORANGE, "gram": GREEN}
    for ax, source in zip(axes, ("X", "W")):
        frame = curves[
            (curves.curve_group == "factor")
            & (curves.source == source)
            & (curves.split == "combined")
        ]
        for item, group in frame.groupby("curve_name"):
            median = group.groupby("rank", as_index=False).rho.median()
            ax.plot(
                median["rank"], median.rho, linewidth=2, color=palette.get(item, PURPLE),
                label={"factor1": "error/activation factor", "factor2": "mapping/error factor", "gram": "Functional Gram"}.get(item, item),
            )
        ax.set(xlabel="rank", ylabel="structural coverage", ylim=(0, 1.01), title=f"{source}-source")
        ax.grid(axis="y", color=GRID)
        ax.legend(frameon=False)
    fig.suptitle("Figure A — Factor spectra", x=0.06, ha="left", color=INK, fontweight="bold")
    _finish(fig, path)
    result["figure_A_factor_spectra"] = str(path)

    # Figure B -- effective-rank decomposition.
    path = figures / "figure_B_effective_rank_decomposition.png"
    combined = factors[factors.split == "combined"].copy()
    pivot = combined.pivot_table(
        index=["module", "source"], columns="factor_role", values="entropy_effective_rank", aggfunc="first"
    ).reset_index()
    labels = [f"{row.module_type if hasattr(row, 'module_type') else row.module.split('.')[-1]}-{row.source}" for row in pivot.itertuples()]
    xloc = np.arange(len(pivot))
    fig, ax = plt.subplots(figsize=(max(10, len(pivot) * 0.8), 5.4))
    for offset, role, color in ((-0.25, "factor1", BLUE), (0, "factor2", ORANGE), (0.25, "gram", GREEN)):
        ax.bar(xloc + offset, pivot.get(role, pd.Series(np.nan, index=pivot.index)), 0.24, label=role, color=color)
    ax.set_xticks(xloc, labels, rotation=35, ha="right")
    ax.set_ylabel("entropy effective rank")
    ax.set_title("Figure B — Effective-rank decomposition", loc="left", color=INK, fontweight="bold")
    ax.grid(axis="y", color=GRID)
    ax.legend(frameon=False, ncol=3)
    _finish(fig, path)
    result["figure_B_effective_rank_decomposition"] = str(path)

    # Figure C -- rank inflation heatmap.
    path = figures / "figure_C_rank_inflation_heatmap.png"
    frame = expansion[expansion.split == "combined"].copy()
    rows = sorted(frame.layer.unique())
    columns = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    fig, axes = plt.subplots(1, 2, figsize=(13, max(3.5, len(rows) * 1.1)), sharey=True)
    for ax, source in zip(axes, ("X", "W")):
        values = np.full((len(rows), len(columns)), np.nan)
        for row in frame[frame.source == source].itertuples():
            if row.module_type in columns:
                values[rows.index(row.layer), columns.index(row.module_type)] = row.rank_inflation_ratio
        image = ax.imshow(np.ma.masked_invalid(values), aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(columns)), columns, rotation=35, ha="right")
        ax.set_yticks(range(len(rows)), rows)
        ax.set_title(f"{source}-source")
        for i in range(len(rows)):
            for j in range(len(columns)):
                if np.isfinite(values[i, j]):
                    ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", fontsize=8)
        fig.colorbar(image, ax=ax, fraction=0.035, pad=0.03)
    fig.suptitle("Figure C — Rank-inflation ratio m", x=0.06, ha="left", color=INK, fontweight="bold")
    _finish(fig, path)
    result["figure_C_rank_inflation_heatmap"] = str(path)

    # Figure D -- J versus h.
    path = figures / "figure_D_J_vs_h.png"
    frame = channels[channels.split == "split_a"].copy()
    epsilon = np.finfo(float).tiny
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=False, sharey=False)
    for ax, source in zip(axes, ("X", "W")):
        source_frame = frame[frame.source == source]
        ordinary = (source_frame.rank_J > 32) & (source_frame.rank_h > 32)
        ax.scatter(np.log10(source_frame.loc[ordinary, "diag_J"].clip(lower=epsilon)), np.log10(source_frame.loc[ordinary, "tail_energy_h"].clip(lower=epsilon)), s=8, alpha=0.25, color="#7f8c8d")
        top_j = source_frame.rank_J <= 32
        top_h = source_frame.rank_h <= 32
        ax.scatter(np.log10(source_frame.loc[top_j, "diag_J"].clip(lower=epsilon)), np.log10(source_frame.loc[top_j, "tail_energy_h"].clip(lower=epsilon)), s=22, alpha=0.8, color=ORANGE, label="top-J")
        ax.scatter(np.log10(source_frame.loc[top_h, "diag_J"].clip(lower=epsilon)), np.log10(source_frame.loc[top_h, "tail_energy_h"].clip(lower=epsilon)), s=22, alpha=0.8, color=GREEN, label="top-h")
        ax.set(xlabel="log10 J", ylabel="log10 h", title=f"{source}-source")
        ax.grid(color=GRID)
        ax.legend(frameon=False)
    fig.suptitle("Figure D — Functional magnitude vs spectral-outlier energy", x=0.06, ha="left", color=INK, fontweight="bold")
    _finish(fig, path)
    result["figure_D_J_vs_h"] = str(path)

    # Figure E -- collapse curve.
    path = figures / "figure_E_spectral_collapse_curve.png"
    frame = collapse[(collapse.split == "split_a") & (collapse.random_seed.astype(str).isin(("aggregate", "none")))]
    colors = {"top_h": GREEN, "top_tail_ratio": PURPLE, "top_J": ORANGE, "random": "#7f8c8d"}
    fig, ax = plt.subplots(figsize=(9, 5.4))
    for method, group in frame.groupby("removal_method"):
        median = group.groupby("remove_budget", as_index=False).remain_rho_func_256.median()
        ax.plot(median.remove_budget, median.remain_rho_func_256, marker="o", linewidth=2, color=colors.get(method, BLUE), label=method)
    ax.set(xlabel="removed channels S", ylabel="remaining functional coverage @256", ylim=(0, 1.01))
    ax.grid(color=GRID)
    ax.legend(frameon=False, ncol=2)
    ax.set_title("Figure E — Spectral collapse curve", loc="left", color=INK, fontweight="bold")
    _finish(fig, path)
    result["figure_E_spectral_collapse_curve"] = str(path)

    # Figure F -- remaining spectra.
    path = figures / "figure_F_remaining_spectrum.png"
    frame = curves[
        (curves.curve_group == "remaining")
        & (curves.source.isin(("X", "W")))
        & (curves.split == "split_a")
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, source in zip(axes, ("X", "W")):
        for budget, color in ((0, BLUE), (64, ORANGE), (128, GREEN)):
            group = frame[(frame.source == source) & (frame.remove_budget == budget)]
            median = group.groupby("rank", as_index=False).rho.median()
            ax.plot(median["rank"], median.rho, color=color, linewidth=2, label=f"remove {budget}")
        ax.set(xlabel="rank", ylabel="structural coverage", ylim=(0, 1.01), title=f"{source}-source")
        ax.grid(color=GRID)
        ax.legend(frameon=False)
    fig.suptitle("Figure F — Remaining spectrum before/after top-h removal", x=0.06, ha="left", color=INK, fontweight="bold")
    _finish(fig, path)
    result["figure_F_remaining_spectrum"] = str(path)

    # Figure G -- split stability.
    path = figures / "figure_G_support_stability.png"
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for source, color in (("X", BLUE), ("W", ORANGE)):
        group = stability[stability.source == source]
        median = group.groupby("budget", as_index=False).jaccard.median()
        ax.plot(median.budget, median.jaccard, marker="o", linewidth=2, color=color, label=source)
    ax.set(xlabel="top-h support budget", ylabel="Split A/B Jaccard", ylim=(0, 1.01))
    ax.grid(color=GRID)
    ax.legend(frameon=False)
    ax.set_title("Figure G — Spectral-outlier support stability", loc="left", color=INK, fontweight="bold")
    _finish(fig, path)
    result["figure_G_support_stability"] = str(path)
    return result


__all__ = ["render_stage1_6_figures"]
