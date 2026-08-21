"""Output-causal NVFP4 RTN attribution and equal-budget K+S interventions.

This experiment keeps the main RTN layout at identity so that compensation can
be isolated from the previously observed reorder regression.  For a linear map

    Y = X W^T,   Y_q = Q(X) Q(W)^T,

the output residual is decomposed exactly as

    Y - Y_q = (X-QX) W^T + QX (W-QW)^T.

Each side is then split into mutually exclusive zero/non-zero E2M1 buckets,
and in a second view into exact-block-scale E2M1 error plus UE4M3 scale-rounding
perturbation.  Signed contributions use <component, total residual>, so they
sum to the total squared error even when components cancel.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utilize import get_wikitext2, load_model  # noqa: E402
from scripts.analyze_rtn_reorder_arc import (  # noqa: E402
    first_layer_inputs,
    nvfp4_quantize,
    pick_rows,
)


STRATEGIES = (
    "paper_activation",
    "activation_energy",
    "activation_greedy",
    "zero_greedy",
    "grid_greedy",
    "weight_greedy",
    "joint_greedy",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=str(
            REPO_ROOT.parent / "modelzoo" / "Qwen" / "Qwen2.5-1.5B-Instruct"
        ),
    )
    parser.add_argument("--saved-dir", default=str(REPO_ROOT / "saved"))
    parser.add_argument(
        "--baseline-metrics",
        default=str(REPO_ROOT / "analysis" / "rtn_reorder_arc" / "module_metrics.csv"),
    )
    parser.add_argument(
        "--output-dir", default=str(REPO_ROOT / "analysis" / "rtn_error_ks")
    )
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--rows-per-module", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--metric", default="max")
    parser.add_argument(
        "--wikitext-cache-dir",
        default=os.environ.get("ARCQUANT_WIKITEXT_CACHE_DIR"),
    )
    return parser.parse_args()


def safe_ratio(numerator: float, denominator: float) -> float:
    if not math.isfinite(denominator) or abs(denominator) < 1e-30:
        return float("nan")
    return numerator / denominator


@torch.no_grad()
def component_attribution(
    total_residual: torch.Tensor,
    components: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    total_residual = total_residual.float()
    total_sse = total_residual.square().sum().item()
    reconstructed = torch.zeros_like(total_residual)
    result: Dict[str, float] = {}
    for name, component in components.items():
        component = component.float()
        reconstructed.add_(component)
        result[f"{name}_signed_contribution"] = safe_ratio(
            (component * total_residual).sum().item(), total_sse
        )
        result[f"{name}_energy_ratio"] = safe_ratio(
            component.square().sum().item(), total_sse
        )
    result["signed_contribution_sum"] = sum(
        value
        for key, value in result.items()
        if key.endswith("_signed_contribution")
    )
    result["component_energy_sum"] = sum(
        value for key, value in result.items() if key.endswith("_energy_ratio")
    )
    result["cross_term_ratio"] = 1.0 - result["component_energy_sum"]
    result["closure_relative_sse"] = safe_ratio(
        (reconstructed - total_residual).square().sum().item(), total_sse
    )
    return result


@torch.no_grad()
def single_rank1_gain_score(
    total_residual: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Exact one-column ideal SSE-reduction score and component energy."""
    projection = total_residual.float().matmul(right.float())
    dot = (left.float() * projection).sum(dim=0)
    energy = left.float().square().sum(dim=0) * right.float().square().sum(dim=0)
    return 2.0 * dot - energy, energy


def top_indices(score: torch.Tensor, count: int) -> torch.Tensor:
    if count <= 0:
        return torch.empty(0, device=score.device, dtype=torch.long)
    return torch.topk(score, k=min(count, score.numel()), largest=True).indices


@torch.no_grad()
def build_augmented_pair(
    strategy: str,
    *,
    select_num: int,
    paper_indices: torch.Tensor,
    ex: torch.Tensor,
    qx: torch.Tensor,
    weight: torch.Tensor,
    ew: torch.Tensor,
    activation_energy_score: torch.Tensor,
    activation_gain_score: torch.Tensor,
    zero_gain_score: torch.Tensor,
    grid_gain_score: torch.Tensor,
    weight_gain_score: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    width = ex.shape[1]
    if strategy == "paper_activation":
        indices = paper_indices[-select_num:]
        left = ex.index_select(1, indices)
        right = weight.index_select(1, indices)
        branch = torch.zeros(select_num, device=ex.device, dtype=torch.long)
        return left, right, indices, select_num, 0

    activation_scores = {
        "activation_energy": activation_energy_score,
        "activation_greedy": activation_gain_score,
        "zero_greedy": zero_gain_score,
        "grid_greedy": grid_gain_score,
    }
    if strategy in activation_scores:
        indices = top_indices(activation_scores[strategy], select_num)
        left = ex.index_select(1, indices)
        right = weight.index_select(1, indices)
        return left, right, indices, select_num, 0

    if strategy == "weight_greedy":
        indices = top_indices(weight_gain_score, select_num)
        left = qx.index_select(1, indices)
        right = ew.index_select(1, indices)
        return left, right, indices + width, 0, select_num

    if strategy != "joint_greedy":
        raise ValueError(f"Unknown strategy: {strategy}")

    combined = torch.cat([activation_gain_score, weight_gain_score], dim=0)
    preliminary = top_indices(combined, select_num)
    activation_count = int((preliminary < width).sum().item())
    # Keep the two numerically different branch families in separate 16-wide
    # blocks. This avoids mixing small residual columns with main-activation
    # columns under one NVFP4 block scale.
    activation_count = int(round(activation_count / 16.0) * 16)
    activation_count = max(0, min(select_num, activation_count))
    weight_count = select_num - activation_count
    activation_indices = top_indices(activation_gain_score, activation_count)
    weight_indices = top_indices(weight_gain_score, weight_count)
    left = torch.cat(
        [
            ex.index_select(1, activation_indices),
            qx.index_select(1, weight_indices),
        ],
        dim=1,
    )
    right = torch.cat(
        [
            weight.index_select(1, activation_indices),
            ew.index_select(1, weight_indices),
        ],
        dim=1,
    )
    encoded_indices = torch.cat(
        [activation_indices, weight_indices + width], dim=0
    )
    return left, right, encoded_indices, activation_count, weight_count


@torch.no_grad()
def analyze_linear(
    *,
    name: str,
    module: nn.Linear,
    x: torch.Tensor,
    x_global_scale: torch.Tensor,
    paper_permutation: torch.Tensor,
    select_num: int,
) -> Tuple[Dict[str, Any], list[Dict[str, Any]]]:
    device = x.device
    width = x.shape[1]
    select_num = int(max(0, min(select_num, width)))
    if select_num % 16:
        raise ValueError(f"select_num must be 16-aligned, got {select_num} for {name}")
    paper_permutation = paper_permutation.to(device=device, dtype=torch.long)
    weight = module.weight.detach().float()
    bias = module.bias.detach().float() if module.bias is not None else None
    x = x.float()
    weight_global_scale = weight.abs().max() / (448.0 * 6.0)

    qx, _ = nvfp4_quantize(x, global_scale=x_global_scale)
    qx_exact, _ = nvfp4_quantize(
        x, global_scale=x_global_scale, exact_block_scale=True
    )
    qw, _ = nvfp4_quantize(weight, global_scale=weight_global_scale)
    qw_exact, _ = nvfp4_quantize(
        weight, global_scale=weight_global_scale, exact_block_scale=True
    )
    y_reference = F.linear(x, weight, bias).float()
    y_rtn = F.linear(qx, qw, bias).float()
    total_residual = y_reference - y_rtn
    total_sse = total_residual.square().sum().item()
    output_elements = total_residual.numel()

    ex = x - qx
    ew = weight - qw
    activation_zero_mask = (qx == 0) & (x != 0)
    weight_zero_mask = (qw == 0) & (weight != 0)
    ex_zero = torch.where(activation_zero_mask, ex, torch.zeros_like(ex))
    ex_grid = ex - ex_zero
    ew_zero = torch.where(weight_zero_mask, ew, torch.zeros_like(ew))
    ew_grid = ew - ew_zero

    observed_components = {
        "activation_zero": F.linear(ex_zero, weight, bias=None),
        "activation_grid": F.linear(ex_grid, weight, bias=None),
        "weight_zero": F.linear(qx, ew_zero, bias=None),
        "weight_grid": F.linear(qx, ew_grid, bias=None),
    }
    observed = component_attribution(total_residual, observed_components)
    del observed_components

    ex_exact_grid = x - qx_exact
    ex_scale_delta = qx_exact - qx
    ew_exact_grid = weight - qw_exact
    ew_scale_delta = qw_exact - qw
    format_components = {
        "activation_e2m1": F.linear(ex_exact_grid, weight, bias=None),
        "activation_ue4m3_scale": F.linear(ex_scale_delta, weight, bias=None),
        "weight_e2m1": F.linear(qx, ew_exact_grid, bias=None),
        "weight_ue4m3_scale": F.linear(qx, ew_scale_delta, bias=None),
    }
    format_view = component_attribution(total_residual, format_components)
    del format_components, qx_exact, qw_exact, ex_exact_grid, ex_scale_delta
    del ew_exact_grid, ew_scale_delta

    activation_gain_score, activation_energy_score = single_rank1_gain_score(
        total_residual, ex, weight
    )
    zero_gain_score, _ = single_rank1_gain_score(total_residual, ex_zero, weight)
    grid_gain_score, _ = single_rank1_gain_score(total_residual, ex_grid, weight)
    weight_gain_score, weight_energy_score = single_rank1_gain_score(
        total_residual, qx, ew
    )

    attribution_row: Dict[str, Any] = {
        "module": name,
        "layer": int(name.split(".")[1]),
        "module_type": name.split(".")[-1],
        "in_features": int(module.in_features),
        "out_features": int(module.out_features),
        "sample_rows": int(x.shape[0]),
        "output_elements": int(output_elements),
        "select_num": select_num,
        "select_ratio": select_num / width,
        "output_reference_energy": y_reference.square().mean().item(),
        "identity_rtn_mse": total_sse / output_elements,
        "identity_rtn_nmse": safe_ratio(
            total_sse, y_reference.square().sum().item()
        ),
        "activation_zero_rate": activation_zero_mask.float().mean().item(),
        "weight_zero_rate": weight_zero_mask.float().mean().item(),
        "positive_activation_candidates": int((activation_gain_score > 0).sum().item()),
        "positive_weight_candidates": int((weight_gain_score > 0).sum().item()),
    }
    attribution_row.update({f"observed_{k}": v for k, v in observed.items()})
    attribution_row.update({f"format_{k}": v for k, v in format_view.items()})

    paper_selected = paper_permutation[-select_num:]
    strategy_rows: list[Dict[str, Any]] = []
    for strategy in STRATEGIES:
        left_aug, right_aug, encoded_indices, activation_count, weight_count = (
            build_augmented_pair(
                strategy,
                select_num=select_num,
                paper_indices=paper_permutation,
                ex=ex,
                qx=qx,
                weight=weight,
                ew=ew,
                activation_energy_score=activation_energy_score,
                activation_gain_score=activation_gain_score,
                zero_gain_score=zero_gain_score,
                grid_gain_score=grid_gain_score,
                weight_gain_score=weight_gain_score,
            )
        )
        ideal_correction = F.linear(left_aug, right_aug, bias=None).float()
        ideal_residual = total_residual - ideal_correction
        ideal_sse = ideal_residual.square().sum().item()
        qleft, _ = nvfp4_quantize(left_aug, global_scale=x_global_scale)
        qright, _ = nvfp4_quantize(
            right_aug, global_scale=weight_global_scale
        )
        quantized_correction = F.linear(qleft, qright, bias=None).float()
        corrected_residual = total_residual - quantized_correction
        corrected_sse = corrected_residual.square().sum().item()

        activation_indices = encoded_indices[encoded_indices < width]
        overlap = 0
        if activation_indices.numel() and paper_selected.numel():
            overlap = int(
                torch.isin(activation_indices, paper_selected).sum().item()
            )
        strategy_rows.append(
            {
                "module": name,
                "layer": attribution_row["layer"],
                "module_type": attribution_row["module_type"],
                "strategy": strategy,
                "in_features": width,
                "out_features": int(module.out_features),
                "sample_rows": int(x.shape[0]),
                "output_elements": int(output_elements),
                "select_num": select_num,
                "select_ratio": select_num / width,
                "activation_branch_num": activation_count,
                "weight_branch_num": weight_count,
                "paper_activation_overlap": overlap,
                "paper_activation_overlap_rate": safe_ratio(
                    overlap, max(1, activation_count)
                ),
                "baseline_sse": total_sse,
                "corrected_sse": corrected_sse,
                "corrected_mse": corrected_sse / output_elements,
                "gain_vs_identity_rtn": 1.0 - safe_ratio(corrected_sse, total_sse),
                "ideal_corrected_sse": ideal_sse,
                "ideal_gain_vs_identity_rtn": 1.0
                - safe_ratio(ideal_sse, total_sse),
                "correction_quantization_gap": safe_ratio(
                    corrected_sse - ideal_sse, total_sse
                ),
            }
        )
        del left_aug, right_aug, ideal_correction, ideal_residual
        del qleft, qright, quantized_correction, corrected_residual

    del y_reference, y_rtn, total_residual, ex, ew, ex_zero, ex_grid
    del ew_zero, ew_grid, activation_gain_score, activation_energy_score
    del zero_gain_score, grid_gain_score, weight_gain_score, weight_energy_score
    return attribution_row, strategy_rows


def aggregate_attribution(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    total_sse = (df.identity_rtn_mse * df.output_elements).sum()
    component_names = sorted(
        {
            column[len(prefix) : -len("_signed_contribution")]
            for column in df.columns
            if column.startswith(prefix) and column.endswith("_signed_contribution")
            and "sum" not in column
        }
    )
    records = []
    for component in component_names:
        signed = (
            df[f"{prefix}{component}_signed_contribution"]
            * df.identity_rtn_mse
            * df.output_elements
        ).sum() / total_sse
        energy = (
            df[f"{prefix}{component}_energy_ratio"]
            * df.identity_rtn_mse
            * df.output_elements
        ).sum() / total_sse
        records.append(
            {
                "component": component,
                "signed_contribution": signed,
                "energy_ratio": energy,
            }
        )
    return pd.DataFrame(records)


def aggregate_strategies(
    strategy_df: pd.DataFrame, baseline_df: pd.DataFrame
) -> pd.DataFrame:
    records = []
    total_baseline = strategy_df[
        strategy_df.strategy == STRATEGIES[0]
    ].baseline_sse.sum()
    for strategy, group in strategy_df.groupby("strategy"):
        records.append(
            {
                "strategy": strategy,
                "gain": 1.0 - group.corrected_sse.sum() / total_baseline,
                "ideal_gain": 1.0
                - group.ideal_corrected_sse.sum() / total_baseline,
                "positive_module_rate": float(
                    (group.gain_vs_identity_rtn > 0).mean()
                ),
                "mean_activation_branch_fraction": float(
                    (group.activation_branch_num / group.select_num).mean()
                ),
                "mean_paper_overlap_rate": float(
                    group.paper_activation_overlap_rate.mean()
                ),
                "modules": int(len(group)),
            }
        )
    if not baseline_df.empty:
        elements = baseline_df.sample_rows * baseline_df.out_features
        identity_sse = (baseline_df.output_identity_rtn_mse * elements).sum()
        reorder_arc_sse = (baseline_df.output_arc_mse * elements).sum()
        records.append(
            {
                "strategy": "paper_reorder_arc",
                "gain": 1.0 - reorder_arc_sse / identity_sse,
                "ideal_gain": float("nan"),
                "positive_module_rate": float(
                    (baseline_df.arc_gain_vs_identity > 0).mean()
                ),
                "mean_activation_branch_fraction": 1.0,
                "mean_paper_overlap_rate": 1.0,
                "modules": int(len(baseline_df)),
            }
        )
    return pd.DataFrame(records).sort_values("gain", ascending=False)


def make_charts(
    attribution_df: pd.DataFrame,
    strategy_df: pd.DataFrame,
    strategy_summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 13})
    colors = {
        "activation_zero": "#C2410C",
        "activation_grid": "#2563EB",
        "weight_zero": "#F59E0B",
        "weight_grid": "#64748B",
        "activation_e2m1": "#2563EB",
        "activation_ue4m3_scale": "#C2410C",
        "weight_e2m1": "#64748B",
        "weight_ue4m3_scale": "#F59E0B",
    }

    observed = aggregate_attribution(attribution_df, "observed_")
    format_view = aggregate_attribution(attribution_df, "format_")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    for ax, frame, title in (
        (axes[0], observed, "Observed zero vs non-zero grid buckets"),
        (axes[1], format_view, "E2M1 data grid vs UE4M3 scale rounding"),
    ):
        frame = frame.sort_values("signed_contribution")
        bars = ax.barh(
            frame.component,
            frame.signed_contribution,
            color=[colors.get(name, "#64748B") for name in frame.component],
        )
        ax.axvline(0, color="#111827", lw=1)
        ax.set_xlabel("Signed contribution to W4A4 output SSE")
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.2)
        ax.bar_label(bars, fmt="%+.1f%%", labels=[f"{v:+.1%}" for v in frame.signed_contribution])
    fig.suptitle("Exact additive attribution of identity NVFP4 RTN output error")
    fig.tight_layout()
    fig.savefig(output_dir / "output_error_attribution.png", dpi=180)
    plt.close(fig)

    plot_summary = strategy_summary.dropna(subset=["gain"]).sort_values("gain")
    labels = plot_summary.strategy.str.replace("_", " ")
    fig, ax = plt.subplots(figsize=(10, 6.2))
    y = np.arange(len(plot_summary))
    ax.barh(y, plot_summary.gain * 100.0, color="#2563EB", label="Quantized K+S")
    ideal_mask = plot_summary.ideal_gain.notna()
    ax.scatter(
        plot_summary.loc[ideal_mask, "ideal_gain"] * 100.0,
        y[ideal_mask.to_numpy()],
        marker="|",
        s=180,
        linewidths=2,
        color="#C2410C",
        label="Same selected columns, FP correction",
    )
    ax.axvline(0, color="#111827", lw=1)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Gain versus identity RTN output SSE (%)")
    ax.set_title("Equal-S K+S compensation strategies")
    ax.grid(axis="x", alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "ks_strategy_gain.png", dpi=180)
    plt.close(fig)

    type_rows = []
    for (strategy, module_type), group in strategy_df.groupby(
        ["strategy", "module_type"]
    ):
        type_rows.append(
            {
                "strategy": strategy,
                "module_type": module_type,
                "gain": 1.0 - group.corrected_sse.sum() / group.baseline_sse.sum(),
            }
        )
    matrix = pd.DataFrame(type_rows).pivot(
        index="strategy", columns="module_type", values="gain"
    )
    matrix = matrix.loc[strategy_summary[strategy_summary.strategy.isin(matrix.index)].strategy]
    fig, ax = plt.subplots(figsize=(11, 6.2))
    image = ax.imshow(matrix.to_numpy() * 100.0, cmap="RdBu_r", vmin=-50, vmax=50, aspect="auto")
    ax.set_xticks(np.arange(len(matrix.columns)), matrix.columns, rotation=30, ha="right")
    ax.set_yticks(
        np.arange(len(matrix.index)), [value.replace("_", " ") for value in matrix.index]
    )
    ax.set_title("K+S gain by module type and selector")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix.iloc[row, column] * 100.0
            ax.text(column, row, f"{value:+.0f}%", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="Gain vs identity RTN output SSE (%)")
    fig.tight_layout()
    fig.savefig(output_dir / "ks_gain_by_module_type.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.wikitext_cache_dir:
        os.environ["ARCQUANT_WIKITEXT_CACHE_DIR"] = args.wikitext_cache_dir
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA is required for this layer-wise experiment")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.time()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cuda_index = torch.device(args.device).index or 0
    torch.cuda.set_device(cuda_index)
    torch.cuda.reset_peak_memory_stats()

    model_path = Path(args.model)
    model_name = model_path.name.lower()
    saved_dir = Path(args.saved_dir)
    reorder_path = saved_dir / f"{model_name}_reorder_index_wikitext2_{args.metric}.pt"
    select_path = saved_dir / f"{model_name}_select_num_wikitext2_{args.metric}.pt"
    for required in (reorder_path, select_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    reorder_index = torch.load(reorder_path, map_location="cpu")
    select_nums = torch.load(select_path, map_location="cpu")
    baseline_path = Path(args.baseline_metrics)
    baseline_df = pd.read_csv(baseline_path) if baseline_path.is_file() else pd.DataFrame()

    model, tokenizer = load_model(str(model_path))
    dataloader, _ = get_wikitext2(
        nsamples=args.samples,
        seed=args.seed,
        seqlen=args.seqlen,
        tokenizer=tokenizer,
    )
    inps, attention_mask, position_ids = first_layer_inputs(
        model, dataloader, args.device
    )
    del dataloader, tokenizer

    attribution_rows: list[Dict[str, Any]] = []
    strategy_rows: list[Dict[str, Any]] = []
    layers = model.model.layers
    for layer_index, layer in enumerate(layers):
        print(f"Analyzing layer {layer_index + 1}/{len(layers)}", flush=True)
        layer = layer.to(args.device)
        sampled_inputs: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        hooks = []

        def capture_hook(
            _module: nn.Module,
            inputs: Tuple[torch.Tensor, ...],
            _output: torch.Tensor,
            *,
            full_name: str,
        ) -> None:
            value = inputs[0]
            sampled_inputs[full_name] = (
                pick_rows(value, args.rows_per_module),
                value.detach().abs().amax().float(),
            )

        for local_name, module in layer.named_modules():
            if isinstance(module, nn.Linear):
                full_name = f"layers.{layer_index}.{local_name}"
                hooks.append(
                    module.register_forward_hook(
                        lambda m, i, o, n=full_name: capture_hook(
                            m, i, o, full_name=n
                        )
                    )
                )

        inps = inps.to(args.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(args.device)
        if position_ids is not None:
            position_ids = position_ids.to(args.device)
        next_inps = layer(
            inps, attention_mask=attention_mask, position_ids=position_ids
        )[0]
        for hook in hooks:
            hook.remove()

        modules = dict(layer.named_modules())
        for full_name, (sample, full_absmax) in sampled_inputs.items():
            local_name = full_name.split(f"layers.{layer_index}.", 1)[1]
            key = full_name + ".input"
            attribution, interventions = analyze_linear(
                name=full_name,
                module=modules[local_name],
                x=sample,
                x_global_scale=full_absmax / (448.0 * 6.0),
                paper_permutation=reorder_index[key],
                select_num=int(select_nums[key]),
            )
            attribution_rows.append(attribution)
            strategy_rows.extend(interventions)
            best = max(interventions, key=lambda row: row["gain_vs_identity_rtn"])
            print(
                f"  {local_name}: best={best['strategy']} "
                f"gain={best['gain_vs_identity_rtn']:+.2%}",
                flush=True,
            )
            pd.DataFrame(attribution_rows).to_csv(
                output_dir / "module_attribution.csv", index=False
            )
            pd.DataFrame(strategy_rows).to_csv(
                output_dir / "ks_strategy_metrics.csv", index=False
            )
            del sample, attribution, interventions
            gc.collect()
            torch.cuda.empty_cache()

        layers[layer_index] = layer.cpu()
        inps = next_inps
        del layer, sampled_inputs, modules
        gc.collect()
        torch.cuda.empty_cache()

    attribution_df = pd.DataFrame(attribution_rows)
    strategy_df = pd.DataFrame(strategy_rows)
    observed_summary = aggregate_attribution(attribution_df, "observed_")
    format_summary = aggregate_attribution(attribution_df, "format_")
    observed_summary.to_csv(output_dir / "observed_error_summary.csv", index=False)
    format_summary.to_csv(output_dir / "format_error_summary.csv", index=False)
    strategy_summary = aggregate_strategies(strategy_df, baseline_df)
    strategy_summary.to_csv(output_dir / "ks_strategy_summary.csv", index=False)
    make_charts(attribution_df, strategy_df, strategy_summary, output_dir)

    metadata = {
        "model": str(model_path.resolve()),
        "calibration": {
            "dataset": "WikiText2 train",
            "samples": 128,
            "seqlen": 2048,
            "seed": 0,
            "metric": args.metric,
        },
        "diagnostic": {
            "samples": args.samples,
            "seqlen": args.seqlen,
            "sample_rows_per_module": args.rows_per_module,
            "seed": args.seed,
            "main_layout": "identity",
            "rotation": False,
            "strategies": list(STRATEGIES),
            "budget": "same per-module S as ARCQuant calibration",
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(cuda_index),
        },
        "module_rows": len(attribution_df),
        "strategy_rows": len(strategy_df),
        "elapsed_seconds": time.time() - started_at,
        "peak_gpu_memory_gib": torch.cuda.max_memory_allocated()
        / (1024.0**3),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(strategy_summary.to_string(index=False), flush=True)
    print(f"Wrote results to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
