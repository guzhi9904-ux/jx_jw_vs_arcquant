"""Diagnose NVFP4 RTN error, channel reorder, and ARC K+S correction.

The script intentionally does not apply rotations.  It compares equivalent
linear maps under three layouts:

1. identity RTN:       Q(X) Q(W)^T
2. reordered RTN:      Q(XP) Q(WP)^T
3. reordered ARC K+S:  Q(XP) Q(WP)^T + Q(E_S) Q(WP_S)^T

where P is the calibrated input-channel permutation and E = XP - Q(XP).
It also separates errors that become zero from non-zero E2M1 grid error and
reports the incremental effect of UE4M3 block-scale rounding.
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


E2M1_VALUES = (-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0,
               0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=str(
            REPO_ROOT.parent
            / "modelzoo"
            / "Qwen"
            / "Qwen2.5-1.5B-Instruct"
        ),
    )
    parser.add_argument("--saved-dir", default=str(REPO_ROOT / "saved"))
    parser.add_argument(
        "--output-dir", default=str(REPO_ROOT / "analysis" / "rtn_reorder_arc")
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


def nearest_e2m1(x: torch.Tensor) -> torch.Tensor:
    values = torch.tensor(E2M1_VALUES, device=x.device, dtype=x.dtype)
    boundaries = (values[:-1] + values[1:]) * 0.5
    indices = torch.bucketize(x.contiguous(), boundaries, right=False)
    return values[indices]


def ue4m3_round(x: torch.Tensor) -> torch.Tensor:
    x = torch.clamp(x, min=2e-3, max=448.0)
    exponent = torch.floor(torch.log2(x + 1e-9))
    mantissa = x / torch.pow(2.0, exponent) - 1.0
    mantissa = torch.round(mantissa * 8.0) / 8.0
    return (1.0 + mantissa) * torch.pow(2.0, exponent)


@torch.no_grad()
def nvfp4_quantize(
    x: torch.Tensor,
    *,
    global_scale: torch.Tensor | float | None = None,
    exact_block_scale: bool = False,
    group_size: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Memory-efficient numerical equivalent of the repository fake quantizer."""
    x = x.float()
    if global_scale is None:
        global_scale = x.abs().max() / (448.0 * 6.0)
    global_scale = torch.as_tensor(global_scale, device=x.device, dtype=torch.float32)
    global_scale = torch.where(
        global_scale > 0, global_scale, torch.ones_like(global_scale)
    )

    original_shape = x.shape
    padding = (-x.shape[-1]) % group_size
    x_scaled = x / global_scale
    if padding:
        x_scaled = F.pad(x_scaled, (0, padding))
    blocks = x_scaled.reshape(-1, group_size)
    exact_scale = blocks.abs().amax(dim=1, keepdim=True) / 6.0
    exact_scale = torch.where(
        exact_scale > 0, exact_scale, torch.full_like(exact_scale, 1e-9)
    )
    block_scale = exact_scale if exact_block_scale else ue4m3_round(exact_scale)
    quantized = nearest_e2m1(blocks / block_scale) * block_scale
    quantized = quantized.reshape(x_scaled.shape)
    if padding:
        quantized = quantized[..., :-padding]
    return (quantized.reshape(original_shape) * global_scale), global_scale


def safe_ratio(numerator: float, denominator: float) -> float:
    if not math.isfinite(denominator) or abs(denominator) < 1e-30:
        return float("nan")
    return numerator / denominator


@torch.no_grad()
def tensor_error_stats(
    x: torch.Tensor,
    qx: torch.Tensor,
    qx_exact_scale: torch.Tensor,
    selected_channels: torch.Tensor,
    channel_scores: torch.Tensor | None,
) -> Dict[str, float]:
    x = x.float()
    qx = qx.float()
    selected_channels = selected_channels.to(device=x.device, dtype=torch.bool)
    squared_error = (x - qx).square()
    exact_squared_error = (x - qx_exact_scale.float()).square()
    total_sse = squared_error.sum().item()
    total_energy = x.square().sum().item()

    zero_mask = (qx == 0) & (x != 0)
    nonzero_mask = ~zero_mask
    selected_mask = selected_channels.reshape(
        *((1,) * (x.ndim - 1)), selected_channels.numel()
    ).expand_as(x)

    groups = selected_channels.reshape(-1, 16)
    group_has_selected = groups.any(dim=1)
    group_all_selected = groups.all(dim=1)
    mixed_groups = group_has_selected & ~group_all_selected
    contaminated_ordinary_channels = (
        mixed_groups[:, None].expand(-1, 16).reshape(-1) & ~selected_channels
    )
    contaminated_mask = contaminated_ordinary_channels.reshape(
        *((1,) * (x.ndim - 1)), contaminated_ordinary_channels.numel()
    ).expand_as(x)

    blocks = x.abs().reshape(-1, 16)
    block_max = blocks.amax(dim=1)
    block_median = blocks.median(dim=1).values
    dominance_log2 = torch.log2((block_max + 1e-12) / (block_median + 1e-12))
    exact_zero_region = blocks < (block_max[:, None] / 24.0)

    result = {
        "elements": float(x.numel()),
        "mse": squared_error.mean().item(),
        "nmse": safe_ratio(total_sse, total_energy),
        "zero_drop_rate": zero_mask.float().mean().item(),
        "zero_error_share": safe_ratio(
            squared_error.masked_select(zero_mask).sum().item(), total_sse
        ),
        "nonzero_grid_error_share": safe_ratio(
            squared_error.masked_select(nonzero_mask).sum().item(), total_sse
        ),
        "exact_scale_mse": exact_squared_error.mean().item(),
        "ue4m3_scale_mse_delta": squared_error.mean().item()
        - exact_squared_error.mean().item(),
        "ue4m3_scale_relative_delta": safe_ratio(
            squared_error.mean().item() - exact_squared_error.mean().item(),
            squared_error.mean().item(),
        ),
        "selected_error_share": safe_ratio(
            squared_error.masked_select(selected_mask).sum().item(), total_sse
        ),
        "ordinary_zero_error_share": safe_ratio(
            squared_error.masked_select(zero_mask & ~selected_mask).sum().item(),
            total_sse,
        ),
        "contaminated_ordinary_error_share": safe_ratio(
            squared_error.masked_select(contaminated_mask).sum().item(), total_sse
        ),
        "contaminated_ordinary_zero_error_share": safe_ratio(
            squared_error.masked_select(contaminated_mask & zero_mask).sum().item(),
            total_sse,
        ),
        "mixed_block_fraction": mixed_groups.float().mean().item(),
        "ordinary_channel_contamination_rate": contaminated_ordinary_channels.float()
        .mean()
        .item(),
        "block_dominance_log2_mean": dominance_log2.mean().item(),
        "block_dominance_log2_p95": torch.quantile(dominance_log2, 0.95).item(),
        "exact_zero_region_rate": exact_zero_region.float().mean().item(),
    }

    if channel_scores is not None:
        channel_scores = channel_scores.to(device=x.device, dtype=torch.float32)
        score_blocks = channel_scores.clamp_min(1e-30).log10().reshape(-1, 16)
        score_span = score_blocks.amax(dim=1) - score_blocks.amin(dim=1)
        result["channel_score_log10_span_mean"] = score_span.mean().item()
        result["channel_score_log10_span_p95"] = torch.quantile(
            score_span, 0.95
        ).item()
    else:
        result["channel_score_log10_span_mean"] = float("nan")
        result["channel_score_log10_span_p95"] = float("nan")
    return result


def prefix_stats(prefix: str, stats: Dict[str, float]) -> Dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in stats.items()}


@torch.no_grad()
def output_nmse(
    y: torch.Tensor, y_reference: torch.Tensor, reference_energy: float
) -> Tuple[float, float]:
    mse = (y.float() - y_reference).square().mean().item()
    return mse, safe_ratio(mse, reference_energy)


@torch.no_grad()
def analyze_linear(
    *,
    name: str,
    module: nn.Linear,
    x: torch.Tensor,
    x_global_scale: torch.Tensor,
    permutation: torch.Tensor,
    select_num: int,
    channel_scores: torch.Tensor | None,
) -> Dict[str, Any]:
    device = x.device
    width = x.shape[-1]
    permutation = permutation.to(device=device, dtype=torch.long)
    identity = torch.arange(width, device=device)
    select_num = int(max(0, min(select_num, width)))
    selected_original = torch.zeros(width, device=device, dtype=torch.bool)
    if select_num:
        selected_original[permutation[-select_num:]] = True
    selected_reordered = torch.zeros(width, device=device, dtype=torch.bool)
    if select_num:
        selected_reordered[-select_num:] = True

    weight = module.weight.detach().float()
    bias = module.bias.detach().float() if module.bias is not None else None
    x = x.float()
    xr = x.index_select(1, permutation)
    wr = weight.index_select(1, permutation)
    scores_identity = channel_scores
    scores_reordered = (
        channel_scores.index_select(0, permutation.cpu())
        if channel_scores is not None
        else None
    )

    weight_global_scale = weight.abs().max() / (448.0 * 6.0)
    qx_i, _ = nvfp4_quantize(x, global_scale=x_global_scale)
    qx_i_exact, _ = nvfp4_quantize(
        x, global_scale=x_global_scale, exact_block_scale=True
    )
    qx_r, _ = nvfp4_quantize(xr, global_scale=x_global_scale)
    qx_r_exact, _ = nvfp4_quantize(
        xr, global_scale=x_global_scale, exact_block_scale=True
    )

    qw_i, _ = nvfp4_quantize(weight, global_scale=weight_global_scale)
    qw_i_exact, _ = nvfp4_quantize(
        weight, global_scale=weight_global_scale, exact_block_scale=True
    )
    qw_r, _ = nvfp4_quantize(wr, global_scale=weight_global_scale)
    qw_r_exact, _ = nvfp4_quantize(
        wr, global_scale=weight_global_scale, exact_block_scale=True
    )

    row: Dict[str, Any] = {
        "module": name,
        "layer": int(name.split(".")[1]),
        "module_type": name.split(".")[-1],
        "in_features": int(module.in_features),
        "out_features": int(module.out_features),
        "sample_rows": int(x.shape[0]),
        "select_num": select_num,
        "select_ratio": select_num / width,
        "average_bits": 4.5 * (width + select_num) / width,
    }
    row.update(
        prefix_stats(
            "activation_identity",
            tensor_error_stats(
                x, qx_i, qx_i_exact, selected_original, scores_identity
            ),
        )
    )
    row.update(
        prefix_stats(
            "activation_reorder",
            tensor_error_stats(
                xr, qx_r, qx_r_exact, selected_reordered, scores_reordered
            ),
        )
    )
    row.update(
        prefix_stats(
            "weight_identity",
            tensor_error_stats(
                weight, qw_i, qw_i_exact, selected_original, scores_identity
            ),
        )
    )
    row.update(
        prefix_stats(
            "weight_reorder",
            tensor_error_stats(
                wr, qw_r, qw_r_exact, selected_reordered, scores_reordered
            ),
        )
    )

    y_reference = F.linear(x, weight, bias).float()
    reference_energy = y_reference.square().mean().item()
    row["output_reference_energy"] = reference_energy

    output_variants = {
        "identity_activation_only": F.linear(qx_i, weight, bias),
        "identity_weight_only": F.linear(x, qw_i, bias),
        "identity_rtn": F.linear(qx_i, qw_i, bias),
        "reorder_activation_only": F.linear(qx_r, wr, bias),
        "reorder_weight_only": F.linear(xr, qw_r, bias),
    }
    for variant_name, output in output_variants.items():
        mse, nmse = output_nmse(output, y_reference, reference_energy)
        row[f"output_{variant_name}_mse"] = mse
        row[f"output_{variant_name}_nmse"] = nmse
    del output_variants

    y_reorder = F.linear(qx_r, qw_r, bias).float()
    reorder_mse, reorder_nmse = output_nmse(
        y_reorder, y_reference, reference_energy
    )
    row["output_reorder_rtn_mse"] = reorder_mse
    row["output_reorder_rtn_nmse"] = reorder_nmse

    if select_num:
        residual_selected = (xr - qx_r)[:, -select_num:]
        weight_selected = wr[:, -select_num:]
        q_residual_selected, _ = nvfp4_quantize(
            residual_selected, global_scale=x_global_scale
        )
        q_weight_selected, _ = nvfp4_quantize(
            weight_selected, global_scale=weight_global_scale
        )
        correction = F.linear(
            q_residual_selected, q_weight_selected, bias=None
        ).float()
    else:
        correction = torch.zeros_like(y_reorder)
    y_arc = y_reorder + correction
    arc_mse, arc_nmse = output_nmse(y_arc, y_reference, reference_energy)
    row["output_arc_mse"] = arc_mse
    row["output_arc_nmse"] = arc_nmse

    identity_mse = row["output_identity_rtn_mse"]
    row["reorder_gain_vs_identity"] = 1.0 - safe_ratio(
        reorder_mse, identity_mse
    )
    row["arc_gain_vs_identity"] = 1.0 - safe_ratio(arc_mse, identity_mse)
    row["arc_gain_vs_reorder"] = 1.0 - safe_ratio(arc_mse, reorder_mse)
    row["reorder_activation_gain_vs_identity"] = 1.0 - safe_ratio(
        row["output_reorder_activation_only_mse"],
        row["output_identity_activation_only_mse"],
    )
    row["reorder_weight_gain_vs_identity"] = 1.0 - safe_ratio(
        row["output_reorder_weight_only_mse"],
        row["output_identity_weight_only_mse"],
    )
    row["identity_interaction_delta_mse"] = (
        identity_mse
        - row["output_identity_activation_only_mse"]
        - row["output_identity_weight_only_mse"]
    )
    row["reorder_interaction_delta_mse"] = (
        reorder_mse
        - row["output_reorder_activation_only_mse"]
        - row["output_reorder_weight_only_mse"]
    )

    first_stage_error = y_reference - y_reorder
    correction_norm = correction.square().sum().sqrt().item()
    error_norm = first_stage_error.square().sum().sqrt().item()
    dot = (correction * first_stage_error).sum().item()
    row["arc_correction_error_cosine"] = safe_ratio(
        dot, correction_norm * error_norm
    )
    row["arc_correction_to_error_energy"] = safe_ratio(
        correction.square().sum().item(), first_stage_error.square().sum().item()
    )
    row["peak_gpu_memory_gib"] = (
        torch.cuda.max_memory_allocated(device) / (1024.0**3)
        if x.is_cuda
        else 0.0
    )
    return row


def pick_rows(x: torch.Tensor, count: int) -> torch.Tensor:
    flat = x.reshape(-1, x.shape[-1])
    if flat.shape[0] <= count:
        return flat.detach().float()
    indices = torch.linspace(
        0, flat.shape[0] - 1, count, device=flat.device
    ).round().long()
    return flat.index_select(0, indices).detach().float()


@torch.no_grad()
def first_layer_inputs(
    model: nn.Module, token_batches: Iterable[Tuple[torch.Tensor, torch.Tensor]], device: str
) -> Tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    layers = model.model.layers
    cache: Dict[str, Any] = {}

    class Catcher(nn.Module):
        def __init__(self, module: nn.Module):
            super().__init__()
            self.module = module

        def forward(self, inp: torch.Tensor, **kwargs: Any) -> torch.Tensor:
            cache["inps"] = inp
            cache["attention_mask"] = kwargs.get("attention_mask")
            cache["position_ids"] = kwargs.get("position_ids")
            raise ValueError("captured first-layer input")

    layers[0] = Catcher(layers[0])
    model.model.embed_tokens = model.model.embed_tokens.to(device)
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.to(device)
    input_ids = torch.stack([batch[0] for batch in token_batches], dim=0).squeeze(1)
    try:
        model(input_ids.to(device))
    except ValueError as error:
        if str(error) != "captured first-layer input":
            raise
    layers[0] = layers[0].module
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.cpu()
    torch.cuda.empty_cache()
    return cache["inps"], cache["attention_mask"], cache["position_ids"]


def flatten_quantization_attribution(module_df: pd.DataFrame) -> pd.DataFrame:
    records = []
    prefixes = (
        "activation_identity",
        "activation_reorder",
        "weight_identity",
        "weight_reorder",
    )
    for _, row in module_df.iterrows():
        for prefix in prefixes:
            entity, layout = prefix.split("_", 1)
            record = {
                "module": row["module"],
                "layer": row["layer"],
                "module_type": row["module_type"],
                "entity": entity,
                "layout": layout,
            }
            for column in module_df.columns:
                marker = prefix + "_"
                if column.startswith(marker):
                    record[column[len(marker):]] = row[column]
            records.append(record)
    return pd.DataFrame.from_records(records)


def make_charts(module_df: pd.DataFrame, output_dir: Path) -> None:
    colors = {
        "Identity RTN": "#6B7280",
        "Reorder RTN": "#2563EB",
        "Reorder + ARC": "#D97706",
    }
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 13})

    layer_df = module_df.groupby("layer", as_index=False).agg(
        identity_nmse=("output_identity_rtn_nmse", "median"),
        reorder_nmse=("output_reorder_rtn_nmse", "median"),
        arc_nmse=("output_arc_nmse", "median"),
    )
    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.plot(layer_df.layer, layer_df.identity_nmse, marker="o", ms=3,
            label="Identity RTN", color=colors["Identity RTN"])
    ax.plot(layer_df.layer, layer_df.reorder_nmse, marker="o", ms=3,
            label="Reorder RTN", color=colors["Reorder RTN"])
    ax.plot(layer_df.layer, layer_df.arc_nmse, marker="o", ms=3,
            label="Reorder + ARC", color=colors["Reorder + ARC"])
    ax.set_yscale("log")
    ax.set_xlabel("Transformer layer")
    ax.set_ylabel("Median linear-output NMSE (log scale)")
    ax.set_title("Output error across Qwen2.5-1.5B layers")
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "output_nmse_by_layer.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    module_types = sorted(module_df.module_type.unique())
    palette = plt.get_cmap("tab10")
    for index, module_type in enumerate(module_types):
        subset = module_df[module_df.module_type == module_type]
        ax.scatter(
            subset.activation_identity_contaminated_ordinary_zero_error_share,
            subset.reorder_gain_vs_identity * 100.0,
            s=30,
            alpha=0.7,
            label=module_type,
            color=palette(index % 10),
        )
    ax.axhline(0, color="#111827", lw=1)
    ax.set_xlabel("Identity RTN error from zeroed ordinary values in mixed blocks")
    ax.set_ylabel("Reorder gain in linear-output MSE (%)")
    ax.set_title("Does isolating high-importance channels explain reorder gains?")
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=False, ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "reorder_gain_vs_pollution.png", dpi=180)
    plt.close(fig)

    attribution = flatten_quantization_attribution(module_df)
    attribution_records = []
    for entity in ("activation", "weight"):
        entity_rows = attribution[attribution.entity == entity]
        identity_rows = entity_rows[entity_rows.layout == "identity"]
        identity_total = (identity_rows.mse * identity_rows.elements).sum()
        for layout in ("identity", "reorder"):
            rows = entity_rows[entity_rows.layout == layout]
            error_weight = rows.mse * rows.elements
            attribution_records.append(
                {
                    "entity": entity,
                    "layout": layout,
                    "zero_error": (
                        rows.zero_error_share * error_weight
                    ).sum()
                    / identity_total,
                    "nonzero_grid_error": (
                        rows.nonzero_grid_error_share * error_weight
                    ).sum()
                    / identity_total,
                }
            )
    attribution_summary = pd.DataFrame(attribution_records)
    labels = [
        f"{row.entity.title()}\n{row.layout.title()}"
        for row in attribution_summary.itertuples()
    ]
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    x_positions = np.arange(len(labels))
    zero_share = attribution_summary.zero_error.to_numpy()
    grid_share = attribution_summary.nonzero_grid_error.to_numpy()
    ax.bar(x_positions, zero_share, label="Quantized to zero", color="#DC2626")
    ax.bar(
        x_positions,
        grid_share,
        bottom=zero_share,
        label="Non-zero E2M1 grid error",
        color="#2563EB",
    )
    ax.set_xticks(x_positions, labels)
    ax.set_ylabel("Squared error relative to identity total")
    ax.set_title("Reorder trades zeroing error for non-zero grid error")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "rtn_error_attribution.png", dpi=180)
    plt.close(fig)

    ranked = module_df.copy()
    ranked["arc_gain_percent"] = ranked.arc_gain_vs_identity * 100.0
    ranked["reorder_gain_percent"] = ranked.reorder_gain_vs_identity * 100.0
    ranked = pd.concat(
        [ranked.nsmallest(10, "arc_gain_percent"), ranked.nlargest(10, "arc_gain_percent")]
    ).drop_duplicates("module").sort_values("arc_gain_percent")
    y_positions = np.arange(len(ranked))
    fig, ax = plt.subplots(figsize=(10, 7.2))
    ax.barh(
        y_positions - 0.18,
        ranked.reorder_gain_percent,
        height=0.34,
        color=colors["Reorder RTN"],
        label="Reorder only",
    )
    ax.barh(
        y_positions + 0.18,
        ranked.arc_gain_percent,
        height=0.34,
        color=colors["Reorder + ARC"],
        label="Reorder + ARC",
    )
    ax.axvline(0, color="#111827", lw=1)
    ax.set_yticks(y_positions, ranked.module)
    ax.set_xlabel("Gain versus identity RTN output MSE (%)")
    ax.set_title("Best and worst module-level gains")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "arc_gain_ranked.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.wikitext_cache_dir:
        os.environ["ARCQUANT_WIKITEXT_CACHE_DIR"] = args.wikitext_cache_dir
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA is required for this layer-wise diagnostic")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.time()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        cuda_index = torch.device(args.device).index or 0
        torch.cuda.set_device(cuda_index)
        torch.cuda.reset_peak_memory_stats()

    model_path = Path(args.model)
    model_name = model_path.name.lower()
    saved_dir = Path(args.saved_dir)
    reorder_path = saved_dir / f"{model_name}_reorder_index_wikitext2_{args.metric}.pt"
    select_path = saved_dir / f"{model_name}_select_num_wikitext2_{args.metric}.pt"
    scores_path = saved_dir / f"{model_name}_act_scales_wikitext2_{args.metric}.pt"
    for required in (reorder_path, select_path, scores_path):
        if not required.is_file():
            raise FileNotFoundError(
                f"Calibration artifact not found: {required}. Finish 128x2048 calibration first."
            )

    reorder_index = torch.load(reorder_path, map_location="cpu")
    select_nums = torch.load(select_path, map_location="cpu")
    channel_scores = torch.load(scores_path, map_location="cpu")
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

    rows = []
    layers = model.model.layers
    for layer_index, layer in enumerate(layers):
        print(f"Analyzing layer {layer_index + 1}/{len(layers)}", flush=True)
        layer = layer.to(args.device)
        sampled_inputs: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        hooks = []

        def capture_hook(
            module: nn.Module,
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
        for full_name, (sample, sample_global_scale) in sampled_inputs.items():
            local_name = full_name.split(f"layers.{layer_index}.", 1)[1]
            module = modules[local_name]
            key = full_name + ".input"
            if key not in reorder_index or key not in select_nums:
                print(f"Skipping {full_name}: missing calibration metadata", flush=True)
                continue
            score = channel_scores.get(key)
            row = analyze_linear(
                name=full_name,
                module=module,
                x=sample,
                x_global_scale=sample_global_scale / (448.0 * 6.0),
                permutation=reorder_index[key],
                select_num=int(select_nums[key]),
                channel_scores=score,
            )
            rows.append(row)
            print(
                f"  {local_name}: reorder={row['reorder_gain_vs_identity']:+.2%}, "
                f"ARC={row['arc_gain_vs_identity']:+.2%}",
                flush=True,
            )
            pd.DataFrame(rows).to_csv(output_dir / "module_metrics.csv", index=False)
            del sample, row
            gc.collect()
            torch.cuda.empty_cache()

        layers[layer_index] = layer.cpu()
        inps = next_inps
        del layer, sampled_inputs, modules
        gc.collect()
        torch.cuda.empty_cache()

    module_df = pd.DataFrame(rows)
    attribution_df = flatten_quantization_attribution(module_df)
    attribution_df.to_csv(output_dir / "quantization_attribution.csv", index=False)
    numeric_columns = module_df.select_dtypes(include=[np.number]).columns.tolist()
    aggregate_df = module_df.groupby("module_type", as_index=False)[numeric_columns].mean()
    aggregate_df.insert(1, "module_count", module_df.groupby("module_type").size().values)
    aggregate_df.to_csv(output_dir / "aggregate_by_module_type.csv", index=False)
    make_charts(module_df, output_dir)

    metadata = {
        "model": str(model_path.resolve()),
        "calibration": {
            "dataset": "WikiText2 train",
            "samples": 128,
            "seqlen": 2048,
            "seed": 0,
            "metric": args.metric,
            "reorder_index": str(reorder_path.resolve()),
            "select_num": str(select_path.resolve()),
        },
        "diagnostic": {
            "samples": args.samples,
            "seqlen": args.seqlen,
            "sample_rows_per_module": args.rows_per_module,
            "seed": args.seed,
            "rotation": False,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(torch.device(args.device).index or 0)
            if torch.cuda.is_available()
            else None,
        },
        "module_rows": len(module_df),
        "elapsed_seconds": time.time() - started_at,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote diagnostics to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
