"""Train/holdout validation for a deployable activation+weight K+S proxy.

The selector never reads holdout output residuals.  It uses a fixed half/half
budget and diagonal second-moment proxies:

    activation: ||X-QX||_2^2 * ||W_j||_2^2
    weight:     ||QX_j||_2^2 * ||W-QW||_2^2

An output-residual oracle selected on train rows and an intentionally leaked
oracle selected on holdout rows are retained only as diagnostic ceilings.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utilize import get_wikitext2, load_model  # noqa: E402
from scripts.analyze_rtn_error_ks import (  # noqa: E402
    single_rank1_gain_score,
    top_indices,
)
from scripts.analyze_rtn_reorder_arc import (  # noqa: E402
    first_layer_inputs,
    nvfp4_quantize,
    pick_rows,
)


BASE_STRATEGIES = (
    "paper_activation_identity",
    "paper_reorder_arc",
    "activation_energy_train",
    "weight_energy_train",
    "proxy_fixed_half_train",
    "proxy_energy_adaptive_train",
    "oracle_joint_train",
    "oracle_joint_holdout_leaked",
)

DUAL_INDEX_ABLATION_STRATEGIES = (
    "activation_energy_train",
    "weight_energy_train",
    "proxy_fixed_half_train",
    "proxy_shared_train",
    "random_fixed_half",
    "oracle_independent_fixed_half_train",
    "oracle_shared_train",
    "oracle_independent_fixed_half_holdout_leaked",
    "oracle_shared_holdout_leaked",
)

# One server run should compare the original paper path against the frozen V2
# ablations on exactly the same selection/holdout rows.  Keeping this as a
# distinct experiment label avoids changing either historical result table.
SERVER_COMPARISON_STRATEGIES = (
    "paper_reorder_arc",
    *DUAL_INDEX_ABLATION_STRATEGIES,
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
        "--output-dir",
        default=str(REPO_ROOT / "analysis" / "rtn_proxy_ks_holdout"),
    )
    parser.add_argument("--selection-samples", type=int, default=4)
    parser.add_argument("--holdout-samples", type=int, default=4)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--rows-per-split", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--metric", default="max")
    parser.add_argument(
        "--experiment",
        choices=("baseline", "dual-index-ablation", "server-comparison"),
        default="baseline",
        help=(
            "baseline reproduces the original proxy experiment; "
            "dual-index-ablation freezes V2 and compares independent, shared, "
            "single-branch, random, and output-aware diagnostic selectors; "
            "server-comparison adds the original paper reorder+ARC baseline "
            "on the same rows."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Reuse complete per-module rows already present in module_metrics.csv. "
            "Completed layers are only forwarded to reconstruct later activations."
        ),
    )
    parser.add_argument(
        "--wikitext-cache-dir",
        default=os.environ.get("ARCQUANT_WIKITEXT_CACHE_DIR"),
    )
    return parser.parse_args()


def safe_ratio(numerator: float, denominator: float) -> float:
    if not math.isfinite(denominator) or abs(denominator) < 1e-30:
        return float("nan")
    return numerator / denominator


def encoded_atoms(
    activation_indices: torch.Tensor,
    weight_indices: torch.Tensor,
    width: int,
) -> torch.Tensor:
    return torch.cat([activation_indices, weight_indices + width], dim=0)


def atom_overlap(left: torch.Tensor, right: torch.Tensor) -> int:
    if left.numel() == 0 or right.numel() == 0:
        return 0
    return int(torch.isin(left, right).sum().item())


def stable_random_indices(
    width: int,
    count: int,
    *,
    seed: int,
    key: str,
    device: torch.device,
) -> torch.Tensor:
    """Return resume-stable random indices without consuming global RNG state."""
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
    local_seed = int.from_bytes(digest[:8], byteorder="little", signed=False)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(local_seed)
    return torch.randperm(width, generator=generator)[:count].to(device=device)


def split_from_combined_scores(
    activation_score: torch.Tensor,
    weight_score: torch.Tensor,
    select_num: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    width = activation_score.numel()
    combined = torch.cat([activation_score, weight_score], dim=0)
    preliminary = top_indices(combined, select_num)
    activation_count = int((preliminary < width).sum().item())
    activation_count = int(math.floor((activation_count + 8) / 16.0) * 16)
    activation_count = max(0, min(select_num, activation_count))
    weight_count = select_num - activation_count
    return (
        top_indices(activation_score, activation_count),
        top_indices(weight_score, weight_count),
    )


@torch.no_grad()
def build_state(
    x: torch.Tensor,
    *,
    x_global_scale: torch.Tensor,
    weight: torch.Tensor,
    qw: torch.Tensor,
    bias: torch.Tensor | None,
) -> Dict[str, torch.Tensor | float]:
    qx, _ = nvfp4_quantize(x, global_scale=x_global_scale)
    ex = x - qx
    ew = weight - qw
    y_reference = F.linear(x, weight, bias).float()
    y_rtn = F.linear(qx, qw, bias).float()
    residual = y_reference - y_rtn
    activation_gain, activation_energy = single_rank1_gain_score(
        residual, ex, weight
    )
    weight_gain, weight_energy = single_rank1_gain_score(residual, qx, ew)
    # For a shared channel j, the correction is the sum of two rank-1 atoms:
    #   EX_j W_j^T + QX_j EW_j^T.
    # Its cross term still factorizes by channel, so this is an exact
    # one-channel ideal energy/gain score and does not read the holdout split
    # when computed on calibration rows.
    paired_cross = (ex.float() * qx.float()).sum(dim=0) * (
        weight.float() * ew.float()
    ).sum(dim=0)
    shared_energy = (
        activation_energy + weight_energy + 2.0 * paired_cross
    ).clamp_min(0.0)
    shared_gain = activation_gain + weight_gain - 2.0 * paired_cross
    return {
        "x": x,
        "qx": qx,
        "ex": ex,
        "residual": residual,
        "baseline_sse": residual.square().sum().item(),
        "reference_sse": y_reference.square().sum().item(),
        "activation_gain": activation_gain,
        "activation_energy": activation_energy,
        "weight_gain": weight_gain,
        "weight_energy": weight_energy,
        "shared_gain": shared_gain,
        "shared_energy": shared_energy,
    }


@torch.no_grad()
def evaluate_identity_correction(
    state: Dict[str, torch.Tensor | float],
    *,
    weight: torch.Tensor,
    ew: torch.Tensor,
    activation_indices: torch.Tensor,
    weight_indices: torch.Tensor,
    x_global_scale: torch.Tensor,
    weight_global_scale: torch.Tensor,
) -> Dict[str, float]:
    ex = state["ex"]
    qx = state["qx"]
    residual = state["residual"]
    assert isinstance(ex, torch.Tensor)
    assert isinstance(qx, torch.Tensor)
    assert isinstance(residual, torch.Tensor)
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
    ideal_correction = F.linear(left, right, bias=None).float()
    ideal_sse = (residual - ideal_correction).square().sum().item()
    qleft, _ = nvfp4_quantize(left, global_scale=x_global_scale)
    qright, _ = nvfp4_quantize(right, global_scale=weight_global_scale)
    correction = F.linear(qleft, qright, bias=None).float()
    corrected_sse = (residual - correction).square().sum().item()
    return {
        "corrected_sse": corrected_sse,
        "ideal_corrected_sse": ideal_sse,
    }


@torch.no_grad()
def evaluate_paper_reorder(
    x: torch.Tensor,
    *,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    permutation: torch.Tensor,
    select_num: int,
    x_global_scale: torch.Tensor,
    weight_global_scale: torch.Tensor,
) -> Dict[str, float]:
    xr = x.index_select(1, permutation)
    wr = weight.index_select(1, permutation)
    qxr, _ = nvfp4_quantize(xr, global_scale=x_global_scale)
    qwr, _ = nvfp4_quantize(wr, global_scale=weight_global_scale)
    y_reference = F.linear(x, weight, bias).float()
    y_main = F.linear(qxr, qwr, bias).float()
    ex_selected = (xr - qxr)[:, -select_num:]
    weight_selected = wr[:, -select_num:]
    ideal = F.linear(ex_selected, weight_selected, bias=None).float()
    ideal_sse = (y_reference - (y_main + ideal)).square().sum().item()
    qleft, _ = nvfp4_quantize(ex_selected, global_scale=x_global_scale)
    qright, _ = nvfp4_quantize(
        weight_selected, global_scale=weight_global_scale
    )
    correction = F.linear(qleft, qright, bias=None).float()
    corrected_sse = (y_reference - (y_main + correction)).square().sum().item()
    return {
        "corrected_sse": corrected_sse,
        "ideal_corrected_sse": ideal_sse,
    }


def strategy_indices(
    *,
    width: int,
    select_num: int,
    paper_permutation: torch.Tensor,
    train_state: Dict[str, torch.Tensor | float],
    holdout_state: Dict[str, torch.Tensor | float],
    random_seed: int,
    random_key: str,
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    empty = torch.empty(0, device=paper_permutation.device, dtype=torch.long)
    paper = paper_permutation[-select_num:]
    half = select_num // 2
    train_activation_energy = train_state["activation_energy"]
    train_weight_energy = train_state["weight_energy"]
    train_activation_gain = train_state["activation_gain"]
    train_weight_gain = train_state["weight_gain"]
    train_shared_energy = train_state["shared_energy"]
    train_shared_gain = train_state["shared_gain"]
    holdout_activation_gain = holdout_state["activation_gain"]
    holdout_weight_gain = holdout_state["weight_gain"]
    holdout_shared_gain = holdout_state["shared_gain"]
    for value in (
        train_activation_energy,
        train_weight_energy,
        train_activation_gain,
        train_weight_gain,
        train_shared_energy,
        train_shared_gain,
        holdout_activation_gain,
        holdout_weight_gain,
        holdout_shared_gain,
    ):
        assert isinstance(value, torch.Tensor)
    energy_adaptive = split_from_combined_scores(
        train_activation_energy, train_weight_energy, select_num
    )
    oracle_train = split_from_combined_scores(
        train_activation_gain, train_weight_gain, select_num
    )
    oracle_holdout = split_from_combined_scores(
        holdout_activation_gain, holdout_weight_gain, select_num
    )
    shared_energy_indices = top_indices(train_shared_energy, half)
    shared_oracle_train_indices = top_indices(train_shared_gain, half)
    shared_oracle_holdout_indices = top_indices(holdout_shared_gain, half)
    random_activation = stable_random_indices(
        width,
        half,
        seed=random_seed,
        key=f"{random_key}:activation",
        device=paper_permutation.device,
    )
    random_weight = stable_random_indices(
        width,
        select_num - half,
        seed=random_seed,
        key=f"{random_key}:weight",
        device=paper_permutation.device,
    )
    return {
        "paper_activation_identity": (paper, empty),
        "paper_reorder_arc": (paper, empty),
        "activation_energy_train": (
            top_indices(train_activation_energy, select_num),
            empty,
        ),
        "weight_energy_train": (
            empty,
            top_indices(train_weight_energy, select_num),
        ),
        "proxy_fixed_half_train": (
            top_indices(train_activation_energy, half),
            top_indices(train_weight_energy, select_num - half),
        ),
        "proxy_shared_train": (
            shared_energy_indices,
            shared_energy_indices,
        ),
        "random_fixed_half": (random_activation, random_weight),
        "proxy_energy_adaptive_train": energy_adaptive,
        "oracle_joint_train": oracle_train,
        "oracle_joint_holdout_leaked": oracle_holdout,
        "oracle_independent_fixed_half_train": (
            top_indices(train_activation_gain, half),
            top_indices(train_weight_gain, select_num - half),
        ),
        "oracle_shared_train": (
            shared_oracle_train_indices,
            shared_oracle_train_indices,
        ),
        "oracle_independent_fixed_half_holdout_leaked": (
            top_indices(holdout_activation_gain, half),
            top_indices(holdout_weight_gain, select_num - half),
        ),
        "oracle_shared_holdout_leaked": (
            shared_oracle_holdout_indices,
            shared_oracle_holdout_indices,
        ),
    }


def aggregate_strategies(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for strategy, group in frame.groupby("strategy"):
        train_baseline = group.train_baseline_sse.sum()
        holdout_baseline = group.holdout_baseline_sse.sum()
        rows.append(
            {
                "strategy": strategy,
                "train_gain": 1.0 - group.train_corrected_sse.sum() / train_baseline,
                "holdout_gain": 1.0
                - group.holdout_corrected_sse.sum() / holdout_baseline,
                "holdout_ideal_gain": 1.0
                - group.holdout_ideal_corrected_sse.sum() / holdout_baseline,
                "generalization_gap": (
                    1.0 - group.train_corrected_sse.sum() / train_baseline
                )
                - (
                    1.0 - group.holdout_corrected_sse.sum() / holdout_baseline
                ),
                "holdout_positive_rate": float(
                    (group.holdout_gain_vs_identity > 0).mean()
                ),
                "mean_activation_fraction": float(
                    (group.activation_branch_num / group.select_num).mean()
                ),
                "mean_holdout_oracle_overlap": float(
                    group.holdout_oracle_overlap_rate.mean()
                ),
                "modules": int(len(group)),
            }
        )
    return pd.DataFrame(rows).sort_values("holdout_gain", ascending=False)


def aggregate_module_types(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (strategy, module_type), group in frame.groupby(
        ["strategy", "module_type"]
    ):
        rows.append(
            {
                "strategy": strategy,
                "module_type": module_type,
                "holdout_gain": 1.0
                - group.holdout_corrected_sse.sum()
                / group.holdout_baseline_sse.sum(),
                "holdout_positive_rate": float(
                    (group.holdout_gain_vs_identity > 0).mean()
                ),
                "mean_activation_fraction": float(
                    (group.activation_branch_num / group.select_num).mean()
                ),
                "modules": int(len(group)),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["strategy", "holdout_gain"], ascending=[True, False]
    )


def main() -> None:
    args = parse_args()
    strategies = {
        "baseline": BASE_STRATEGIES,
        "dual-index-ablation": DUAL_INDEX_ABLATION_STRATEGIES,
        "server-comparison": SERVER_COMPARISON_STRATEGIES,
    }[args.experiment]
    if args.wikitext_cache_dir:
        os.environ["ARCQUANT_WIKITEXT_CACHE_DIR"] = args.wikitext_cache_dir
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA is required for the layer-wise holdout experiment")
    if args.selection_samples <= 0 or args.holdout_samples <= 0:
        raise ValueError("Both train and holdout sample counts must be positive")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.time()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    torch.cuda.set_device(device.index or 0)
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

    model, tokenizer = load_model(str(model_path))
    total_samples = args.selection_samples + args.holdout_samples
    dataloader, _ = get_wikitext2(
        nsamples=total_samples,
        seed=args.seed,
        seqlen=args.seqlen,
        tokenizer=tokenizer,
    )
    inps, attention_mask, position_ids = first_layer_inputs(
        model, dataloader, args.device
    )
    del dataloader, tokenizer

    metrics_path = output_dir / "module_metrics.csv"
    selection_path = output_dir / "selection_indices.pt"
    strategy_rows: list[Dict[str, Any]] = []
    completed_modules: set[str] = set()
    selection_indices: Dict[str, Dict[str, Any]] = {}
    if args.resume and selection_path.is_file():
        selection_indices = torch.load(selection_path, map_location="cpu")
    if args.resume and metrics_path.is_file():
        existing = pd.read_csv(metrics_path)
        counts = existing.groupby("module").strategy.nunique()
        completed_modules = set(counts[counts == len(strategies)].index)
        existing = existing[existing.module.isin(completed_modules)].copy()
        strategy_rows = existing.to_dict(orient="records")
        print(
            f"Resuming from {len(completed_modules)} complete modules "
            f"({len(strategy_rows)} strategy rows)",
            flush=True,
        )
    layers = model.model.layers
    for layer_index, layer in enumerate(layers):
        expected_modules = {
            f"layers.{layer_index}.{local_name}"
            for local_name, module in layer.named_modules()
            if isinstance(module, nn.Linear)
        }
        layer_complete = expected_modules.issubset(
            completed_modules
        ) and expected_modules.issubset(selection_indices)
        action = "Forwarding" if layer_complete else "Analyzing"
        print(f"{action} layer {layer_index + 1}/{len(layers)}", flush=True)
        layer = layer.to(args.device)
        sampled_inputs: Dict[
            str,
            Tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
            ],
        ] = {}
        hooks = []

        def capture_hook(
            _module: nn.Module,
            inputs: Tuple[torch.Tensor, ...],
            _output: torch.Tensor,
            *,
            full_name: str,
        ) -> None:
            value = inputs[0].detach()
            if value.shape[0] != total_samples:
                raise ValueError(
                    f"Expected batch dimension {total_samples}, got {value.shape[0]}"
                )
            train_value = value[: args.selection_samples]
            holdout_value = value[args.selection_samples : total_samples]
            sampled_inputs[full_name] = (
                pick_rows(train_value, args.rows_per_split),
                train_value.abs().amax().float(),
                pick_rows(holdout_value, args.rows_per_split),
                holdout_value.abs().amax().float(),
            )

        for local_name, module in layer.named_modules():
            if isinstance(module, nn.Linear) and not layer_complete:
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
        with torch.no_grad():
            next_inps = layer(
                inps, attention_mask=attention_mask, position_ids=position_ids
            )[0].detach()
        for hook in hooks:
            hook.remove()

        modules = dict(layer.named_modules())
        for full_name, values in sampled_inputs.items():
            needs_metrics = full_name not in completed_modules
            needs_indices = full_name not in selection_indices
            if not needs_metrics and not needs_indices:
                continue
            train_x, train_absmax, holdout_x, holdout_absmax = values
            local_name = full_name.split(f"layers.{layer_index}.", 1)[1]
            key = full_name + ".input"
            module = modules[local_name]
            weight = module.weight.detach().float()
            bias = module.bias.detach().float() if module.bias is not None else None
            width = int(module.in_features)
            select_num = int(max(0, min(int(select_nums[key]), width)))
            if select_num % 64:
                raise ValueError(
                    f"select_num must be 64-aligned, got {select_num} for {full_name}"
                )
            if (select_num // 2) % 32:
                raise ValueError(
                    f"half budget must be 32-aligned, got S={select_num}"
                )
            permutation = reorder_index[key].to(device=device, dtype=torch.long)
            train_scale = train_absmax / (448.0 * 6.0)
            holdout_scale = holdout_absmax / (448.0 * 6.0)
            weight_scale = weight.abs().max() / (448.0 * 6.0)
            qw, _ = nvfp4_quantize(weight, global_scale=weight_scale)
            ew = weight - qw
            train_state = build_state(
                train_x,
                x_global_scale=train_scale,
                weight=weight,
                qw=qw,
                bias=bias,
            )
            holdout_state = build_state(
                holdout_x,
                x_global_scale=holdout_scale,
                weight=weight,
                qw=qw,
                bias=bias,
            )
            indices = strategy_indices(
                width=width,
                select_num=select_num,
                paper_permutation=permutation,
                train_state=train_state,
                holdout_state=holdout_state,
                random_seed=args.seed,
                random_key=full_name,
            )
            stored_strategies = (
                ("proxy_fixed_half_train", "proxy_energy_adaptive_train")
                if args.experiment == "baseline"
                else tuple(
                    name
                    for name in strategies
                    if not name.endswith("_holdout_leaked")
                )
            )
            selection_indices[full_name] = {"select_num": select_num}
            for stored_strategy in stored_strategies:
                selected_activation, selected_weight = indices[stored_strategy]
                selection_indices[full_name][stored_strategy] = {
                    "activation": selected_activation.detach().cpu(),
                    "weight": selected_weight.detach().cpu(),
                }
            torch.save(selection_indices, selection_path)
            if not needs_metrics:
                del train_x, holdout_x, weight, bias, qw, ew
                del train_state, holdout_state, indices
                gc.collect()
                torch.cuda.empty_cache()
                continue
            holdout_oracle_name = (
                "oracle_joint_holdout_leaked"
                if args.experiment == "baseline"
                else "oracle_independent_fixed_half_holdout_leaked"
            )
            holdout_oracle_atoms = encoded_atoms(
                *indices[holdout_oracle_name], width
            )

            best_strategy = ""
            best_holdout_gain = -float("inf")
            for strategy in strategies:
                activation_indices, weight_indices = indices[strategy]
                if activation_indices.numel() + weight_indices.numel() != select_num:
                    raise AssertionError(f"Budget mismatch for {strategy}")
                if strategy == "paper_reorder_arc":
                    train_result = evaluate_paper_reorder(
                        train_x,
                        weight=weight,
                        bias=bias,
                        permutation=permutation,
                        select_num=select_num,
                        x_global_scale=train_scale,
                        weight_global_scale=weight_scale,
                    )
                    holdout_result = evaluate_paper_reorder(
                        holdout_x,
                        weight=weight,
                        bias=bias,
                        permutation=permutation,
                        select_num=select_num,
                        x_global_scale=holdout_scale,
                        weight_global_scale=weight_scale,
                    )
                else:
                    train_result = evaluate_identity_correction(
                        train_state,
                        weight=weight,
                        ew=ew,
                        activation_indices=activation_indices,
                        weight_indices=weight_indices,
                        x_global_scale=train_scale,
                        weight_global_scale=weight_scale,
                    )
                    holdout_result = evaluate_identity_correction(
                        holdout_state,
                        weight=weight,
                        ew=ew,
                        activation_indices=activation_indices,
                        weight_indices=weight_indices,
                        x_global_scale=holdout_scale,
                        weight_global_scale=weight_scale,
                    )
                train_baseline = float(train_state["baseline_sse"])
                holdout_baseline = float(holdout_state["baseline_sse"])
                train_gain = 1.0 - safe_ratio(
                    train_result["corrected_sse"], train_baseline
                )
                holdout_gain = 1.0 - safe_ratio(
                    holdout_result["corrected_sse"], holdout_baseline
                )
                atoms = encoded_atoms(
                    activation_indices, weight_indices, width
                )
                overlap = atom_overlap(atoms, holdout_oracle_atoms)
                strategy_rows.append(
                    {
                        "module": full_name,
                        "layer": layer_index,
                        "module_type": full_name.split(".")[-1],
                        "strategy": strategy,
                        "selection_split": (
                            "holdout_leaked"
                            if strategy.endswith("_holdout_leaked")
                            else "train_or_paper"
                        ),
                        "in_features": width,
                        "out_features": int(module.out_features),
                        "selection_rows": int(train_x.shape[0]),
                        "holdout_rows": int(holdout_x.shape[0]),
                        "select_num": select_num,
                        "select_ratio": select_num / width,
                        "activation_branch_num": int(
                            activation_indices.numel()
                        ),
                        "weight_branch_num": int(weight_indices.numel()),
                        "train_baseline_sse": train_baseline,
                        "train_corrected_sse": train_result["corrected_sse"],
                        "train_ideal_corrected_sse": train_result[
                            "ideal_corrected_sse"
                        ],
                        "train_gain_vs_identity": train_gain,
                        "holdout_baseline_sse": holdout_baseline,
                        "holdout_corrected_sse": holdout_result[
                            "corrected_sse"
                        ],
                        "holdout_ideal_corrected_sse": holdout_result[
                            "ideal_corrected_sse"
                        ],
                        "holdout_gain_vs_identity": holdout_gain,
                        "holdout_oracle_overlap": overlap,
                        "holdout_oracle_overlap_rate": overlap / select_num,
                    }
                )
                if (
                    not strategy.endswith("_holdout_leaked")
                    and holdout_gain > best_holdout_gain
                ):
                    best_holdout_gain = holdout_gain
                    best_strategy = strategy

            pd.DataFrame(strategy_rows).to_csv(metrics_path, index=False)
            print(
                f"  {local_name}: best_holdout={best_strategy} "
                f"gain={best_holdout_gain:+.2%}",
                flush=True,
            )
            del train_x, holdout_x, weight, bias, qw, ew
            del train_state, holdout_state, indices, holdout_oracle_atoms
            gc.collect()
            torch.cuda.empty_cache()

        layers[layer_index] = layer.cpu()
        inps = next_inps
        del layer, sampled_inputs, modules
        gc.collect()
        torch.cuda.empty_cache()

    module_frame = pd.DataFrame(strategy_rows)
    strategy_summary = aggregate_strategies(module_frame)
    module_type_summary = aggregate_module_types(module_frame)
    strategy_summary.to_csv(output_dir / "strategy_summary.csv", index=False)
    module_type_summary.to_csv(
        output_dir / "module_type_summary.csv", index=False
    )
    metadata = {
        "model": str(model_path.resolve()),
        "dataset": "WikiText2 train",
        "selection": {
            "samples": args.selection_samples,
            "seqlen": args.seqlen,
            "rows_per_module": args.rows_per_split,
            "sample_positions": [0, args.selection_samples - 1],
        },
        "holdout": {
            "samples": args.holdout_samples,
            "seqlen": args.seqlen,
            "rows_per_module": args.rows_per_split,
            "sample_positions": [args.selection_samples, total_samples - 1],
        },
        "seed": args.seed,
        "experiment": args.experiment,
        "main_layout": "identity except paper_reorder_arc comparator",
        "rotation": False,
        "proxy": {
            "activation_score": "sum((X-QX)^2) * sum(W_col^2)",
            "weight_score": "sum(QX_col^2) * sum((W-QW)_col^2)",
            "fixed_budget": "S/2 activation atoms + S/2 weight atoms",
            "reads_holdout_residual": False,
            "shared_score": (
                "energy(EX_j W_j^T + QX_j EW_j^T); both atoms consume S total"
            ),
            "diagnostic_output_aware": (
                "one-shot per-channel gain on selection rows; leaked holdout "
                "variants are labeled and excluded from deployable comparisons"
            ),
        },
        "strategies": list(strategies),
        "module_rows": int(module_frame.module.nunique()),
        "strategy_rows": int(len(module_frame)),
        "selection_index_file": str(selection_path.resolve()),
        "elapsed_seconds": time.time() - started_at,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device.index or 0),
            "peak_gpu_memory_gib": torch.cuda.max_memory_allocated()
            / (1024.0**3),
        },
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(strategy_summary.to_string(index=False), flush=True)
    print(f"Wrote results to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
