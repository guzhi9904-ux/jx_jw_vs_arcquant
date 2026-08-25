"""Capture decoder Linear operands and quantize them with the audited NVFP4 path."""

from __future__ import annotations

import gc
import hashlib
import os
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn

from fp4_residual_carrier.common.reproducibility import sha256_file
from functional_gram.model_inputs import (
    first_layer_inputs,
    load_model,
    load_wikitext_samples,
    model_weight_files,
    model_weight_fingerprint,
    pick_rows_with_ids,
)
from functional_gram.quantization import compute_global_scale, quantize_matrix_chunked


MODULE_TYPES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def initial_cost_control_modules(num_layers: int) -> tuple[str, ...]:
    """Cover all seven Linear types across early, middle, and late layers."""

    if num_layers < 3:
        raise ValueError("at least three decoder layers are required")
    early, middle, late = 0, num_layers // 2, num_layers - 1
    return (
        f"layers.{early}.self_attn.q_proj",
        f"layers.{early}.self_attn.k_proj",
        f"layers.{middle}.self_attn.v_proj",
        f"layers.{middle}.self_attn.o_proj",
        f"layers.{middle}.mlp.gate_proj",
        f"layers.{late}.mlp.up_proj",
        f"layers.{late}.mlp.down_proj",
    )


def depth_control_modules(num_layers: int) -> tuple[str, ...]:
    """Return all seven Linear types at matched early/middle/late depths."""

    if num_layers < 3:
        raise ValueError("at least three decoder layers are required")
    layers = tuple(dict.fromkeys((0, num_layers // 2, num_layers - 1)))
    local_names = (
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    )
    return tuple(
        f"layers.{layer}.{local_name}"
        for layer in layers
        for local_name in local_names
    )


def all_linears_modules(num_layers: int) -> tuple[str, ...]:
    local_names = (
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    )
    return tuple(
        f"layers.{layer}.{local_name}"
        for layer in range(num_layers)
        for local_name in local_names
    )


def _positive_scale(maximum: torch.Tensor) -> torch.Tensor:
    scale = maximum.float() / (448.0 * 6.0)
    return torch.where(scale > 0, scale, torch.ones_like(scale)).reshape(())


def _artifact_name(module_name: str) -> str:
    return module_name.replace(".", "__") + ".pt"


def _validate_modules(modules: Iterable[str], num_layers: int) -> tuple[str, ...]:
    result = tuple(dict.fromkeys(modules))
    if not result:
        raise ValueError("at least one target module is required")
    for name in result:
        parts = name.split(".")
        if len(parts) < 4 or parts[0] != "layers" or not parts[1].isdigit():
            raise ValueError(f"invalid module name: {name}")
        if not 0 <= int(parts[1]) < num_layers:
            raise ValueError(f"module layer outside model: {name}")
        if parts[-1] not in MODULE_TYPES:
            raise ValueError(f"unsupported module type: {name}")
    return result


@torch.no_grad()
def collect_decoder_operands(
    *,
    model_path: str | Path,
    wikitext_cache_dir: str | Path,
    output_dir: str | Path,
    modules: Iterable[str] | str | None,
    selection_samples: int,
    holdout_samples: int,
    seqlen: int,
    rows_per_split: int,
    seed: int,
    device: str,
    quant_row_chunk: int,
    resume: bool = True,
) -> dict[str, Any]:
    """Capture split-A/B Linear inputs and save original/quantized operands."""

    model_path = Path(model_path).resolve()
    cache_dir = Path(wikitext_cache_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_weight_files(model_path)
    if not (cache_dir / "wikitext-train.arrow").is_file():
        raise FileNotFoundError(cache_dir / "wikitext-train.arrow")
    if selection_samples <= 0 or holdout_samples <= 0 or seqlen < 2 or rows_per_split <= 0:
        raise ValueError("sample counts, seqlen, and rows_per_split must be positive")
    if not torch.cuda.is_available() and str(device).startswith("cuda"):
        raise RuntimeError("CUDA device requested but CUDA is unavailable")

    os.environ["ARCQUANT_WIKITEXT_CACHE_DIR"] = str(cache_dir)
    model_digest, model_weight_inventory = model_weight_fingerprint(model_path)
    dataset_digest = sha256_file(cache_dir / "wikitext-train.arrow")
    model, tokenizer = load_model(str(model_path))
    num_layers = len(model.model.layers)
    if modules is None:
        requested_modules: Iterable[str] = initial_cost_control_modules(num_layers)
    elif modules == "depth-control":
        requested_modules = depth_control_modules(num_layers)
    elif isinstance(modules, str):
        raise ValueError(f"unknown module preset: {modules}")
    else:
        requested_modules = modules
    targets = _validate_modules(requested_modules, num_layers)
    config = model.config
    hidden_size = int(getattr(config, "hidden_size"))
    attention_heads = int(getattr(config, "num_attention_heads"))
    key_value_heads = int(getattr(config, "num_key_value_heads", attention_heads))
    head_dim = int(getattr(config, "head_dim", 0) or hidden_size // attention_heads)
    expected = {name: output_dir / _artifact_name(name) for name in targets}
    completed = {name for name, path in expected.items() if resume and path.is_file()}
    pending = set(targets) - completed
    if not pending:
        del model, tokenizer
        return {
            "status": "complete",
            "modules": list(targets),
            "artifacts": {name: str(path) for name, path in expected.items()},
            "resumed": True,
        }

    batches, split_manifest = load_wikitext_samples(
        tokenizer,
        selection_samples=selection_samples,
        holdout_samples=holdout_samples,
        seed=seed,
        seqlen=seqlen,
    )
    inps, attention_mask, position_ids = first_layer_inputs(model, batches, device)
    del tokenizer, batches
    layers = model.model.layers
    total_samples = selection_samples + holdout_samples
    target_layers = {name: int(name.split(".")[1]) for name in pending}
    last_layer = max(target_layers.values())

    for layer_index in range(last_layer + 1):
        layer = layers[layer_index].to(device)
        names_here = {name for name in pending if target_layers[name] == layer_index}
        captured: dict[str, tuple[torch.Tensor, ...]] = {}
        handles: list[Any] = []

        def capture(name: str, value: torch.Tensor) -> None:
            if value.shape[0] != total_samples:
                raise ValueError(f"unexpected sample dimension for {name}: {value.shape}")
            split_a = value[:selection_samples]
            split_b = value[selection_samples:]
            xa, ids_a = pick_rows_with_ids(split_a, rows_per_split)
            xb, ids_b = pick_rows_with_ids(split_b, rows_per_split)
            captured[name] = (
                xa,
                xb,
                split_a.abs().amax().float(),
                split_b.abs().amax().float(),
                ids_a,
                ids_b,
            )

        for local_name, module in layer.named_modules():
            full_name = f"layers.{layer_index}.{local_name}"
            if isinstance(module, nn.Linear) and full_name in names_here:
                handles.append(
                    module.register_forward_pre_hook(
                        lambda _module, inputs, name=full_name: capture(name, inputs[0].detach())
                    )
                )

        layer_input = inps.to(device)
        mask = attention_mask.to(device) if attention_mask is not None else None
        positions = position_ids.to(device) if position_ids is not None else None
        next_inps = layer(layer_input, attention_mask=mask, position_ids=positions)[0].detach()
        for handle in handles:
            handle.remove()
        local_modules = dict(layer.named_modules())
        if set(captured) != names_here:
            raise RuntimeError(f"missing captures at layer {layer_index}: {sorted(names_here - set(captured))}")

        for full_name, values in captured.items():
            xa, xb, absmax_a, absmax_b, ids_a, ids_b = values
            local_name = full_name.split(f"layers.{layer_index}.", 1)[1]
            module = local_modules[local_name]
            weight = module.weight.detach().float()
            weight_scale = compute_global_scale(weight)
            scale_a, scale_b = _positive_scale(absmax_a), _positive_scale(absmax_b)
            qweight, weight_diag = quantize_matrix_chunked(
                weight, global_scale=weight_scale, row_chunk=quant_row_chunk
            )
            qxa, activation_diag_a = quantize_matrix_chunked(
                xa, global_scale=scale_a, row_chunk=quant_row_chunk
            )
            qxb, activation_diag_b = quantize_matrix_chunked(
                xb, global_scale=scale_b, row_chunk=quant_row_chunk
            )
            payload = {
                "schema_version": 1,
                "module": full_name,
                "module_type": full_name.split(".")[-1],
                "layer": layer_index,
                "K": int(module.in_features),
                "out_features": int(module.out_features),
                "attention_heads": (
                    attention_heads if full_name.endswith("q_proj") else
                    key_value_heads if full_name.endswith(("k_proj", "v_proj")) else None
                ),
                "head_dim": (
                    head_dim if full_name.endswith(("q_proj", "k_proj", "v_proj")) else None
                ),
                "split_a": {"x": xa.cpu(), "qx": qxa.cpu(), "row_ids": ids_a},
                "split_b": {"x": xb.cpu(), "qx": qxb.cpu(), "row_ids": ids_b},
                "weight": weight.cpu(),
                "qweight": qweight.cpu(),
                "quantizer": {
                    "implementation": "fp4_residual_carrier.common.nvfp4_reference.quantize_nvfp4",
                    "format": "NVFP4 E2M1 + UE4M3, group_size=16",
                    "weight": weight_diag,
                    "split_a_activation": activation_diag_a,
                    "split_b_activation": activation_diag_b,
                },
                "provenance": {
                    "model": str(model_path),
                    "model_sha256": model_digest,
                    "model_weight_inventory": model_weight_inventory,
                    "architecture": type(model).__name__,
                    "dataset": str(cache_dir / "wikitext-train.arrow"),
                    "dataset_sha256": dataset_digest,
                    "seed": seed,
                    "selection_samples": selection_samples,
                    "holdout_samples": holdout_samples,
                    "seqlen": seqlen,
                    "rows_per_split": rows_per_split,
                    "split_manifest": split_manifest,
                },
            }
            if payload["attention_heads"] is not None and (
                int(payload["attention_heads"]) * int(payload["head_dim"])
                != int(module.out_features)
            ):
                raise ValueError(
                    f"attention head metadata does not match {full_name}: "
                    f"{payload['attention_heads']} x {payload['head_dim']} != {module.out_features}"
                )
            torch.save(payload, expected[full_name])
            print(f"captured {full_name}: X={tuple(xa.shape)}, W={tuple(weight.shape)}", flush=True)
            del xa, xb, qxa, qxb, weight, qweight, payload
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        layers[layer_index] = layer.cpu()
        inps = next_inps.cpu()
        del layer, layer_input, next_inps, captured, local_modules
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    del model, inps, attention_mask, position_ids
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    missing = [name for name, path in expected.items() if not path.is_file()]
    if missing:
        raise RuntimeError(f"operand collection incomplete: {missing}")
    return {
        "status": "complete",
        "modules": list(targets),
        "artifacts": {name: str(path) for name, path in expected.items()},
        "resumed": bool(completed),
        "split_manifest": split_manifest,
        "input_fingerprint": hashlib.sha256(
            (model_digest + dataset_digest).encode()
        ).hexdigest(),
    }


# Backward-compatible spelling used by the original Qwen Stage-1 runner.
collect_qwen_operands = collect_decoder_operands
