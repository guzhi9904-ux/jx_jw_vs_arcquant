"""Layer-wise WikiText2 perplexity for RTN, ARC, and joint K+S proxies.

All quantized methods share the same fast fake-NVFP4 numerical backend.  The
main model stays in the original channel order except for the paper ARC
comparator, which uses its calibrated permutation.  Joint methods add both
activation-residual and weight-residual rank-1 columns to the same K+S GEMM.
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
from typing import Any, Dict

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_rtn_reorder_arc import nvfp4_quantize  # noqa: E402
from utilize import _load_wikitext2_split  # noqa: E402


METHODS = {
    "bf16",
    "rtn_identity",
    "paper_reorder_arc",
    "proxy_fixed_half",
    "proxy_energy_adaptive",
    "proxy_shared",
    "activation_only",
    "weight_only",
    "random_fixed_half",
    "oracle_fixed_half",
    "oracle_shared",
    "hw_block_fixed_half",
    "hw_block_shared_fixed_half",
    "hw_block_shared_same_set",
}

JOINT_METHOD_TO_STRATEGY = {
    "proxy_fixed_half": "proxy_fixed_half_train",
    "proxy_energy_adaptive": "proxy_energy_adaptive_train",
    "proxy_shared": "proxy_shared_train",
    "activation_only": "activation_energy_train",
    "weight_only": "weight_energy_train",
    "random_fixed_half": "random_fixed_half",
    "oracle_fixed_half": "oracle_independent_fixed_half_train",
    "oracle_shared": "oracle_shared_train",
    "hw_block_fixed_half": "block_fixed_half",
    "hw_block_shared_fixed_half": "block_shared_fixed_half",
    "hw_block_shared_same_set": "block_shared_same_set",
}


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
        "--selection-indices",
        default=str(
            REPO_ROOT
            / "analysis"
            / "rtn_proxy_ks_holdout"
            / "selection_indices.pt"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "analysis" / "rtn_joint_ks_ppl"),
    )
    parser.add_argument("--method", required=True, choices=sorted(METHODS))
    parser.add_argument("--metric", default="max")
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument(
        "--max-windows",
        type=int,
        default=0,
        help="0 evaluates every complete WikiText2 test window.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--checkpoint-every", type=int, default=7)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--wikitext-cache-dir",
        default=os.environ.get("ARCQUANT_WIKITEXT_CACHE_DIR"),
    )
    return parser.parse_args()


class FakeNVFP4KPlusSLinear(nn.Module):
    """Numerical W4A4 linear layer with an optional quantized K+S correction."""

    def __init__(
        self,
        original: nn.Linear,
        *,
        method: str,
        permutation: torch.Tensor | None = None,
        activation_indices: torch.Tensor | None = None,
        weight_indices: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if method not in METHODS - {"bf16"}:
            raise ValueError(method)
        self.method = method
        self.in_features = int(original.in_features)
        self.out_features = int(original.out_features)
        device = original.weight.device
        dtype = original.weight.dtype
        weight = original.weight.detach().float()
        weight_scale = weight.abs().amax() / (448.0 * 6.0)

        if method == "paper_reorder_arc":
            if permutation is None:
                raise ValueError("paper ARC requires a permutation")
            permutation = permutation.to(device=device, dtype=torch.long)
            weight_main = weight.index_select(1, permutation)
            activation_indices = activation_indices.to(
                device=device, dtype=torch.long
            )
            if activation_indices.numel() == 0:
                raise ValueError("paper ARC requires selected channels")
            correction_right = weight_main.index_select(1, activation_indices)
            self.register_buffer("permutation", permutation)
        else:
            weight_main = weight
            self.permutation = None

        qweight_float, _ = nvfp4_quantize(
            weight_main, global_scale=weight_scale
        )
        self.register_buffer("qweight", qweight_float.to(dtype))
        self.register_buffer("weight_scale", weight_scale.float())

        if method in JOINT_METHOD_TO_STRATEGY:
            if activation_indices is None or weight_indices is None:
                raise ValueError("joint K+S requires both index tensors")
            activation_indices = activation_indices.to(
                device=device, dtype=torch.long
            )
            weight_indices = weight_indices.to(device=device, dtype=torch.long)
            if (activation_indices.numel() + weight_indices.numel()) % 16:
                raise ValueError("K+S correction width must be 16-aligned")
            if activation_indices.numel() % 16 or weight_indices.numel() % 16:
                raise ValueError("Each K+S branch must be 16-aligned")
            weight_error = weight - qweight_float
            correction_right = torch.cat(
                [
                    weight.index_select(1, activation_indices),
                    weight_error.index_select(1, weight_indices),
                ],
                dim=1,
            )
        elif method == "rtn_identity":
            activation_indices = torch.empty(
                0, device=device, dtype=torch.long
            )
            weight_indices = torch.empty(0, device=device, dtype=torch.long)
            correction_right = None
        else:
            weight_indices = torch.empty(0, device=device, dtype=torch.long)

        self.register_buffer("activation_indices", activation_indices)
        self.register_buffer("weight_indices", weight_indices)
        if correction_right is not None:
            qright, _ = nvfp4_quantize(
                correction_right, global_scale=weight_scale
            )
            self.register_buffer("qright", qright.to(dtype))
        else:
            self.qright = None

        if original.bias is None:
            self.bias = None
        else:
            self.register_buffer("bias", original.bias.detach().clone())

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        x_float = x.reshape(-1, self.in_features).float()
        if self.method == "paper_reorder_arc":
            x_main = x_float.index_select(1, self.permutation)
        else:
            x_main = x_float
        x_scale = x_main.abs().amax() / (448.0 * 6.0)
        qx_float, _ = nvfp4_quantize(x_main, global_scale=x_scale)
        output = F.linear(qx_float.to(x.dtype), self.qweight)

        if self.qright is not None:
            if self.method == "paper_reorder_arc":
                error = x_main - qx_float
                left = error.index_select(1, self.activation_indices)
            else:
                error = x_float - qx_float
                left = torch.cat(
                    [
                        error.index_select(1, self.activation_indices),
                        qx_float.index_select(1, self.weight_indices),
                    ],
                    dim=1,
                )
            qleft, _ = nvfp4_quantize(left, global_scale=x_scale)
            output = output + F.linear(qleft.to(x.dtype), self.qright)
        if self.bias is not None:
            output = output + self.bias
        return output.reshape(*original_shape[:-1], self.out_features)


def module_key(layer_index: int, local_name: str) -> str:
    return f"layers.{layer_index}.{local_name}"


def replace_linears(
    layer: nn.Module,
    *,
    layer_index: int,
    method: str,
    reorder_index: Dict[str, torch.Tensor],
    select_nums: Dict[str, int],
    selection_indices: Dict[str, Dict[str, Any]],
) -> None:
    names = [
        name
        for name, module in layer.named_modules()
        if name and isinstance(module, nn.Linear)
    ]
    for local_name in names:
        parent_name, child_name = local_name.rsplit(".", 1)
        parent = layer.get_submodule(parent_name)
        original = getattr(parent, child_name)
        full_name = module_key(layer_index, local_name)
        cache_key = full_name + ".input"
        kwargs: Dict[str, Any] = {}
        if method == "paper_reorder_arc":
            select_num = int(select_nums[cache_key])
            permutation = reorder_index[cache_key]
            kwargs = {
                "permutation": permutation,
                "activation_indices": torch.arange(
                    original.in_features - select_num,
                    original.in_features,
                    dtype=torch.long,
                ),
                "weight_indices": torch.empty(0, dtype=torch.long),
            }
        elif method in JOINT_METHOD_TO_STRATEGY:
            strategy = JOINT_METHOD_TO_STRATEGY[method]
            selected = selection_indices[full_name][strategy]
            kwargs = {
                "activation_indices": selected["activation"],
                "weight_indices": selected["weight"],
            }
            expected = int(select_nums[cache_key])
            actual = int(
                selected["activation"].numel() + selected["weight"].numel()
            )
            if actual != expected:
                raise ValueError(
                    f"Budget mismatch for {full_name}: {actual} != {expected}"
                )
        wrapper = FakeNVFP4KPlusSLinear(
            original, method=method, **kwargs
        )
        setattr(parent, child_name, wrapper)
        del original
        gc.collect()
        torch.cuda.empty_cache()


@torch.no_grad()
def capture_first_layer_inputs(
    model: nn.Module,
    token_ids: torch.Tensor,
    *,
    seqlen: int,
    windows: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    layers = model.model.layers
    original_layer = layers[0]
    dtype = next(model.parameters()).dtype
    inputs = torch.empty(
        (windows, seqlen, model.config.hidden_size),
        dtype=dtype,
        device="cpu",
    )
    cache: Dict[str, Any] = {"index": 0}

    class Catcher(nn.Module):
        def __init__(self, module: nn.Module) -> None:
            super().__init__()
            self.module = module

        def forward(self, hidden_states: torch.Tensor, **kwargs: Any) -> Any:
            index = int(cache["index"])
            inputs[index].copy_(hidden_states[0].detach().cpu())
            cache["index"] = index + 1
            if index == 0:
                attention_mask = kwargs.get("attention_mask")
                position_ids = kwargs.get("position_ids")
                cache["attention_mask"] = (
                    attention_mask.detach().cpu()
                    if attention_mask is not None
                    else None
                )
                cache["position_ids"] = (
                    position_ids.detach().cpu()
                    if position_ids is not None
                    else None
                )
            raise ValueError("captured-first-layer-input")

    layers[0] = Catcher(original_layer)
    model.model.embed_tokens = model.model.embed_tokens.to(device)
    try:
        for index in range(windows):
            batch = token_ids[
                :, index * seqlen : (index + 1) * seqlen
            ].to(device)
            try:
                model(input_ids=batch, use_cache=False)
            except ValueError as error:
                if str(error) != "captured-first-layer-input":
                    raise
    finally:
        layers[0] = original_layer
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        torch.cuda.empty_cache()
    if cache["index"] != windows:
        raise AssertionError(f"Captured {cache['index']} of {windows} windows")
    return inputs, cache["attention_mask"], cache["position_ids"]


def save_checkpoint(
    path: Path,
    *,
    next_layer: int,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    position_ids: torch.Tensor | None,
    windows: int,
) -> None:
    torch.save(
        {
            "next_layer": next_layer,
            "hidden_states": hidden_states,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "windows": windows,
        },
        path,
    )


def load_test_tokens(
    model_path: Path, tokenizer: AutoTokenizer
) -> torch.Tensor:
    test_data = _load_wikitext2_split("test")
    return tokenizer(
        "\n\n".join(test_data["text"]), return_tensors="pt"
    ).input_ids


def main() -> None:
    args = parse_args()
    if args.wikitext_cache_dir:
        os.environ["ARCQUANT_WIKITEXT_CACHE_DIR"] = args.wikitext_cache_dir
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA is required")
    device = torch.device(args.device)
    torch.cuda.set_device(device.index or 0)
    torch.cuda.reset_peak_memory_stats()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"checkpoint_{args.method}.pt"
    result_path = output_dir / f"result_{args.method}.json"
    timing_path = output_dir / f"layer_timing_{args.method}.csv"
    started_at = time.time()

    model_path = Path(args.model)
    model_name = model_path.name.lower()
    saved_dir = Path(args.saved_dir)
    reorder_path = (
        saved_dir / f"{model_name}_reorder_index_wikitext2_{args.metric}.pt"
    )
    select_path = (
        saved_dir / f"{model_name}_select_num_wikitext2_{args.metric}.pt"
    )
    reorder_index: Dict[str, torch.Tensor] = {}
    select_nums: Dict[str, int] = {}
    needs_arc_artifacts = (
        args.method == "paper_reorder_arc"
        or args.method in JOINT_METHOD_TO_STRATEGY
    )
    if needs_arc_artifacts:
        for required in (reorder_path, select_path):
            if not required.is_file():
                raise FileNotFoundError(required)
        reorder_index = torch.load(reorder_path, map_location="cpu")
        select_nums = torch.load(select_path, map_location="cpu")
    selection_path = Path(args.selection_indices)
    selection_indices: Dict[str, Dict[str, Any]] = {}
    if args.method in JOINT_METHOD_TO_STRATEGY:
        if not selection_path.is_file():
            raise FileNotFoundError(selection_path)
        selection_indices = torch.load(selection_path, map_location="cpu")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, use_fast=False, legacy=False
    )
    token_ids = load_test_tokens(model_path, tokenizer)
    available_windows = token_ids.numel() // args.seqlen
    windows = (
        min(args.max_windows, available_windows)
        if args.max_windows > 0
        else available_windows
    )
    token_ids = token_ids[:, : windows * args.seqlen].contiguous()
    if windows <= 0:
        raise ValueError("No complete test windows")
    print(
        f"method={args.method} windows={windows}/{available_windows} "
        f"tokens={windows * args.seqlen}",
        flush=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    model.config.use_cache = False
    for parameter in model.parameters():
        parameter.requires_grad = False
    layers = model.model.layers
    expected_modules = {
        module_key(layer_index, local_name)
        for layer_index, layer in enumerate(layers)
        for local_name, module in layer.named_modules()
        if local_name and isinstance(module, nn.Linear)
    }
    expected_input_keys = {name + ".input" for name in expected_modules}
    if args.method in JOINT_METHOD_TO_STRATEGY:
        actual_modules = set(selection_indices)
        if actual_modules != expected_modules:
            missing = sorted(expected_modules - actual_modules)
            extra = sorted(actual_modules - expected_modules)
            raise ValueError(
                "Selection/model module mismatch: "
                f"expected={len(expected_modules)} actual={len(actual_modules)} "
                f"missing={missing[:3]} extra={extra[:3]}"
            )
    if needs_arc_artifacts:
        missing_reorder = sorted(expected_input_keys - set(reorder_index))
        missing_budget = sorted(expected_input_keys - set(select_nums))
        if missing_reorder or missing_budget:
            raise ValueError(
                "ARC calibration/model module mismatch: "
                f"missing_reorder={missing_reorder[:3]} "
                f"missing_select_num={missing_budget[:3]}"
            )

    start_layer = 0
    if args.resume and checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if int(checkpoint["windows"]) != windows:
            raise ValueError("Checkpoint window count does not match")
        hidden_states = checkpoint["hidden_states"]
        attention_mask = checkpoint["attention_mask"]
        position_ids = checkpoint["position_ids"]
        start_layer = int(checkpoint["next_layer"])
        print(f"Resuming at layer {start_layer + 1}", flush=True)
        for index in range(start_layer):
            layers[index] = nn.Identity()
    else:
        hidden_states, attention_mask, position_ids = capture_first_layer_inputs(
            model,
            token_ids,
            seqlen=args.seqlen,
            windows=windows,
            device=device,
        )

    attention_mask_device = (
        attention_mask.to(device) if attention_mask is not None else None
    )
    position_ids_device = (
        position_ids.to(device) if position_ids is not None else None
    )
    outputs = torch.empty_like(hidden_states)
    timing_rows = []
    if timing_path.is_file() and start_layer:
        timing_rows = pd.read_csv(timing_path).to_dict(orient="records")
        timing_rows = [row for row in timing_rows if int(row["layer"]) < start_layer]

    for layer_index in range(start_layer, len(layers)):
        layer_started = time.time()
        layer = layers[layer_index].to(device)
        if args.method != "bf16":
            replace_linears(
                layer,
                layer_index=layer_index,
                method=args.method,
                reorder_index=reorder_index,
                select_nums=select_nums,
                selection_indices=selection_indices,
            )
        torch.cuda.reset_peak_memory_stats()
        for window_index in range(windows):
            value = hidden_states[window_index : window_index + 1].to(device)
            with torch.no_grad():
                result = layer(
                    value,
                    attention_mask=attention_mask_device,
                    position_ids=position_ids_device,
                    use_cache=False,
                )[0]
            outputs[window_index].copy_(result[0].detach().cpu())
            del value, result
        torch.cuda.synchronize(device)
        elapsed = time.time() - layer_started
        peak_gib = torch.cuda.max_memory_allocated(device) / (1024.0**3)
        timing_rows.append(
            {
                "method": args.method,
                "layer": layer_index,
                "windows": windows,
                "elapsed_seconds": elapsed,
                "peak_gpu_memory_gib": peak_gib,
            }
        )
        pd.DataFrame(timing_rows).to_csv(timing_path, index=False)
        layers[layer_index] = nn.Identity()
        del layer
        hidden_states, outputs = outputs, hidden_states
        gc.collect()
        torch.cuda.empty_cache()
        print(
            f"layer={layer_index + 1}/{len(layers)} "
            f"seconds={elapsed:.1f} peak_gib={peak_gib:.2f}",
            flush=True,
        )
        next_layer = layer_index + 1
        if (
            args.checkpoint_every > 0
            and (
                next_layer % args.checkpoint_every == 0
                or next_layer == len(layers)
            )
        ):
            save_checkpoint(
                checkpoint_path,
                next_layer=next_layer,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                windows=windows,
            )

    model.model.norm = model.model.norm.to(device)
    model.lm_head = model.lm_head.to(device)
    total_nll = 0.0
    total_predictions = 0
    for window_index in range(windows):
        states = hidden_states[window_index : window_index + 1].to(device)
        with torch.no_grad():
            states = model.model.norm(states)
            logits = model.lm_head(states)
            labels = token_ids[
                :, window_index * args.seqlen : (window_index + 1) * args.seqlen
            ].to(device)
            loss = F.cross_entropy(
                logits[:, :-1, :].float().reshape(-1, logits.shape[-1]),
                labels[:, 1:].reshape(-1),
                reduction="sum",
            )
        total_nll += float(loss.item())
        total_predictions += args.seqlen - 1
        del states, logits, labels, loss
    ppl = math.exp(total_nll / total_predictions)
    elapsed_seconds = time.time() - started_at
    result = {
        "status": "completed",
        "method": args.method,
        "perplexity": ppl,
        "negative_log_likelihood": total_nll,
        "predicted_tokens": total_predictions,
        "windows": windows,
        "available_windows": available_windows,
        "seqlen": args.seqlen,
        "full_wikitext2_test": windows == available_windows,
        "model": str(model_path.resolve()),
        "dataset": "WikiText2 test",
        "quantization": (
            "BF16"
            if args.method == "bf16"
            else "W4A4 fake NVFP4 with BF16 GEMM operands"
        ),
        "main_layout": (
            "paper calibrated reorder"
            if args.method == "paper_reorder_arc"
            else "identity"
        ),
        "rotation": False,
        "selection_indices": (
            str(selection_path.resolve())
            if args.method in JOINT_METHOD_TO_STRATEGY
            else None
        ),
        "elapsed_seconds": elapsed_seconds,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device.index or 0),
        },
    }
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if checkpoint_path.is_file():
        checkpoint_path.unlink()


if __name__ == "__main__":
    main()
