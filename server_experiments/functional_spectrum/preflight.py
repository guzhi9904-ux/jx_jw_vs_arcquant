"""Fail-fast checks for the Llama-3.1-8B functional-spectrum server run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from functional_gram.collect_stats import (  # noqa: E402
    depth_control_modules,
    initial_cost_control_modules,
)
from functional_gram.model_inputs import model_weight_files  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--wikitext-cache-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--minimum-vram-gib", type=float, default=20.0)
    args = parser.parse_args()

    from transformers import AutoConfig

    model_path = Path(args.model).resolve()
    cache_dir = Path(args.wikitext_cache_dir).resolve()
    weights = model_weight_files(model_path)
    train_arrow = cache_dir / "wikitext-train.arrow"
    if not train_arrow.is_file():
        raise FileNotFoundError(train_arrow)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    vram_gib = torch.cuda.get_device_properties(0).total_memory / 2**30
    if vram_gib < args.minimum_vram_gib:
        raise RuntimeError(
            f"{vram_gib:.1f} GiB VRAM is below the {args.minimum_vram_gib:.1f} GiB safety floor"
        )
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    num_layers = int(getattr(config, "num_hidden_layers"))
    hidden_size = int(getattr(config, "hidden_size"))
    intermediate_size = int(getattr(config, "intermediate_size"))
    attention_heads = int(getattr(config, "num_attention_heads"))
    key_value_heads = int(getattr(config, "num_key_value_heads", attention_heads))
    head_dim = int(getattr(config, "head_dim", 0) or hidden_size // attention_heads)
    if num_layers < 3 or hidden_size < 1024 or intermediate_size <= hidden_size:
        raise ValueError("checkpoint does not look like the expected causal decoder")
    modules = initial_cost_control_modules(num_layers)
    depth_modules = depth_control_modules(num_layers)
    maximum_k = max(hidden_size, intermediate_size)
    gram_gib = maximum_k * maximum_k * 4 / 2**30
    payload = {
        "status": "passed",
        "model": str(model_path),
        "model_type": getattr(config, "model_type", None),
        "architectures": getattr(config, "architectures", None),
        "num_hidden_layers": num_layers,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "attention_layout": {
            "query_heads": attention_heads,
            "key_value_heads": key_value_heads,
            "head_dim": head_dim,
            "q_projection_rows": attention_heads * head_dim,
            "k_v_projection_rows": key_value_heads * head_dim,
        },
        "target_modules": modules,
        "depth_control_target_modules": depth_modules,
        "weight_files": [
            {"name": path.name, "bytes": path.stat().st_size} for path in weights
        ],
        "wikitext_train_arrow": str(train_arrow),
        "cuda": {
            "device": torch.cuda.get_device_name(0),
            "vram_gib": vram_gib,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
        },
        "largest_single_fp32_k_by_k_gram_gib": gram_gib,
        "solver": {
            "K_le_4096": "randomized top-512",
            "K_gt_4096": "randomized top-1024 (includes equal-fraction rank 896)",
            "randomized_min_K": 2048,
        },
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
