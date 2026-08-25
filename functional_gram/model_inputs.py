"""Model-agnostic decoder input capture for Qwen2 and Llama3 experiments."""

from __future__ import annotations

import hashlib
import os
import random
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn

from fp4_residual_carrier.common.reproducibility import sha256_file


def model_weight_files(model_path: str | Path) -> tuple[Path, ...]:
    """Return monolithic or sharded Hugging Face weight files."""

    root = Path(model_path).resolve()
    candidates = tuple(sorted(root.glob("*.safetensors"))) or tuple(
        sorted(root.glob("pytorch_model*.bin"))
    )
    if not candidates:
        raise FileNotFoundError(f"no Hugging Face weight files found under {root}")
    return candidates


def model_weight_fingerprint(model_path: str | Path) -> tuple[str, list[dict[str, Any]]]:
    """Hash every immutable weight shard and return an aggregate digest."""

    records: list[dict[str, Any]] = []
    aggregate = hashlib.sha256()
    for path in model_weight_files(model_path):
        digest = sha256_file(path)
        record = {"name": path.name, "bytes": path.stat().st_size, "sha256": digest}
        records.append(record)
        aggregate.update(path.name.encode())
        aggregate.update(str(path.stat().st_size).encode())
        aggregate.update(digest.encode())
    return aggregate.hexdigest(), records


def load_model(model_path: str) -> tuple[nn.Module, Any]:
    """Load a local Hugging Face causal decoder on CPU."""

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config.use_cache = False
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        trust_remote_code=True,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, use_fast=True, trust_remote_code=False
    )
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise TypeError("expected a Hugging Face decoder with model.layers")
    return model, tokenizer


def _load_wikitext2_split(split: str) -> dict[str, list[str]] | Any:
    cache_dir = os.environ.get("ARCQUANT_WIKITEXT_CACHE_DIR")
    if cache_dir:
        path = Path(cache_dir) / f"wikitext-{split}.arrow"
        if not path.is_file():
            raise FileNotFoundError(f"missing cached WikiText2 split: {path}")
        import pyarrow as pa
        import pyarrow.ipc as ipc

        with pa.memory_map(str(path), "r") as source:
            table = ipc.open_stream(source).read_all()
        return {"text": table.column("text").to_pylist()}
    from datasets import load_dataset

    return load_dataset("wikitext", "wikitext-2-raw-v1", split=split)


def load_wikitext_samples(
    tokenizer: Any,
    *,
    selection_samples: int,
    holdout_samples: int,
    seed: int,
    seqlen: int,
) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], dict[str, Any]]:
    """Load deterministic interval-disjoint WikiText2 train windows."""

    train_data = _load_wikitext2_split("train")
    encoded = tokenizer("\n\n".join(train_data["text"]), return_tensors="pt")
    token_ids = encoded.input_ids
    generator = random.Random(seed)
    total = selection_samples + holdout_samples
    starts: list[int] = []
    attempts = 0
    while len(starts) < total:
        attempts += 1
        if attempts > 100_000:
            raise RuntimeError("failed to sample disjoint WikiText2 windows")
        candidate = generator.randint(0, token_ids.shape[1] - seqlen - 1)
        if all(candidate + seqlen <= start or start + seqlen <= candidate for start in starts):
            starts.append(candidate)
    batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    hashes: list[str] = []
    for start in starts:
        value = token_ids[:, start : start + seqlen].contiguous()
        target = value.clone()
        target[:, :-1] = -100
        batches.append((value, target))
        hashes.append(hashlib.sha256(value.numpy().tobytes()).hexdigest())
    selection_ids = range(selection_samples)
    holdout_ids = range(selection_samples, total)
    split_manifest = {
        "token_count": int(token_ids.numel()),
        "seed": seed,
        "seqlen": seqlen,
        "selection": [
            {"sample_id": i, "start": starts[i], "end": starts[i] + seqlen, "sha256": hashes[i]}
            for i in selection_ids
        ],
        "holdout": [
            {"sample_id": i, "start": starts[i], "end": starts[i] + seqlen, "sha256": hashes[i]}
            for i in holdout_ids
        ],
        "selection_holdout_exact_disjoint": True,
        "selection_holdout_interval_disjoint": True,
        "all_windows_interval_disjoint": True,
    }
    del encoded, token_ids, train_data
    return batches, split_manifest


def pick_rows_with_ids(value: torch.Tensor, count: int) -> tuple[torch.Tensor, torch.Tensor]:
    flat = value.reshape(-1, value.shape[-1])
    count = min(count, flat.shape[0])
    indices = (
        torch.arange(count, device=flat.device)
        if count == flat.shape[0]
        else torch.linspace(0, flat.shape[0] - 1, count, device=flat.device).round().long()
    )
    if indices.unique().numel() != indices.numel():
        raise AssertionError("row sampler produced duplicate token rows")
    return flat.index_select(0, indices).detach().float(), indices.detach().cpu()


@torch.no_grad()
def first_layer_inputs(
    model: nn.Module,
    token_batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    device: str,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Capture first-decoder-layer inputs for Qwen2/Llama3 style models."""

    layers = model.model.layers
    cache: dict[str, Any] = {}

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
    finally:
        layers[0] = layers[0].module
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        if hasattr(model.model, "rotary_emb"):
            model.model.rotary_emb = model.model.rotary_emb.cpu()
    if "inps" not in cache:
        raise RuntimeError("failed to capture first-layer inputs")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return cache["inps"], cache["attention_mask"], cache["position_ids"]


__all__ = [
    "first_layer_inputs",
    "load_model",
    "load_wikitext_samples",
    "model_weight_files",
    "model_weight_fingerprint",
    "pick_rows_with_ids",
]
