"""Paper-aligned lm-eval for BF16, RTN, ARC, and dual-source K+S.

The script deliberately reuses the fake-NVFP4 linear wrapper from the PPL
evaluator so every quantized method has the same numerical backend.  It runs
the five zero-shot tasks reported in ARCQuant and 5-shot MMLU, saves each
suite independently for resume, and then emits one compact result JSON for
server aggregation.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_joint_ks_ppl import (  # noqa: E402
    JOINT_METHOD_TO_STRATEGY,
    METHODS,
    module_key,
    replace_linears,
)


DEFAULT_ZERO_SHOT_TASKS = (
    "arc_challenge",
    "hellaswag",
    "lambada_openai",
    "piqa",
    "winogrande",
)
PAPER_METRICS = {
    "arc_challenge": "acc_norm,none",
    "hellaswag": "acc_norm,none",
    "lambada_openai": "acc,none",
    "piqa": "acc_norm,none",
    "winogrande": "acc,none",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run paper-aligned downstream tasks with one fake-NVFP4 backend."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--saved-dir", required=True)
    parser.add_argument("--selection-indices")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--method", required=True, choices=sorted(METHODS))
    parser.add_argument("--metric", default="max")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--zero-shot-tasks",
        default=",".join(DEFAULT_ZERO_SHOT_TASKS),
    )
    parser.add_argument("--mmlu-task", default="mmlu")
    parser.add_argument("--mmlu-num-fewshot", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--zero-shot-batch-size", default="auto")
    parser.add_argument("--mmlu-batch-size", default="2")
    parser.add_argument("--max-batch-size", type=int, default=64)
    parser.add_argument("--bootstrap-iters", type=int, default=0)
    parser.add_argument("--selection-seed", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def completed_json(path: Path, expected: dict[str, Any] | None = None) -> bool:
    if not path.is_file():
        return False
    try:
        payload = read_json(path)
    except (OSError, json.JSONDecodeError):
        return False
    if payload.get("status") != "completed":
        return False
    return expected is None or all(payload.get(key) == value for key, value in expected.items())


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def parse_batch_size(raw: str) -> int | str:
    return int(raw) if raw.isdigit() else raw


def import_lm_eval() -> tuple[Any, Any, Any, str]:
    try:
        from lm_eval import evaluator
        from lm_eval.models.huggingface import HFLM
        from lm_eval.tasks import TaskManager
    except ImportError as error:
        raise RuntimeError(
            "lm-eval is required; install the pinned server requirements "
            "(lm-eval==0.4.8)"
        ) from error
    version = importlib.metadata.version("lm-eval")
    if version != "0.4.8":
        raise RuntimeError(f"Expected lm-eval==0.4.8, got {version}")
    return evaluator, HFLM, TaskManager, version


def prepare_artifacts(
    *,
    model: nn.Module,
    model_path: Path,
    saved_dir: Path,
    selection_path: Path | None,
    method: str,
    metric: str,
) -> tuple[dict[str, torch.Tensor], dict[str, int], dict[str, dict[str, Any]]]:
    model_name = model_path.name.lower()
    reorder_path = saved_dir / f"{model_name}_reorder_index_wikitext2_{metric}.pt"
    select_path = saved_dir / f"{model_name}_select_num_wikitext2_{metric}.pt"
    needs_arc_artifacts = (
        method == "paper_reorder_arc" or method in JOINT_METHOD_TO_STRATEGY
    )
    reorder_index: dict[str, torch.Tensor] = {}
    select_nums: dict[str, int] = {}
    if needs_arc_artifacts:
        for required in (reorder_path, select_path):
            if not required.is_file():
                raise FileNotFoundError(required)
        reorder_index = torch.load(reorder_path, map_location="cpu")
        select_nums = torch.load(select_path, map_location="cpu")

    selection_indices: dict[str, dict[str, Any]] = {}
    if method in JOINT_METHOD_TO_STRATEGY:
        if selection_path is None or not selection_path.is_file():
            raise FileNotFoundError(
                selection_path or "--selection-indices is required for joint K+S"
            )
        selection_indices = torch.load(selection_path, map_location="cpu")

    layers = model.model.layers
    expected_modules = {
        module_key(layer_index, local_name)
        for layer_index, layer in enumerate(layers)
        for local_name, module in layer.named_modules()
        if local_name and isinstance(module, nn.Linear)
    }
    expected_input_keys = {name + ".input" for name in expected_modules}
    if method in JOINT_METHOD_TO_STRATEGY:
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
    return reorder_index, select_nums, selection_indices


@torch.no_grad()
def apply_method(
    model: nn.Module,
    *,
    method: str,
    reorder_index: dict[str, torch.Tensor],
    select_nums: dict[str, int],
    selection_indices: dict[str, dict[str, Any]],
) -> None:
    if method == "bf16":
        return
    layers = model.model.layers
    for layer_index, layer in enumerate(layers):
        replace_linears(
            layer,
            layer_index=layer_index,
            method=method,
            reorder_index=reorder_index,
            select_nums=select_nums,
            selection_indices=selection_indices,
        )
        print(
            f"prepared fake-NVFP4 layer={layer_index + 1}/{len(layers)}",
            flush=True,
        )
        gc.collect()


def run_suite(
    *,
    evaluator: Any,
    task_manager: Any,
    lm: Any,
    suite: str,
    tasks: list[str],
    num_fewshot: int,
    batch_size: int | str,
    max_batch_size: int,
    limit: int,
    bootstrap_iters: int,
    output_path: Path,
    resume: bool,
) -> dict[str, Any]:
    expected = {
        "suite": suite,
        "tasks": tasks,
        "num_fewshot": num_fewshot,
        "limit": limit,
    }
    if resume and completed_json(output_path, expected):
        print(f"Skipping completed task suite: {suite}", flush=True)
        return read_json(output_path)
    started = time.time()
    raw = evaluator.simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        max_batch_size=max_batch_size,
        device=str(lm.device),
        limit=None if limit <= 0 else limit,
        bootstrap_iters=bootstrap_iters,
        log_samples=False,
        task_manager=task_manager,
        random_seed=0,
        numpy_random_seed=0,
        torch_random_seed=0,
        fewshot_random_seed=0,
    )
    payload = {
        "status": "completed",
        "suite": suite,
        "tasks": tasks,
        "num_fewshot": num_fewshot,
        "batch_size": batch_size,
        "limit": limit,
        "full_evaluation": limit <= 0,
        "elapsed_seconds": time.time() - started,
        "raw": raw,
    }
    write_json(output_path, payload)
    return payload


def metric_from_row(row: dict[str, Any], preferred: str) -> tuple[str, float]:
    candidates = (preferred, preferred.split(",", 1)[0])
    for key in candidates:
        value = row.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return key, float(value)
    raise KeyError(f"Missing metric {preferred}; available={sorted(row)}")


def task_metric(raw: dict[str, Any], task: str, preferred: str) -> tuple[str, float]:
    results = raw.get("results", {})
    if task not in results:
        raise KeyError(f"Missing task {task}; available={sorted(results)}")
    return metric_from_row(results[task], preferred)


def mmlu_metric(raw: dict[str, Any], group_name: str) -> tuple[str, float]:
    for container_name in ("groups", "results"):
        container = raw.get(container_name, {})
        if group_name in container:
            try:
                return metric_from_row(container[group_name], "acc,none")
            except KeyError:
                pass

    results = raw.get("results", {})
    samples = raw.get("n-samples", {})
    weighted_sum = 0.0
    total = 0
    used_key = "acc,none"
    for task, row in results.items():
        if not task.startswith(group_name + "_"):
            continue
        metric_key, value = metric_from_row(row, "acc,none")
        sample_info = samples.get(task, {})
        count = int(sample_info.get("effective", sample_info.get("original", 1)))
        weighted_sum += value * count
        total += count
        used_key = metric_key
    if total == 0:
        raise KeyError(
            f"Could not find group {group_name} or its subtasks in lm-eval output"
        )
    return used_key, weighted_sum / total


def main() -> None:
    args = parse_args()
    if args.method == "bf16":
        quantization = "BF16"
    else:
        quantization = "W4A4 fake NVFP4 with BF16 GEMM operands"
    if args.limit < 0:
        raise ValueError("--limit must be 0 (full) or a positive smoke-test count")
    zero_tasks = [item.strip() for item in args.zero_shot_tasks.split(",") if item.strip()]
    if not zero_tasks:
        raise ValueError("At least one zero-shot task is required")
    unknown = sorted(set(zero_tasks) - set(PAPER_METRICS))
    if unknown:
        raise ValueError(
            f"No paper metric mapping for {unknown}; supported={sorted(PAPER_METRICS)}"
        )
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA is required")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device.index or 0)
        torch.cuda.reset_peak_memory_stats(device)

    evaluator, HFLM, TaskManager, lm_eval_version = import_lm_eval()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"result_tasks_{args.method}.json"
    expected_result = {
        "method": args.method,
        "selection_seed": args.selection_seed,
        "zero_shot_tasks": zero_tasks,
        "mmlu_task": args.mmlu_task,
        "mmlu_num_fewshot": args.mmlu_num_fewshot,
        "limit": args.limit,
    }
    if args.resume and completed_json(result_path, expected_result):
        print(json.dumps(read_json(result_path), ensure_ascii=False, indent=2))
        return

    model_path = Path(args.model).expanduser().resolve()
    saved_dir = Path(args.saved_dir).expanduser().resolve()
    selection_path = (
        Path(args.selection_indices).expanduser().resolve()
        if args.selection_indices
        else None
    )
    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        use_fast=False,
        legacy=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    model.config.use_cache = True
    for parameter in model.parameters():
        parameter.requires_grad = False

    reorder_index, select_nums, selection_indices = prepare_artifacts(
        model=model,
        model_path=model_path,
        saved_dir=saved_dir,
        selection_path=selection_path,
        method=args.method,
        metric=args.metric,
    )
    apply_method(
        model,
        method=args.method,
        reorder_index=reorder_index,
        select_nums=select_nums,
        selection_indices=selection_indices,
    )
    del reorder_index, select_nums, selection_indices
    gc.collect()
    model.to(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    zero_batch_size = parse_batch_size(args.zero_shot_batch_size)
    mmlu_batch_size = parse_batch_size(args.mmlu_batch_size)
    zero_lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=zero_batch_size,
        max_batch_size=args.max_batch_size,
        device=str(device),
        dtype=torch.bfloat16,
        use_fast_tokenizer=False,
    )
    mmlu_lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=mmlu_batch_size,
        max_batch_size=args.max_batch_size,
        device=str(device),
        dtype=torch.bfloat16,
        use_fast_tokenizer=False,
    )
    task_manager = TaskManager()
    zero_payload = run_suite(
        evaluator=evaluator,
        task_manager=task_manager,
        lm=zero_lm,
        suite="paper_zero_shot",
        tasks=zero_tasks,
        num_fewshot=0,
        batch_size=zero_batch_size,
        max_batch_size=args.max_batch_size,
        limit=args.limit,
        bootstrap_iters=args.bootstrap_iters,
        output_path=output_dir / f"raw_zero_shot_{args.method}.json",
        resume=args.resume,
    )
    mmlu_payload = run_suite(
        evaluator=evaluator,
        task_manager=task_manager,
        lm=mmlu_lm,
        suite="paper_mmlu_5shot",
        tasks=[args.mmlu_task],
        num_fewshot=args.mmlu_num_fewshot,
        batch_size=mmlu_batch_size,
        max_batch_size=args.max_batch_size,
        limit=args.limit,
        bootstrap_iters=args.bootstrap_iters,
        output_path=output_dir / f"raw_mmlu_{args.method}.json",
        resume=args.resume,
    )

    zero_raw = zero_payload["raw"]
    metrics: dict[str, dict[str, Any]] = {}
    for task in zero_tasks:
        metric_key, value = task_metric(zero_raw, task, PAPER_METRICS[task])
        metrics[task] = {"metric": metric_key, "value": value}
    zero_average = sum(item["value"] for item in metrics.values()) / len(metrics)
    mmlu_key, mmlu_value = mmlu_metric(mmlu_payload["raw"], args.mmlu_task)
    metrics[args.mmlu_task] = {
        "metric": mmlu_key,
        "value": mmlu_value,
        "num_fewshot": args.mmlu_num_fewshot,
    }

    peak_gib = (
        torch.cuda.max_memory_allocated(device) / 2**30
        if device.type == "cuda"
        else None
    )
    result = {
        "status": "completed",
        "method": args.method,
        "selection_seed": args.selection_seed,
        "paper_zero_shot_average": zero_average,
        "mmlu_5shot_accuracy": mmlu_value,
        "task_metrics": metrics,
        "zero_shot_tasks": zero_tasks,
        "mmlu_task": args.mmlu_task,
        "mmlu_num_fewshot": args.mmlu_num_fewshot,
        "limit": args.limit,
        "full_evaluation": args.limit <= 0,
        "model": str(model_path),
        "quantization": quantization,
        "main_layout": (
            "paper calibrated reorder"
            if args.method == "paper_reorder_arc"
            else "identity"
        ),
        "rotation": False,
        "selection_indices": (
            str(selection_path)
            if args.method in JOINT_METHOD_TO_STRATEGY
            else None
        ),
        "elapsed_seconds": time.time() - started,
        "peak_gpu_memory_gib": peak_gib,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "transformers": importlib.metadata.version("transformers"),
            "lm_eval": lm_eval_version,
            "gpu": (
                torch.cuda.get_device_name(device.index or 0)
                if device.type == "cuda"
                else None
            ),
            "hf_datasets_offline": os.environ.get("HF_DATASETS_OFFLINE"),
        },
    }
    write_json(result_path, result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
