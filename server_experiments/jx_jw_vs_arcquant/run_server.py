"""Single entry point for ARCQuant versus independent J_X/J_W server runs."""

from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BUNDLE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = BUNDLE_ROOT.parents[1]
CALIBRATE_SCRIPT = REPO_ROOT / "reorder_indices.py"
UTILIZE_SCRIPT = REPO_ROOT / "utilize.py"
QUANTIZE_SCRIPT = REPO_ROOT / "model" / "quantize.py"
KV_CACHE_SCRIPT = REPO_ROOT / "model" / "kv_cache.py"
LOCAL_SCRIPT = REPO_ROOT / "scripts" / "analyze_proxy_joint_ks_holdout.py"
PPL_SCRIPT = REPO_ROOT / "scripts" / "evaluate_joint_ks_ppl.py"
SPLIT_VALIDATOR = REPO_ROOT / "scripts" / "validate_proxy_split_disjoint.py"
SEED_VALIDATOR = BUNDLE_ROOT / "validate_seed.py"
AGGREGATOR = BUNDLE_ROOT / "aggregate_results.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run full ARC calibration, local output-SSE comparisons, and "
            "WikiText2 PPL from an isolated server run directory."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--model",
        help="Local Hugging Face model directory; overrides config/env.",
    )
    parser.add_argument("--run-dir")
    parser.add_argument(
        "--stage",
        choices=("preflight", "calibrate", "local", "ppl", "aggregate", "all"),
        default="all",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wikitext-cache-dir")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="2-sample/128-token/2-window smoke run in a separate *_quick dir.",
    )
    parser.add_argument("--seeds", help="Override local seeds, e.g. 0,1,2,3,4")
    parser.add_argument("--ppl-seeds", help="Override J_X/J_W PPL seeds")
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--with-ppl-diagnostics", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def parse_int_list(raw: str | None, fallback: list[int]) -> list[int]:
    if raw is None:
        values = fallback
    else:
        values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("Seed list cannot be empty")
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate seeds are not allowed: {values}")
    return values


def completed_json(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        return read_json(path).get("status") == "completed"
    except (OSError, json.JSONDecodeError):
        return False


def display_command(command: list[str]) -> str:
    return shlex.join(command)


def run_command(
    command: list[str],
    *,
    log_path: Path,
    env: dict[str, str],
    dry_run: bool,
) -> None:
    rendered = display_command(command)
    print(f"\n$ {rendered}", flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        with log_path.open("a", encoding="utf-8") as sink:
            sink.write(f"[dry-run] {rendered}\n")
        return
    with log_path.open("a", encoding="utf-8") as sink:
        sink.write(f"\n[{datetime.now(timezone.utc).isoformat()}] $ {rendered}\n")
        sink.flush()
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            sink.write(line)
            sink.flush()
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def git_snapshot() -> dict[str, Any]:
    def capture(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            return result.stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = capture("status", "--short")
    return {
        "commit": capture("rev-parse", "HEAD"),
        "dirty": bool(status),
        "changed_path_count": 0 if not status else len(status.splitlines()),
    }


def artifact_paths(
    model_path: Path, artifact_dir: Path, dataset: str, metric: str
) -> tuple[Path, Path]:
    prefix = model_path.name.lower()
    return (
        artifact_dir / f"{prefix}_reorder_index_{dataset}_{metric}.pt",
        artifact_dir / f"{prefix}_select_num_{dataset}_{metric}.pt",
    )


def apply_quick_overrides(config: dict[str, Any]) -> None:
    config["calibration"].update(
        {"samples": 2, "select_samples": 2, "seqlen": 128}
    )
    config["local"].update(
        {
            "seeds": [config["local"]["seeds"][0]],
            "selection_samples": 1,
            "holdout_samples": 1,
            "rows_per_split": 32,
            "seqlen": 128,
        }
    )
    config["ppl"].update(
        {
            "dual_seeds": [config["local"]["seeds"][0]],
            "seqlen": 128,
            "max_windows": 2,
            "checkpoint_every": 0,
        }
    )


def preflight(
    *,
    config: dict[str, Any],
    model_path: Path,
    run_dir: Path,
    device: str,
    cache_dir: Path | None,
    offline: bool,
) -> dict[str, Any]:
    missing_code = [
        str(path)
        for path in (
            CALIBRATE_SCRIPT,
            UTILIZE_SCRIPT,
            QUANTIZE_SCRIPT,
            KV_CACHE_SCRIPT,
            LOCAL_SCRIPT,
            PPL_SCRIPT,
            SPLIT_VALIDATOR,
            SEED_VALIDATOR,
            AGGREGATOR,
        )
        if not path.is_file()
    ]
    if missing_code:
        raise FileNotFoundError(f"Missing experiment code: {missing_code}")
    source_contracts = {
        CALIBRATE_SCRIPT: ("--saved-dir", "--select-samples"),
        UTILIZE_SCRIPT: ("if metric == 'frobenius'",),
        QUANTIZE_SCRIPT: (
            "def quantize_nvfp4_tensor",
            "scale = torch.max(x.abs())",
        ),
        KV_CACHE_SCRIPT: ("except (ImportError, OSError)",),
        LOCAL_SCRIPT: ("server-comparison", "SERVER_COMPARISON_STRATEGIES"),
        PPL_SCRIPT: ("Selection/model module mismatch",),
    }
    stale_sources = []
    for path, markers in source_contracts.items():
        source = path.read_text(encoding="utf-8")
        missing_markers = [marker for marker in markers if marker not in source]
        if missing_markers:
            stale_sources.append({"path": str(path), "missing": missing_markers})
    if stale_sources:
        raise RuntimeError(
            "Repository is missing server-bundle source updates: "
            f"{stale_sources}. Commit the files listed in SOURCE_DEPENDENCIES.md."
        )
    if not model_path.is_dir():
        raise FileNotFoundError(
            f"Model must be a local Hugging Face directory: {model_path}"
        )
    if cache_dir is not None:
        missing_cache = [
            str(cache_dir / f"wikitext-{split}.arrow")
            for split in ("train", "test")
            if not (cache_dir / f"wikitext-{split}.arrow").is_file()
        ]
        if missing_cache:
            raise FileNotFoundError(f"Incomplete WikiText2 cache: {missing_cache}")
    elif offline:
        print(
            "Warning: --offline without --wikitext-cache-dir relies on the "
            "Hugging Face datasets cache.",
            flush=True,
        )

    import torch
    import transformers
    from transformers import AutoConfig

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    if device.startswith("cuda"):
        index = int(device.split(":", 1)[1]) if ":" in device else 0
        if index >= torch.cuda.device_count():
            raise ValueError(
                f"Requested {device}, but only {torch.cuda.device_count()} GPUs exist"
            )
        gpu_name = torch.cuda.get_device_name(index)
        gpu_total_gib = torch.cuda.get_device_properties(index).total_memory / 2**30
    else:
        gpu_name = None
        gpu_total_gib = None

    hf_config = AutoConfig.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=True
    )
    expected_family = config.get("model_family")
    if expected_family and hf_config.model_type != expected_family:
        raise ValueError(
            f"Config expects {expected_family}, model reports {hf_config.model_type}"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    disk = shutil.disk_usage(run_dir)
    result = {
        "status": "passed",
        "model": str(model_path),
        "model_type": hf_config.model_type,
        "architectures": getattr(hf_config, "architectures", None),
        "hidden_size": getattr(hf_config, "hidden_size", None),
        "num_hidden_layers": getattr(hf_config, "num_hidden_layers", None),
        "python": platform.python_version(),
        "executable": sys.executable,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "device": device,
        "gpu": gpu_name,
        "gpu_total_gib": gpu_total_gib,
        "disk_free_gib": disk.free / 2**30,
        "wikitext_cache_dir": str(cache_dir) if cache_dir else None,
        "offline": offline,
    }
    write_json(run_dir / "preflight.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config = copy.deepcopy(read_json(config_path))
    if args.quick:
        apply_quick_overrides(config)
    run_name = config["name"] + ("_quick" if args.quick else "")
    run_dir = (
        Path(args.run_dir).expanduser().resolve()
        if args.run_dir
        else (BUNDLE_ROOT / "runs" / run_name).resolve()
    )
    if args.stage == "aggregate":
        run_command(
            [sys.executable, str(AGGREGATOR), "--run-dir", str(run_dir)],
            log_path=run_dir / "logs" / "aggregate.log",
            env=os.environ.copy(),
            dry_run=args.dry_run,
        )
        return

    model_value = args.model or os.environ.get("ARCQUANT_MODEL") or config.get("model")
    if not model_value:
        raise ValueError(
            "Pass --model /absolute/model/path or set ARCQUANT_MODEL; "
            "Git configs intentionally do not store machine-specific paths."
        )
    model_path = Path(model_value).expanduser().resolve()
    cache_dir = (
        Path(args.wikitext_cache_dir).expanduser().resolve()
        if args.wikitext_cache_dir
        else None
    )
    local_seeds = parse_int_list(args.seeds, list(config["local"]["seeds"]))
    ppl_seeds = parse_int_list(
        args.ppl_seeds, list(config["ppl"]["dual_seeds"])
    )
    if args.quick:
        local_seeds = local_seeds[:1]
        ppl_seeds = ppl_seeds[:1]
    max_windows = (
        args.max_windows
        if args.max_windows is not None
        else int(config["ppl"]["max_windows"])
    )

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if cache_dir is not None:
        env["ARCQUANT_WIKITEXT_CACHE_DIR"] = str(cache_dir)
    if args.offline or cache_dir is not None:
        env["HF_DATASETS_OFFLINE"] = "1"
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"

    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "bundle_version": "v1",
        "config_file": str(config_path),
        "effective_config": config,
        "model": str(model_path),
        "run_dir": str(run_dir),
        "device": args.device,
        "local_seeds": local_seeds,
        "ppl_seeds": ppl_seeds,
        "quick": args.quick,
        "offline": args.offline or cache_dir is not None,
        "git": git_snapshot(),
    }
    write_json(run_dir / "run_manifest.json", manifest)
    preflight(
        config=config,
        model_path=model_path,
        run_dir=run_dir,
        device=args.device,
        cache_dir=cache_dir,
        offline=args.offline or cache_dir is not None,
    )
    if args.stage == "preflight":
        return

    calibration = config["calibration"]
    artifact_dir = run_dir / "artifacts"
    reorder_path, select_path = artifact_paths(
        model_path,
        artifact_dir,
        calibration["dataset"],
        calibration["metric"],
    )

    if args.stage in {"calibrate", "all"}:
        if args.resume and reorder_path.is_file() and select_path.is_file():
            print(f"Skipping completed calibration in {artifact_dir}", flush=True)
        else:
            command = [
                sys.executable,
                str(CALIBRATE_SCRIPT),
                "--model",
                str(model_path),
                "--dataset",
                str(calibration["dataset"]),
                "--act_sort_metric",
                str(calibration["metric"]),
                "--samples",
                str(calibration["samples"]),
                "--select-samples",
                str(calibration["select_samples"]),
                "--seqlen",
                str(calibration["seqlen"]),
                "--seed",
                str(calibration["seed"]),
                "--device",
                args.device,
                "--saved-dir",
                str(artifact_dir),
            ]
            if cache_dir is not None:
                command.extend(["--wikitext-cache-dir", str(cache_dir)])
            run_command(
                command,
                log_path=run_dir / "logs" / "calibrate.log",
                env=env,
                dry_run=args.dry_run,
            )
        if args.stage == "calibrate":
            return

    if args.stage in {"local", "ppl", "all"} and not args.dry_run:
        for required in (reorder_path, select_path):
            if not required.is_file():
                raise FileNotFoundError(
                    f"Missing calibration artifact {required}; run --stage calibrate first"
                )

    local_config = config["local"]
    if args.stage in {"local", "all"}:
        for seed in local_seeds:
            seed_dir = run_dir / "local" / f"seed_{seed}"
            complete = all(
                (seed_dir / name).is_file()
                for name in (
                    "metadata.json",
                    "module_metrics.csv",
                    "strategy_summary.csv",
                    "selection_indices.pt",
                    "server_validation.json",
                    "split_validation.json",
                )
            )
            if args.resume and complete:
                print(f"Skipping completed local seed {seed}", flush=True)
                continue
            command = [
                sys.executable,
                str(LOCAL_SCRIPT),
                "--model",
                str(model_path),
                "--saved-dir",
                str(artifact_dir),
                "--output-dir",
                str(seed_dir),
                "--selection-samples",
                str(local_config["selection_samples"]),
                "--holdout-samples",
                str(local_config["holdout_samples"]),
                "--seqlen",
                str(local_config["seqlen"]),
                "--rows-per-split",
                str(local_config["rows_per_split"]),
                "--seed",
                str(seed),
                "--device",
                args.device,
                "--metric",
                str(calibration["metric"]),
                "--experiment",
                "server-comparison",
            ]
            if args.resume:
                command.append("--resume")
            if cache_dir is not None:
                command.extend(["--wikitext-cache-dir", str(cache_dir)])
            run_command(
                command,
                log_path=run_dir / "logs" / f"local_seed_{seed}.log",
                env=env,
                dry_run=args.dry_run,
            )
            run_command(
                [sys.executable, str(SEED_VALIDATOR), "--analysis-dir", str(seed_dir)],
                log_path=run_dir / "logs" / f"validate_seed_{seed}.log",
                env=env,
                dry_run=args.dry_run,
            )
            run_command(
                [
                    sys.executable,
                    str(SPLIT_VALIDATOR),
                    "--analysis-dir",
                    str(seed_dir),
                    "--selection-samples",
                    str(local_config["selection_samples"]),
                    "--holdout-samples",
                    str(local_config["holdout_samples"]),
                    "--seqlen",
                    str(local_config["seqlen"]),
                    "--seed",
                    str(seed),
                ],
                log_path=run_dir / "logs" / f"validate_split_{seed}.log",
                env=env,
                dry_run=args.dry_run,
            )
        if args.stage == "local":
            return

    if args.stage in {"ppl", "all"}:
        ppl_config = config["ppl"]
        first_selection = run_dir / "local" / f"seed_{local_seeds[0]}" / "selection_indices.pt"

        def run_ppl(method: str, output_dir: Path, selection_path: Path) -> None:
            result_path = output_dir / f"result_{method}.json"
            if args.resume and completed_json(result_path):
                print(f"Skipping completed PPL: {method} in {output_dir}", flush=True)
                return
            command = [
                sys.executable,
                str(PPL_SCRIPT),
                "--model",
                str(model_path),
                "--saved-dir",
                str(artifact_dir),
                "--selection-indices",
                str(selection_path),
                "--output-dir",
                str(output_dir),
                "--method",
                method,
                "--metric",
                str(calibration["metric"]),
                "--seqlen",
                str(ppl_config["seqlen"]),
                "--max-windows",
                str(max_windows),
                "--device",
                args.device,
                "--checkpoint-every",
                str(ppl_config["checkpoint_every"]),
            ]
            if args.resume:
                command.append("--resume")
            if cache_dir is not None:
                command.extend(["--wikitext-cache-dir", str(cache_dir)])
            run_command(
                command,
                log_path=run_dir / "logs" / f"ppl_{output_dir.name}_{method}.log",
                env=env,
                dry_run=args.dry_run,
            )

        for method in ppl_config["baseline_methods"]:
            run_ppl(method, run_dir / "ppl" / "baselines", first_selection)
        for seed in ppl_seeds:
            selection_path = (
                run_dir / "local" / f"seed_{seed}" / "selection_indices.pt"
            )
            if not args.dry_run and not selection_path.is_file():
                raise FileNotFoundError(
                    f"Missing J_X/J_W selection for PPL seed {seed}: {selection_path}"
                )
            run_ppl(
                str(ppl_config["dual_method"]),
                run_dir / "ppl" / f"dual_seed_{seed}",
                selection_path,
            )
        if args.with_ppl_diagnostics:
            diagnostic_seed = local_seeds[0]
            selection_path = (
                run_dir
                / "local"
                / f"seed_{diagnostic_seed}"
                / "selection_indices.pt"
            )
            for method in ppl_config.get("diagnostic_methods", []):
                run_ppl(
                    method,
                    run_dir / "ppl" / f"diagnostics_seed_{diagnostic_seed}",
                    selection_path,
                )
        if args.stage == "ppl":
            return

    if args.stage in {"aggregate", "all"}:
        run_command(
            [sys.executable, str(AGGREGATOR), "--run-dir", str(run_dir)],
            log_path=run_dir / "logs" / "aggregate.log",
            env=env,
            dry_run=args.dry_run,
        )
    elapsed = time.time() - os.path.getmtime(run_dir / "run_manifest.json")
    print(f"Finished requested stages in {elapsed / 60.0:.1f} minutes", flush=True)


if __name__ == "__main__":
    main()
