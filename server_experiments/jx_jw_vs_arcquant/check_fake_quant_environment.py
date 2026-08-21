"""Verify CUDA, BF16, and the fake-NVFP4 path before an expensive run."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


BUNDLE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = BUNDLE_ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_rtn_reorder_arc import nvfp4_quantize  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--require-blackwell", action="store_true")
    return parser.parse_args()


def nvidia_smi() -> dict[str, str] | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        name, driver, memory_mib = [
            item.strip() for item in result.stdout.splitlines()[0].split(",")
        ]
        return {
            "name": name,
            "driver": driver,
            "memory_mib": memory_mib,
        }
    except (OSError, subprocess.CalledProcessError, IndexError, ValueError):
        return None


def version_tuple(raw: str | None) -> tuple[int, int]:
    if not raw:
        return (0, 0)
    parts = raw.split(".")
    return (int(parts[0]), int(parts[1]))


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is false")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    index = device.index or 0
    name = torch.cuda.get_device_name(index)
    capability = torch.cuda.get_device_capability(index)
    cuda_runtime = torch.version.cuda
    if args.require_blackwell and version_tuple(cuda_runtime) < (12, 8):
        raise RuntimeError(
            f"RTX 5090 fake quant requires a cu128-or-newer wheel; got {cuda_runtime}"
        )
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 is not supported by this PyTorch/GPU combination")
    if args.require_blackwell and capability < (12, 0):
        raise RuntimeError(
            f"Expected Blackwell compute capability >= 12.0, got {capability}"
        )

    torch.manual_seed(0)
    x = torch.randn(128, 256, device=device, dtype=torch.bfloat16)
    w = torch.randn(192, 256, device=device, dtype=torch.bfloat16)
    x_scale = x.float().abs().amax() / (448.0 * 6.0)
    w_scale = w.float().abs().amax() / (448.0 * 6.0)
    qx, _ = nvfp4_quantize(x.float(), global_scale=x_scale)
    qw, _ = nvfp4_quantize(w.float(), global_scale=w_scale)
    output = F.linear(qx.to(torch.bfloat16), qw.to(torch.bfloat16))
    torch.cuda.synchronize(device)
    if not torch.isfinite(output).all():
        raise AssertionError("fake-NVFP4 BF16 GEMM produced non-finite values")

    result = {
        "status": "passed",
        "torch": torch.__version__,
        "torch_cuda_runtime": cuda_runtime,
        "device": str(device),
        "gpu": name,
        "compute_capability": list(capability),
        "bf16_supported": torch.cuda.is_bf16_supported(),
        "torch_arch_list": torch.cuda.get_arch_list(),
        "nvidia_smi": nvidia_smi(),
        "fake_nvfp4_shape": list(output.shape),
        "fake_nvfp4_abs_mean": float(output.float().abs().mean().item()),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
