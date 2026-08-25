#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to the local Meta-Llama-3.1-8B directory}"

OUTPUT_ROOT="${OUTPUT_ROOT:-analysis/factor_rank_spectral_outlier/llama31_8b}"
OPERANDS_DIR="${OPERANDS_DIR:-${OUTPUT_ROOT}/operands}"
DEVICE="${DEVICE:-cuda:0}"
CPU_THREADS="${CPU_THREADS:-0}"

args=(
  python -m factor_rank_spectral_outlier.run_stage1_6
  --model "${MODEL_PATH}"
  --output-root "${OUTPUT_ROOT}"
  --operands-dir "${OPERANDS_DIR}"
  --device "${DEVICE}"
  --analysis-device "${DEVICE}"
  --exact-max-dimension 4096
  --removal-exact-max-dimension 2048
  --bulk-rank 256
  --removal-budgets 16,32,64,128,256
  --random-seeds 10
  --randomized-oversample 16
  --randomized-power-iterations 1
  --slq-probes 8
  --slq-steps 32
  --cpu-threads "${CPU_THREADS}"
)

if [[ "${SKIP_COLLECTION:-0}" == "1" ]]; then
  args+=(--skip-collection)
fi

"${args[@]}"
