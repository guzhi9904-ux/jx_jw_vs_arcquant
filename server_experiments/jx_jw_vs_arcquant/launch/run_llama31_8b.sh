#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${1:?usage: run_llama31_8b.sh MODEL_PATH [WIKITEXT_ARROW_DIR] [RUN_DIR]}"
WIKITEXT_CACHE="${2:-}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
RUN_DIR="${3:-${BUNDLE_DIR}/runs/llama31_8b}"
CONDA_ENV="${ARCQUANT_CONDA_ENV:-ptq}"

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"

CACHE_ARGS=()
if [[ -n "${WIKITEXT_CACHE}" ]]; then
  CACHE_ARGS+=(--wikitext-cache-dir "${WIKITEXT_CACHE}" --offline)
fi

python "${BUNDLE_DIR}/run_server.py" \
  --config "${BUNDLE_DIR}/configs/llama31_8b.json" \
  --model "${MODEL_PATH}" \
  --run-dir "${RUN_DIR}" \
  --stage all \
  --device "${ARCQUANT_DEVICE:-cuda:0}" \
  --resume \
  "${CACHE_ARGS[@]}"
