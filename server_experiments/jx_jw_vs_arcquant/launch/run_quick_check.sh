#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:?usage: run_quick_check.sh CONFIG_JSON MODEL_PATH [WIKITEXT_ARROW_DIR]}"
MODEL_PATH="${2:?usage: run_quick_check.sh CONFIG_JSON MODEL_PATH [WIKITEXT_ARROW_DIR]}"
WIKITEXT_CACHE="${3:-}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CONDA_ENV="${ARCQUANT_CONDA_ENV:-ptq}"

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"

CACHE_ARGS=()
if [[ -n "${WIKITEXT_CACHE}" ]]; then
  CACHE_ARGS+=(--wikitext-cache-dir "${WIKITEXT_CACHE}" --offline)
fi

python "${BUNDLE_DIR}/run_server.py" \
  --config "${CONFIG}" \
  --model "${MODEL_PATH}" \
  --stage all \
  --device "${ARCQUANT_DEVICE:-cuda:0}" \
  --quick \
  --resume \
  "${CACHE_ARGS[@]}"
