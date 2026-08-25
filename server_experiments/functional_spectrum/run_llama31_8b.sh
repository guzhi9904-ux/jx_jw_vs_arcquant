#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/models/Llama-3.1-8B}"
WIKITEXT_CACHE_DIR="${WIKITEXT_CACHE_DIR:-/root/autodl-tmp/cache/wikitext-2-raw-v1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/results/llama31_functional_spectrum}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-ptq}"
DEVICE="${DEVICE:-cuda:0}"

if [[ -n "${CONDA_ENV_NAME}" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_NAME}"
fi

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export ARCQUANT_WIKITEXT_CACHE_DIR="${WIKITEXT_CACHE_DIR}"
mkdir -p "${OUTPUT_ROOT}"

python server_experiments/functional_spectrum/preflight.py \
  --model "${MODEL_PATH}" \
  --wikitext-cache-dir "${WIKITEXT_CACHE_DIR}" \
  --output "${OUTPUT_ROOT}/preflight.json"

python -m unittest discover -s functional_gram/tests -v
python -m unittest discover -s dual_source_spectral_shaping/tests -v

python -m functional_gram.run_stage1 \
  --model "${MODEL_PATH}" \
  --wikitext-cache-dir "${WIKITEXT_CACHE_DIR}" \
  --output-root "${OUTPUT_ROOT}/stage1" \
  --device "${DEVICE}" \
  --randomized-min-k 2048 \
  --randomized-oversample 16 \
  --randomized-power-iterations 2 \
  --gram-dtype float32 \
  --omit-full-gram \
  --quant-row-chunk 128

python -m dual_source_spectral_shaping.run_stage1_5 \
  --model "${MODEL_PATH}" \
  --operands-dir "${OUTPUT_ROOT}/stage1/operands" \
  --output-root "${OUTPUT_ROOT}/stage1_5" \
  --device "${DEVICE}" \
  --exact-max-dimension 2048 \
  --oversample 16 \
  --raw-power-iterations 2 \
  --scaling-power-iterations 2 \
  --quant-row-chunk 128

python -m dual_source_spectral_shaping.validate_stage1_5 \
  --output-root "${OUTPUT_ROOT}/stage1_5"

printf '%s\n' "Completed Stage 1 and Stage 1.5: ${OUTPUT_ROOT}"
