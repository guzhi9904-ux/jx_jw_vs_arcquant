#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/models/Llama-3.1-8B}"
WIKITEXT_CACHE_DIR="${WIKITEXT_CACHE_DIR:-/root/autodl-tmp/datasets/wikitext2_arrow}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/arcquant_runs/llama31_full_depth_scan}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-ptq}"
DEVICE="${DEVICE:-cuda:0}"
HEAD_ANALYSIS_LAYERS="${HEAD_ANALYSIS_LAYERS:-0,4,8,12,16,20,24,28,31}"

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

# Phase A: all 32 Llama attention depths, with per-head work restricted to the
# preregistered nine-depth grid so the expensive head solver stays bounded.
printf '%s\n' "Phase A/2: full-depth q/k/v/o; per-head layers=${HEAD_ANALYSIS_LAYERS}"
python -m functional_gram.run_stage1 \
  --model "${MODEL_PATH}" \
  --wikitext-cache-dir "${WIKITEXT_CACHE_DIR}" \
  --output-root "${OUTPUT_ROOT}/stage1_attention_all" \
  --modules attention-all \
  --head-analysis \
  --head-analysis-layers "${HEAD_ANALYSIS_LAYERS}" \
  --device "${DEVICE}" \
  --randomized-min-k 2048 \
  --randomized-oversample 16 \
  --randomized-power-iterations 2 \
  --head-oversample 16 \
  --head-power-iterations 2 \
  --head-overlap-rank 32 \
  --gram-dtype float32 \
  --omit-full-gram \
  --quant-row-chunk 128

python -c 'import json, sys; data=json.load(open(sys.argv[1], encoding="utf-8")); assert data["status"] == "passed", data' \
  "${OUTPUT_ROOT}/stage1_attention_all/validation.json"

# Phase B: down_proj on the same nine-depth grid. Its K=14336 solver reports
# rank 896 (the 6.25% control) and rank 1024 in a separate output tree.
printf '%s\n' "Phase B/2: down_proj depth scan at the preregistered grid"
python -m functional_gram.run_stage1 \
  --model "${MODEL_PATH}" \
  --wikitext-cache-dir "${WIKITEXT_CACHE_DIR}" \
  --output-root "${OUTPUT_ROOT}/stage1_down_depth_scan" \
  --modules down-depth-scan \
  --device "${DEVICE}" \
  --randomized-min-k 2048 \
  --randomized-oversample 16 \
  --randomized-power-iterations 2 \
  --gram-dtype float32 \
  --omit-full-gram \
  --quant-row-chunk 128

python -c 'import json, sys; data=json.load(open(sys.argv[1], encoding="utf-8")); assert data["status"] == "passed", data' \
  "${OUTPUT_ROOT}/stage1_down_depth_scan/validation.json"

printf '%s\n' "Completed full-depth attention and down-projection scans: ${OUTPUT_ROOT}"
