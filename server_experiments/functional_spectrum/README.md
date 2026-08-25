# Llama-3.1-8B Functional Spectrum server run

This directory runs the frozen Stage 1 and Stage 1.5 protocol on a local
Llama-3.1-8B Base checkpoint. It does not tune ranks, modules, splits, or gates
after seeing the Qwen result.

The seven cost-controlled targets are generated from the decoder depth:

- layer 0: `q_proj`, `k_proj`
- middle layer (`num_layers // 2`): `v_proj`, `o_proj`, `gate_proj`
- last layer: `up_proj`, `down_proj`

For a 32-layer Llama model this gives layers 0, 16, and 31. Stage 1.5 reuses
the exact Stage-1 operands, so its Split A/B samples and 512 retained rows are
identical.

## Run

```bash
cd /path/to/jx_jw_vs_arcquant
chmod +x server_experiments/functional_spectrum/run_llama31_8b.sh

MODEL_PATH=/root/autodl-tmp/models/Llama-3.1-8B \
WIKITEXT_CACHE_DIR=/root/autodl-tmp/cache/wikitext-2-raw-v1 \
OUTPUT_ROOT=/root/autodl-tmp/results/llama31_functional_spectrum \
CONDA_ENV_NAME=ptq \
bash server_experiments/functional_spectrum/run_llama31_8b.sh
```

The cache directory must contain `wikitext-train.arrow`. If the active shell
already has the desired environment, set `CONDA_ENV_NAME=` to skip activation.

## Outputs

```text
OUTPUT_ROOT/
├── preflight.json
├── stage1/
│   ├── operands/
│   ├── gram_artifacts/
│   ├── stage1_summary.csv
│   ├── stage1_gate_summary.json
│   ├── validation.json
│   └── STAGE1_REPORT.md
└── stage1_5/
    ├── stage1_5A_joint_spectrum.csv
    ├── stage1_5B_scaling_sweep.csv
    ├── stage1_5_gate_summary.json
    ├── validation.json
    ├── artifact_validation.json
    └── STAGE1_5_REPORT.md
```

Stage 1.5B runs only if Stage 1.5A returns `CONDITIONAL_SHAPING`. A header-only
scaling CSV and explicit `not_run` gate are the correct outputs for any direct
GO or `RAW_NO_GO` result.

## Matched-depth and per-head Stage-1 extension

The frozen seven-module Stage 1/1.5 result remains unchanged. To remove the
module-type/depth confound, the diagnostic extension captures all seven Linear
types at layers 0, 16, and 31 (21 modules for a 32-layer model). It also splits
every q/k/v weight into exact `head_dim` row blocks and analyzes each head on
the combined calibration split:

```bash
MODEL_PATH=/root/autodl-tmp/models/Llama-3.1-8B \
WIKITEXT_CACHE_DIR=/root/autodl-tmp/datasets/wikitext2_arrow \
OUTPUT_ROOT=/root/autodl-tmp/arcquant_runs/llama31_stage1_depth_control \
CONDA_ENV_NAME=ptq \
bash server_experiments/functional_spectrum/run_llama31_depth_control.sh
```

Additional outputs include:

```text
stage1_depth_control/
├── stage1_summary.csv
├── figures/figure_F_qkvo_depth_heatmaps_rank512.png
├── figures/figure_G_equal_fraction_heatmaps.png
└── head_analysis/
    ├── head_spectrum_summary.csv
    ├── head_module_summary.csv
    ├── head_analysis_summary.json
    └── figure_H_per_head_diagnostics.png
```

The equal-fraction figure uses rank 256 for K=4096 and rank 896 for the
K=14336 down projection. This extension is diagnostic and must not be mixed
with the preregistered seven-module gate.

## Solver and hardware notes

Llama-3.1-8B has K=4096 for attention/input projections and approximately
K=14336 for `down_proj`. The server command therefore uses deterministic
randomized top-512 subspace iteration for K >= 2048 and does not serialize full
K x K Stage-1 Grams. Stage 1.5 never materializes its full 2K x 2K joint Gram.
The matched-depth extension raises only large-K down projections to top-1024
so it can report the equal-fraction rank 896 point; K=4096 modules remain
top-512.

A CUDA GPU with at least 24 GiB is recommended; the supplied preflight fails
early below 20 GiB. These are offline fake-NVFP4 mechanism diagnostics and do
not establish Blackwell kernel throughput.
