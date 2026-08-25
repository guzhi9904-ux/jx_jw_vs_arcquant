# Stage 1.6 — Factor rank and spectral-outlier analysis

This package implements the preregistered mechanism chain:

```text
factor spectrum -> Hadamard rank expansion -> spectral-outlier removal
```

It uses the existing audited NVFP4 reference and the exact decomposition
`XW - qXqW = (X-qX)W + qX(W-qW)`. In particular, the W-source activation
factor is `qX.T @ qX`, not `X.T @ X`.

## Local Qwen2.5-1.5B

Activate the requested environment and run the tests:

```powershell
conda activate ptq
python -m unittest discover -s factor_rank_spectral_outlier/tests -v
```

If the audited Stage-1 operands already exist, reuse them without loading the
model again:

```powershell
python -m factor_rank_spectral_outlier.run_stage1_6 `
  --model ..\modelzoo\Qwen\Qwen2.5-1.5B-Instruct `
  --operands-dir analysis\functional_gram\stage1_qwen25_15b\operands `
  --output-root analysis\factor_rank_spectral_outlier\qwen25_15b `
  --skip-collection
```

Without `--skip-collection`, the runner captures the preregistered seven
early/middle/late q/k/v/o/gate/up/down operands and quantizes them through
`fp4_residual_carrier.common.nvfp4_reference.quantize_nvfp4`.

For a short end-to-end environment check, select one real module and reduce
only the intervention grid. The resulting gate is deliberately labelled
`SMOKE_ONLY`:

```powershell
python -m factor_rank_spectral_outlier.run_stage1_6 `
  --model ..\modelzoo\Qwen\Qwen2.5-1.5B-Instruct `
  --operands-dir analysis\functional_gram\stage1_qwen25_15b\operands `
  --output-root analysis\factor_rank_spectral_outlier\qwen25_15b_smoke `
  --modules layers.0.self_attn.k_proj --skip-collection `
  --removal-budgets 16 --random-seeds 1 `
  --removal-exact-max-dimension 0 --randomized-power-iterations 0
```

## Llama-3.1-8B server run

Use the launcher in `server_experiments/factor_rank_spectral_outlier`. It keeps
the gate and experimental definition unchanged:

```bash
MODEL_PATH=/models/Meta-Llama-3.1-8B \
OUTPUT_ROOT=analysis/factor_rank_spectral_outlier/llama31_8b \
bash server_experiments/factor_rank_spectral_outlier/run_llama31_8b.sh
```

Set `OPERANDS_DIR` and `SKIP_COLLECTION=1` to reuse existing audited operands.
The runner is module-resumable through ignored `module_artifacts/*.pt` files.

## Numerical paths

- `K <= 4096`: FP64 Gram accumulation and exact `torch.linalg.eigh` for the raw
  spectrum.
- `K > 4096`: FP32 explicit Gram, randomized top-r subspace iteration, exact
  trace/participation rank, and labelled stochastic-Lanczos entropy-rank
  estimation.
- Factor spectra always use the smaller of `F F.T` and `F.T F`, so wide
  `down_proj` factors do not require a second K x K covariance.
- Removal spectra use exact eigendecomposition through K=2048 and the scalable
  top-256 solver above that threshold.

Qwen2.5-1.5B has `down_proj K=8960`; therefore it also exercises the scalable
path. Every CSV records the spectral and entropy method used per row.

## Outputs

The runner writes the four required tables, `stage1_6_gate_summary.json`, all
Figure A--G plots, validation details, a run manifest, and a technical report.
Two small additional tables hold complete plotting curves and Split-A/B
support Jaccard values. A noncanonical module/budget/seed selection can never
emit the formal GO/WEAK/NO-GO verdict; it emits `SMOKE_ONLY` plus a clearly
labelled provisional diagnostic.

Channel removal here is a mechanism intervention only. It does not claim that
the removed atoms have already been corrected by a runtime K+S kernel.
