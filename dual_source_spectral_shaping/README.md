# Stage 1.5 — Dual-source spectrum and FP4 error consolidation

This package implements the preregistered Stage 1.5 experiment using the exact
decomposition `D = (X-qX)W + qX(W-qW)`.  It reuses the seven audited Stage-1
operand artifacts, keeps Split A/B disjoint, and calls the repository NVFP4
reference afresh for every non-identity scaling candidate.

```powershell
& 'D:\anaconda\shell\condabin\conda-hook.ps1'
conda activate ptq
python -m unittest discover -s dual_source_spectral_shaping/tests -v
python -m dual_source_spectral_shaping.run_stage1_5
python -m dual_source_spectral_shaping.validate_stage1_5
```

The full run writes the required CSVs, gate JSON, validation record, Figure
A--F PNGs, selected per-module scales, spectral artifacts, and a technical
report under `analysis/dual_source_spectral_shaping/stage1_5_qwen25_15b`.

Raw Grams below `--exact-max-dimension` use exact FP64 eigendecomposition.
Larger pair/joint diagnostics use deterministic block-randomized top-512
subspace iteration and record the Ritz residual. Scaling sweeps use the same
bounded top-512 solver, allowing the same code to cover Qwen2.5-1.5B and
Llama-3.1-8B without materializing a 2K x 2K joint matrix.
