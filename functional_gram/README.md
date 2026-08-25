# Stage 1 Functional Gram validation

This package implements the preregistered activation-side and weight-side
Functional Gram experiment without changing the repository NVFP4 quantizer.

The default run uses local Qwen2.5-1.5B-Instruct weights, while `--model`
also supports local sharded Hugging Face decoders such as Llama-3.1-8B. Both use two disjoint
WikiText2 calibration splits (4 x 2048 tokens each), 512 sampled rows per
split, and a cost-controlled early/middle/late set covering q/k/v/o/gate/up/down.

```powershell
conda run --no-capture-output -n ptq python -m functional_gram.run_stage1
```

For an 8B model, use the scalable solver and avoid serializing a multi-GB full
Gram:

```bash
python -m functional_gram.run_stage1 \
  --model /models/Meta-Llama-3.1-8B \
  --output-root analysis/functional_gram/llama31_8b \
  --randomized-min-k 2048 --gram-dtype float32 --omit-full-gram
```

The runner is resumable. Use `--skip-collection` to reuse saved operands and
`--modules` with a comma-separated list to expand the target set. Core outputs
are `stage1_summary.csv`, `stage1_gate_summary.json`, `validation.json`, source
spectral artifacts, Figure A--E PNGs, and the technical report. Full combined
Grams are included unless `--omit-full-gram` is supplied.

For the post-gate matched-depth diagnostic, `--modules depth-control` captures
q/k/v/o/gate/up/down at the early, middle, and final decoder layers. Add
`--head-analysis` to split q/k/v weights into their exact contiguous
`head_dim` row blocks. This writes per-head coverage, pairwise projector
overlap, and aggregate-subspace capture under `head_analysis/`.

Stage-1 summary ranks are dimension aware: K=4096 modules retain ranks through
512, while large-K modules additionally report ranks 896 and 1024. The
`equal_fraction_reference` row is 6.25% of K (rank 256 for K=4096 and rank 896
for K=14336). The preregistered GO/NO-GO gate remains rank 256.
