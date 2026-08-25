# Stage 1.5 reproduction

Environment and model:

```powershell
conda activate ptq
$env:MODEL_PATH = 'C:\path\to\local\model'
python -m dual_source_spectral_shaping.run_stage1_5 `
  --model $env:MODEL_PATH
```

The runner deliberately expects exactly the seven operands from the completed
Stage-1 preregistration.  Stage 1.5B runs only when the raw gate is
`CONDITIONAL_SHAPING`, unless `--force-stage1-5b` is explicitly supplied for a
diagnostic run.  Candidate selection uses Split A only; the selected diagonal
is then applied unchanged to Split B.
