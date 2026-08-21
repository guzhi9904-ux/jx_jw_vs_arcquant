# Source dependency manifest

The server bundle deliberately calls the verified repository implementation instead of maintaining a second copy of ARCQuant. Commit these files together with this directory:

```text
reorder_indices.py
utilize.py
model/quantize.py
model/kv_cache.py
scripts/analyze_proxy_joint_ks_holdout.py
scripts/analyze_rtn_error_ks.py
scripts/analyze_rtn_reorder_arc.py
scripts/evaluate_joint_ks_ppl.py
scripts/validate_proxy_split_disjoint.py
server_experiments/jx_jw_vs_arcquant/**
```

Why each update matters:

- `reorder_indices.py`: configurable artifact directory/device/cache and unchanged default `128 × 2048` calibration.
- `utilize.py`: avoids retaining every layer's unused concatenated weight copy on GPU during activation-max calibration.
- `model/quantize.py` and `model/kv_cache.py`: keep fake-NVFP4 analysis importable without a prebuilt `agemm` extension and preserve the corrected magnitude-based tensor scale.
- `analyze_proxy_joint_ks_holdout.py`: adds `server-comparison`, so paper ARC and frozen V2 use the same rows.
- `evaluate_joint_ks_ppl.py`: removes the Qwen-1.5B-only `196` Linear assertion and checks the actual model modules.
- The two imported analysis helpers provide the shared fake-NVFP4 backend and output-SSE scoring primitives.

`run_server.py --stage preflight` checks markers for these source updates and fails early if an incomplete Git upload restored an older file.
