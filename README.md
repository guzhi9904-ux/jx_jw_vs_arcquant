# J_X/J_W vs. ARCQuant

This repository is a frozen server-experiment version for comparing two NVFP4 residual-compensation designs:

- the original ARCQuant `activation-max reorder + ARC` method;
- an identity-layout dual-source method with independent offline indices `J_X/J_W` and a fixed `S/2 + S/2` budget.

It also includes BF16 and ordinary RTN baselines, shared-index/single-branch/random ablations, output-aware diagnostic ceilings, five-seed local output-SSE evaluation, and five-seed WikiText2 perplexity evaluation.

The code is based on the official [ARCQuant repository](https://github.com/actypedef/ARCQuant). Its original README is preserved in [ARCQUANT_UPSTREAM_README.md](ARCQUANT_UPSTREAM_README.md).

## Start here

The complete Chinese experiment guide is in [server_experiments/jx_jw_vs_arcquant/README.md](server_experiments/jx_jw_vs_arcquant/README.md).

Activate the aligned environment and run a small non-paper smoke test first:

```bash
conda activate ptq
pip install -r server_experiments/jx_jw_vs_arcquant/requirements-server.txt

bash server_experiments/jx_jw_vs_arcquant/launch/run_quick_check.sh \
  server_experiments/jx_jw_vs_arcquant/configs/qwen25_7b.json \
  /models/Qwen2.5-7B-Instruct \
  /data/wikitext2_arrow
```

Then launch the complete `128 × 2048` Qwen2.5-7B experiment:

```bash
bash server_experiments/jx_jw_vs_arcquant/launch/run_qwen25_7b.sh \
  /models/Qwen2.5-7B-Instruct \
  /data/wikitext2_arrow \
  /data/arcquant_runs/qwen25_7b
```

Llama-3.1-8B has an equivalent launcher:

```bash
bash server_experiments/jx_jw_vs_arcquant/launch/run_llama31_8b.sh \
  /models/Meta-Llama-3.1-8B \
  /data/wikitext2_arrow \
  /data/arcquant_runs/llama31_8b
```

Generated models, calibration tensors, checkpoints, logs, and run directories are excluded from Git.

## Scope

The W4A4 evaluations in this version use one shared fake-NVFP4 numerical backend. They are intended to compare quantization error and perplexity, not real Blackwell kernel latency. Rotation, online dynamic selection, score tuning, and new hardware-block selectors are intentionally outside this frozen v1.
