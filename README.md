# J_X/J_W vs. ARCQuant

This repository is a frozen server-experiment version for comparing two NVFP4 residual-compensation designs:

- the original ARCQuant `activation-max reorder + ARC` method;
- an identity-layout dual-source method with independent offline indices `J_X/J_W` and a fixed `S/2 + S/2` budget.

It also includes BF16 and ordinary RTN baselines, shared-index/single-branch/random ablations, output-aware diagnostic ceilings, five-seed local output-SSE evaluation, and five-seed WikiText2 perplexity evaluation.

The code is based on the official [ARCQuant repository](https://github.com/actypedef/ARCQuant). Its original README is preserved in [ARCQUANT_UPSTREAM_README.md](ARCQUANT_UPSTREAM_README.md).

## Start here

For an RTX 5090, follow the complete Chinese setup and execution guide in [RTX5090_FAKE_QUANT_GUIDE.md](server_experiments/jx_jw_vs_arcquant/RTX5090_FAKE_QUANT_GUIDE.md). The algorithm and output layout are documented in [the experiment README](server_experiments/jx_jw_vs_arcquant/README.md).

PyTorch is installed separately because RTX 5090 needs a Blackwell-compatible CUDA 12.8 build:

```bash
conda create -n ptq python=3.10 -y
conda activate ptq
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r server_experiments/jx_jw_vs_arcquant/requirements-server.txt
python server_experiments/jx_jw_vs_arcquant/check_fake_quant_environment.py \
  --device cuda:0 --require-blackwell
```

Generated models, calibration tensors, checkpoints, logs, and run directories are excluded from Git.

## Scope

The W4A4 evaluations in this version use one shared fake-NVFP4 numerical backend. They are intended to compare quantization error and perplexity, not real Blackwell kernel latency. Rotation, online dynamic selection, score tuning, and new hardware-block selectors are intentionally outside this frozen v1.
