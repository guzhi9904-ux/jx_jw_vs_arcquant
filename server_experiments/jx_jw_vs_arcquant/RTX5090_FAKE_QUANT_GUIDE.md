# RTX 5090：从 Conda 环境到完整 fake-quant 实验

这份流程只验证量化算法的数值效果：权重和激活会先在 PyTorch 中模拟成 NVFP4，再用 BF16 矩阵乘法计算输出。它能比较 output SSE 和 WikiText2 perplexity，但不能代表真实 NVFP4 内核的速度、显存占用或吞吐。

因此本阶段不需要编译 `agemm`、CUTLASS 或任何 CUDA 扩展，也不要运行仓库根目录的 `evaluate.sh`、`model/main.py` 或 `kernels/remake.sh`。

## 1. 租机器时怎么选

推荐配置：

- Ubuntu 22.04 或 24.04；
- RTX 5090 32 GB；
- NVIDIA 驱动不低于 570.26；
- CPU 内存至少 64 GB，推荐 96 GB；
- 至少 150 GB 持久化磁盘；
- 8 个以上 CPU 核。

RTX 5090 的计算能力是 12.0。这里使用 PyTorch 2.7.1 的 CUDA 12.8 wheel。fake quant 不编译 CUDA 代码，因此服务器不必另装 CUDA Toolkit，`nvcc` 是否存在也不重要；能正常工作的 NVIDIA 驱动才是关键。

登录机器后先确认：

```bash
nvidia-smi
```

如果驱动低于 570.26，优先换租带有新驱动的镜像，不建议在临时租用实例里手工升级驱动。

## 2. 拉取代码

下面假设持久化盘挂载在 `/data`。如果服务商给的是别的目录，只需统一替换 `/data`。

```bash
cd /data
git clone -b main https://github.com/guzhi9904-ux/jx_jw_vs_arcquant.git
cd /data/jx_jw_vs_arcquant
git rev-parse HEAD
```

后续所有命令都在 `/data/jx_jw_vs_arcquant` 执行。

## 3. 创建 Conda 环境

如果镜像已经安装 Miniconda：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda create -n ptq python=3.10 -y
conda activate ptq
python -m pip install --upgrade pip setuptools wheel
```

5090 的 PyTorch 必须单独安装，不能沿用旧环境中的 PyTorch 2.5.1：

```bash
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r server_experiments/jx_jw_vs_arcquant/requirements-server.txt
python -m pip check
```

本项目不使用 `torchvision` 和 `torchaudio`，所以不必安装它们。若镜像预装了其他版本的 PyTorch也没有关系，新建的 `ptq` 环境会与它隔离。

建议为缓存和运行结果使用持久化磁盘：

```bash
export HF_HOME=/data/hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
mkdir -p /data/hf_cache /data/models /data/datasets /data/arcquant_runs
```

每次重新连接服务器后，都要重新执行 `conda activate ptq` 和这些 `export`。也可以把它们放进自己的 shell 启动文件。

## 4. 验证 5090 和 fake-NVFP4 路径

```bash
python server_experiments/jx_jw_vs_arcquant/check_fake_quant_environment.py \
  --device cuda:0 \
  --require-blackwell
```

正常输出至少应包含：

```text
"status": "passed"
"torch": "2.7.1+cu128"
"torch_cuda_runtime": "12.8"
"compute_capability": [12, 0]
"bf16_supported": true
```

这个检查会真正完成一次小规模 fake-NVFP4 量化和 BF16 GEMM。如果出现 `no kernel image is available`，通常是装了不支持 5090 的旧 PyTorch wheel；删掉当前环境重建，确认安装命令最后是 `cu128`。如果 `torch.cuda.is_available()` 为 false，先检查 `nvidia-smi` 和租用镜像的驱动，而不是安装 `nvcc`。

## 5. 下载模型

建议先跑 Qwen2.5-7B，再跑 Llama-3.1-8B。基座模型比 Instruct 模型更适合做标准语言模型 PPL；同一组比较中不要混用 Base 和 Instruct。

```bash
huggingface-cli download Qwen/Qwen2.5-7B \
  --local-dir /data/models/Qwen2.5-7B
```

Llama-3.1-8B 是 gated 模型，需要先在 Hugging Face 页面接受许可并登录：

```bash
huggingface-cli login
huggingface-cli download meta-llama/Llama-3.1-8B \
  --local-dir /data/models/Meta-Llama-3.1-8B
```

下载完成后，实验始终传本地模型目录，不传 Hugging Face 仓库名。这样正式运行时不会因为断网或远端文件变化而中断。

## 6. 固定 WikiText2 数据缓存

联网状态下执行一次：

```bash
python server_experiments/jx_jw_vs_arcquant/prepare_wikitext2_cache.py \
  --output-dir /data/datasets/wikitext2_arrow
```

完成后应有：

```text
/data/datasets/wikitext2_arrow/wikitext-train.arrow
/data/datasets/wikitext2_arrow/wikitext-test.arrow
/data/datasets/wikitext2_arrow/wikitext-validation.arrow
/data/datasets/wikitext2_arrow/manifest.json
```

正式实验传入这个目录后会进入离线模式，不再访问 Hugging Face。

## 7. Qwen2.5-7B：正式运行前检查

先统一路径：

```bash
export PROJECT=/data/jx_jw_vs_arcquant
export MODEL=/data/models/Qwen2.5-7B
export WT2_CACHE=/data/datasets/wikitext2_arrow
export RUN_DIR=/data/arcquant_runs/qwen25_7b
cd "$PROJECT"
```

只做模型、数据、GPU、磁盘和依赖检查：

```bash
python server_experiments/jx_jw_vs_arcquant/run_server.py \
  --config server_experiments/jx_jw_vs_arcquant/configs/qwen25_7b.json \
  --model "$MODEL" \
  --run-dir "$RUN_DIR" \
  --wikitext-cache-dir "$WT2_CACHE" \
  --offline --device cuda:0 --stage preflight
```

然后做隔离的快速冒烟实验：

```bash
python server_experiments/jx_jw_vs_arcquant/run_server.py \
  --config server_experiments/jx_jw_vs_arcquant/configs/qwen25_7b.json \
  --model "$MODEL" \
  --run-dir "${RUN_DIR}_quick" \
  --wikitext-cache-dir "$WT2_CACHE" \
  --offline --device cuda:0 --quick --resume --stage all
```

`quick` 只验证流程能走通，数据量和 PPL window 都很小，不能放进论文表格，也不能与完整实验数值比较。

## 8. Qwen2.5-7B：完整实验

长任务建议放在 `tmux` 中：

```bash
tmux new -s qwen_arc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ptq
export HF_HOME=/data/hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
cd /data/jx_jw_vs_arcquant
```

推荐分四个阶段运行。这样每一步完成后都能先检查结果，服务器中断时也容易续跑。

### 阶段 A：128×2048 全量校准

```bash
python server_experiments/jx_jw_vs_arcquant/run_server.py \
  --config server_experiments/jx_jw_vs_arcquant/configs/qwen25_7b.json \
  --model /data/models/Qwen2.5-7B \
  --run-dir /data/arcquant_runs/qwen25_7b \
  --wikitext-cache-dir /data/datasets/wikitext2_arrow \
  --offline --device cuda:0 --resume --stage calibrate
```

### 阶段 B：5 个 seed 的局部 output-SSE 消融

```bash
python server_experiments/jx_jw_vs_arcquant/run_server.py \
  --config server_experiments/jx_jw_vs_arcquant/configs/qwen25_7b.json \
  --model /data/models/Qwen2.5-7B \
  --run-dir /data/arcquant_runs/qwen25_7b \
  --wikitext-cache-dir /data/datasets/wikitext2_arrow \
  --offline --device cuda:0 --resume --stage local
```

这一阶段比较论文 `reorder + ARC`、独立 `J_X/J_W`、共享 `J`、只补激活、只补权重、随机双集合和诊断 oracle。它回答“局部 output SSE 到底改善了多少”。

### 阶段 C：完整 WikiText2 PPL

```bash
python server_experiments/jx_jw_vs_arcquant/run_server.py \
  --config server_experiments/jx_jw_vs_arcquant/configs/qwen25_7b.json \
  --model /data/models/Qwen2.5-7B \
  --run-dir /data/arcquant_runs/qwen25_7b \
  --wikitext-cache-dir /data/datasets/wikitext2_arrow \
  --offline --device cuda:0 --resume --stage ppl
```

默认 PPL 主实验包含 BF16、普通 RTN、论文 ARC 各一次，以及独立 `J_X/J_W` 的 5 个 selection seed。不要给正式 PPL 加 `--max-windows`；它会把完整测试集截断成 smoke 结果。

如主实验完成后还想给共享 `J`、只补激活、只补权重和随机集合补跑 seed 0 的 PPL，再执行：

```bash
python server_experiments/jx_jw_vs_arcquant/run_server.py \
  --config server_experiments/jx_jw_vs_arcquant/configs/qwen25_7b.json \
  --model /data/models/Qwen2.5-7B \
  --run-dir /data/arcquant_runs/qwen25_7b \
  --wikitext-cache-dir /data/datasets/wikitext2_arrow \
  --offline --device cuda:0 --resume --stage ppl --with-ppl-diagnostics
```

### 阶段 D：汇总

```bash
python server_experiments/jx_jw_vs_arcquant/run_server.py \
  --config server_experiments/jx_jw_vs_arcquant/configs/qwen25_7b.json \
  --model /data/models/Qwen2.5-7B \
  --run-dir /data/arcquant_runs/qwen25_7b \
  --wikitext-cache-dir /data/datasets/wikitext2_arrow \
  --offline --device cuda:0 --resume --stage aggregate
```

也可以用 `--stage all` 一次跑完，但分阶段更容易核查。

## 9. 中断后怎么续跑

所有正式命令都保留 `--resume`。SSH 断开或实例重启后，重新激活环境、恢复环境变量，然后重跑同一条命令即可：

- 完成的整个阶段会跳过；
- 局部实验会按已完成的 module 继续；
- PPL 会按 layer checkpoint 继续；
- 不要删除原来的 `RUN_DIR`，也不要换模型目录。

查看进度：

```bash
nvidia-smi -l 2
ls -lt /data/arcquant_runs/qwen25_7b/logs
tail -f /data/arcquant_runs/qwen25_7b/logs/*.log
```

退出 tmux 而不停止任务：按 `Ctrl-b`，再按 `d`。回来时：

```bash
tmux attach -t qwen_arc
```

## 10. 结果在哪里

首先看：

```text
/data/arcquant_runs/qwen25_7b/summary/REPORT.md
/data/arcquant_runs/qwen25_7b/summary/local_strategy_aggregate.csv
/data/arcquant_runs/qwen25_7b/summary/local_key_comparisons.csv
/data/arcquant_runs/qwen25_7b/summary/ppl_aggregate.csv
```

原始结果还包括每个 seed 的选择索引、每层 output-SSE 和每次 PPL 的 JSON。`run_manifest.json` 会记录配置、模型路径、软件版本和 Git 状态，回传结果时不要漏掉。

建议任务结束后打包整个运行目录；已完成阶段的大 checkpoint 会自动清理，命令仍额外排除残留 checkpoint：

```bash
tar --exclude='checkpoint_*.pt' -czf /data/qwen25_7b_fake_quant_results.tar.gz \
  -C /data/arcquant_runs qwen25_7b
```

## 11. 再跑 Llama-3.1-8B

Qwen 全流程确认无误后，再把模型、配置和运行目录替换为：

```text
模型：/data/models/Meta-Llama-3.1-8B
配置：server_experiments/jx_jw_vs_arcquant/configs/llama31_8b.json
结果：/data/arcquant_runs/llama31_8b
```

其余 `preflight → quick → calibrate → local → ppl → aggregate` 流程完全相同。单张 5090 上不要同时跑 Qwen 和 Llama。

## 12. 时间和资源预期

这套实现优先保证算法可核查，不是优化过的 GPU 内核。5090 的计算很快，但 Python 调度、CPU 内存搬运和多次完整 PPL 仍会占主要时间。初步租卡时可以按下面预留：

- quick：约 10–30 分钟；
- 128×2048 校准：约 2–6 小时；
- 5 seed 局部消融：约 2–6 小时；
- 默认 8 次完整 PPL：约 8–20 小时；
- 单模型合计：约 12–30 小时。

这是保守估算，不是性能承诺。先记录 quick 和校准前几层的实际速度，再决定续租时长。CPU 内存不足会比 GPU 算力更早成为问题，因此宁可多租内存，也不要为了省钱选 32 GB 系统内存。

## 13. 常见问题

- `No module named agemm`：这条 fake-quant 路径不应依赖真实内核。先 `git pull` 确认使用最新代码，不要转去编译 kernel。
- `CUDA out of memory`：确认只有一个实验进程；先不加 `--with-ppl-diagnostics`；不要通过缩短正式 PPL window 来制造可比较结果。
- 系统内存或进程被杀：提高租用机 CPU RAM/Swap，保持 `--resume` 原目录续跑。
- 找不到 WikiText Arrow：重新运行 `prepare_wikitext2_cache.py`，不要把 Hugging Face 的内部 cache 目录直接当成这里需要的固定 Arrow 目录。
- Llama 下载 401/403：先在模型页面接受许可，再执行 `huggingface-cli login`。
- 离线模式找不到模型：检查传入的是完整本地目录，且 tokenizer、config 和所有权重 shard 都已下载。
- 结果目录已经有另一模型的数据：换一个全新的 `RUN_DIR`，不要混写。

## 14. 官方环境依据

- NVIDIA 的 GPU 计算能力表：<https://developer.nvidia.com/cuda/gpus>
- PyTorch 2.7.1 CUDA 12.8 安装命令：<https://pytorch.org/get-started/previous-versions/>
- CUDA 12.8 对应的最低 Linux 驱动版本：<https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html>
