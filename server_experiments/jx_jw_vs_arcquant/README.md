# J_X/J_W vs. ARCQuant 服务器实验包

这个目录是服务器实验的唯一操作入口。模型、校准缓存、日志和结果都写入独立的 `runs/<model>/`，不会再读取本地已有的 `saved/*.pt`，也默认不会被 Git 收进去。

当前版本冻结算法，不再调分数。它回答两个问题：

1. 双来源残差补偿是否稳定优于论文原版 `reorder + ARC`、只补激活和只补权重？
2. 独立 `J_X/J_W` 是否稳定优于共享 `J`，差距能否达到值得作为论文中心的约 3–5 个百分点，并传递到 PPL？

## 固定实验方法

| 名称 | 大白话 | 是否用于主结论 |
|---|---|---|
| `bf16` | 不量化，用来给 PPL 提供上界 | 是 |
| `rtn_identity` | 所有激活和权重直接做普通 NVFP4 RTN，不重排、不补偿 | 是 |
| `paper_reorder_arc` | 按激活最大值重排，把论文选中的尾部通道作为 ARC 补偿列 | 是，原论文基线 |
| `proxy_fixed_half` | 不重排；用离线统计分别选 `J_X` 和 `J_W`，各占 `S/2` | 是，我们的冻结 V2 |
| `proxy_shared` | 只选 `S/2` 个共享通道，每个通道同时放激活、权重两种残差 | 局部主消融，PPL 可选 |
| `activation_only` / `weight_only` | 全部预算只给一个残差来源 | 局部主消融，PPL 可选 |
| `random_fixed_half` | 激活、权重各随机选 `S/2` | sanity check |
| output-aware oracle | 直接看 selection 行的 output SSE 再选通道 | 只估计上限，不是算法 |

`J_X` 是“哪些输入通道值得补激活量化残差”的离线索引；`J_W` 是“哪些输入通道值得补权重量化残差”的离线索引。两者都由校准集统计后固定，prefill/decode 时不重新选通道。

局部比较会在完全相同的 selection/holdout 行上一次性计算论文 ARC、独立/共享双索引、单分支、随机和 oracle。端到端 PPL 默认执行 8 次完整模型遍历：BF16、RTN、论文 ARC 各一次，以及独立 `J_X/J_W` 的 5 个 selection seed。下游任务单独作为 `tasks` 阶段运行，默认比较 BF16、RTN、论文 ARC 和 seed 0 的独立 `J_X/J_W`；需要判断方差时再扩成 5 个 seed。

## 目录边界

这个包是“操作和产物独立”，但有意复用仓库根目录中已经核实的 ARCQuant 量化、校准和 layer-wise PPL 实现，避免复制后让论文基线悄悄分叉。因此上传 Git 时要上传整个 ARCQuant 仓库，而不是只复制这个子目录。

关键入口如下：

```text
server_experiments/jx_jw_vs_arcquant/
├── configs/                 # Qwen2.5-7B、Llama-3.1-8B 固定配置
├── launch/                  # Linux 一键启动脚本
├── RTX5090_FAKE_QUANT_GUIDE.md # 5090 从建环境到完整实验的逐步指南
├── check_fake_quant_environment.py # Blackwell/PyTorch/fake-NVFP4 自检
├── prepare_wikitext2_cache.py # 生成可复用的离线 WikiText2 Arrow
├── run_server.py            # 分阶段编排、断点续跑、日志
├── validate_seed.py         # 不写死 28 层/196 Linear 的完整性检查
├── aggregate_results.py     # 多 seed output-SSE/PPL/下游任务汇总
├── requirements-server.txt  # 与本地 ptq 环境对齐的核心版本
└── runs/                    # 运行时生成，已被 .gitignore 排除
```

## 环境与数据

本地复现实验核对过 PyTorch 2.5.1，但 RTX 5090 不能照搬这个旧版本。5090 服务器要单独安装支持 Blackwell 的 PyTorch 2.7.1 CUDA 12.8 wheel，再装其余依赖：

```bash
conda create -n ptq python=3.10 -y
conda activate ptq
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r server_experiments/jx_jw_vs_arcquant/requirements-server.txt
python server_experiments/jx_jw_vs_arcquant/check_fake_quant_environment.py \
  --device cuda:0 --require-blackwell
```

5090 从租机、下载模型和数据，到分阶段运行、续跑和回传结果的完整命令见 [`RTX5090_FAKE_QUANT_GUIDE.md`](RTX5090_FAKE_QUANT_GUIDE.md)。本阶段只做 fake quant，不需要系统 CUDA Toolkit、`nvcc` 或真实 NVFP4 内核。

模型必须是服务器上的本地 Hugging Face 目录。配置文件故意不保存机器路径，通过 `--model` 或 `ARCQUANT_MODEL` 传入。

推荐把 WikiText2 的 Arrow 缓存也放到服务器固定目录，里面至少有：

```text
wikitext-train.arrow
wikitext-test.arrow
```

传入这个目录后会自动开启 Hugging Face/Transformers 离线模式，避免一次实验中数据版本或网络状态变化。

## 推荐运行顺序

先做一个隔离的快速检查。它只用 2 条、128 token 和 2 个 PPL window，结果不能进论文：

```bash
bash server_experiments/jx_jw_vs_arcquant/launch/run_quick_check.sh \
  server_experiments/jx_jw_vs_arcquant/configs/qwen25_7b.json \
  /models/Qwen2.5-7B-Instruct \
  /data/wikitext2_arrow
```

快速检查通过后启动完整 Qwen2.5-7B。完整校准严格使用 `128 × 2048`，ARC 宽度搜索使用论文代码原本的 32 条样本：

```bash
bash server_experiments/jx_jw_vs_arcquant/launch/run_qwen25_7b.sh \
  /models/Qwen2.5-7B-Instruct \
  /data/wikitext2_arrow \
  /data/arcquant_runs/qwen25_7b
```

再用同样设置跑 Llama-3.1-8B：

```bash
bash server_experiments/jx_jw_vs_arcquant/launch/run_llama31_8b.sh \
  /models/Meta-Llama-3.1-8B \
  /data/wikitext2_arrow \
  /data/arcquant_runs/llama31_8b
```

两个启动脚本都会执行 `conda activate ptq`。可以用 `ARCQUANT_CONDA_ENV` 换环境，用 `ARCQUANT_DEVICE=cuda:1` 换 GPU。

## 分阶段运行和断点续跑

一键脚本内部等价于：

```bash
python server_experiments/jx_jw_vs_arcquant/run_server.py \
  --config server_experiments/jx_jw_vs_arcquant/configs/qwen25_7b.json \
  --model /models/Qwen2.5-7B-Instruct \
  --run-dir /data/arcquant_runs/qwen25_7b \
  --wikitext-cache-dir /data/wikitext2_arrow \
  --offline --resume --stage all
```

`--stage` 可以取：

- `preflight`：只核对模型架构、GPU、依赖文件、数据缓存和磁盘。
- `calibrate`：论文 ARCQuant 的 `128 × 2048` 激活统计、重排索引和 `select_num`。
- `local`：5 个 seed 的局部 output-SSE 比较，生成各自 `J_X/J_W`。
- `ppl`：BF16/RTN/论文 ARC，加 5 个 `J_X/J_W` seed 的 WikiText2 PPL。
- `tasks`：论文对齐的五项 zero-shot 任务和 5-shot MMLU。
- `aggregate`：只重新汇总已有结果。
- `all`：运行校准、局部实验、PPL 和汇总；有意不自动运行耗时更长、首次需要下载数据的 `tasks`。

`--resume` 的行为是：完整阶段直接跳过；局部实验按已经完成的 module 续跑；PPL 按 layer checkpoint 续跑。日志始终写到 `<run-dir>/logs/`。

如需给 seed 0 补跑共享/单分支/随机 PPL，加上：

```bash
--with-ppl-diagnostics
```

如只想先看前 8 个 WikiText2 window，可加 `--max-windows 8`；这种 PPL 也只能做 smoke，不能和完整测试集数值混写。

## 论文对齐的下游任务

任务口径固定为 ARC-Challenge、HellaSwag、LAMBADA、PIQA、Winogrande 的 zero-shot 准确率，以及 MMLU 5-shot。五任务平均分别使用 ARC/HellaSwag/PIQA 的 normalized accuracy 和 LAMBADA/Winogrande 的 accuracy，和论文表格保持同一口径。评测依赖固定为 `lm-eval==0.4.8`。

第一次运行要允许 lm-eval 把数据集写入持久化的 `HF_HOME`。先做每个任务最多 2 条样本的流程检查；MMLU 的 limit 是“每个子任务 2 条”，所以仍会遍历所有 MMLU 子类：

```bash
export HF_HOME=/root/autodl-tmp/hf_cache
python server_experiments/jx_jw_vs_arcquant/run_server.py \
  --config server_experiments/jx_jw_vs_arcquant/configs/llama31_8b.json \
  --model /root/autodl-tmp/jx_jw_vs_arcquant/Llama-3.1-8B \
  --run-dir /root/autodl-tmp/arcquant_runs/llama31_8b \
  --wikitext-cache-dir /root/autodl-tmp/datasets/wikitext2_arrow \
  --offline --device cuda:0 --resume --stage tasks \
  --task-limit 2 --allow-task-downloads
```

确认 smoke 完成后，去掉 limit 跑完整主比较。数据已进入 `HF_HOME` 后不再需要 `--allow-task-downloads`：

```bash
python server_experiments/jx_jw_vs_arcquant/run_server.py \
  --config server_experiments/jx_jw_vs_arcquant/configs/llama31_8b.json \
  --model /root/autodl-tmp/jx_jw_vs_arcquant/Llama-3.1-8B \
  --run-dir /root/autodl-tmp/arcquant_runs/llama31_8b \
  --wikitext-cache-dir /root/autodl-tmp/datasets/wikitext2_arrow \
  --offline --device cuda:0 --resume --stage tasks --task-limit 0
```

默认只跑 seed 0 的独立 `J_X/J_W`，先验证方法方向。若要给独立方法报告多 seed 均值，增加 `--task-seeds 0,1,2,3,4`；若要同时补跑共享 `J`，增加 `--with-task-diagnostics`。`--task-limit` 为正数的结果会明确标记 `full_evaluation=false`，不能和正式结果混用。每个方法的 zero-shot 与 MMLU 原始结果分开保存，因此中断后加 `--resume` 会从未完成的套件继续。

## 结果目录

```text
<run-dir>/
├── preflight.json
├── run_manifest.json
├── artifacts/                         # 本次运行独有 ARC 校准产物
├── local/seed_0..4/
│   ├── selection_indices.pt           # 含独立 J_X/J_W、共享 J 等索引
│   ├── strategy_summary.csv
│   ├── module_type_summary.csv
│   ├── server_validation.json
│   └── split_validation.json
├── ppl/
│   ├── baselines/
│   └── dual_seed_0..4/
├── tasks/
│   ├── baselines/
│   ├── dual_seed_0..4/
│   └── diagnostics_seed_0/            # 仅在显式要求时生成
├── logs/
└── summary/
    ├── REPORT.md
    ├── local_all_seeds.csv
    ├── local_strategy_aggregate.csv
    ├── local_key_comparisons.csv
    ├── ppl_runs.csv
    ├── ppl_aggregate.csv
    ├── tasks_runs.csv
    ├── tasks_task_metrics.csv
    └── tasks_aggregate.csv
```

每个 run 都保存 Git commit/dirty 状态、最终配置、模型路径、GPU 和软件版本。服务器回传时至少保留整个 `summary/`、各 seed 的 `strategy_summary.csv`/`metadata.json`/验证 JSON，以及所有 PPL result JSON；大体积 checkpoint 不需要回传。

## 口径和限制

- 全部 W4A4 结果使用同一个 fake NVFP4 数值后端，适合判断算法误差和 PPL，不是 Blackwell kernel 的真实速度/显存测试。
- 当前没有 rotation；论文基线的 reorder 只属于 ARCQuant 自己的通道重排。
- `paper_reorder_arc` 使用论文的 activation-max 排序、尾部 ARC 通道和同一 `select_num` 预算。
- 独立 `J_X/J_W` 保持原通道顺序，把总预算固定拆成 `S/2 + S/2`，没有在线贪心选列。
- output-aware oracle 读取 output residual，只能当诊断上限，不能当部署方法。
- 建议服务器至少 24 GiB GPU、64 GiB CPU RAM。代码逐层搬模型，但完整 PPL 会在 CPU 保存多组 hidden states，7/8B 全窗口时不适合低内存机器。

## Git 上传

`runs/`、模型权重、`.pt` 和日志已经在本目录 `.gitignore` 中排除。上传前建议先检查：

```bash
git status --short
git check-ignore -v server_experiments/jx_jw_vs_arcquant/runs/test.pt
```

不要把本地模型路径写回两个 JSON 配置；服务器路径继续通过命令行传入即可。
