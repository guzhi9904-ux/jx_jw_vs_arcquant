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

局部比较会在完全相同的 selection/holdout 行上一次性计算论文 ARC、独立/共享双索引、单分支、随机和 oracle。端到端 PPL 默认执行 8 次完整模型遍历：BF16、RTN、论文 ARC 各一次，以及独立 `J_X/J_W` 的 5 个 selection seed。

## 目录边界

这个包是“操作和产物独立”，但有意复用仓库根目录中已经核实的 ARCQuant 量化、校准和 layer-wise PPL 实现，避免复制后让论文基线悄悄分叉。因此上传 Git 时要上传整个 ARCQuant 仓库，而不是只复制这个子目录。

关键入口如下：

```text
server_experiments/jx_jw_vs_arcquant/
├── configs/                 # Qwen2.5-7B、Llama-3.1-8B 固定配置
├── launch/                  # Linux 一键启动脚本
├── run_server.py            # 分阶段编排、断点续跑、日志
├── validate_seed.py         # 不写死 28 层/196 Linear 的完整性检查
├── aggregate_results.py     # 多 seed output-SSE/PPL 汇总
├── requirements-server.txt  # 与本地 ptq 环境对齐的核心版本
└── runs/                    # 运行时生成，已被 .gitignore 排除
```

## 环境与数据

本地已经核对过的核心版本是 Python 环境 `ptq`、PyTorch 2.5.1、Transformers 4.44.0、Datasets 3.4.0、Pandas 2.3.3、NumPy 2.2.6。服务器优先复用同版本环境：

```bash
conda activate ptq
pip install -r server_experiments/jx_jw_vs_arcquant/requirements-server.txt
```

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
- `aggregate`：只重新汇总已有结果。
- `all`：按上述顺序全部执行。

`--resume` 的行为是：完整阶段直接跳过；局部实验按已经完成的 module 续跑；PPL 按 layer checkpoint 续跑。日志始终写到 `<run-dir>/logs/`。

如需给 seed 0 补跑共享/单分支/随机 PPL，加上：

```bash
--with-ppl-diagnostics
```

如只想先看前 8 个 WikiText2 window，可加 `--max-windows 8`；这种 PPL 也只能做 smoke，不能和完整测试集数值混写。

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
├── logs/
└── summary/
    ├── REPORT.md
    ├── local_all_seeds.csv
    ├── local_strategy_aggregate.csv
    ├── local_key_comparisons.csv
    ├── ppl_runs.csv
    └── ppl_aggregate.csv
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
