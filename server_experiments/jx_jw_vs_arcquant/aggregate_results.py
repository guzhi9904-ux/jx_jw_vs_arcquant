"""Aggregate local output-SSE, PPL, and downstream-task server results."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


FRIENDLY = {
    "bf16": "BF16",
    "rtn_identity": "普通 RTN",
    "proxy_fixed_half": "独立 J_X/J_W（各 S/2）",
    "proxy_shared": "共享 J（同一通道双残差）",
    "paper_reorder_arc": "论文 reorder + ARC",
    "proxy_fixed_half_train": "独立 J_X/J_W（各 S/2）",
    "proxy_shared_train": "共享 J（同一通道双残差）",
    "activation_energy_train": "只补激活残差",
    "weight_energy_train": "只补权重残差",
    "random_fixed_half": "随机双集合",
    "oracle_independent_fixed_half_train": "output-aware oracle（selection）",
    "oracle_shared_train": "共享 output-aware oracle（selection）",
    "oracle_independent_fixed_half_holdout_leaked": "oracle（holdout 泄漏上限）",
    "oracle_shared_holdout_leaked": "共享 oracle（holdout 泄漏上限）",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    return parser.parse_args()


def seed_from_name(name: str) -> int | None:
    match = re.search(r"seed_(\d+)", name)
    return int(match.group(1)) if match else None


def percent(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:.2f}%"


def ppl_text(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def accuracy_text(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:.2f}%"


def main() -> None:
    run_dir = Path(parse_args().run_dir)
    summary_dir = run_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    local_rows: list[dict] = []
    for path in sorted((run_dir / "local").glob("seed_*/strategy_summary.csv")):
        seed = seed_from_name(path.parent.name)
        frame = pd.read_csv(path)
        for row in frame.to_dict(orient="records"):
            row["seed"] = seed
            row["analysis_dir"] = str(path.parent.resolve())
            row["method_cn"] = FRIENDLY.get(row["strategy"], row["strategy"])
            local_rows.append(row)

    local_frame = pd.DataFrame(local_rows)
    local_aggregate = pd.DataFrame()
    comparisons = pd.DataFrame()
    if not local_frame.empty:
        local_frame.to_csv(summary_dir / "local_all_seeds.csv", index=False)
        local_aggregate = (
            local_frame.groupby(["strategy", "method_cn"], as_index=False)
            .agg(
                seeds=("seed", "nunique"),
                train_gain_mean=("train_gain", "mean"),
                train_gain_std=("train_gain", "std"),
                holdout_gain_mean=("holdout_gain", "mean"),
                holdout_gain_std=("holdout_gain", "std"),
                holdout_gain_min=("holdout_gain", "min"),
                holdout_gain_max=("holdout_gain", "max"),
            )
            .fillna(0.0)
            .sort_values("holdout_gain_mean", ascending=False)
        )
        local_aggregate.to_csv(
            summary_dir / "local_strategy_aggregate.csv", index=False
        )
        pivot = local_frame.pivot(
            index="seed", columns="strategy", values="holdout_gain"
        )
        needed = {
            "proxy_fixed_half_train",
            "paper_reorder_arc",
            "proxy_shared_train",
        }
        if needed.issubset(pivot.columns):
            comparisons = pd.DataFrame(
                {
                    "seed": pivot.index,
                    "independent_jx_jw": pivot["proxy_fixed_half_train"],
                    "paper_reorder_arc": pivot["paper_reorder_arc"],
                    "shared_j": pivot["proxy_shared_train"],
                }
            ).reset_index(drop=True)
            comparisons["independent_minus_paper_pp"] = 100.0 * (
                comparisons.independent_jx_jw - comparisons.paper_reorder_arc
            )
            comparisons["independent_minus_shared_pp"] = 100.0 * (
                comparisons.independent_jx_jw - comparisons.shared_j
            )
            comparisons.to_csv(
                summary_dir / "local_key_comparisons.csv", index=False
            )

    ppl_rows: list[dict] = []
    for path in sorted((run_dir / "ppl").glob("**/result_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["result_file"] = str(path.resolve())
        payload["scope"] = path.parent.name
        payload["selection_seed"] = seed_from_name(path.parent.name)
        ppl_rows.append(payload)
    ppl_frame = pd.DataFrame(ppl_rows)
    ppl_aggregate = pd.DataFrame()
    if not ppl_frame.empty:
        ppl_frame.to_csv(summary_dir / "ppl_runs.csv", index=False)
        ppl_aggregate = (
            ppl_frame.groupby("method", as_index=False)
            .agg(
                runs=("perplexity", "count"),
                perplexity_mean=("perplexity", "mean"),
                perplexity_std=("perplexity", "std"),
                perplexity_min=("perplexity", "min"),
                perplexity_max=("perplexity", "max"),
                windows=("windows", "min"),
            )
            .fillna(0.0)
            .sort_values("perplexity_mean")
        )
        ppl_aggregate.to_csv(summary_dir / "ppl_aggregate.csv", index=False)

    task_rows: list[dict] = []
    task_metric_rows: list[dict] = []
    for path in sorted((run_dir / "tasks").glob("**/result_tasks_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "completed":
            continue
        row = {
            key: value
            for key, value in payload.items()
            if key not in {"task_metrics", "environment"}
        }
        row["result_file"] = str(path.resolve())
        row["scope"] = path.parent.name
        row["method_cn"] = FRIENDLY.get(payload["method"], payload["method"])
        for task, metric in payload.get("task_metrics", {}).items():
            row[f"task_{task}"] = metric["value"]
            task_metric_rows.append(
                {
                    "method": payload["method"],
                    "method_cn": row["method_cn"],
                    "selection_seed": payload.get("selection_seed"),
                    "task": task,
                    "metric": metric["metric"],
                    "value": metric["value"],
                    "num_fewshot": metric.get("num_fewshot", 0),
                    "full_evaluation": payload.get("full_evaluation"),
                    "limit": payload.get("limit"),
                    "result_file": str(path.resolve()),
                }
            )
        task_rows.append(row)

    task_frame = pd.DataFrame(task_rows)
    task_aggregate = pd.DataFrame()
    if not task_frame.empty:
        task_frame.to_csv(summary_dir / "tasks_runs.csv", index=False)
        pd.DataFrame(task_metric_rows).to_csv(
            summary_dir / "tasks_task_metrics.csv", index=False
        )
        metric_columns = [
            "paper_zero_shot_average",
            "mmlu_5shot_accuracy",
            *sorted(
                column for column in task_frame.columns if column.startswith("task_")
            ),
        ]
        aggregation: dict[str, tuple[str, str]] = {
            "runs": ("method", "count"),
        }
        for column in metric_columns:
            if column not in task_frame:
                continue
            aggregation[f"{column}_mean"] = (column, "mean")
            aggregation[f"{column}_std"] = (column, "std")
            aggregation[f"{column}_min"] = (column, "min")
            aggregation[f"{column}_max"] = (column, "max")
        task_aggregate = (
            task_frame.groupby(
                ["method", "method_cn", "full_evaluation", "limit"],
                as_index=False,
            )
            .agg(**aggregation)
            .fillna(0.0)
            .sort_values("paper_zero_shot_average_mean", ascending=False)
        )
        task_aggregate.to_csv(
            summary_dir / "tasks_aggregate.csv", index=False
        )

    def local_value(strategy: str) -> float | None:
        if local_aggregate.empty:
            return None
        rows = local_aggregate[local_aggregate.strategy == strategy]
        return None if rows.empty else float(rows.iloc[0].holdout_gain_mean)

    def ppl_value(method: str) -> float | None:
        if ppl_aggregate.empty:
            return None
        rows = ppl_aggregate[ppl_aggregate.method == method]
        return None if rows.empty else float(rows.iloc[0].perplexity_mean)

    def task_value(method: str, column: str) -> float | None:
        if task_aggregate.empty:
            return None
        rows = task_aggregate[
            (task_aggregate.method == method)
            & (task_aggregate.full_evaluation == True)  # noqa: E712
        ]
        return None if rows.empty else float(rows.iloc[0][column])

    independent_shared_delta = (
        None
        if comparisons.empty
        else float(comparisons.independent_minus_shared_pp.mean())
    )
    report = f"""# 服务器实验汇总

## 先看结论

- 论文原版 `reorder + ARC` 的局部 output-SSE 恢复率：{percent(local_value('paper_reorder_arc'))}
- 独立 `J_X/J_W`（固定 1:1）的局部恢复率：{percent(local_value('proxy_fixed_half_train'))}
- 共享 `J` 的局部恢复率：{percent(local_value('proxy_shared_train'))}
- 独立集合相对共享集合：{'n/a' if independent_shared_delta is None else f'{independent_shared_delta:+.2f} 个百分点'}
- selection-row output-aware oracle：{percent(local_value('oracle_independent_fixed_half_train'))}

## WikiText2 perplexity

- BF16：{ppl_text(ppl_value('bf16'))}
- 普通 RTN：{ppl_text(ppl_value('rtn_identity'))}
- 论文 `reorder + ARC`：{ppl_text(ppl_value('paper_reorder_arc'))}
- 独立 `J_X/J_W`（多 seed 均值）：{ppl_text(ppl_value('proxy_fixed_half'))}

## 论文对齐下游任务

- BF16 五任务 zero-shot 平均：{accuracy_text(task_value('bf16', 'paper_zero_shot_average_mean'))}
- 普通 RTN 五任务 zero-shot 平均：{accuracy_text(task_value('rtn_identity', 'paper_zero_shot_average_mean'))}
- 论文 `reorder + ARC` 五任务 zero-shot 平均：{accuracy_text(task_value('paper_reorder_arc', 'paper_zero_shot_average_mean'))}
- 独立 `J_X/J_W` 五任务 zero-shot 平均：{accuracy_text(task_value('proxy_fixed_half', 'paper_zero_shot_average_mean'))}
- BF16 MMLU 5-shot：{accuracy_text(task_value('bf16', 'mmlu_5shot_accuracy_mean'))}
- 普通 RTN MMLU 5-shot：{accuracy_text(task_value('rtn_identity', 'mmlu_5shot_accuracy_mean'))}
- 论文 `reorder + ARC` MMLU 5-shot：{accuracy_text(task_value('paper_reorder_arc', 'mmlu_5shot_accuracy_mean'))}
- 独立 `J_X/J_W` MMLU 5-shot：{accuracy_text(task_value('proxy_fixed_half', 'mmlu_5shot_accuracy_mean'))}

## 口径

局部百分比表示相对普通 RTN output SSE 恢复了多少，越大越好；PPL 越小越好。下游五任务平均严格使用 ARC-Challenge、HellaSwag、LAMBADA、PIQA、Winogrande 的论文指标；MMLU 使用 5-shot。汇总表会保留 smoke 行供排错，但上面的正式结论只读取 `full_evaluation=true` 的结果。局部 oracle 只用于看简单统计分数离上限还有多远，不是可部署算法。所有 W4A4 数值结果使用仓库统一的 fake NVFP4 后端，不代表真实 kernel 延迟。
"""
    (summary_dir / "REPORT.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"Wrote aggregate results to {summary_dir}")


if __name__ == "__main__":
    main()
