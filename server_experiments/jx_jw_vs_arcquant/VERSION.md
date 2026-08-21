# Version record

## v1 — frozen V2 server comparison

- ARCQuant baseline: activation-max reorder plus paper ARC tail compensation.
- Proposed method: identity layout, independent `J_X/J_W`, fixed `S/2 + S/2` budget.
- Local evaluation: 5 calibration seeds with disjoint selection/holdout rows.
- End-to-end evaluation: WikiText2 full-window PPL for BF16, RTN, paper ARC, and 5 independent-index seeds.
- Diagnostics: shared index, activation-only, weight-only, random, and output-aware ceilings.
- Explicit non-goals: rotation, score tuning, dynamic online selection, and real-kernel latency.

Any later scoring rule or hardware-constrained block selector should be added as v2 instead of silently changing these configs.

## v1.1 — paper-aligned downstream evaluation

- Adds a separate `tasks` stage for ARC-Challenge, HellaSwag, LAMBADA, PIQA and Winogrande zero-shot plus MMLU 5-shot.
- Uses the same fake-NVFP4 model wrappers as WikiText2 PPL for BF16, RTN, paper ARC, independent `J_X/J_W`, and optional shared `J`.
- Pins `lm-eval==0.4.8`, records raw suite JSON, supports suite-level resume, and separates limited smoke results from full evaluation.
- Keeps `tasks` outside `stage all` because first use needs dataset downloads and full evaluation is substantially more expensive.
