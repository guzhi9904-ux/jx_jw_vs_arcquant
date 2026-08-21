# Version record

## v1 — frozen V2 server comparison

- ARCQuant baseline: activation-max reorder plus paper ARC tail compensation.
- Proposed method: identity layout, independent `J_X/J_W`, fixed `S/2 + S/2` budget.
- Local evaluation: 5 calibration seeds with disjoint selection/holdout rows.
- End-to-end evaluation: WikiText2 full-window PPL for BF16, RTN, paper ARC, and 5 independent-index seeds.
- Diagnostics: shared index, activation-only, weight-only, random, and output-aware ceilings.
- Explicit non-goals: rotation, score tuning, dynamic online selection, and real-kernel latency.

Any later scoring rule or hardware-constrained block selector should be added as v2 instead of silently changing these configs.
