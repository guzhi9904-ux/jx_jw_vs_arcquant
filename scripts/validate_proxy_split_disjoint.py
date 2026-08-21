"""Reconstruct WikiText2 sample offsets and verify train/holdout disjointness."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

from transformers import AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utilize import _load_wikitext2_split  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--analysis-dir",
        default=str(REPO_ROOT / "analysis" / "rtn_proxy_ks_holdout"),
    )
    parser.add_argument("--selection-samples", type=int, default=4)
    parser.add_argument("--holdout-samples", type=int, default=4)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analysis_dir = Path(args.analysis_dir)
    metadata = json.loads(
        (analysis_dir / "metadata.json").read_text(encoding="utf-8")
    )
    tokenizer = AutoTokenizer.from_pretrained(
        metadata["model"], local_files_only=True
    )
    train_data = _load_wikitext2_split("train")
    token_ids = tokenizer(
        "\n\n".join(train_data["text"]), return_tensors="pt"
    ).input_ids[0]

    random.seed(args.seed)
    count = args.selection_samples + args.holdout_samples
    starts = [
        random.randint(0, token_ids.numel() - args.seqlen - 1)
        for _ in range(count)
    ]
    hashes = []
    for start in starts:
        raw = token_ids[start : start + args.seqlen].numpy().tobytes()
        hashes.append(hashlib.sha256(raw).hexdigest())

    selection = list(range(args.selection_samples))
    holdout = list(range(args.selection_samples, count))
    cross_pairs = [(left, right) for left in selection for right in holdout]
    exact_disjoint = all(hashes[left] != hashes[right] for left, right in cross_pairs)
    interval_disjoint = all(
        starts[left] + args.seqlen <= starts[right]
        or starts[right] + args.seqlen <= starts[left]
        for left, right in cross_pairs
    )
    minimum_gap = min(
        max(starts[left], starts[right])
        - min(starts[left] + args.seqlen, starts[right] + args.seqlen)
        for left, right in cross_pairs
    )
    result = {
        "status": "passed" if exact_disjoint and interval_disjoint else "failed",
        "token_count": int(token_ids.numel()),
        "seed": args.seed,
        "seqlen": args.seqlen,
        "selection_starts": [starts[index] for index in selection],
        "holdout_starts": [starts[index] for index in holdout],
        "selection_hashes": [hashes[index] for index in selection],
        "holdout_hashes": [hashes[index] for index in holdout],
        "exact_sequence_disjoint": exact_disjoint,
        "token_intervals_disjoint": interval_disjoint,
        "minimum_cross_split_token_gap": int(minimum_gap),
    }
    if result["status"] != "passed":
        raise AssertionError(result)
    (analysis_dir / "split_validation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
