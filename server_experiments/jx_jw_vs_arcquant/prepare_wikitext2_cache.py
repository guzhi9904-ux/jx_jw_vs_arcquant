"""Download and freeze WikiText2 as single Arrow files for offline runs."""

from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path

import pyarrow as pa
from datasets import Dataset, load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--source-cache-dir",
        help=(
            "Optional existing directory containing wikitext-*.arrow. "
            "Used for an offline copy/validation instead of downloading."
        ),
    )
    return parser.parse_args()


def write_arrow(dataset: Dataset, destination: Path) -> None:
    temporary = destination.with_suffix(".arrow.tmp")
    table = dataset.data.table
    with pa.OSFile(str(temporary), "wb") as sink:
        with pa.ipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table)
    os.replace(temporary, destination)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_dir = (
        Path(args.source_cache_dir).expanduser().resolve()
        if args.source_cache_dir
        else None
    )
    manifest: dict[str, object] = {
        "dataset": "wikitext/wikitext-2-raw-v1",
        "output_dir": str(output_dir),
        "python": platform.python_version(),
        "splits": {},
    }
    for split in ("train", "test", "validation"):
        if source_dir is None:
            dataset = load_dataset(
                "wikitext", "wikitext-2-raw-v1", split=split
            )
            source = "Hugging Face Hub/cache"
        else:
            source_path = source_dir / f"wikitext-{split}.arrow"
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            dataset = Dataset.from_file(str(source_path))
            source = str(source_path)
        destination = output_dir / f"wikitext-{split}.arrow"
        write_arrow(dataset, destination)
        verified = Dataset.from_file(str(destination))
        if len(verified) != len(dataset) or "text" not in verified.column_names:
            raise AssertionError(f"Arrow validation failed: {destination}")
        manifest["splits"][split] = {
            "rows": len(verified),
            "bytes": destination.stat().st_size,
            "source": source,
            "file": str(destination),
        }
        print(
            f"{split}: rows={len(verified)} "
            f"size_mib={destination.stat().st_size / 2**20:.2f}"
        )
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"WikiText2 Arrow cache is ready: {output_dir}")


if __name__ == "__main__":
    main()
