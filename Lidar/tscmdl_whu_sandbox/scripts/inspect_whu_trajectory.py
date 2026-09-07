"""Inspect one WHU-STree trajectory using bounded, memory-mapped reads."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


SANDBOX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SANDBOX_ROOT / "src"))

from tscmdl_whu import open_whu_trajectory  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--road", required=True)
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument("--sample-size", type=int, default=4096)
    parser.add_argument("--no-reference", action="store_true")
    return parser.parse_args()


def describe_chunk(chunk) -> dict[str, object]:
    sample_count = min(len(chunk), 4096)
    records = chunk.records[:sample_count]
    result: dict[str, object] = {
        "start": chunk.start,
        "stop": chunk.stop,
        "length": len(chunk),
        "sample_count": sample_count,
        "xyz_first": [float(records[name][0]) for name in ("x", "y", "z")],
        "intensity_sample_range": [
            float(np.min(records["intensity"])),
            float(np.max(records["intensity"])),
        ],
    }
    if chunk.tree is not None and chunk.label is not None:
        trees = chunk.tree[:sample_count]
        labels = chunk.label[:sample_count]
        result["tree_sample_range"] = [float(np.min(trees)), float(np.max(trees))]
        result["label_sample_range"] = [int(np.min(labels)), int(np.max(labels))]
    return result


def main() -> None:
    args = parse_args()
    if args.sample_size <= 0:
        raise ValueError("sample-size must be positive")

    with open_whu_trajectory(
        args.dataset_root,
        args.road,
        args.trajectory,
        chunk_size=args.chunk_size,
        use_reference=not args.no_reference,
    ) as reader:
        first_stop = min(args.sample_size, len(reader))
        last_start = max(0, len(reader) - args.sample_size)
        first_chunk = reader.read_chunk(0, first_stop)
        last_chunk = reader.read_chunk(last_start, len(reader))
        result = {
            "road_id": args.road,
            "trajectory_id": args.trajectory,
            "ply_path": str(reader.header.path),
            "format": reader.header.format,
            "vertex_count": len(reader),
            "properties": [list(item) for item in reader.header.properties],
            "record_size": reader.header.record_size,
            "data_offset": reader.header.data_offset,
            "file_size": reader.header.path.stat().st_size,
            "expected_file_size": reader.header.expected_file_size,
            "chunk_size": reader.chunk_size,
            "chunk_count": reader.chunk_count,
            "annotation_source": reader.annotation_source,
            "reference_path": str(reader.reference_path) if reader.reference_path else None,
            "first_sample": describe_chunk(first_chunk),
            "last_sample": describe_chunk(last_chunk),
        }

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
