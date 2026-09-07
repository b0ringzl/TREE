"""Build a bounded-memory tree-instance profile for one WHU-STree trajectory."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path


SANDBOX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SANDBOX_ROOT / "src"))

from tscmdl_whu import open_whu_trajectory, scan_tree_instances  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--road", required=True)
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    with open_whu_trajectory(
        args.dataset_root,
        args.road,
        args.trajectory,
        chunk_size=args.chunk_size,
    ) as reader:
        scan = scan_tree_instances(reader)
        result = scan.to_dict()
        result.update(
            {
                "road_id": args.road,
                "trajectory_id": args.trajectory,
                "ply_path": str(reader.header.path),
                "annotation_source": reader.annotation_source,
                "chunk_size": reader.chunk_size,
                "chunk_count": reader.chunk_count,
            }
        )

    benchmark_histogram = Counter(
        instance.benchmark_label_id
        for instance in scan.instances
        if instance.is_classification_valid
    )
    result["benchmark_instance_histogram"] = {
        str(label): count for label, count in sorted(benchmark_histogram.items())
    }
    result["elapsed_seconds"] = round(time.perf_counter() - started, 4)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
