"""Build a resumable inventory of selected WHU-STree benchmark classes."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path


SANDBOX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SANDBOX_ROOT / "src"))

from tscmdl_whu import open_whu_trajectory, scan_tree_instances  # noqa: E402


FORMAT_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--trajectory-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", type=int, action="append", dest="labels")
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {
        "road_id",
        "trajectory_id",
        "ply_vertex_count",
        "image_count",
        "is_test_reference",
        "structural_status",
    }
    if not rows:
        raise ValueError(f"Trajectory manifest is empty: {path}")
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Trajectory manifest is missing columns: {sorted(missing)}")
    invalid = [
        f"{row['road_id']}/{row['trajectory_id']}"
        for row in rows
        if row["structural_status"] != "ok"
    ]
    if invalid:
        raise ValueError(f"Structurally invalid trajectories: {invalid}")
    return rows


def new_checkpoint(labels: tuple[int, ...], row_count: int) -> dict[str, object]:
    return {
        "format_version": FORMAT_VERSION,
        "target_benchmark_labels": list(labels),
        "trajectory_count": row_count,
        "completed_trajectories": [],
        "trajectory_summaries": [],
        "instances": [],
    }


def load_checkpoint(
    path: Path, labels: tuple[int, ...], row_count: int
) -> dict[str, object]:
    if not path.is_file():
        return new_checkpoint(labels, row_count)
    checkpoint = json.loads(path.read_text(encoding="utf-8"))
    if checkpoint.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"Unsupported checkpoint format: {checkpoint.get('format_version')}")
    if checkpoint.get("target_benchmark_labels") != list(labels):
        raise ValueError("Checkpoint target labels do not match the requested labels")
    if checkpoint.get("trajectory_count") != row_count:
        raise ValueError("Checkpoint trajectory count does not match the current manifest")
    return checkpoint


def write_checkpoint(path: Path, checkpoint: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{path}.tmp")
    temporary.write_text(
        json.dumps(checkpoint, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def bool_value(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized not in {"true", "false"}:
        raise ValueError(f"Expected a boolean value, got {value!r}")
    return normalized == "true"


def finalize_summary(checkpoint: dict[str, object]) -> None:
    instances = checkpoint["instances"]
    summaries = checkpoint["trajectory_summaries"]
    histogram = Counter(int(item["benchmark_label_id"]) for item in instances)
    checkpoint["summary"] = {
        "completed_trajectory_count": len(checkpoint["completed_trajectories"]),
        "instance_count": len(instances),
        "class_histogram": {
            str(label): count for label, count in sorted(histogram.items())
        },
        "total_scanned_point_count": sum(
            int(item["total_point_count"]) for item in summaries
        ),
        "total_scan_seconds": round(
            sum(float(item["elapsed_seconds"]) for item in summaries), 4
        ),
    }


def main() -> None:
    args = parse_args()
    if args.chunk_size <= 0:
        raise ValueError("chunk-size must be positive")
    labels = tuple(sorted(set(args.labels or [1, 2, 13])))
    if not labels or any(label < 0 or label > 18 for label in labels):
        raise ValueError("Target labels must be benchmark labels in [0, 18]")

    rows = load_rows(args.trajectory_manifest)
    checkpoint = load_checkpoint(args.output, labels, len(rows))
    completed = set(str(item) for item in checkpoint["completed_trajectories"])
    pending_count = len(rows) - len(completed)
    print(
        f"WHU inventory: {len(completed)}/{len(rows)} complete, "
        f"{pending_count} pending, labels={labels}",
        flush=True,
    )
    run_started = time.perf_counter()
    run_completed = 0

    for row_index, row in enumerate(rows, start=1):
        road_id = row["road_id"]
        trajectory_id = row["trajectory_id"]
        key = f"{road_id}/{trajectory_id}"
        if key in completed:
            continue

        started = time.perf_counter()
        with open_whu_trajectory(
            args.dataset_root,
            road_id,
            trajectory_id,
            chunk_size=args.chunk_size,
        ) as reader:
            scan = scan_tree_instances(reader)
            annotation_source = reader.annotation_source

        selected = [
            instance
            for instance in scan.instances
            if instance.is_classification_valid
            and instance.benchmark_label_id in labels
        ]
        official_test = bool_value(row["is_test_reference"])
        elapsed = time.perf_counter() - started
        histogram = Counter(
            int(instance.benchmark_label_id) for instance in selected
        )
        checkpoint["instances"].extend(
            {
                "sample_key": f"{road_id}_{trajectory_id}_{instance.tree_id}",
                "road_id": road_id,
                "trajectory_id": trajectory_id,
                "tree_id": instance.tree_id,
                "raw_label_id": instance.raw_label_id,
                "benchmark_label_id": instance.benchmark_label_id,
                "point_count": instance.point_count,
                "min_xyz": list(instance.min_xyz),
                "max_xyz": list(instance.max_xyz),
                "center_xyz": [
                    (low + high) / 2.0
                    for low, high in zip(instance.min_xyz, instance.max_xyz)
                ],
                "dimensions_xyz": list(instance.dimensions),
                "annotation_source": annotation_source,
                "is_official_test": official_test,
            }
            for instance in selected
        )
        checkpoint["trajectory_summaries"].append(
            {
                "trajectory_key": key,
                "road_id": road_id,
                "trajectory_id": trajectory_id,
                "annotation_source": annotation_source,
                "is_official_test": official_test,
                "total_point_count": scan.total_point_count,
                "positive_instance_count": len(scan.instances),
                "classification_valid_count": scan.classification_valid_count,
                "unlabeled_count": scan.unlabeled_count,
                "inconsistent_count": scan.inconsistent_count,
                "target_instance_count": len(selected),
                "target_class_histogram": {
                    str(label): count for label, count in sorted(histogram.items())
                },
                "image_count": int(row["image_count"]),
                "elapsed_seconds": round(elapsed, 4),
            }
        )
        checkpoint["completed_trajectories"].append(key)
        completed.add(key)
        finalize_summary(checkpoint)
        write_checkpoint(args.output, checkpoint)

        run_completed += 1
        run_elapsed = time.perf_counter() - run_started
        average = run_elapsed / run_completed
        remaining = len(rows) - len(completed)
        eta = average * remaining
        hist_text = ", ".join(
            f"{label}:{histogram.get(label, 0)}" for label in labels
        )
        print(
            f"[{len(completed):3d}/{len(rows)}] {key:<8} "
            f"points={scan.total_point_count:>10,d} targets={len(selected):>4d} "
            f"({hist_text}) time={elapsed:6.2f}s eta={eta/60:5.1f}m",
            flush=True,
        )

    finalize_summary(checkpoint)
    write_checkpoint(args.output, checkpoint)
    print(json.dumps(checkpoint["summary"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
