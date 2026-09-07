"""Extract selected WHU-STree instances and prepare reproducible 8192-point samples."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np


SANDBOX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SANDBOX_ROOT / "src"))

from tscmdl_whu import (  # noqa: E402
    extract_tree_instances,
    normalize_unit_sphere,
    open_whu_trajectory,
    scan_tree_instances,
    uniform_sample_indices,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--road", required=True)
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--tree-id", type=int, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-points", type=int, default=8192)
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=20260714)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def main() -> None:
    args = parse_args()
    tree_ids = sorted(set(args.tree_id))
    if any(tree_id <= 0 for tree_id in tree_ids):
        raise ValueError("tree-id values must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    with open_whu_trajectory(
        args.dataset_root,
        args.road,
        args.trajectory,
        chunk_size=args.chunk_size,
    ) as reader:
        scan = scan_tree_instances(reader)
        by_tree_id = scan.by_tree_id()
        missing = [tree_id for tree_id in tree_ids if tree_id not in by_tree_id]
        if missing:
            raise ValueError(f"Unknown tree IDs: {missing}")
        selected_stats = [by_tree_id[tree_id] for tree_id in tree_ids]
        invalid = [stats.tree_id for stats in selected_stats if not stats.is_classification_valid]
        if invalid:
            raise ValueError(f"Trees are not valid classification instances: {invalid}")
        instances = extract_tree_instances(reader, selected_stats)

    outputs = []
    for instance in instances:
        tree_seed = args.seed + instance.stats.tree_id
        sample_indices = uniform_sample_indices(
            instance.stats.point_count,
            args.target_points,
            tree_seed,
        )
        sampled_xyz = instance.xyz[sample_indices].astype(np.float32, copy=False)
        sampled_intensity = instance.intensity[sample_indices].astype(np.float32, copy=False)
        normalized_xyz, transform = normalize_unit_sphere(sampled_xyz)
        used_replacement = instance.stats.point_count < args.target_points

        filename = f"{args.road}_{args.trajectory}_tree_{instance.stats.tree_id}.npz"
        output_path = args.output_dir / filename
        np.savez_compressed(
            output_path,
            points_xyz_raw=sampled_xyz,
            points_xyz_normalized=normalized_xyz,
            intensity_raw=sampled_intensity,
            sample_indices=sample_indices,
            tree_id=np.int64(instance.stats.tree_id),
            raw_label_id=np.int64(instance.stats.raw_label_id),
            benchmark_label_id=np.int64(instance.stats.benchmark_label_id),
            source_point_count=np.int64(instance.stats.point_count),
            target_point_count=np.int64(args.target_points),
            used_replacement=np.bool_(used_replacement),
            seed=np.int64(tree_seed),
            normalization_centroid=np.asarray(transform.centroid, dtype=np.float64),
            normalization_scale=np.float64(transform.scale),
        )
        outputs.append(
            {
                "tree_id": instance.stats.tree_id,
                "raw_label_id": instance.stats.raw_label_id,
                "benchmark_label_id": instance.stats.benchmark_label_id,
                "source_point_count": instance.stats.point_count,
                "target_point_count": args.target_points,
                "used_replacement": used_replacement,
                "unique_sample_index_count": int(np.unique(sample_indices).size),
                "seed": tree_seed,
                "normalization_centroid": list(transform.centroid),
                "normalization_scale": transform.scale,
                "normalized_centroid_max_abs": float(
                    np.max(np.abs(normalized_xyz.mean(axis=0)))
                ),
                "normalized_max_radius": float(
                    np.linalg.norm(normalized_xyz, axis=1).max()
                ),
                "output_path": str(output_path),
                "output_size_bytes": output_path.stat().st_size,
                "sha256": file_sha256(output_path),
            }
        )

    manifest = {
        "road_id": args.road,
        "trajectory_id": args.trajectory,
        "sampling_strategy": "uniform random without replacement; retain all then pad with replacement when short",
        "normalization_strategy": "sample centroid plus unit maximum radius",
        "base_seed": args.seed,
        "target_point_count": args.target_points,
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "outputs": outputs,
    }
    manifest_path = args.output_dir / "a3_sample_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
