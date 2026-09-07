"""Export all C1 shared point-cloud and image assets with trajectory shards."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image


SANDBOX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SANDBOX_ROOT / "src"))

from tscmdl_whu import (  # noqa: E402
    TreeInstanceStats,
    extract_tree_instances,
    nearest_poses,
    normalize_unit_sphere,
    open_whu_trajectory,
    orientation_column_diagnostics,
    project_equirectangular,
    projection_crop_bounds,
    read_trajectory_csv,
    uniform_sample_indices,
    validate_shared_asset_paths,
)
from tscmdl_whu.export_assets import (  # noqa: E402
    atomic_jpeg,
    atomic_json,
    atomic_npz,
    crop_visible_fraction,
    directory_bytes,
    export_timestamp,
    overlay_points,
    sha256_file,
    sha256_json,
    wrapped_crop,
)


FORMAT_VERSION = 1
EXPECTED_SAMPLE_COUNT = 17134
EXPECTED_TRAJECTORY_COUNT = 110
TARGET_POINT_COUNT = 8192
OUTPUT_IMAGE_SIZE = (768, 512)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--shared-assets", type=Path, required=True)
    parser.add_argument("--benchmark-manifest", type=Path, required=True)
    parser.add_argument("--road-manifest", type=Path, required=True)
    parser.add_argument("--classes-19", type=Path, required=True)
    parser.add_argument("--classes-16", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument("--view-candidates", type=int, default=3)
    parser.add_argument("--previews-per-class", type=int, default=2)
    parser.add_argument("--image-workers", type=int, default=4)
    parser.add_argument("--max-new-trajectories", type=int)
    return parser.parse_args()


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def atomic_csv(
    path: Path,
    records: list[dict[str, object]],
    fields: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{path}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(path)


def stable_rank(sample_key: str, namespace: str) -> bytes:
    return hashlib.sha256(f"c1-full|{namespace}|{sample_key}".encode("utf-8")).digest()


def build_sources(args: argparse.Namespace) -> dict[str, dict[str, str]]:
    paths = {
        "inventory": args.inventory,
        "shared_assets": args.shared_assets,
        "benchmark_manifest": args.benchmark_manifest,
        "road_manifest": args.road_manifest,
        "classes_19": args.classes_19,
        "classes_16": args.classes_16,
    }
    return {
        name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
        for name, path in paths.items()
    }


def build_plan(
    args: argparse.Namespace,
    shared_rows: list[dict[str, str]],
    sources: dict[str, dict[str, str]],
) -> dict[str, object]:
    if len(shared_rows) != EXPECTED_SAMPLE_COUNT:
        raise ValueError(
            f"Shared asset count mismatch: {len(shared_rows)} != "
            f"{EXPECTED_SAMPLE_COUNT}"
        )
    by_label: dict[int, list[dict[str, str]]] = defaultdict(list)
    trajectories = set()
    for row in shared_rows:
        validate_shared_asset_paths(row)
        by_label[int(row["benchmark_label_id"])].append(row)
        trajectories.add((row["road_id"], row["trajectory_id"]))
    if sorted(by_label) != list(range(19)):
        raise ValueError("Shared asset manifest does not contain labels 0 through 18")
    if len(trajectories) != EXPECTED_TRAJECTORY_COUNT:
        raise ValueError(
            f"Trajectory count mismatch: {len(trajectories)} != "
            f"{EXPECTED_TRAJECTORY_COUNT}"
        )

    preview_keys = []
    for label in range(19):
        development = [
            row for row in by_label[label] if int(row["is_official_test"]) == 0
        ]
        ranked = sorted(
            development,
            key=lambda row: stable_rank(row["sample_key"], f"preview-{label}"),
        )
        if len(ranked) < args.previews_per_class:
            raise ValueError(f"Not enough development previews for label {label}")
        preview_keys.extend(
            row["sample_key"] for row in ranked[: args.previews_per_class]
        )

    return {
        "format_version": FORMAT_VERSION,
        "stage": "C1",
        "sources": sources,
        "settings": {
            "point_count_per_sample": TARGET_POINT_COUNT,
            "output_image_size": list(OUTPUT_IMAGE_SIZE),
            "view_candidates": args.view_candidates,
            "chunk_size": args.chunk_size,
            "previews_per_class": args.previews_per_class,
            "physical_assets_written_once": True,
        },
        "summary": {
            "sample_count": len(shared_rows),
            "trajectory_count": len(trajectories),
            "class_count": len(by_label),
            "preview_count": len(preview_keys),
        },
        "quality_preview_keys": sorted(preview_keys),
    }


def verify_completed_trajectories(
    output_root: Path, completed: dict[str, dict[str, object]]
) -> int:
    reused = 0
    for trajectory_key, entry in completed.items():
        shard_path = output_root / str(entry["shard_path"])
        if not shard_path.is_file():
            raise FileNotFoundError(
                f"Completed trajectory shard missing for {trajectory_key}: {shard_path}"
            )
        if sha256_file(shard_path) != str(entry["shard_sha256"]):
            raise ValueError(f"Completed trajectory shard changed: {shard_path}")
        shard = read_json(shard_path)
        if not isinstance(shard, dict):
            raise ValueError(f"Invalid trajectory shard: {shard_path}")
        records = list(shard["records"])
        if len(records) != int(entry["sample_count"]):
            raise ValueError(f"Trajectory shard count changed: {shard_path}")
        for record in records:
            for prefix in ("point", "image"):
                path = output_root / str(record[f"{prefix}_path"])
                if not path.is_file():
                    raise FileNotFoundError(f"Completed asset missing: {path}")
                if path.stat().st_size != int(record[f"{prefix}_bytes"]):
                    raise ValueError(f"Completed asset size changed: {path}")
                if path.stat().st_mtime_ns != int(record[f"{prefix}_mtime_ns"]):
                    raise ValueError(f"Completed asset mtime changed: {path}")
        reused += len(records)
    return reused


def write_progress(
    path: Path,
    *,
    status: str,
    completed_tracks: int,
    total_tracks: int,
    completed_samples: int,
    total_samples: int,
    current_trajectory: str,
    started_perf: float,
    new_tracks: int,
    free_disk_bytes: int,
    output_bytes: int,
) -> None:
    elapsed = time.perf_counter() - started_perf
    remaining_tracks = total_tracks - completed_tracks
    eta = elapsed / new_tracks * remaining_tracks if new_tracks else None
    atomic_json(
        path,
        {
            "stage": "C1",
            "status": status,
            "updated_at": export_timestamp(),
            "current_trajectory": current_trajectory,
            "completed_trajectories": completed_tracks,
            "total_trajectories": total_tracks,
            "completed_samples": completed_samples,
            "total_samples": total_samples,
            "progress_fraction": completed_samples / total_samples,
            "elapsed_seconds_this_run": round(elapsed, 3),
            "eta_seconds_this_run": round(eta, 3) if eta is not None else None,
            "free_disk_bytes": free_disk_bytes,
            "output_bytes": output_bytes,
        },
    )


def enrich_source_records(
    shared_rows: list[dict[str, str]],
    inventory: dict[str, object],
    benchmark_rows: list[dict[str, str]],
    road_rows: list[dict[str, str]],
) -> tuple[
    dict[tuple[str, str], list[dict[str, object]]],
    dict[str, dict[str, str]],
    dict[str, list[dict[str, str]]],
]:
    inventory_by_key = {
        str(row["sample_key"]): row for row in inventory["instances"]
    }
    benchmark_by_key = {row["sample_key"]: row for row in benchmark_rows}
    road_by_key: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in road_rows:
        road_by_key[row["sample_key"]].append(row)

    by_trajectory: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(
        list
    )
    for shared in shared_rows:
        sample_key = shared["sample_key"]
        inventory_row = inventory_by_key[sample_key]
        benchmark = benchmark_by_key[sample_key]
        for field in ("point_path", "image_path"):
            if shared[field] != benchmark[field]:
                raise ValueError(f"Benchmark {field} mismatch for {sample_key}")
        memberships = sorted(
            road_by_key.get(sample_key, []), key=lambda row: int(row["run_index"])
        )
        label = int(shared["benchmark_label_id"])
        expected_memberships = (
            0
            if int(shared["is_official_test"]) == 1 or label in {12, 16, 17}
            else 3
        )
        if len(memberships) != expected_memberships:
            raise ValueError(f"Road-domain membership mismatch for {sample_key}")
        for membership in memberships:
            for field in ("point_path", "image_path"):
                if shared[field] != membership[field]:
                    raise ValueError(
                        f"Road-domain {field} mismatch for {sample_key}"
                    )

        record = {
            "sample_key": sample_key,
            "class_index": label,
            "benchmark_label_id": label,
            "scientific_name": shared["scientific_name"],
            "road_id": shared["road_id"],
            "trajectory_id": shared["trajectory_id"],
            "tree_id": int(shared["tree_id"]),
            "raw_label_id": int(inventory_row["raw_label_id"]),
            "source_point_count": int(inventory_row["point_count"]),
            "min_xyz": [float(value) for value in inventory_row["min_xyz"]],
            "max_xyz": [float(value) for value in inventory_row["max_xyz"]],
            "sample_seed": int(shared["sample_seed"]),
            "point_path": shared["point_path"],
            "image_path": shared["image_path"],
            "is_official_test": int(shared["is_official_test"]),
            "benchmark_split": benchmark["split"],
            "benchmark_balanced_evaluation_selected": int(
                benchmark["balanced_evaluation_selected"]
            ),
            "road_membership": [
                {
                    "run_index": int(row["run_index"]),
                    "split": row["split"],
                    "class_index": int(row["class_index"]),
                    "balanced_evaluation_selected": int(
                        row["balanced_evaluation_selected"]
                    ),
                }
                for row in memberships
            ],
        }
        by_trajectory[(shared["road_id"], shared["trajectory_id"])].append(record)
    return by_trajectory, benchmark_by_key, road_by_key


def export_trajectory(
    args: argparse.Namespace,
    records: list[dict[str, object]],
    preview_keys: set[str],
) -> list[dict[str, object]]:
    road_id = str(records[0]["road_id"])
    trajectory_id = str(records[0]["trajectory_id"])
    trajectory_key = f"{road_id}/{trajectory_id}"
    stats = [
        TreeInstanceStats(
            tree_id=int(record["tree_id"]),
            point_count=int(record["source_point_count"]),
            label_counts=(
                (
                    int(record["raw_label_id"]),
                    int(record["source_point_count"]),
                ),
            ),
            min_xyz=tuple(float(value) for value in record["min_xyz"]),
            max_xyz=tuple(float(value) for value in record["max_xyz"]),
        )
        for record in records
    ]
    with open_whu_trajectory(
        args.dataset_root,
        road_id,
        trajectory_id,
        chunk_size=args.chunk_size,
    ) as reader:
        instances = extract_tree_instances(reader, stats)
    instances_by_id = {instance.stats.tree_id: instance for instance in instances}

    trajectory_csv = (
        args.dataset_root / road_id / "hdi" / trajectory_id / "traj.csv"
    )
    diagnostics = orientation_column_diagnostics(trajectory_csv)
    if diagnostics["heading_column"] != 3:
        raise ValueError(
            f"Unexpected heading column for {trajectory_key}: {diagnostics}"
        )
    poses = read_trajectory_csv(trajectory_csv)
    image_dir = args.dataset_root / road_id / "image" / trajectory_id
    with Image.open(image_dir / poses[0].image_name) as first_image:
        panorama_width, panorama_height = first_image.size

    image_jobs: dict[str, list[dict[str, object]]] = defaultdict(list)
    outputs: list[dict[str, object]] = []
    for record in records:
        instance = instances_by_id[int(record["tree_id"])]
        indices = uniform_sample_indices(
            instance.stats.point_count,
            TARGET_POINT_COUNT,
            int(record["sample_seed"]),
        )
        sampled_raw = instance.xyz[indices]
        normalized, transform = normalize_unit_sphere(sampled_raw)
        point_path = args.output_root / str(record["point_path"])
        image_path = args.output_root / str(record["image_path"])
        unique_count = int(len(np.unique(indices)))
        atomic_npz(
            point_path,
            points_xyz=normalized.astype(np.float32, copy=False),
            class_index=np.asarray(record["class_index"], dtype=np.int64),
            benchmark_label_id=np.asarray(
                record["benchmark_label_id"], dtype=np.int64
            ),
            raw_label_id=np.asarray(record["raw_label_id"], dtype=np.int64),
            centroid_xyz=np.asarray(transform.centroid, dtype=np.float64),
            scale=np.asarray(transform.scale, dtype=np.float64),
            source_point_count=np.asarray(
                instance.stats.point_count, dtype=np.int64
            ),
            sampled_unique_point_count=np.asarray(unique_count, dtype=np.int64),
            sample_seed=np.asarray(record["sample_seed"], dtype=np.uint32),
        )

        center = instance.xyz.astype(np.float64).mean(axis=0)
        candidates = []
        for pose, camera_distance in nearest_poses(
            center, poses, count=min(args.view_candidates, len(poses))
        ):
            u, v, _ = project_equirectangular(
                instance.xyz,
                pose,
                panorama_width,
                panorama_height,
                apply_tilt=True,
            )
            crop = projection_crop_bounds(u, v, panorama_width, panorama_height)
            panorama_visible = float(
                np.mean((v >= 0) & (v < panorama_height))
            )
            candidates.append(
                {
                    "score": (
                        round(panorama_visible, 6),
                        crop.width_px * crop.height_px,
                        -camera_distance,
                    ),
                    "pose": pose,
                    "camera_distance": camera_distance,
                    "crop": crop,
                    "crop_visible_fraction": crop_visible_fraction(
                        u, v, crop, panorama_width
                    ),
                    "panorama_visible_fraction": panorama_visible,
                }
            )
        best = max(candidates, key=lambda item: item["score"])
        output = {
            **record,
            "sampled_unique_point_count": unique_count,
            "image_name": best["pose"].image_name,
            "camera_pose": best["pose"].to_dict(),
            "camera_distance_m": best["camera_distance"],
            "panorama_size": [panorama_width, panorama_height],
            "crop": best["crop"].to_dict(),
            "crop_visible_fraction": best["crop_visible_fraction"],
            "panorama_visible_fraction": best["panorama_visible_fraction"],
            "view_candidate_count": len(candidates),
            "point_sha256": sha256_file(point_path),
            "point_mtime_ns": point_path.stat().st_mtime_ns,
            "point_bytes": point_path.stat().st_size,
        }
        image_jobs[best["pose"].image_name].append(
            {
                "output": output,
                "image_path": image_path,
                "crop": best["crop"],
                "pose": best["pose"],
                "sampled_raw": (
                    sampled_raw
                    if str(record["sample_key"]) in preview_keys
                    else None
                ),
            }
        )
        outputs.append(output)

    def process_image_group(
        item: tuple[str, list[dict[str, object]]],
    ) -> None:
        image_name, jobs = item
        with Image.open(image_dir / image_name) as source:
            panorama = source.convert("RGB")
        if panorama.size != (panorama_width, panorama_height):
            raise ValueError(
                f"Panorama size changed within {trajectory_key}: {panorama.size}"
            )
        for job in jobs:
            crop = job["crop"]
            crop_image = wrapped_crop(
                panorama,
                crop.left_unwrapped_px,
                crop.top_px,
                crop.width_px,
                crop.height_px,
            ).resize(OUTPUT_IMAGE_SIZE, Image.Resampling.LANCZOS)
            atomic_jpeg(job["image_path"], crop_image)
            job["output"].update(
                {
                    "image_sha256": sha256_file(job["image_path"]),
                    "image_mtime_ns": job["image_path"].stat().st_mtime_ns,
                    "image_bytes": job["image_path"].stat().st_size,
                }
            )
            if job["sampled_raw"] is not None:
                preview = overlay_points(
                    crop_image,
                    job["sampled_raw"],
                    job["pose"],
                    crop,
                    panorama_width,
                    panorama_height,
                )
                preview_path = (
                    args.output_root
                    / "quality_previews"
                    / f'{job["output"]["sample_key"]}_overlay.jpg'
                )
                atomic_jpeg(preview_path, preview)
                job["output"]["quality_preview_path"] = preview_path.relative_to(
                    args.output_root
                ).as_posix()
    image_groups = sorted(image_jobs.items())
    if args.image_workers == 1:
        for item in image_groups:
            process_image_group(item)
    else:
        with ThreadPoolExecutor(max_workers=args.image_workers) as executor:
            list(executor.map(process_image_group, image_groups))
    return sorted(outputs, key=lambda record: str(record["sample_key"]))


def build_final_manifests(
    args: argparse.Namespace,
    checkpoint: dict[str, object],
    plan: dict[str, object],
    benchmark_by_key: dict[str, dict[str, str]],
    road_by_key: dict[str, list[dict[str, str]]],
) -> dict[str, object]:
    records = []
    for trajectory_key in sorted(checkpoint["completed_trajectories"]):
        entry = checkpoint["completed_trajectories"][trajectory_key]
        shard = read_json(args.output_root / str(entry["shard_path"]))
        records.extend(shard["records"])
    records.sort(key=lambda record: str(record["sample_key"]))
    if len(records) != EXPECTED_SAMPLE_COUNT:
        raise ValueError(
            f"Final shared record count mismatch: {len(records)} != "
            f"{EXPECTED_SAMPLE_COUNT}"
        )
    by_key = {str(record["sample_key"]): record for record in records}
    if len(by_key) != len(records):
        raise ValueError("Final shared manifest contains duplicate sample keys")

    shared_fields = [
        "sample_key",
        "class_index",
        "benchmark_label_id",
        "scientific_name",
        "road_id",
        "trajectory_id",
        "tree_id",
        "raw_label_id",
        "source_point_count",
        "sampled_unique_point_count",
        "is_official_test",
        "benchmark_split",
        "point_path",
        "image_path",
        "image_name",
        "camera_distance_m",
        "crop_visible_fraction",
        "point_bytes",
        "image_bytes",
        "point_sha256",
        "image_sha256",
    ]
    atomic_csv(args.output_root / "shared_manifest.csv", records, shared_fields)

    benchmark_records = []
    for sample_key, source in sorted(benchmark_by_key.items()):
        asset = by_key[sample_key]
        benchmark_records.append(
            {
                **source,
                "source_point_count": asset["source_point_count"],
                "sampled_unique_point_count": asset[
                    "sampled_unique_point_count"
                ],
                "image_name": asset["image_name"],
                "camera_distance_m": asset["camera_distance_m"],
                "crop_visible_fraction": asset["crop_visible_fraction"],
                "point_sha256": asset["point_sha256"],
                "image_sha256": asset["image_sha256"],
            }
        )
    benchmark_fields = list(benchmark_records[0])
    atomic_csv(
        args.output_root / "benchmark_19_manifest.csv",
        benchmark_records,
        benchmark_fields,
    )

    road_records = []
    for sample_key in sorted(road_by_key):
        asset = by_key[sample_key]
        for source in sorted(
            road_by_key[sample_key], key=lambda row: int(row["run_index"])
        ):
            road_records.append(
                {
                    **source,
                    "source_point_count": asset["source_point_count"],
                    "sampled_unique_point_count": asset[
                        "sampled_unique_point_count"
                    ],
                    "image_name": asset["image_name"],
                    "camera_distance_m": asset["camera_distance_m"],
                    "crop_visible_fraction": asset["crop_visible_fraction"],
                    "point_sha256": asset["point_sha256"],
                    "image_sha256": asset["image_sha256"],
                }
            )
    road_fields = list(road_records[0])
    atomic_csv(
        args.output_root / "road_domain_16_manifest.csv",
        road_records,
        road_fields,
    )

    classes_19 = read_json(args.classes_19)
    classes_16 = read_json(args.classes_16)
    atomic_json(args.output_root / "classes_19.json", classes_19)
    atomic_json(args.output_root / "classes_16.json", classes_16)
    class_histogram = Counter(
        int(record["benchmark_label_id"]) for record in records
    )
    split_histogram = Counter(str(record["benchmark_split"]) for record in records)
    shared_asset_bytes = sum(
        int(record["point_bytes"]) + int(record["image_bytes"])
        for record in records
    )
    summary = {
        "sample_count": len(records),
        "trajectory_count": len(checkpoint["completed_trajectories"]),
        "class_count": len(class_histogram),
        "class_histogram": {
            str(label): class_histogram[label] for label in sorted(class_histogram)
        },
        "benchmark_split_histogram": {
            split: split_histogram[split]
            for split in ("train", "val", "test")
        },
        "road_manifest_row_count": len(road_records),
        "quality_preview_count": sum(
            "quality_preview_path" in record for record in records
        ),
        "minimum_crop_visible_fraction": min(
            float(record["crop_visible_fraction"]) for record in records
        ),
        "shared_asset_bytes": shared_asset_bytes,
        "output_directory_bytes_before_final_json": directory_bytes(
            args.output_root
        ),
        "free_disk_bytes": shutil.disk_usage(args.output_root).free,
        "run_count": len(checkpoint["run_events"]),
    }
    manifest = {
        "format_version": FORMAT_VERSION,
        "stage": "C1",
        "status": "complete",
        "source_plan": str((args.output_root / "export_plan.json").resolve()),
        "source_plan_sha256": plan["plan_sha256"],
        "point_count_per_sample": TARGET_POINT_COUNT,
        "point_npz_key": "points_xyz",
        "image_size": list(OUTPUT_IMAGE_SIZE),
        "summary": summary,
        "records": records,
    }
    atomic_json(args.output_root / "shared_manifest.json", manifest)
    atomic_json(args.output_root / "summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    if args.chunk_size <= 0 or args.view_candidates <= 0:
        raise ValueError("chunk-size and view-candidates must be positive")
    if args.previews_per_class <= 0:
        raise ValueError("previews-per-class must be positive")
    if args.image_workers <= 0:
        raise ValueError("image-workers must be positive")
    if args.max_new_trajectories is not None and args.max_new_trajectories <= 0:
        raise ValueError("max-new-trajectories must be positive when provided")

    args.output_root.mkdir(parents=True, exist_ok=True)
    shared_rows = read_csv(args.shared_assets)
    inventory = read_json(args.inventory)
    benchmark_rows = read_csv(args.benchmark_manifest)
    road_rows = read_csv(args.road_manifest)
    if not isinstance(inventory, dict):
        raise ValueError("Invalid C1a inventory")
    sources = build_sources(args)
    plan_without_hash = build_plan(args, shared_rows, sources)
    plan_hash = sha256_json(plan_without_hash)
    plan = {**plan_without_hash, "plan_sha256": plan_hash}
    plan_path = args.output_root / "export_plan.json"
    atomic_json(plan_path, plan)

    by_trajectory, benchmark_by_key, road_by_key = enrich_source_records(
        shared_rows, inventory, benchmark_rows, road_rows
    )
    tracks = sorted(by_trajectory)
    checkpoint_path = args.output_root / "export_checkpoint.json"
    if checkpoint_path.is_file():
        checkpoint = read_json(checkpoint_path)
        if not isinstance(checkpoint, dict):
            raise ValueError("Invalid C1 checkpoint")
        if checkpoint.get("source_plan_sha256") != plan_hash:
            raise ValueError("Existing checkpoint belongs to a different C1 plan")
    else:
        checkpoint = {
            "format_version": FORMAT_VERSION,
            "stage": "C1",
            "status": "planned",
            "source_plan": str(plan_path.resolve()),
            "source_plan_sha256": plan_hash,
            "completed_trajectories": {},
            "run_events": [],
        }
    completed = checkpoint["completed_trajectories"]
    reused_samples = verify_completed_trajectories(args.output_root, completed)
    completed_samples = sum(
        int(entry["sample_count"]) for entry in completed.values()
    )
    preview_keys = set(str(value) for value in plan["quality_preview_keys"])
    free_before = shutil.disk_usage(args.output_root).free
    output_before = directory_bytes(args.output_root)
    started_at = export_timestamp()
    started_perf = time.perf_counter()
    new_tracks = 0
    new_samples = 0

    print(
        f"C1 full export: {len(completed)}/{len(tracks)} trajectories, "
        f"{completed_samples}/{len(shared_rows)} samples, reused={reused_samples}",
        flush=True,
    )
    write_progress(
        args.output_root / "progress.json",
        status="running",
        completed_tracks=len(completed),
        total_tracks=len(tracks),
        completed_samples=completed_samples,
        total_samples=len(shared_rows),
        current_trajectory="",
        started_perf=started_perf,
        new_tracks=new_tracks,
        free_disk_bytes=free_before,
        output_bytes=output_before,
    )

    for road_id, trajectory_id in tracks:
        trajectory_key = f"{road_id}/{trajectory_id}"
        if trajectory_key in completed:
            continue
        track_started = time.perf_counter()
        records = by_trajectory[(road_id, trajectory_id)]
        write_progress(
            args.output_root / "progress.json",
            status="running",
            completed_tracks=len(completed),
            total_tracks=len(tracks),
            completed_samples=completed_samples,
            total_samples=len(shared_rows),
            current_trajectory=trajectory_key,
            started_perf=started_perf,
            new_tracks=new_tracks,
            free_disk_bytes=shutil.disk_usage(args.output_root).free,
            output_bytes=directory_bytes(args.output_root),
        )
        outputs = export_trajectory(args, records, preview_keys)
        shard_relative = Path("trajectory_records") / (
            f"{road_id}_{trajectory_id}.json"
        )
        shard_path = args.output_root / shard_relative
        shard = {
            "format_version": FORMAT_VERSION,
            "stage": "C1",
            "trajectory_key": trajectory_key,
            "sample_count": len(outputs),
            "records": outputs,
        }
        atomic_json(shard_path, shard)
        completed[trajectory_key] = {
            "shard_path": shard_relative.as_posix(),
            "shard_sha256": sha256_file(shard_path),
            "sample_count": len(outputs),
            "completed_at": export_timestamp(),
        }
        completed_samples += len(outputs)
        new_samples += len(outputs)
        new_tracks += 1
        checkpoint["status"] = "running"
        atomic_json(checkpoint_path, checkpoint)

        elapsed = time.perf_counter() - started_perf
        remaining = len(tracks) - len(completed)
        eta = elapsed / new_tracks * remaining if new_tracks else 0.0
        print(
            f"[{len(completed):3d}/{len(tracks)}] {trajectory_key:<9} "
            f"samples={len(outputs):>3d} total={completed_samples:>5d}/"
            f"{len(shared_rows)} time={time.perf_counter()-track_started:6.1f}s "
            f"eta={eta/60:5.1f}m",
            flush=True,
        )
        write_progress(
            args.output_root / "progress.json",
            status="running",
            completed_tracks=len(completed),
            total_tracks=len(tracks),
            completed_samples=completed_samples,
            total_samples=len(shared_rows),
            current_trajectory=trajectory_key,
            started_perf=started_perf,
            new_tracks=new_tracks,
            free_disk_bytes=shutil.disk_usage(args.output_root).free,
            output_bytes=directory_bytes(args.output_root),
        )
        if (
            args.max_new_trajectories is not None
            and new_tracks >= args.max_new_trajectories
        ):
            break

    complete = len(completed) == len(tracks)
    status = "complete" if complete else "partial"
    run_event = {
        "started_at": started_at,
        "finished_at": export_timestamp(),
        "status": status,
        "completed_before": len(completed) - new_tracks,
        "completed_after": len(completed),
        "reused_sample_count": reused_samples,
        "new_trajectory_count": new_tracks,
        "new_sample_count": new_samples,
        "free_disk_before_bytes": free_before,
        "free_disk_after_bytes": shutil.disk_usage(args.output_root).free,
        "output_bytes_before": output_before,
        "output_bytes_after": directory_bytes(args.output_root),
        "elapsed_seconds": round(time.perf_counter() - started_perf, 3),
    }
    checkpoint["run_events"].append(run_event)
    checkpoint["status"] = status
    if complete:
        checkpoint["summary"] = build_final_manifests(
            args, checkpoint, plan, benchmark_by_key, road_by_key
        )
    atomic_json(checkpoint_path, checkpoint)
    write_progress(
        args.output_root / "progress.json",
        status=status,
        completed_tracks=len(completed),
        total_tracks=len(tracks),
        completed_samples=completed_samples,
        total_samples=len(shared_rows),
        current_trajectory="",
        started_perf=started_perf,
        new_tracks=new_tracks,
        free_disk_bytes=shutil.disk_usage(args.output_root).free,
        output_bytes=directory_bytes(args.output_root),
    )
    print(
        json.dumps(
            {
                "status": status,
                "completed_trajectories": len(completed),
                "total_trajectories": len(tracks),
                "completed_samples": completed_samples,
                "total_samples": len(shared_rows),
                "reused_sample_count": reused_samples,
                "output_root": str(args.output_root.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
