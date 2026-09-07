"""Export a resumable, class-complete C1 shared-asset smoke dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


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
    select_shared_smoke_records,
    uniform_sample_indices,
    validate_shared_asset_paths,
)


FORMAT_VERSION = 1
TARGET_POINT_COUNT = 8192
OUTPUT_IMAGE_SIZE = (768, 512)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--shared-assets", type=Path, required=True)
    parser.add_argument("--benchmark-manifest", type=Path, required=True)
    parser.add_argument("--road-manifest", type=Path, required=True)
    parser.add_argument("--classes", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--samples-per-class", type=int, default=1)
    parser.add_argument("--max-trajectories", type=int, default=3)
    parser.add_argument("--max-new-trajectories", type=int)
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument("--view-candidates", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260731)
    return parser.parse_args()


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{path}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{path}.tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
    temporary.replace(path)


def atomic_jpeg(path: Path, image: Image.Image, *, quality: int = 92) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{path}.tmp")
    image.save(temporary, format="JPEG", quality=quality)
    temporary.replace(path)


def atomic_csv(path: Path, records: list[dict[str, object]]) -> None:
    fields = [
        "sample_key",
        "class_index",
        "benchmark_label_id",
        "scientific_name",
        "road_id",
        "trajectory_id",
        "tree_id",
        "source_point_count",
        "sampled_unique_point_count",
        "benchmark_split",
        "road_run_count",
        "road_splits",
        "point_path",
        "image_path",
        "image_name",
        "camera_distance_m",
        "crop_visible_fraction",
        "point_sha256",
        "image_sha256",
    ]
    temporary = Path(f"{path}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            row = dict(record)
            memberships = list(record["road_membership"])
            row["road_run_count"] = len(memberships)
            row["road_splits"] = ";".join(
                f'{item["run_index"]}:{item["split"]}' for item in memberships
            )
            writer.writerow(row)
    temporary.replace(path)


def wrapped_crop(
    image: Image.Image, left: float, top: float, width: float, height: float
) -> Image.Image:
    image_width, image_height = image.size
    crop_width = max(1, min(int(round(width)), image_width))
    crop_height = max(1, min(int(round(height)), image_height))
    top_int = max(0, min(int(round(top)), image_height - crop_height))
    left_mod = int(round(left)) % image_width
    if left_mod + crop_width <= image_width:
        return image.crop(
            (left_mod, top_int, left_mod + crop_width, top_int + crop_height)
        )
    first_width = image_width - left_mod
    output = Image.new(image.mode, (crop_width, crop_height))
    output.paste(
        image.crop((left_mod, top_int, image_width, top_int + crop_height)),
        (0, 0),
    )
    output.paste(
        image.crop((0, top_int, crop_width - first_width, top_int + crop_height)),
        (first_width, 0),
    )
    return output


def crop_visible_fraction(
    u: np.ndarray, v: np.ndarray, crop: object, panorama_width: int
) -> float:
    x = np.mod(u - crop.left_unwrapped_px, panorama_width)
    y = v - crop.top_px
    visible = (
        (x >= 0)
        & (x < crop.width_px)
        & (y >= 0)
        & (y < crop.height_px)
    )
    return float(np.mean(visible))


def overlay_points(
    crop_image: Image.Image,
    sampled_points: np.ndarray,
    pose: object,
    crop: object,
    panorama_width: int,
    panorama_height: int,
) -> Image.Image:
    output = crop_image.convert("RGBA")
    layer = Image.new("RGBA", output.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    u, v, _ = project_equirectangular(
        sampled_points,
        pose,
        panorama_width,
        panorama_height,
        apply_tilt=True,
    )
    x = np.mod(u - crop.left_unwrapped_px, panorama_width)
    y = v - crop.top_px
    visible = (
        (x >= 0)
        & (x < crop.width_px)
        & (y >= 0)
        & (y < crop.height_px)
    )
    x = x[visible] * output.width / crop.width_px
    y = y[visible] * output.height / crop.height_px
    for px, py in zip(x[::2], y[::2]):
        draw.ellipse(
            (px - 1.5, py - 1.5, px + 1.5, py + 1.5),
            fill=(255, 35, 20, 175),
        )
    return Image.alpha_composite(output, layer).convert("RGB")


def directory_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def build_plan(args: argparse.Namespace) -> dict[str, object]:
    shared_rows = read_csv(args.shared_assets)
    classes = read_json(args.classes)
    inventory = read_json(args.inventory)
    if not isinstance(classes, list) or len(classes) != 19:
        raise ValueError("C1c smoke export requires the 19-class definition")
    if not isinstance(inventory, dict):
        raise ValueError("Invalid C1a inventory")

    labels = [int(item["benchmark_label_id"]) for item in classes]
    selected, selection_summary = select_shared_smoke_records(
        shared_rows,
        labels,
        samples_per_class=args.samples_per_class,
        max_trajectories=args.max_trajectories,
        seed=args.seed,
        allow_official_test=False,
    )
    inventory_by_key = {
        str(item["sample_key"]): item for item in inventory["instances"]
    }
    benchmark_by_key = {
        str(item["sample_key"]): item for item in read_csv(args.benchmark_manifest)
    }
    road_by_key: dict[str, list[dict[str, str]]] = defaultdict(list)
    for item in read_csv(args.road_manifest):
        road_by_key[str(item["sample_key"])].append(item)

    records: list[dict[str, object]] = []
    for selected_row in selected:
        sample_key = str(selected_row["sample_key"])
        validate_shared_asset_paths(selected_row)
        inventory_row = inventory_by_key[sample_key]
        benchmark_row = benchmark_by_key[sample_key]
        if bool(inventory_row["is_official_test"]):
            raise ValueError(f"Official test sample selected for C1c: {sample_key}")
        for field in ("point_path", "image_path"):
            if str(benchmark_row[field]) != str(selected_row[field]):
                raise ValueError(f"Benchmark {field} mismatch for {sample_key}")

        road_membership = []
        for row in sorted(road_by_key.get(sample_key, []), key=lambda item: int(item["run_index"])):
            for field in ("point_path", "image_path"):
                if str(row[field]) != str(selected_row[field]):
                    raise ValueError(f"Road-domain {field} mismatch for {sample_key}")
            road_membership.append(
                {
                    "run_index": int(row["run_index"]),
                    "split": str(row["split"]),
                    "class_index": int(row["class_index"]),
                    "balanced_evaluation_selected": int(
                        row["balanced_evaluation_selected"]
                    ),
                }
            )

        label = int(selected_row["benchmark_label_id"])
        expected_road_runs = 0 if label in {12, 16, 17} else 3
        if len(road_membership) != expected_road_runs:
            raise ValueError(
                f"Road-domain membership count mismatch for {sample_key}: "
                f"{len(road_membership)} != {expected_road_runs}"
            )
        records.append(
            {
                "sample_key": sample_key,
                "class_index": int(selected_row["class_index"]),
                "benchmark_label_id": label,
                "scientific_name": str(selected_row["scientific_name"]),
                "road_id": str(selected_row["road_id"]),
                "trajectory_id": str(selected_row["trajectory_id"]),
                "tree_id": int(selected_row["tree_id"]),
                "raw_label_id": int(inventory_row["raw_label_id"]),
                "source_point_count": int(inventory_row["point_count"]),
                "min_xyz": [float(value) for value in inventory_row["min_xyz"]],
                "max_xyz": [float(value) for value in inventory_row["max_xyz"]],
                "sample_seed": int(selected_row["sample_seed"]),
                "point_path": str(selected_row["point_path"]),
                "image_path": str(selected_row["image_path"]),
                "benchmark_split": str(benchmark_row["split"]),
                "benchmark_balanced_evaluation_selected": int(
                    benchmark_row["balanced_evaluation_selected"]
                ),
                "road_membership": road_membership,
            }
        )
    records.sort(key=lambda item: str(item["sample_key"]))

    return {
        "format_version": FORMAT_VERSION,
        "stage": "C1c",
        "policy": {
            "official_test_samples_allowed": False,
            "physical_assets_written_once": True,
            "point_count_per_sample": TARGET_POINT_COUNT,
            "output_image_size": list(OUTPUT_IMAGE_SIZE),
        },
        "seed": args.seed,
        "sources": {
            "inventory": {
                "path": str(args.inventory.resolve()),
                "sha256": sha256_file(args.inventory),
            },
            "shared_assets": {
                "path": str(args.shared_assets.resolve()),
                "sha256": sha256_file(args.shared_assets),
            },
            "benchmark_manifest": {
                "path": str(args.benchmark_manifest.resolve()),
                "sha256": sha256_file(args.benchmark_manifest),
            },
            "road_manifest": {
                "path": str(args.road_manifest.resolve()),
                "sha256": sha256_file(args.road_manifest),
            },
            "classes": {
                "path": str(args.classes.resolve()),
                "sha256": sha256_file(args.classes),
            },
        },
        "summary": selection_summary,
        "records": records,
    }


def verify_reusable_records(
    output_root: Path, records: list[dict[str, object]]
) -> int:
    verified = 0
    for record in records:
        for prefix in ("point", "image"):
            path = output_root / str(record[f"{prefix}_path"])
            if not path.is_file():
                raise FileNotFoundError(f"Completed checkpoint asset is missing: {path}")
            if sha256_file(path) != str(record[f"{prefix}_sha256"]):
                raise ValueError(f"Completed checkpoint asset hash changed: {path}")
            if path.stat().st_mtime_ns != int(record[f"{prefix}_mtime_ns"]):
                raise ValueError(f"Completed checkpoint asset mtime changed: {path}")
        verified += 1
    return verified


def write_progress(
    path: Path,
    *,
    status: str,
    completed: int,
    total: int,
    samples: int,
    free_bytes: int,
) -> None:
    atomic_json(
        path,
        {
            "stage": "C1c",
            "status": status,
            "updated_at": now(),
            "completed_trajectories": completed,
            "total_trajectories": total,
            "exported_samples": samples,
            "free_disk_bytes": free_bytes,
        },
    )


def main() -> None:
    args = parse_args()
    if args.chunk_size <= 0 or args.view_candidates <= 0:
        raise ValueError("chunk-size and view-candidates must be positive")
    if args.max_new_trajectories is not None and args.max_new_trajectories <= 0:
        raise ValueError("max-new-trajectories must be positive when provided")

    args.output_root.mkdir(parents=True, exist_ok=True)
    plan = build_plan(args)
    plan_hash = sha256_json(plan)
    plan_path = args.output_root / "selection_plan.json"
    atomic_json(plan_path, {**plan, "plan_sha256": plan_hash})

    checkpoint_path = args.output_root / "export_checkpoint.json"
    if checkpoint_path.is_file():
        checkpoint = read_json(checkpoint_path)
        if not isinstance(checkpoint, dict):
            raise ValueError("Invalid C1c checkpoint")
        if checkpoint.get("source_plan_sha256") != plan_hash:
            raise ValueError("Existing checkpoint belongs to a different C1c plan")
    else:
        checkpoint = {
            "format_version": FORMAT_VERSION,
            "stage": "C1c",
            "source_plan": str(plan_path.resolve()),
            "source_plan_sha256": plan_hash,
            "status": "planned",
            "completed_trajectories": [],
            "records": [],
            "run_events": [],
        }

    plan_records = list(plan["records"])
    by_trajectory: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for record in plan_records:
        by_trajectory[(str(record["road_id"]), str(record["trajectory_id"]))].append(
            record
        )
    tracks = sorted(by_trajectory)
    completed = set(str(value) for value in checkpoint["completed_trajectories"])
    existing_records = list(checkpoint["records"])
    verified_reused = verify_reusable_records(args.output_root, existing_records)
    started_at = now()
    free_before = shutil.disk_usage(args.output_root).free
    size_before = directory_bytes(args.output_root)
    started = time.perf_counter()
    new_tracks = 0

    print(
        f"C1c smoke export: {len(completed)}/{len(tracks)} trajectories complete, "
        f"{len(plan_records)} samples, reused={verified_reused}",
        flush=True,
    )
    write_progress(
        args.output_root / "progress.json",
        status="running",
        completed=len(completed),
        total=len(tracks),
        samples=len(existing_records),
        free_bytes=free_before,
    )

    for road_id, trajectory_id in tracks:
        trajectory_key = f"{road_id}/{trajectory_id}"
        if trajectory_key in completed:
            continue
        track_started = time.perf_counter()
        records = by_trajectory[(road_id, trajectory_id)]
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
        instances_by_id = {
            instance.stats.tree_id: instance for instance in instances
        }

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
        track_outputs: list[dict[str, object]] = []
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
            unique_sampled_count = int(len(np.unique(indices)))
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
                sampled_unique_point_count=np.asarray(
                    unique_sampled_count, dtype=np.int64
                ),
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
                crop = projection_crop_bounds(
                    u, v, panorama_width, panorama_height
                )
                panorama_visible = float(
                    np.mean((v >= 0) & (v < panorama_height))
                )
                visible_fraction = crop_visible_fraction(
                    u, v, crop, panorama_width
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
                        "crop_visible_fraction": visible_fraction,
                        "panorama_visible_fraction": panorama_visible,
                    }
                )
            best = max(candidates, key=lambda item: item["score"])
            output = {
                **record,
                "sampled_unique_point_count": unique_sampled_count,
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
                    "sampled_raw": sampled_raw,
                }
            )
            track_outputs.append(output)

        for image_name, jobs in image_jobs.items():
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
                job["output"].update(
                    {
                        "image_sha256": sha256_file(job["image_path"]),
                        "image_mtime_ns": job["image_path"].stat().st_mtime_ns,
                        "image_bytes": job["image_path"].stat().st_size,
                        "quality_preview_path": preview_path.relative_to(
                            args.output_root
                        ).as_posix(),
                    }
                )

        checkpoint["records"].extend(track_outputs)
        checkpoint["completed_trajectories"].append(trajectory_key)
        completed.add(trajectory_key)
        checkpoint["status"] = "running"
        atomic_json(checkpoint_path, checkpoint)
        new_tracks += 1
        elapsed = time.perf_counter() - started
        remaining = len(tracks) - len(completed)
        eta = elapsed / new_tracks * remaining if new_tracks else 0.0
        print(
            f"[{len(completed)}/{len(tracks)}] {trajectory_key:<8} "
            f"samples={len(records):>2d} images={len(image_jobs):>2d} "
            f"time={time.perf_counter()-track_started:6.2f}s eta={eta/60:5.1f}m",
            flush=True,
        )
        write_progress(
            args.output_root / "progress.json",
            status="running",
            completed=len(completed),
            total=len(tracks),
            samples=len(checkpoint["records"]),
            free_bytes=shutil.disk_usage(args.output_root).free,
        )
        if (
            args.max_new_trajectories is not None
            and new_tracks >= args.max_new_trajectories
        ):
            break

    complete = len(completed) == len(tracks)
    status = "complete" if complete else "partial"
    free_after = shutil.disk_usage(args.output_root).free
    run_event = {
        "started_at": started_at,
        "finished_at": now(),
        "status": status,
        "completed_before": len(completed) - new_tracks,
        "completed_after": len(completed),
        "verified_reused_asset_count": verified_reused,
        "new_trajectory_count": new_tracks,
        "free_disk_before_bytes": free_before,
        "free_disk_after_bytes": free_after,
        "output_bytes_before": size_before,
        "output_bytes_after": directory_bytes(args.output_root),
        "elapsed_seconds": round(time.perf_counter() - started, 4),
    }
    checkpoint["run_events"].append(run_event)
    checkpoint["status"] = status

    if complete:
        output_records = sorted(
            checkpoint["records"], key=lambda item: str(item["sample_key"])
        )
        if len(output_records) != len(plan_records):
            raise ValueError(
                f"Exported record count mismatch: {len(output_records)} != "
                f"{len(plan_records)}"
            )
        histogram = Counter(
            int(record["benchmark_label_id"]) for record in output_records
        )
        summary = {
            "sample_count": len(output_records),
            "class_count": len(histogram),
            "trajectory_count": len(tracks),
            "class_histogram": {
                str(label): histogram[label] for label in sorted(histogram)
            },
            "minimum_crop_visible_fraction": min(
                float(record["crop_visible_fraction"])
                for record in output_records
            ),
            "shared_asset_bytes": sum(
                int(record["point_bytes"]) + int(record["image_bytes"])
                for record in output_records
            ),
            "quality_preview_count": sum(
                "quality_preview_path" in record for record in output_records
            ),
            "resume_run_count": len(checkpoint["run_events"]),
            "free_disk_before_bytes": checkpoint["run_events"][0][
                "free_disk_before_bytes"
            ],
            "free_disk_after_bytes": free_after,
            "output_directory_bytes": directory_bytes(args.output_root),
        }
        manifest = {
            "format_version": FORMAT_VERSION,
            "stage": "C1c",
            "status": "complete",
            "source_plan": str(plan_path.resolve()),
            "source_plan_sha256": plan_hash,
            "point_count_per_sample": TARGET_POINT_COUNT,
            "point_npz_key": "points_xyz",
            "image_size": list(OUTPUT_IMAGE_SIZE),
            "summary": summary,
            "records": output_records,
        }
        atomic_json(args.output_root / "manifest.json", manifest)
        atomic_csv(args.output_root / "manifest.csv", output_records)
        atomic_json(args.output_root / "summary.json", summary)
        checkpoint["summary"] = summary
    atomic_json(checkpoint_path, checkpoint)
    write_progress(
        args.output_root / "progress.json",
        status=status,
        completed=len(completed),
        total=len(tracks),
        samples=len(checkpoint["records"]),
        free_bytes=free_after,
    )
    print(
        json.dumps(
            {
                "status": status,
                "completed_trajectories": len(completed),
                "total_trajectories": len(tracks),
                "exported_samples": len(checkpoint["records"]),
                "verified_reused_asset_count": verified_reused,
                "output_root": str(args.output_root.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
