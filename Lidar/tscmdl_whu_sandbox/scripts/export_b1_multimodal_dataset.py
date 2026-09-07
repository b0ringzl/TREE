"""Export the selected B1 point-cloud and panorama classification samples."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
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
    uniform_sample_indices,
)


FORMAT_VERSION = 1
TARGET_POINT_COUNT = 8192


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--selection-plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument("--view-candidates", type=int, default=3)
    parser.add_argument("--preview-per-class-split", type=int, default=2)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{path}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
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


def wrapped_crop(
    image: Image.Image, left: float, top: float, width: float, height: float
) -> Image.Image:
    image_width, image_height = image.size
    crop_width = max(1, min(int(round(width)), image_width))
    crop_height = max(1, min(int(round(height)), image_height))
    top_int = max(0, min(int(round(top)), image_height - crop_height))
    left_mod = int(round(left)) % image_width
    if left_mod + crop_width <= image_width:
        return image.crop((left_mod, top_int, left_mod + crop_width, top_int + crop_height))
    first_width = image_width - left_mod
    output = Image.new(image.mode, (crop_width, crop_height))
    output.paste(image.crop((left_mod, top_int, image_width, top_int + crop_height)), (0, 0))
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
        sampled_points, pose, panorama_width, panorama_height, apply_tilt=True
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


def write_manifest_csv(path: Path, records: list[dict[str, object]]) -> None:
    fields = [
        "sample_key",
        "split",
        "class_index",
        "benchmark_label_id",
        "scientific_name",
        "road_id",
        "trajectory_id",
        "tree_id",
        "source_point_count",
        "sampled_unique_point_count",
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
        writer.writerows(records)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.chunk_size <= 0 or args.view_candidates <= 0:
        raise ValueError("chunk-size and view-candidates must be positive")
    if args.preview_per_class_split < 0:
        raise ValueError("preview-per-class-split cannot be negative")

    plan_hash = sha256_file(args.selection_plan)
    plan = json.loads(args.selection_plan.read_text(encoding="utf-8"))
    plan_records = plan["records"]
    if len(plan_records) != plan["summary"]["selected_count"]:
        raise ValueError("Selection plan record count is inconsistent")
    by_trajectory: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for record in plan_records:
        key = (str(record["road_id"]), str(record["trajectory_id"]))
        by_trajectory[key].append(record)

    preview_keys: set[str] = set()
    preview_groups: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for record in plan_records:
        preview_groups[(str(record["split"]), int(record["class_index"]))].append(record)
    for group in preview_groups.values():
        ordered = sorted(group, key=lambda item: str(item["sample_key"]))
        preview_keys.update(
            str(item["sample_key"])
            for item in ordered[: args.preview_per_class_split]
        )

    checkpoint_path = args.output_root / "export_checkpoint.json"
    if checkpoint_path.is_file():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("source_plan_sha256") != plan_hash:
            raise ValueError("Existing checkpoint belongs to a different selection plan")
    else:
        checkpoint = {
            "format_version": FORMAT_VERSION,
            "source_plan": str(args.selection_plan),
            "source_plan_sha256": plan_hash,
            "completed_trajectories": [],
            "records": [],
        }
    completed = set(str(item) for item in checkpoint["completed_trajectories"])
    tracks = sorted(by_trajectory)
    started = time.perf_counter()
    run_completed = 0
    print(
        f"B1 export: {len(completed)}/{len(tracks)} trajectories complete, "
        f"{len(plan_records)} selected samples",
        flush=True,
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
                point_count=int(record["point_count"]),
                label_counts=((int(record["raw_label_id"]), int(record["point_count"])),),
                min_xyz=tuple(float(value) for value in record["min_xyz"]),
                max_xyz=tuple(float(value) for value in record["max_xyz"]),
            )
            for record in records
        ]
        with open_whu_trajectory(
            args.dataset_root, road_id, trajectory_id, chunk_size=args.chunk_size
        ) as reader:
            instances = extract_tree_instances(reader, stats)
        instances_by_id = {instance.stats.tree_id: instance for instance in instances}

        trajectory_csv = args.dataset_root / road_id / "hdi" / trajectory_id / "traj.csv"
        diagnostics = orientation_column_diagnostics(trajectory_csv)
        if diagnostics["heading_column"] != 3:
            raise ValueError(f"Unexpected heading column for {trajectory_key}: {diagnostics}")
        poses = read_trajectory_csv(trajectory_csv)
        image_dir = args.dataset_root / road_id / "image" / trajectory_id
        with Image.open(image_dir / poses[0].image_name) as first_image:
            panorama_width, panorama_height = first_image.size

        image_jobs: dict[str, list[dict[str, object]]] = defaultdict(list)
        track_outputs: list[dict[str, object]] = []
        for record in records:
            instance = instances_by_id[int(record["tree_id"])]
            sample_key = str(record["sample_key"])
            indices = uniform_sample_indices(
                instance.stats.point_count,
                TARGET_POINT_COUNT,
                int(record["sample_seed"]),
            )
            sampled_raw = instance.xyz[indices]
            normalized, transform = normalize_unit_sphere(sampled_raw)

            class_dir = f"class_{int(record['class_index'])}"
            point_relative = (
                Path("points")
                / str(record["split"])
                / class_dir
                / f"{sample_key}.npz"
            )
            image_relative = (
                Path("images")
                / str(record["split"])
                / class_dir
                / f"{sample_key}.jpg"
            )
            point_path = args.output_root / point_relative
            image_path = args.output_root / image_relative
            unique_sampled_count = int(len(np.unique(indices)))
            atomic_npz(
                point_path,
                points_xyz=normalized.astype(np.float32, copy=False),
                class_index=np.asarray(record["class_index"], dtype=np.int64),
                benchmark_label_id=np.asarray(record["benchmark_label_id"], dtype=np.int64),
                centroid_xyz=np.asarray(transform.centroid, dtype=np.float64),
                scale=np.asarray(transform.scale, dtype=np.float64),
                source_point_count=np.asarray(instance.stats.point_count, dtype=np.int64),
                sampled_unique_point_count=np.asarray(unique_sampled_count, dtype=np.int64),
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
                panorama_visible = float(np.mean((v >= 0) & (v < panorama_height)))
                crop_fraction = crop_visible_fraction(u, v, crop, panorama_width)
                score = (
                    round(panorama_visible, 6),
                    crop.width_px * crop.height_px,
                    -camera_distance,
                )
                candidates.append(
                    {
                        "score": score,
                        "pose": pose,
                        "camera_distance": camera_distance,
                        "crop": crop,
                        "crop_visible_fraction": crop_fraction,
                        "panorama_visible_fraction": panorama_visible,
                    }
                )
            best = max(candidates, key=lambda item: item["score"])
            output = {
                **record,
                "source_point_count": instance.stats.point_count,
                "sampled_unique_point_count": unique_sampled_count,
                "point_path": point_relative.as_posix(),
                "image_path": image_relative.as_posix(),
                "image_name": best["pose"].image_name,
                "camera_pose": best["pose"].to_dict(),
                "camera_distance_m": best["camera_distance"],
                "panorama_size": [panorama_width, panorama_height],
                "crop": best["crop"].to_dict(),
                "crop_visible_fraction": best["crop_visible_fraction"],
                "panorama_visible_fraction": best["panorama_visible_fraction"],
                "view_candidate_count": len(candidates),
                "point_sha256": sha256_file(point_path),
            }
            image_jobs[best["pose"].image_name].append(
                {
                    "output": output,
                    "image_path": image_path,
                    "crop": best["crop"],
                    "pose": best["pose"],
                    "sampled_raw": sampled_raw if sample_key in preview_keys else None,
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
                ).resize((768, 512), Image.Resampling.LANCZOS)
                atomic_jpeg(job["image_path"], crop_image)
                job["output"]["image_sha256"] = sha256_file(job["image_path"])
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
                        / f"{job['output']['sample_key']}_overlay.jpg"
                    )
                    atomic_jpeg(preview_path, preview)
                    job["output"]["quality_preview_path"] = preview_path.relative_to(
                        args.output_root
                    ).as_posix()

        checkpoint["records"].extend(track_outputs)
        checkpoint["completed_trajectories"].append(trajectory_key)
        completed.add(trajectory_key)
        atomic_json(checkpoint_path, checkpoint)
        run_completed += 1
        elapsed = time.perf_counter() - started
        remaining = len(tracks) - len(completed)
        eta = elapsed / run_completed * remaining if run_completed else 0.0
        print(
            f"[{len(completed):2d}/{len(tracks)}] {trajectory_key:<8} "
            f"samples={len(records):>3d} images={len(image_jobs):>3d} "
            f"time={time.perf_counter()-track_started:6.2f}s eta={eta/60:5.1f}m",
            flush=True,
        )

    output_records = checkpoint["records"]
    if len(output_records) != len(plan_records):
        raise ValueError(
            f"Exported record count mismatch: {len(output_records)} != {len(plan_records)}"
        )
    output_records.sort(key=lambda item: str(item["sample_key"]))
    split_histogram: dict[str, Counter[int]] = defaultdict(Counter)
    for record in output_records:
        split_histogram[str(record["split"])][int(record["benchmark_label_id"])] += 1
    summary = {
        "sample_count": len(output_records),
        "trajectory_count": len(tracks),
        "split_histogram": {
            split: {str(label): count for label, count in sorted(hist.items())}
            for split, hist in sorted(split_histogram.items())
        },
        "minimum_crop_visible_fraction": min(
            float(record["crop_visible_fraction"]) for record in output_records
        ),
        "preview_count": sum(
            "quality_preview_path" in record for record in output_records
        ),
    }
    manifest = {
        "format_version": FORMAT_VERSION,
        "source_plan": str(args.selection_plan),
        "source_plan_sha256": plan_hash,
        "point_count_per_sample": TARGET_POINT_COUNT,
        "point_npz_key": "points_xyz",
        "image_size": [768, 512],
        "projection_policy": (
            "best of three nearest poses by vertical visibility, projected crop area, then distance"
        ),
        "summary": summary,
        "records": output_records,
    }
    atomic_json(args.output_root / "manifest.json", manifest)
    atomic_json(args.output_root / "summary.json", summary)
    atomic_json(
        args.output_root / "classes.json",
        [
            {
                "class_index": int(index),
                "benchmark_label_id": int(label),
                "scientific_name": plan["class_names"][str(label)],
            }
            for index, label in plan["class_index_to_benchmark_label"].items()
        ],
    )
    write_manifest_csv(args.output_root / "manifest.csv", output_records)
    checkpoint["status"] = "complete"
    checkpoint["summary"] = summary
    atomic_json(checkpoint_path, checkpoint)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
