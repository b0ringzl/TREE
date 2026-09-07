"""Prepare traceable D1 alternate-view and recrop candidates for re-review.

The source C1 assets, raw WHU-STree files, and the user-owned D1 checkpoint are
read-only.  Every generated image and manifest is written under a new batch
directory so the replacement decision can be reviewed independently.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont


SANDBOX_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_ROOT = PROJECT_ROOT / "lidar data" / "whu"
DERIVED_ROOT = RAW_ROOT / "derived" / "tscmdl"
DEFAULT_C1_ROOT = DERIVED_ROOT / "c1_full_shared_dataset"
DEFAULT_D1_ROOT = DERIVED_ROOT / "d1_four_class_image_quality"
OUTPUT_IMAGE_SIZE = (768, 512)
VALID_ACTIONS = {"retry_alternate_view", "manual_recrop"}
EXPECTED_CLASSES = {
    "Cinnamomum camphora",
    "Lagerstroemia indica",
    "Magnolia grandiflora",
    "Other",
}

sys.path.insert(0, str(SANDBOX_ROOT / "src"))

from tscmdl_whu.export_assets import (  # noqa: E402
    atomic_jpeg,
    atomic_json,
    crop_visible_fraction,
    overlay_points,
    wrapped_crop,
)
from tscmdl_whu.instances import TreeInstanceStats, extract_tree_instances  # noqa: E402
from tscmdl_whu.io import open_whu_trajectory  # noqa: E402
from tscmdl_whu.panorama import (  # noqa: E402
    CameraPose,
    nearest_poses,
    orientation_column_diagnostics,
    project_equirectangular,
    projection_crop_bounds,
    read_trajectory_csv,
)


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def batch_stamp() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def atomic_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    materialized = list(rows)
    fields: list[str] = []
    for row in materialized:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{path}.tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(materialized)
    temporary.replace(path)


def latest_action_dir(d1_root: Path) -> Path:
    candidates = sorted(
        path
        for path in (d1_root / "review_actions").glob("*")
        if (path / "evaluation_rework_queue.csv").is_file()
    )
    if not candidates:
        raise FileNotFoundError("No exported D1 evaluation rework queue was found")
    return candidates[-1]


def image_metrics(image: Image.Image) -> dict[str, object]:
    grayscale = image.convert("L")
    grayscale.thumbnail((256, 256), Image.Resampling.BILINEAR)
    array = np.asarray(grayscale, dtype=np.float32)
    p01, p05, p50, p95, p99 = np.percentile(array, [1, 5, 50, 95, 99])
    dx = np.abs(np.diff(array, axis=1)).mean() if array.shape[1] > 1 else 0.0
    dy = np.abs(np.diff(array, axis=0)).mean() if array.shape[0] > 1 else 0.0
    return {
        "width": image.width,
        "height": image.height,
        "mean_luminance": round(float(array.mean()), 4),
        "luminance_std": round(float(array.std()), 4),
        "p01": round(float(p01), 4),
        "p05": round(float(p05), 4),
        "p50": round(float(p50), 4),
        "p95": round(float(p95), 4),
        "p99": round(float(p99), 4),
        "dynamic_range_90": round(float(p95 - p05), 4),
        "edge_energy": round(float(dx + dy), 4),
        "dark_ratio": round(float((array <= 15).mean()), 6),
        "bright_ratio": round(float((array >= 240).mean()), 6),
    }


def quality_risk(metrics: dict[str, object]) -> tuple[float, list[str]]:
    score = 0.0
    reasons = []
    dynamic_range = float(metrics["dynamic_range_90"])
    dark_ratio = float(metrics["dark_ratio"])
    bright_ratio = float(metrics["bright_ratio"])
    edge_energy = float(metrics["edge_energy"])
    if dynamic_range < 45.0:
        reasons.append("low_dynamic_range")
        score += (45.0 - dynamic_range) / 2.0
    if dark_ratio > 0.35:
        reasons.append("excessive_dark_pixels")
        score += (dark_ratio - 0.35) * 40.0
    if bright_ratio > 0.35:
        reasons.append("excessive_bright_pixels")
        score += (bright_ratio - 0.35) * 40.0
    if edge_energy < 8.0:
        reasons.append("low_edge_energy")
        score += 8.0 - edge_energy
    return round(score, 4), reasons


def font(size: int = 18) -> ImageFont.ImageFont:
    candidates = (
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/segoeui.ttf"),
    )
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def comparison_sheet(
    original_path: Path,
    candidates: list[dict[str, object]],
    selected_id: str,
    output_path: Path,
) -> None:
    tiles: list[tuple[str, Image.Image, bool]] = []
    with Image.open(original_path) as image:
        tiles.append(("ORIGINAL", image.convert("RGB"), False))
    for candidate in candidates:
        with Image.open(Path(str(candidate["overlay_absolute_path"]))) as image:
            label = (
                f'{candidate["candidate_id"]}  '
                f'{float(candidate["camera_distance_m"]):.1f}m  '
                f'risk={float(candidate["quality_risk_score"]):.1f}'
            )
            tiles.append(
                (
                    label,
                    image.convert("RGB"),
                    str(candidate["candidate_id"]) == selected_id,
                )
            )
    columns = 3
    tile_width, image_height, label_height = 512, 342, 34
    rows = (len(tiles) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_width, rows * (image_height + label_height)), "white")
    draw = ImageDraw.Draw(sheet)
    label_font = font(17)
    for index, (label, image, selected) in enumerate(tiles):
        x = (index % columns) * tile_width
        y = (index // columns) * (image_height + label_height)
        tile = image.resize((tile_width, image_height), Image.Resampling.LANCZOS)
        sheet.paste(tile, (x, y + label_height))
        fill = (20, 110, 45) if selected else (25, 32, 39)
        draw.text((x + 8, y + 7), ("SELECTED  " if selected else "") + label, font=label_font, fill=fill)
        draw.rectangle(
            (x, y, x + tile_width - 1, y + image_height + label_height - 1),
            outline=(35, 150, 65) if selected else (175, 181, 185),
            width=4 if selected else 1,
        )
    atomic_jpeg(output_path, sheet, quality=90)


def projection_candidate(
    instance_xyz: np.ndarray,
    pose: CameraPose,
    panorama_size: tuple[int, int],
    candidate_id: str,
    camera_distance: float,
    *,
    margin: float,
    method: str,
) -> dict[str, object]:
    width, height = panorama_size
    u, v, _ = project_equirectangular(instance_xyz, pose, width, height, apply_tilt=True)
    crop = projection_crop_bounds(u, v, width, height, margin=margin)
    panorama_visible = float(np.mean((v >= 0) & (v < height)))
    return {
        "candidate_id": candidate_id,
        "method": method,
        "margin": margin,
        "pose": pose,
        "camera_distance_m": camera_distance,
        "crop": crop,
        "crop_visible_fraction": crop_visible_fraction(u, v, crop, width),
        "panorama_visible_fraction": panorama_visible,
        "projected_crop_area_px": crop.width_px * crop.height_px,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--c1-root", type=Path, default=DEFAULT_C1_ROOT)
    parser.add_argument("--d1-root", type=Path, default=DEFAULT_D1_ROOT)
    parser.add_argument("--action-dir", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--scope", choices=("evaluation", "training_risk"), default="evaluation"
    )
    parser.add_argument("--nearest-count", type=int, default=6)
    parser.add_argument("--alternate-count", type=int, default=5)
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def preflight(args: argparse.Namespace) -> dict[str, object]:
    args.raw_root = args.raw_root.resolve()
    args.c1_root = args.c1_root.resolve()
    args.d1_root = args.d1_root.resolve()
    args.action_dir = (args.action_dir or latest_action_dir(args.d1_root)).resolve()
    if args.nearest_count < 2 or args.alternate_count < 1:
        raise ValueError("nearest-count must be >=2 and alternate-count must be positive")

    action_path = args.action_dir / f"{args.scope}_rework_queue.csv"
    candidate_path = args.d1_root / "candidate_manifest.json"
    review_manifest_path = args.d1_root / "review_manifest.json"
    review_path = args.d1_root / "user_visual_review.json"
    c1_manifest_path = args.c1_root / "shared_manifest.json"
    for path in (action_path, candidate_path, review_manifest_path, review_path, c1_manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    action_rows = read_csv(action_path)
    if not action_rows:
        raise ValueError(f"The D1 {args.scope} rework queue is empty")
    if any(row.get("review_scope") != args.scope for row in action_rows):
        raise ValueError(f"The D1 rework queue contains records outside {args.scope}")
    keys = [row["sample_key"] for row in action_rows]
    if len(keys) != len(set(keys)):
        raise ValueError("The D1 rework queue contains duplicate sample keys")
    actions = Counter(row["action"] for row in action_rows)
    if set(actions) - VALID_ACTIONS:
        raise ValueError(f"Unexpected D1 rework actions: {sorted(set(actions) - VALID_ACTIONS)}")

    candidate_manifest = read_json(candidate_path)
    source_records = candidate_manifest.get("records")
    if not isinstance(source_records, list):
        raise ValueError("Invalid D1 candidate manifest")
    by_key = {str(record["sample_key"]): dict(record) for record in source_records}
    if any(key not in by_key for key in keys):
        raise ValueError("A rework key is missing from the D1 candidate manifest")
    classes = {
        str(record["model_class_name"])
        for record in source_records
        if "model_class_name" in record
    }
    if classes != EXPECTED_CLASSES:
        raise ValueError(f"Unexpected D1 classes: {sorted(classes)}")

    review_state = read_json(review_path)
    if review_state.get("reviewer") != "user":
        raise ValueError("The D1 review checkpoint is not user-owned")
    if review_state.get("source_manifest_sha256") != sha256_file(review_manifest_path):
        raise ValueError("The D1 review checkpoint is bound to a different manifest")
    reviews = review_state.get("reviews")
    if not isinstance(reviews, dict) or any(key not in reviews for key in keys):
        raise ValueError("Every D1 rework item must have an authoritative user review")

    if candidate_manifest.get("source_manifest_sha256") != sha256_file(c1_manifest_path):
        raise ValueError("The local C1 manifest does not match the D1 candidate source hash")
    for row in action_rows:
        record = by_key[row["sample_key"]]
        if row["action"] != reviews[row["sample_key"]].get("action"):
            raise ValueError(f'Action changed for {row["sample_key"]}')
        source_image = args.c1_root / str(record["image_path"])
        if not source_image.is_file() or sha256_file(source_image) != str(record["image_sha256"]):
            raise ValueError(f'Source image mismatch: {row["sample_key"]}')
        if row["image_sha256"] != str(record["image_sha256"]):
            raise ValueError(f'Action-list image hash mismatch: {row["sample_key"]}')
        trajectory_csv = (
            args.raw_root / str(record["road_id"]) / "hdi" / str(record["trajectory_id"]) / "traj.csv"
        )
        image_dir = args.raw_root / str(record["road_id"]) / "image" / str(record["trajectory_id"])
        if not trajectory_csv.is_file() or not image_dir.is_dir():
            raise FileNotFoundError(f'Raw trajectory assets missing: {row["sample_key"]}')

    return {
        "action_path": action_path,
        "candidate_path": candidate_path,
        "review_manifest_path": review_manifest_path,
        "review_path": review_path,
        "c1_manifest_path": c1_manifest_path,
        "action_rows": action_rows,
        "source_records": by_key,
        "candidate_manifest": candidate_manifest,
        "review_state": review_state,
        "source_review_sha256": sha256_file(review_path),
        "counts": dict(actions),
    }


def main() -> None:
    args = parse_args()
    state = preflight(args)
    summary = {
        "status": "preflight_passed",
        "action_dir": str(args.action_dir),
        "review_scope": args.scope,
        "total": len(state["action_rows"]),
        "counts": state["counts"],
        "trajectory_count": len(
            {
                (row["road_id"], row["trajectory_id"])
                for row in state["action_rows"]
            }
        ),
        "source_review_sha256": state["source_review_sha256"],
    }
    if args.check_only:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    output_root = (
        args.output_root
        or args.d1_root
        / ("rework_batches" if args.scope == "evaluation" else "training_rework_batches")
        / batch_stamp()
    ).resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to reuse an existing D1 rework batch: {output_root}")
    output_root.mkdir(parents=True)
    atomic_json(output_root / "progress.json", {**summary, "status": "running", "started_at": timestamp()})

    action_by_key = {row["sample_key"]: row for row in state["action_rows"]}
    source_records = state["source_records"]
    review_state = state["review_state"]
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in state["action_rows"]:
        record = source_records[row["sample_key"]]
        groups[(str(record["road_id"]), str(record["trajectory_id"]))].append(record)

    candidate_rows: list[dict[str, object]] = []
    replacement_records: list[dict[str, object]] = []
    completed = 0
    for (road_id, trajectory_id), records in sorted(groups.items()):
        stats = [
            TreeInstanceStats(
                tree_id=int(record["tree_id"]),
                point_count=int(record["source_point_count"]),
                label_counts=((int(record["raw_label_id"]), int(record["source_point_count"])),),
                min_xyz=tuple(float(value) for value in record["min_xyz"]),
                max_xyz=tuple(float(value) for value in record["max_xyz"]),
            )
            for record in records
        ]
        with open_whu_trajectory(
            args.raw_root, road_id, trajectory_id, chunk_size=args.chunk_size
        ) as reader:
            instances = extract_tree_instances(reader, stats)
        instance_by_id = {instance.stats.tree_id: instance for instance in instances}

        trajectory_csv = args.raw_root / road_id / "hdi" / trajectory_id / "traj.csv"
        diagnostics = orientation_column_diagnostics(trajectory_csv)
        if diagnostics["heading_column"] != 3:
            raise ValueError(f"Unexpected heading column for {road_id}/{trajectory_id}")
        poses = read_trajectory_csv(trajectory_csv)
        pose_by_name = {pose.image_name: pose for pose in poses}
        image_dir = args.raw_root / road_id / "image" / trajectory_id
        with Image.open(image_dir / poses[0].image_name) as first_image:
            panorama_size = first_image.size

        candidates_by_key: dict[str, list[dict[str, object]]] = {}
        jobs_by_panorama: dict[str, list[tuple[str, dict[str, object]]]] = defaultdict(list)
        for record in records:
            key = str(record["sample_key"])
            instance = instance_by_id[int(record["tree_id"])]
            center = instance.xyz.astype(np.float64).mean(axis=0)
            action = action_by_key[key]["action"]
            generated: list[dict[str, object]] = []
            if action == "retry_alternate_view":
                ranked = nearest_poses(center, poses, count=min(args.nearest_count, len(poses)))
                alternates = [item for item in ranked if item[0].image_name != record["image_name"]]
                for index, (pose, distance) in enumerate(alternates[: args.alternate_count], start=1):
                    generated.append(
                        projection_candidate(
                            instance.xyz,
                            pose,
                            panorama_size,
                            f"alternate_{index}",
                            distance,
                            margin=1.25,
                            method="alternate_view",
                        )
                    )
            else:
                pose = pose_by_name.get(str(record["image_name"]))
                if pose is None:
                    raise ValueError(f"Original panorama pose missing for {key}")
                for margin in (1.00, 1.10, 1.40):
                    generated.append(
                        projection_candidate(
                            instance.xyz,
                            pose,
                            panorama_size,
                            f"recrop_margin_{int(round(margin * 100)):03d}",
                            float(record["camera_distance_m"]),
                            margin=margin,
                            method="manual_recrop_candidate",
                        )
                    )
            if not generated:
                raise ValueError(f"No replacement candidates for {key}")
            candidates_by_key[key] = generated
            for candidate in generated:
                jobs_by_panorama[candidate["pose"].image_name].append((key, candidate))

        for image_name, jobs in sorted(jobs_by_panorama.items()):
            with Image.open(image_dir / image_name) as source:
                panorama = source.convert("RGB")
            if panorama.size != panorama_size:
                raise ValueError(f"Panorama size changed in {road_id}/{trajectory_id}")
            for key, candidate in jobs:
                crop = candidate["crop"]
                crop_image = wrapped_crop(
                    panorama,
                    crop.left_unwrapped_px,
                    crop.top_px,
                    crop.width_px,
                    crop.height_px,
                ).resize(OUTPUT_IMAGE_SIZE, Image.Resampling.LANCZOS)
                candidate_dir = output_root / "candidates" / key
                image_path = candidate_dir / f'{candidate["candidate_id"]}.jpg'
                overlay_path = candidate_dir / f'{candidate["candidate_id"]}_overlay.jpg'
                atomic_jpeg(image_path, crop_image)
                instance = instance_by_id[int(source_records[key]["tree_id"])]
                overlay = overlay_points(
                    crop_image,
                    instance.xyz,
                    candidate["pose"],
                    crop,
                    panorama_size[0],
                    panorama_size[1],
                )
                atomic_jpeg(overlay_path, overlay)
                metrics = image_metrics(crop_image)
                risk_score, risk_reasons = quality_risk(metrics)
                candidate.update(
                    {
                        "image_path": image_path.relative_to(output_root).as_posix(),
                        "overlay_path": overlay_path.relative_to(output_root).as_posix(),
                        "image_absolute_path": str(image_path),
                        "overlay_absolute_path": str(overlay_path),
                        "image_sha256": sha256_file(image_path),
                        "image_bytes": image_path.stat().st_size,
                        "automatic_metrics": metrics,
                        "quality_risk_score": risk_score,
                        "quality_risk_reasons": risk_reasons,
                    }
                )

        for record in sorted(records, key=lambda item: str(item["sample_key"])):
            key = str(record["sample_key"])
            candidates = candidates_by_key[key]
            source_action = action_by_key[key]["action"]
            if source_action == "manual_recrop":
                review_reasons = {
                    reason
                    for reason in action_by_key[key].get("reasons", "").split("|")
                    if reason
                }
                if "projection_offset" in review_reasons:
                    preferred_margin = (
                        1.10
                        if review_reasons & {"background_clutter", "neighbor_tree"}
                        else 1.40
                    )
                else:
                    preferred_margin = 1.00
                selected = min(
                    candidates,
                    key=lambda item: (
                        float(item["quality_risk_score"]),
                        abs(float(item["margin"]) - preferred_margin),
                        -round(float(item["panorama_visible_fraction"]), 6),
                        str(item["candidate_id"]),
                    ),
                )
            else:
                selected = min(
                    candidates,
                    key=lambda item: (
                        float(item["quality_risk_score"]),
                        -round(float(item["panorama_visible_fraction"]), 6),
                        -float(item["projected_crop_area_px"]),
                        float(item["camera_distance_m"]),
                        str(item["candidate_id"]),
                    ),
                )
            selected_id = str(selected["candidate_id"])
            selected_image = output_root / str(selected["image_path"])
            final_image = output_root / "assets" / "images" / f"{key}.jpg"
            final_point = output_root / "assets" / "points" / f"{key}.npz"
            final_image.parent.mkdir(parents=True, exist_ok=True)
            final_point.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(selected_image, final_image)
            shutil.copy2(args.c1_root / str(record["point_path"]), final_point)
            comparison_path = output_root / "comparisons" / f"{key}.jpg"
            comparison_sheet(
                args.c1_root / str(record["image_path"]),
                candidates,
                selected_id,
                comparison_path,
            )

            review = dict(review_state["reviews"][key])
            replacement = {
                **record,
                "point_path": final_point.relative_to(output_root).as_posix(),
                "image_path": final_image.relative_to(output_root).as_posix(),
                "image_name": selected["pose"].image_name,
                "camera_pose": selected["pose"].to_dict(),
                "camera_distance_m": selected["camera_distance_m"],
                "panorama_size": list(panorama_size),
                "crop": selected["crop"].to_dict(),
                "crop_visible_fraction": selected["crop_visible_fraction"],
                "panorama_visible_fraction": selected["panorama_visible_fraction"],
                "view_candidate_count": len(candidates),
                "point_sha256": sha256_file(final_point),
                "point_mtime_ns": final_point.stat().st_mtime_ns,
                "point_bytes": final_point.stat().st_size,
                "image_sha256": sha256_file(final_image),
                "image_mtime_ns": final_image.stat().st_mtime_ns,
                "image_bytes": final_image.stat().st_size,
                "review_scope": args.scope,
                "review_required": True,
                "automatic_quality_tier": (
                    "risk" if selected["quality_risk_reasons"] else "passed"
                ),
                "automatic_risk_score": selected["quality_risk_score"],
                "automatic_risk_reasons": selected["quality_risk_reasons"],
                "automatic_metrics": selected["automatic_metrics"],
                "quality_preview_path": selected["overlay_path"],
                "comparison_image_path": comparison_path.relative_to(output_root).as_posix(),
                "replacement_method": selected["method"],
                "replacement_candidate_id": selected_id,
                "replacement_source_action": action_by_key[key]["action"],
                "original_image_path": str(record["image_path"]),
                "original_image_sha256": str(record["image_sha256"]),
                "original_image_name": str(record["image_name"]),
                "original_review": review,
            }
            replacement_records.append(replacement)
            for candidate in candidates:
                candidate_rows.append(
                    {
                        "sample_key": key,
                        "source_action": action_by_key[key]["action"],
                        "candidate_id": candidate["candidate_id"],
                        "selected": candidate is selected,
                        "method": candidate["method"],
                        "margin": candidate["margin"],
                        "image_name": candidate["pose"].image_name,
                        "camera_distance_m": candidate["camera_distance_m"],
                        "crop_visible_fraction": candidate["crop_visible_fraction"],
                        "panorama_visible_fraction": candidate["panorama_visible_fraction"],
                        "projected_crop_area_px": candidate["projected_crop_area_px"],
                        "quality_risk_score": candidate["quality_risk_score"],
                        "quality_risk_reasons": "|".join(candidate["quality_risk_reasons"]),
                        "mean_luminance": candidate["automatic_metrics"]["mean_luminance"],
                        "dynamic_range_90": candidate["automatic_metrics"]["dynamic_range_90"],
                        "edge_energy": candidate["automatic_metrics"]["edge_energy"],
                        "image_path": candidate["image_path"],
                        "overlay_path": candidate["overlay_path"],
                        "image_sha256": candidate["image_sha256"],
                    }
                )
            completed += 1
            atomic_json(
                output_root / "progress.json",
                {
                    **summary,
                    "status": "running",
                    "completed": completed,
                    "total": len(state["action_rows"]),
                    "updated_at": timestamp(),
                },
            )

    replacement_records.sort(key=lambda item: str(item["sample_key"]))
    manifest_path = output_root / "replacement_review_manifest.json"
    manifest = {
        "format_version": 1,
        "stage": f"D1-four-class-image-{args.scope}-rework-review",
        "status": "awaiting_user_review",
        "generated_at": timestamp(),
        "source_d1_review": str(state["review_path"]),
        "source_d1_review_sha256": state["source_review_sha256"],
        "source_action_list": str(state["action_path"]),
        "source_action_list_sha256": sha256_file(state["action_path"]),
        "source_c1_manifest": str(state["c1_manifest_path"]),
        "source_c1_manifest_sha256": sha256_file(state["c1_manifest_path"]),
        "classes": state["candidate_manifest"]["classes"],
        "records": replacement_records,
    }
    atomic_json(manifest_path, manifest)
    review_path = output_root / "replacement_user_review.json"
    now = timestamp()
    atomic_json(
        review_path,
        {
            "schema_version": 1,
            "stage": f"D1-{args.scope}-replacement-human-review",
            "reviewer": "user",
            "dataset_root": str(output_root),
            "source_manifest": str(manifest_path),
            "source_manifest_sha256": sha256_file(manifest_path),
            "created_at": now,
            "updated_at": now,
            "scope_status": {},
            "reviews": {},
        },
    )
    atomic_csv(output_root / "candidate_inventory.csv", candidate_rows)
    atomic_csv(
        output_root / "replacement_selection.csv",
        [
            {
                "sample_key": record["sample_key"],
                "model_class_name": record["model_class_name"],
                "model_split": record["model_split"],
                "source_action": record["replacement_source_action"],
                "selected_candidate": record["replacement_candidate_id"],
                "replacement_method": record["replacement_method"],
                "replacement_image_path": record["image_path"],
                "replacement_image_sha256": record["image_sha256"],
                "comparison_image_path": record["comparison_image_path"],
                "automatic_quality_tier": record["automatic_quality_tier"],
                "automatic_risk_score": record["automatic_risk_score"],
                "automatic_risk_reasons": "|".join(record["automatic_risk_reasons"]),
            }
            for record in replacement_records
        ],
    )

    if sha256_file(state["review_path"]) != state["source_review_sha256"]:
        raise RuntimeError("The user-owned D1 review checkpoint changed during rework generation")
    selected_counts = Counter(record["replacement_method"] for record in replacement_records)
    risk_count = sum(record["automatic_quality_tier"] == "risk" for record in replacement_records)
    final_summary = {
        **summary,
        "status": "complete",
        "completed_at": timestamp(),
        "output_root": str(output_root),
        "replacement_count": len(replacement_records),
        "candidate_count": len(candidate_rows),
        "selected_method_counts": dict(selected_counts),
        "selected_automatic_risk_count": risk_count,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "replacement_review": str(review_path),
        "replacement_review_sha256": sha256_file(review_path),
        "source_review_unchanged": True,
    }
    atomic_json(output_root / "rework_summary.json", final_summary)
    atomic_json(output_root / "progress.json", final_summary)
    print(json.dumps(final_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
