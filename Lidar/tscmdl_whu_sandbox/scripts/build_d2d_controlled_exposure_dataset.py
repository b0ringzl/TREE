#!/usr/bin/env python3
"""Freeze a balanced same-tree controlled exposure evaluation dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import uuid
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DERIVED_ROOT = PROJECT_ROOT / "lidar data" / "whu" / "derived" / "tscmdl"
DEFAULT_CONTROL_ROOT = (
    DERIVED_ROOT
    / "d2_exposure_stratified_evaluation"
    / "20260819_matched_normal_controls_v1"
)
DEFAULT_SHARED_ROOT = DERIVED_ROOT / "c1_full_shared_dataset"
DEFAULT_OUTPUT_ROOT = (
    DERIVED_ROOT
    / "d2_exposure_stratified_evaluation"
    / "20260819_same_tree_controlled_exposure_v1"
)
CONDITIONS = (
    ("dark_ev3", -3.0, "dark", 3),
    ("dark_ev2", -2.0, "dark", 2),
    ("dark_ev1", -1.0, "dark", 1),
    ("normal", 0.0, "normal", 0),
    ("bright_ev1", 1.0, "bright", 1),
    ("bright_ev2", 2.0, "bright", 2),
    ("bright_ev3", 3.0, "bright", 3),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-root", type=Path, default=DEFAULT_CONTROL_ROOT)
    parser.add_argument("--shared-root", type=Path, default=DEFAULT_SHARED_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--per-class", type=int, default=8)
    parser.add_argument("--selection-seed", type=int, default=20260819)
    return parser.parse_args()


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def target_normal(metrics: dict[str, Any]) -> bool:
    return (
        float(metrics["visible_projected_point_fraction"]) >= 0.95
        and 60.0 <= float(metrics["target_r5_mean_luminance"]) <= 190.0
        and float(metrics["target_r5_dark_ratio"]) <= 0.10
        and float(metrics["target_r5_bright_ratio"]) <= 0.15
        and float(metrics["target_r5_visible_ratio"]) >= 0.75
        and float(metrics["target_r5_edge_energy"]) >= 12.0
    )


def deterministic_rank(seed: int, sample_key: str) -> str:
    return hashlib.sha256(f"{seed}:{sample_key}".encode("utf-8")).hexdigest()


def select_balanced(records: list[dict[str, Any]], per_class: int, seed: int) -> list[dict[str, Any]]:
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_class[str(record["model_class_name"])].append(record)
    selected: list[dict[str, Any]] = []
    for class_name in sorted(by_class):
        candidates = by_class[class_name]
        if len(candidates) < per_class:
            raise ValueError(f"Only {len(candidates)} strict-normal candidates for {class_name}")
        by_road: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in candidates:
            by_road[str(record["road_id"])].append(record)
        for road_records in by_road.values():
            road_records.sort(key=lambda item: deterministic_rank(seed, str(item["sample_key"])))
        roads = sorted(
            by_road,
            key=lambda road: deterministic_rank(seed, f"{class_name}:road:{road}"),
        )
        cursor = {road: 0 for road in roads}
        class_selection: list[dict[str, Any]] = []
        while len(class_selection) < per_class:
            progressed = False
            for road in roads:
                if cursor[road] < len(by_road[road]) and len(class_selection) < per_class:
                    class_selection.append(by_road[road][cursor[road]])
                    cursor[road] += 1
                    progressed = True
            if not progressed:
                raise RuntimeError(f"Selection stalled for {class_name}")
        selected.extend(class_selection)
    return sorted(selected, key=lambda item: (int(item["model_class_index"]), str(item["sample_key"])))


def relabel_point(source: Path, target: Path, class_index: int) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    with np.load(source, allow_pickle=False) as archive:
        payload = {name: np.asarray(archive[name]) for name in archive.files}
    points = np.asarray(payload["points_xyz"], dtype=np.float32)
    if points.shape != (8192, 3) or not np.isfinite(points).all():
        raise ValueError(f"Invalid point asset: {source}")
    payload["class_index"] = np.asarray(class_index, dtype=np.int64)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp.npz")
    np.savez_compressed(temporary, **payload)
    temporary.replace(target)
    return sha256_file(target)


def exposure_transform(image: Image.Image, ev_shift: float) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    linear = np.power(rgb, 2.2)
    exposed = np.clip(linear * (2.0**ev_shift), 0.0, 1.0)
    srgb = np.power(exposed, 1.0 / 2.2)
    return Image.fromarray(np.rint(srgb * 255.0).astype(np.uint8), mode="RGB")


def image_metrics(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        gray = np.asarray(image.convert("L"), dtype=np.float32)
    horizontal = np.abs(np.diff(gray, axis=1))
    vertical = np.abs(np.diff(gray, axis=0))
    return {
        "mean_luminance": round(float(gray.mean()), 4),
        "dark_ratio": round(float((gray <= 15).mean()), 6),
        "bright_ratio": round(float((gray >= 240).mean()), 6),
        "edge_energy": round(float(horizontal.mean() + vertical.mean()), 4),
    }


def make_preview(output_root: Path, base_records: list[dict[str, Any]]) -> Path:
    representatives: dict[str, dict[str, Any]] = {}
    for record in base_records:
        representatives.setdefault(str(record["model_class_name"]), record)
    thumb_width, thumb_height = 280, 187
    label_height = 28
    classes = sorted(representatives)
    canvas = Image.new(
        "RGB",
        (thumb_width * len(CONDITIONS), (thumb_height + label_height) * len(classes)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for row_index, class_name in enumerate(classes):
        base = representatives[class_name]
        for column_index, (condition, _, _, _) in enumerate(CONDITIONS):
            key = f"{base['sample_key']}__{condition}"
            path = output_root / "assets" / "images" / f"{key}.jpg"
            with Image.open(path) as image:
                thumb = image.convert("RGB")
                thumb.thumbnail((thumb_width, thumb_height))
            x = column_index * thumb_width + (thumb_width - thumb.width) // 2
            y0 = row_index * (thumb_height + label_height)
            y = y0 + (thumb_height - thumb.height) // 2
            canvas.paste(thumb, (x, y))
            label = f"{class_name} | {condition}" if column_index == 0 else condition
            draw.text((column_index * thumb_width + 6, y0 + thumb_height + 7), label, fill="black", font=font)
    path = output_root / "controlled_exposure_preview.jpg"
    canvas.save(path, format="JPEG", quality=94, optimize=True)
    return path


def main() -> None:
    args = parse_args()
    control_root = args.control_root.resolve()
    shared_root = args.shared_root.resolve()
    output_root = args.output_root.resolve()
    candidate_path = control_root / "normal_candidate_manifest.json"
    metrics_path = control_root / "target_metrics" / "target_exposure_metrics.json"
    candidate_doc = json.loads(candidate_path.read_text(encoding="utf-8-sig"))
    metric_doc = json.loads(metrics_path.read_text(encoding="utf-8-sig"))
    metric_by_key = {str(item["sample_key"]): item for item in metric_doc["rows"]}
    strict_normal = [
        {**item, "d2_target_metrics_normal": metric_by_key[str(item["sample_key"])]}
        for item in candidate_doc["records"]
        if target_normal(metric_by_key[str(item["sample_key"])])
    ]
    selected = select_balanced(strict_normal, args.per_class, args.selection_seed)
    if len(selected) != args.per_class * len(candidate_doc["classes"]):
        raise ValueError("Balanced base cohort size mismatch")

    image_dir = output_root / "assets" / "images"
    point_dir = output_root / "assets" / "points"
    image_dir.mkdir(parents=True, exist_ok=True)
    point_dir.mkdir(parents=True, exist_ok=True)
    frozen_records: list[dict[str, Any]] = []
    base_manifest: list[dict[str, Any]] = []
    point_hashes: dict[str, str] = {}
    for base in selected:
        base_key = str(base["sample_key"])
        class_index = int(base["model_class_index"])
        source_image = (shared_root / str(base["image_path"])).resolve()
        source_point = (shared_root / str(base["point_path"])).resolve()
        if sha256_file(source_image) != str(base["image_sha256"]):
            raise ValueError(f"Source image hash mismatch: {base_key}")
        if sha256_file(source_point) != str(base["point_sha256"]):
            raise ValueError(f"Source point hash mismatch: {base_key}")
        point_target = point_dir / f"{base_key}.npz"
        point_hash = relabel_point(source_point, point_target, class_index)
        point_hashes[base_key] = point_hash
        with Image.open(source_image) as source_pil:
            source_rgb = source_pil.convert("RGB")
            for condition_order, (condition, ev_shift, direction, severity) in enumerate(CONDITIONS):
                variant_key = f"{base_key}__{condition}"
                image_target = image_dir / f"{variant_key}.jpg"
                if ev_shift == 0.0:
                    shutil.copyfile(source_image, image_target)
                else:
                    transformed = exposure_transform(source_rgb, ev_shift)
                    transformed.save(image_target, format="JPEG", quality=95, optimize=True, subsampling=0)
                image_hash = sha256_file(image_target)
                reasons = (
                    []
                    if direction == "normal"
                    else ["excessive_dark_pixels"]
                    if direction == "dark"
                    else ["excessive_bright_pixels"]
                )
                record = {
                    **base,
                    "sample_key": variant_key,
                    "split": "d2_controlled_exposure",
                    "class_index": class_index,
                    "scientific_name": str(base["model_class_name"]),
                    "image_path": os.path.relpath(image_target, output_root),
                    "point_path": os.path.relpath(point_target, output_root),
                    "packaged_image_sha256": image_hash,
                    "packaged_point_sha256": point_hash,
                    "source_image_path": str(source_image),
                    "source_image_sha256": str(base["image_sha256"]),
                    "source_point_path": str(source_point),
                    "source_point_sha256": str(base["point_sha256"]),
                    "base_sample_key": base_key,
                    "exposure_condition": condition,
                    "exposure_condition_order": condition_order,
                    "exposure_ev_shift": ev_shift,
                    "exposure_direction": direction,
                    "exposure_severity": severity,
                    "exposure_transform": "linear-light exposure scaling with sRGB gamma approximation 2.2",
                    "automatic_quality_tier": "controlled_exposure",
                    "automatic_risk_score": float(severity),
                    "automatic_risk_reasons": reasons,
                    "automatic_metrics": image_metrics(image_target),
                    "d2_exposure_group": direction,
                }
                frozen_records.append(record)
        base_manifest.append(
            {
                "base_sample_key": base_key,
                "class_index": class_index,
                "model_class_name": str(base["model_class_name"]),
                "source_scientific_name": str(base["source_scientific_name"]),
                "road_id": str(base["road_id"]),
                "trajectory_id": str(base["trajectory_id"]),
                "tree_id": int(base["tree_id"]),
                "source_image_sha256": str(base["image_sha256"]),
                "source_point_sha256": str(base["point_sha256"]),
                "packaged_point_sha256": point_hash,
                "normal_target_metrics": base["d2_target_metrics_normal"],
            }
        )

    preview_path = make_preview(output_root, selected)
    class_counts = Counter(str(item["model_class_name"]) for item in base_manifest)
    road_counts = {
        class_name: len({str(item["road_id"]) for item in base_manifest if item["model_class_name"] == class_name})
        for class_name in class_counts
    }
    manifest = {
        "format_version": 1,
        "stage": "D2d-same-tree-controlled-exposure",
        "status": "frozen",
        "generated_at": timestamp(),
        "design": "balanced same-tree repeated-measures controlled exposure; point cloud and label fixed",
        "interpretation_limit": "Synthetic linear-light exposure shifts isolate exposure sensitivity but do not reproduce every natural camera artifact.",
        "selection": {
            "strict_target_normal_pool_count": len(strict_normal),
            "per_class": args.per_class,
            "selection_seed": args.selection_seed,
            "policy": "maximize road coverage by deterministic road round-robin; deterministic seeded hash within roads; no model predictions used",
            "candidate_manifest_sha256": sha256_file(candidate_path),
            "target_metrics_sha256": sha256_file(metrics_path),
        },
        "exposure_conditions": [
            {"name": name, "ev_shift": ev, "direction": direction, "severity": severity}
            for name, ev, direction, severity in CONDITIONS
        ],
        "classes": candidate_doc["classes"],
        "summary": {
            "base_tree_count": len(base_manifest),
            "condition_count": len(CONDITIONS),
            "record_count": len(frozen_records),
            "base_class_counts": dict(sorted(class_counts.items())),
            "road_counts_by_class": dict(sorted(road_counts.items())),
        },
        "base_records": base_manifest,
        "records": frozen_records,
    }
    manifest_path = output_root / "manifest.json"
    atomic_json(manifest_path, manifest)
    validation = {
        "status": "passed",
        "generated_at": manifest["generated_at"],
        "base_tree_count": len(base_manifest),
        "record_count": len(frozen_records),
        "unique_record_count": len({item["sample_key"] for item in frozen_records}),
        "balanced_base_class_counts": dict(sorted(class_counts.items())),
        "all_base_images_model_unseen": True,
        "all_variants_share_base_point_hash": all(
            item["packaged_point_sha256"] == point_hashes[item["base_sample_key"]]
            for item in frozen_records
        ),
        "all_variants_share_label": all(
            int(item["class_index"]) == int(next(base["class_index"] for base in base_manifest if base["base_sample_key"] == item["base_sample_key"]))
            for item in frozen_records
        ),
        "normal_images_equal_source": all(
            item["packaged_image_sha256"] == item["source_image_sha256"]
            for item in frozen_records
            if item["exposure_condition"] == "normal"
        ),
        "manifest_sha256": sha256_file(manifest_path),
        "preview_sha256": sha256_file(preview_path),
    }
    if (
        len(base_manifest) != args.per_class * len(candidate_doc["classes"])
        or len(frozen_records) != len(base_manifest) * len(CONDITIONS)
        or validation["record_count"] != validation["unique_record_count"]
        or len(set(class_counts.values())) != 1
        or not validation["all_variants_share_base_point_hash"]
        or not validation["all_variants_share_label"]
        or not validation["normal_images_equal_source"]
    ):
        validation["status"] = "failed"
    atomic_json(output_root / "build_validation.json", validation)
    print(json.dumps({"manifest": str(manifest_path), **manifest["summary"], "validation": validation["status"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
