#!/usr/bin/env python3
"""Build a same-tree controlled point-segmentation-quality ablation dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DERIVED_ROOT = PROJECT_ROOT / "lidar data" / "whu" / "derived" / "tscmdl"
DEFAULT_D2D_ROOT = (
    DERIVED_ROOT
    / "d2_exposure_stratified_evaluation"
    / "20260819_same_tree_controlled_exposure_v1"
)
DEFAULT_OUTPUT_ROOT = (
    DERIVED_ROOT
    / "d2_exposure_stratified_evaluation"
    / "20260819_same_tree_point_quality_ablation_v1"
)
POINT_COUNT = 8192
KEEP_LEVELS = (0.75, 0.50, 0.25)
CONTAMINATION_LEVELS = (0.10, 0.25, 0.40)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d2d-root", type=Path, default=DEFAULT_D2D_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=20260819)
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


def stable_seed(seed: int, *parts: str) -> int:
    value = hashlib.sha256(":".join([str(seed), *parts]).encode("utf-8")).digest()
    return int.from_bytes(value[:8], "little", signed=False)


def normalize_unit_sphere(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    work = np.asarray(points, dtype=np.float64)
    centroid = work.mean(axis=0)
    centered = work - centroid
    scale = float(np.linalg.norm(centered, axis=1).max())
    if not np.isfinite(scale) or scale <= np.finfo(np.float64).eps:
        raise ValueError("Degenerate point cloud")
    normalized = (centered / scale).astype(np.float32)
    return normalized, centroid, scale


def resample_with_replacement(
    points: np.ndarray,
    retained_indices: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    retained = np.asarray(retained_indices, dtype=np.int64)
    if len(retained) > POINT_COUNT or len(retained) == 0:
        raise ValueError("Invalid retained index count")
    extras = rng.choice(retained, size=POINT_COUNT - len(retained), replace=True)
    source_indices = np.concatenate((retained, extras))
    rng.shuffle(source_indices)
    return points[source_indices], source_indices


def save_variant(
    path: Path,
    points: np.ndarray,
    class_index: int,
    base_archive: dict[str, np.ndarray],
    metadata: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    normalized, centroid, scale = normalize_unit_sphere(points)
    if normalized.shape != (POINT_COUNT, 3) or not np.isfinite(normalized).all():
        raise ValueError(f"Invalid variant: {path}")
    payload = {
        **base_archive,
        "points_xyz": normalized,
        "class_index": np.asarray(class_index, dtype=np.int64),
        "centroid_xyz": centroid.astype(np.float64),
        "scale": np.asarray(scale, dtype=np.float64),
        "sampled_unique_point_count": np.asarray(
            len(np.unique(normalized, axis=0)), dtype=np.int64
        ),
        "quality_retained_target_fraction": np.asarray(
            metadata["retained_target_fraction"], dtype=np.float64
        ),
        "quality_contamination_fraction": np.asarray(
            metadata["contamination_fraction"], dtype=np.float64
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp.npz")
    np.savez_compressed(temporary, **payload)
    temporary.replace(path)
    radii = np.linalg.norm(normalized, axis=1)
    metrics = {
        "unique_output_point_count": int(payload["sampled_unique_point_count"]),
        "normalized_centroid_norm": float(np.linalg.norm(normalized.mean(axis=0))),
        "normalized_max_radius": float(radii.max()),
        "normalized_axis_std": [float(value) for value in normalized.std(axis=0)],
        "normalized_axis_range": [
            float(value) for value in (normalized.max(axis=0) - normalized.min(axis=0))
        ],
        "variant_pre_normalization_scale": scale,
    }
    return sha256_file(path), metrics


def donor_assignments(base_records: list[dict[str, Any]], seed: int) -> dict[str, str]:
    assignments: dict[str, str] = {}
    for target in base_records:
        candidates = [
            item
            for item in base_records
            if item["model_class_name"] != target["model_class_name"]
        ]
        candidates.sort(
            key=lambda item: (
                0 if item["road_id"] == target["road_id"] else 1,
                hashlib.sha256(
                    f"{seed}:donor:{target['base_sample_key']}:{item['base_sample_key']}".encode("utf-8")
                ).hexdigest(),
            )
        )
        assignments[str(target["base_sample_key"])] = str(candidates[0]["base_sample_key"])
    return assignments


def base_archive(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as archive:
        points = np.asarray(archive["points_xyz"], dtype=np.float32)
        payload = {name: np.asarray(archive[name]) for name in archive.files}
    if points.shape != (POINT_COUNT, 3) or not np.isfinite(points).all():
        raise ValueError(f"Invalid base points: {path}")
    return points, payload


def quality_conditions() -> list[dict[str, Any]]:
    conditions = [
        {
            "name": "baseline_full",
            "factor": "baseline",
            "level": 0.0,
            "retained_target_fraction": 1.0,
            "contamination_fraction": 0.0,
        }
    ]
    for keep in KEEP_LEVELS:
        suffix = int(round(keep * 100))
        conditions.append(
            {
                "name": f"completeness_keep{suffix}",
                "factor": "completeness",
                "level": 1.0 - keep,
                "retained_target_fraction": keep,
                "contamination_fraction": 0.0,
            }
        )
    for keep in KEEP_LEVELS:
        suffix = int(round(keep * 100))
        conditions.append(
            {
                "name": f"density_keep{suffix}",
                "factor": "density",
                "level": 1.0 - keep,
                "retained_target_fraction": keep,
                "contamination_fraction": 0.0,
            }
        )
    for contamination in CONTAMINATION_LEVELS:
        suffix = int(round(contamination * 100))
        conditions.append(
            {
                "name": f"impurity{suffix}",
                "factor": "impurity",
                "level": contamination,
                "retained_target_fraction": 1.0 - contamination,
                "contamination_fraction": contamination,
            }
        )
    return conditions


def preview(
    output_root: Path,
    records: list[dict[str, Any]],
    class_names: list[str],
) -> Path:
    conditions = (
        "baseline_full",
        "completeness_keep50",
        "completeness_keep25",
        "density_keep25",
        "impurity25",
        "impurity40",
    )
    representatives = {
        class_name: next(item for item in records if item["scientific_name"] == class_name)
        for class_name in class_names
    }
    fig = plt.figure(figsize=(16, 10.5))
    for row_index, class_name in enumerate(class_names):
        base_key = representatives[class_name]["base_sample_key"]
        for column_index, condition in enumerate(conditions):
            record = next(
                item
                for item in records
                if item["base_sample_key"] == base_key and item["point_quality_condition"] == condition
            )
            path = output_root / str(record["point_path"])
            with np.load(path, allow_pickle=False) as archive:
                points = np.asarray(archive["points_xyz"], dtype=np.float32)
            sample = points[::8]
            axis = fig.add_subplot(
                len(class_names), len(conditions), row_index * len(conditions) + column_index + 1, projection="3d"
            )
            axis.scatter(sample[:, 0], sample[:, 1], sample[:, 2], s=1.2, c=sample[:, 2], cmap="viridis")
            axis.view_init(elev=18, azim=-65)
            axis.set_axis_off()
            if row_index == 0:
                axis.set_title(condition.replace("_", "\n"), fontsize=8)
            if column_index == 0:
                axis.text2D(-0.12, 0.5, class_name, transform=axis.transAxes, rotation=90, va="center", fontsize=8)
    fig.suptitle("D2e point-quality ablation preview", fontsize=15)
    fig.tight_layout(rect=(0.02, 0.01, 1.0, 0.96))
    path = output_root / "point_quality_ablation_preview.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    args = parse_args()
    d2d_root = args.d2d_root.resolve()
    output_root = args.output_root.resolve()
    d2d_manifest_path = d2d_root / "manifest.json"
    d2d = json.loads(d2d_manifest_path.read_text(encoding="utf-8-sig"))
    base_records = [dict(item) for item in d2d["base_records"]]
    d2d_records = [dict(item) for item in d2d["records"]]
    normal_by_base = {
        str(item["base_sample_key"]): item
        for item in d2d_records
        if item["exposure_condition"] == "normal"
    }
    class_names = [
        str(item["scientific_name"])
        for item in sorted(d2d["classes"], key=lambda item: int(item["class_index"]))
    ]
    if Counter(item["model_class_name"] for item in base_records) != Counter({name: 8 for name in class_names}):
        raise ValueError("D2d base cohort is not four-class balanced")

    point_dir = output_root / "assets" / "points"
    image_dir = output_root / "assets" / "images"
    point_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    points_by_base: dict[str, np.ndarray] = {}
    archives_by_base: dict[str, dict[str, np.ndarray]] = {}
    for base in base_records:
        key = str(base["base_sample_key"])
        source = d2d_root / "assets" / "points" / f"{key}.npz"
        if sha256_file(source) != str(base["packaged_point_sha256"]):
            raise ValueError(f"D2d base point hash mismatch: {key}")
        points_by_base[key], archives_by_base[key] = base_archive(source)
        normal = normal_by_base[key]
        image_source = (d2d_root / str(normal["image_path"])).resolve()
        image_target = image_dir / f"{key}.jpg"
        shutil.copyfile(image_source, image_target)
        if sha256_file(image_target) != str(normal["packaged_image_sha256"]):
            raise ValueError(f"Normal image hash mismatch: {key}")

    donors = donor_assignments(base_records, args.seed)
    conditions = quality_conditions()
    condition_order = {item["name"]: index for index, item in enumerate(conditions)}
    output_records: list[dict[str, Any]] = []
    base_protocol: list[dict[str, Any]] = []
    for base in base_records:
        key = str(base["base_sample_key"])
        points = points_by_base[key]
        archive = archives_by_base[key]
        class_index = int(base["class_index"])
        image_path = image_dir / f"{key}.jpg"
        image_hash = sha256_file(image_path)
        _, unique_indices = np.unique(points, axis=0, return_index=True)
        unique_indices = np.sort(unique_indices.astype(np.int64))
        spatial_rng = np.random.default_rng(stable_seed(args.seed, key, "spatial_direction"))
        direction = spatial_rng.normal(size=3)
        direction[2] *= 0.65
        direction /= np.linalg.norm(direction)
        spatial_order = unique_indices[
            np.argsort(points[unique_indices] @ direction)[::-1]
        ]
        density_rng = np.random.default_rng(stable_seed(args.seed, key, "density_order"))
        density_order = density_rng.permutation(unique_indices)
        target_mix_rng = np.random.default_rng(stable_seed(args.seed, key, "target_mix_order"))
        target_mix_order = target_mix_rng.permutation(POINT_COUNT)
        donor_key = donors[key]
        donor_points = points_by_base[donor_key]
        donor_mix_rng = np.random.default_rng(stable_seed(args.seed, key, donor_key, "donor_mix_order"))
        donor_order = donor_mix_rng.permutation(POINT_COUNT)
        offset_rng = np.random.default_rng(stable_seed(args.seed, key, donor_key, "donor_offset"))
        angle = float(offset_rng.uniform(0.0, 2.0 * math.pi))
        offset = np.asarray([0.58 * math.cos(angle), 0.58 * math.sin(angle), 0.0], dtype=np.float32)
        placed_donor = donor_points * 0.72 + offset
        base_protocol.append(
            {
                **base,
                "spatial_keep_direction": [float(value) for value in direction],
                "donor_base_sample_key": donor_key,
                "donor_model_class_name": next(
                    item["model_class_name"] for item in base_records if item["base_sample_key"] == donor_key
                ),
                "donor_same_road": bool(
                    next(item["road_id"] for item in base_records if item["base_sample_key"] == donor_key)
                    == base["road_id"]
                ),
                "donor_scale": 0.72,
                "donor_offset": [float(value) for value in offset],
            }
        )
        for condition in conditions:
            name = str(condition["name"])
            variant_key = f"{key}__{name}"
            target_path = point_dir / f"{variant_key}.npz"
            if name == "baseline_full":
                source_path = d2d_root / "assets" / "points" / f"{key}.npz"
                shutil.copyfile(source_path, target_path)
                point_hash = sha256_file(target_path)
                with np.load(target_path, allow_pickle=False) as baseline_archive:
                    normalized = np.asarray(baseline_archive["points_xyz"], dtype=np.float32)
                quality_metrics = {
                    "unique_output_point_count": int(len(np.unique(normalized, axis=0))),
                    "normalized_centroid_norm": float(np.linalg.norm(normalized.mean(axis=0))),
                    "normalized_max_radius": float(np.linalg.norm(normalized, axis=1).max()),
                    "normalized_axis_std": [float(value) for value in normalized.std(axis=0)],
                    "normalized_axis_range": [
                        float(value) for value in (normalized.max(axis=0) - normalized.min(axis=0))
                    ],
                    "variant_pre_normalization_scale": float(archive["scale"]),
                }
            elif condition["factor"] in {"completeness", "density"}:
                keep_count = max(
                    1,
                    int(round(len(unique_indices) * float(condition["retained_target_fraction"]))),
                )
                order = spatial_order if condition["factor"] == "completeness" else density_order
                retained = order[:keep_count]
                rng = np.random.default_rng(stable_seed(args.seed, key, name, "resample"))
                pre_normalized, _ = resample_with_replacement(points, retained, rng)
                point_hash, quality_metrics = save_variant(
                    target_path, pre_normalized, class_index, archive, condition
                )
            elif condition["factor"] == "impurity":
                contamination_count = int(round(POINT_COUNT * float(condition["contamination_fraction"])))
                target_count = POINT_COUNT - contamination_count
                mixed = np.concatenate(
                    (
                        points[target_mix_order[:target_count]],
                        placed_donor[donor_order[:contamination_count]],
                    ),
                    axis=0,
                )
                mix_rng = np.random.default_rng(stable_seed(args.seed, key, name, "mix_shuffle"))
                mix_rng.shuffle(mixed, axis=0)
                point_hash, quality_metrics = save_variant(
                    target_path, mixed, class_index, archive, condition
                )
            else:
                raise ValueError(f"Unknown quality factor: {condition['factor']}")
            output_records.append(
                {
                    **normal_by_base[key],
                    "sample_key": variant_key,
                    "split": "d2_point_quality_controlled",
                    "image_path": os.path.relpath(image_path, output_root),
                    "point_path": os.path.relpath(target_path, output_root),
                    "packaged_image_sha256": image_hash,
                    "packaged_point_sha256": point_hash,
                    "base_sample_key": key,
                    "point_quality_condition": name,
                    "point_quality_condition_order": condition_order[name],
                    "point_quality_factor": condition["factor"],
                    "point_quality_level": float(condition["level"]),
                    "retained_target_fraction": float(condition["retained_target_fraction"]),
                    "contamination_fraction": float(condition["contamination_fraction"]),
                    "point_quality_metrics": quality_metrics,
                    "donor_base_sample_key": donor_key if condition["factor"] == "impurity" else "",
                    "donor_model_class_name": (
                        next(item["model_class_name"] for item in base_records if item["base_sample_key"] == donor_key)
                        if condition["factor"] == "impurity"
                        else ""
                    ),
                    "automatic_risk_reasons": [],
                    "automatic_quality_tier": "controlled_point_quality",
                    "automatic_risk_score": float(condition["level"]),
                }
            )

    output_records.sort(
        key=lambda item: (str(item["base_sample_key"]), int(item["point_quality_condition_order"]))
    )
    preview_path = preview(output_root, output_records, class_names)
    manifest = {
        "format_version": 1,
        "stage": "D2e-same-tree-point-quality-ablation",
        "status": "frozen",
        "generated_at": timestamp(),
        "design": "balanced same-tree repeated-measures point segmentation quality ablation with fixed normal image",
        "interpretation_limit": (
            "Synthetic ablations isolate spatial completeness, matched random density, and cross-class contamination. "
            "They approximate segmentation failures but do not reproduce every real segmentation algorithm error."
        ),
        "source_d2d_manifest": str(d2d_manifest_path),
        "source_d2d_manifest_sha256": sha256_file(d2d_manifest_path),
        "seed": args.seed,
        "classes": d2d["classes"],
        "quality_conditions": conditions,
        "summary": {
            "base_tree_count": len(base_records),
            "condition_count": len(conditions),
            "record_count": len(output_records),
            "base_class_counts": dict(sorted(Counter(item["model_class_name"] for item in base_records).items())),
            "donor_same_road_count": sum(bool(item["donor_same_road"]) for item in base_protocol),
        },
        "base_protocol": base_protocol,
        "records": output_records,
    }
    manifest_path = output_root / "manifest.json"
    atomic_json(manifest_path, manifest)
    condition_counts = Counter(item["point_quality_condition"] for item in output_records)
    baseline_equal = all(
        item["packaged_point_sha256"]
        == next(base["packaged_point_sha256"] for base in base_records if base["base_sample_key"] == item["base_sample_key"])
        for item in output_records
        if item["point_quality_condition"] == "baseline_full"
    )
    validation = {
        "status": "passed",
        "generated_at": manifest["generated_at"],
        "base_tree_count": len(base_records),
        "condition_count": len(conditions),
        "record_count": len(output_records),
        "unique_record_count": len({item["sample_key"] for item in output_records}),
        "condition_counts": dict(sorted(condition_counts.items())),
        "all_conditions_balanced": set(condition_counts.values()) == {len(base_records)},
        "baseline_points_byte_identical": baseline_equal,
        "all_images_fixed_within_tree": all(
            len({item["packaged_image_sha256"] for item in output_records if item["base_sample_key"] == base["base_sample_key"]}) == 1
            for base in base_records
        ),
        "all_point_arrays_valid": all(
            item["point_quality_metrics"]["normalized_max_radius"] <= 1.00001
            and item["point_quality_metrics"]["normalized_centroid_norm"] <= 1e-5
            for item in output_records
        ),
        "all_impurity_donors_cross_class": all(
            item["donor_model_class_name"] != item["scientific_name"]
            for item in output_records
            if item["point_quality_factor"] == "impurity"
        ),
        "completeness_density_unique_counts_matched": all(
            next(
                item["point_quality_metrics"]["unique_output_point_count"]
                for item in output_records
                if item["base_sample_key"] == base["base_sample_key"]
                and item["point_quality_condition"] == f"completeness_keep{level}"
            )
            == next(
                item["point_quality_metrics"]["unique_output_point_count"]
                for item in output_records
                if item["base_sample_key"] == base["base_sample_key"]
                and item["point_quality_condition"] == f"density_keep{level}"
            )
            for base in base_records
            for level in (75, 50, 25)
        ),
        "manifest_sha256": sha256_file(manifest_path),
        "preview_sha256": sha256_file(preview_path),
    }
    if not all(
        (
            validation["record_count"] == validation["unique_record_count"],
            validation["all_conditions_balanced"],
            validation["baseline_points_byte_identical"],
            validation["all_images_fixed_within_tree"],
            validation["all_point_arrays_valid"],
            validation["all_impurity_donors_cross_class"],
            validation["completeness_density_unique_counts_matched"],
        )
    ):
        validation["status"] = "failed"
    atomic_json(output_root / "build_validation.json", validation)
    print(json.dumps({"manifest": str(manifest_path), **manifest["summary"], "validation": validation["status"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
