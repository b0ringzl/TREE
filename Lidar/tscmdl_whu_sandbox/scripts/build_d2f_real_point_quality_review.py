"""Build the D2f real point-cloud quality review cohort and automatic metrics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


SANDBOX_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DERIVED_ROOT = PROJECT_ROOT / "lidar data" / "whu" / "derived" / "tscmdl"
DEFAULT_PACKAGE = (
    DERIVED_ROOT / "d1_four_class_training_package" / "20260817_165127"
)
DEFAULT_OUTPUT = (
    DERIVED_ROOT
    / "d2_exposure_stratified_evaluation"
    / "20260821_real_point_quality_review_v1"
)
DEFAULT_SUITES = {
    "image": DERIVED_ROOT / "d1_resnet50_clean_repeats" / "20260817_protocol_v1",
    "point": DERIVED_ROOT / "d1_ptv2_clean_repeats" / "20260818_protocol_v1",
    "fusion": DERIVED_ROOT / "d1_fusion_clean_repeats" / "20260818_protocol_v1",
}


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def entropy(probabilities: np.ndarray) -> float:
    values = np.clip(probabilities.astype(np.float64), 1e-12, 1.0)
    return float(-(values * np.log(values)).sum() / math.log(len(values)))


def load_ensemble_predictions(suite_root: Path) -> dict[str, dict[str, object]]:
    paths = sorted((suite_root / "runs").glob("seed_*/predictions.csv"))
    if len(paths) != 3:
        raise ValueError(f"expected three prediction files under {suite_root}, found {len(paths)}")
    rows_by_key: dict[str, list[dict[str, str]]] = defaultdict(list)
    for path in paths:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                rows_by_key[str(row["sample_key"])].append(row)
    result: dict[str, dict[str, object]] = {}
    for key, rows in rows_by_key.items():
        if len(rows) != 3:
            raise ValueError(f"{key} has {len(rows)} seeds in {suite_root}")
        true_class = int(rows[0]["true_class"])
        split = str(rows[0]["split"])
        probabilities = np.asarray(
            [[float(row[f"probability_{index}"]) for index in range(4)] for row in rows],
            dtype=np.float64,
        )
        mean_probabilities = probabilities.mean(axis=0)
        prediction = int(mean_probabilities.argmax())
        seed_predictions = probabilities.argmax(axis=1).tolist()
        result[key] = {
            "split": split,
            "true_class": true_class,
            "prediction": prediction,
            "correct": bool(prediction == true_class),
            "true_probability": float(mean_probabilities[true_class]),
            "confidence": float(mean_probabilities[prediction]),
            "entropy_normalized": entropy(mean_probabilities),
            "seed_agreement": float(Counter(seed_predictions).most_common(1)[0][1] / 3.0),
            "mean_probabilities": mean_probabilities.tolist(),
            "seed_predictions": seed_predictions,
            "source_files": [str(path.resolve()) for path in paths],
        }
    return result


def longest_empty_run(occupied: np.ndarray) -> int:
    longest = 0
    current = 0
    for value in occupied.tolist():
        if value:
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest


def point_metrics(point_path: Path) -> dict[str, object]:
    with np.load(point_path) as payload:
        points = np.asarray(payload["points_xyz"], dtype=np.float64)
        source_count = int(np.asarray(payload["source_point_count"]).item())
        stored_unique_count = int(np.asarray(payload["sampled_unique_point_count"]).item())
        centroid = np.asarray(payload["centroid_xyz"], dtype=np.float64).tolist()
        scale = float(np.asarray(payload["scale"]).item())
    if points.shape != (8192, 3) or not np.isfinite(points).all():
        raise ValueError(f"invalid point array: {point_path}")
    unique = np.unique(points, axis=0)
    quantile_low, quantile_high = np.quantile(unique, [0.01, 0.99], axis=0)
    extents = np.maximum(quantile_high - quantile_low, 1e-9)
    centered = unique - unique.mean(axis=0, keepdims=True)
    covariance = np.cov(centered, rowvar=False)
    eigenvalues = np.sort(np.linalg.eigvalsh(covariance))[::-1]
    eigenvalues = np.maximum(eigenvalues, 1e-12)
    eigen_fraction = eigenvalues / eigenvalues.sum()
    linearity = float((eigenvalues[0] - eigenvalues[1]) / eigenvalues[0])
    planarity = float((eigenvalues[1] - eigenvalues[2]) / eigenvalues[0])
    scattering = float(eigenvalues[2] / eigenvalues[0])

    spacing_points = unique
    if len(spacing_points) > 4096:
        indices = np.linspace(0, len(spacing_points) - 1, 4096, dtype=np.int64)
        spacing_points = spacing_points[indices]
    distances, _ = cKDTree(spacing_points).query(spacing_points, k=2, workers=-1)
    nearest = distances[:, 1]
    median_spacing = float(np.median(nearest))
    mad_spacing = float(np.median(np.abs(nearest - median_spacing)))
    isolation_threshold = median_spacing + 4.0 * max(mad_spacing, 1e-9)

    z = unique[:, 2]
    z_min, z_max = float(z.min()), float(z.max())
    z_histogram, _ = np.histogram(z, bins=20, range=(z_min, z_max))
    occupied_z = z_histogram > 0
    xy = unique[:, :2]
    radial = np.linalg.norm(xy - np.median(xy, axis=0, keepdims=True), axis=1)
    angles = np.mod(np.arctan2(xy[:, 1], xy[:, 0]), 2 * np.pi)
    angle_histogram, _ = np.histogram(angles, bins=24, range=(0.0, 2 * np.pi))
    voxel = np.floor((unique - unique.min(axis=0)) / np.maximum(np.ptp(unique, axis=0), 1e-9) * 15.999).astype(np.int16)
    occupied_voxels = int(len(np.unique(voxel, axis=0)))

    return {
        "source_point_count": source_count,
        "stored_sampled_unique_point_count": stored_unique_count,
        "actual_sampled_unique_point_count": int(len(unique)),
        "repeat_fraction": float(1.0 - len(unique) / len(points)),
        "centroid_xyz": centroid,
        "normalization_scale": scale,
        "robust_extent_x": float(extents[0]),
        "robust_extent_y": float(extents[1]),
        "robust_extent_z": float(extents[2]),
        "height_to_horizontal_ratio": float(extents[2] / math.sqrt(extents[0] * extents[1])),
        "pca_eigen_fraction": eigen_fraction.tolist(),
        "pca_linearity": linearity,
        "pca_planarity": planarity,
        "pca_scattering": scattering,
        "nearest_spacing_median": median_spacing,
        "nearest_spacing_p95": float(np.quantile(nearest, 0.95)),
        "nearest_spacing_cv": float(np.std(nearest) / max(np.mean(nearest), 1e-9)),
        "isolated_point_ratio": float(np.mean(nearest > isolation_threshold)),
        "vertical_bin_coverage": float(occupied_z.mean()),
        "vertical_longest_empty_run": int(longest_empty_run(occupied_z)),
        "angular_bin_coverage": float(np.mean(angle_histogram > 0)),
        "radial_p95": float(np.quantile(radial, 0.95)),
        "occupied_voxels_16cubed": occupied_voxels,
        "occupied_voxel_ratio": float(occupied_voxels / 4096.0),
    }


def percentile_ranks(values: list[float]) -> np.ndarray:
    order = np.argsort(np.asarray(values, dtype=np.float64), kind="mergesort")
    ranks = np.empty(len(order), dtype=np.float64)
    ranks[order] = np.arange(len(order), dtype=np.float64)
    return ranks / max(len(order) - 1, 1)


def add_risk_scores(records: list[dict[str, object]]) -> None:
    source_rank = percentile_ranks([float(r["point_metrics"]["source_point_count"]) for r in records])
    unique_rank = percentile_ranks([float(r["point_metrics"]["actual_sampled_unique_point_count"]) for r in records])
    isolation_rank = percentile_ranks([float(r["point_metrics"]["isolated_point_ratio"]) for r in records])
    spacing_rank = percentile_ranks([float(r["point_metrics"]["nearest_spacing_cv"]) for r in records])
    voxel_rank = percentile_ranks([float(r["point_metrics"]["occupied_voxel_ratio"]) for r in records])
    for index, record in enumerate(records):
        score = (
            0.30 * (1.0 - source_rank[index])
            + 0.20 * (1.0 - unique_rank[index])
            + 0.20 * isolation_rank[index]
            + 0.15 * spacing_rank[index]
            + 0.15 * (1.0 - voxel_rank[index])
        )
        record["automatic_point_quality_risk_score"] = float(score)
        record["automatic_point_quality_risk_decile"] = int(min(9, math.floor(score * 10.0)))


def deterministic_review_order(records: list[dict[str, object]]) -> None:
    for record in records:
        token = hashlib.sha256(
            f"d2f-review-v1|{record['sample_key']}".encode("utf-8")
        ).hexdigest()
        record["review_random_key"] = token
    records.sort(
        key=lambda row: (
            -int(row["automatic_point_quality_risk_decile"]),
            str(row["split"]),
            int(row["class_index"]),
            str(row["review_random_key"]),
        )
    )
    # Interleave the sorted strata so the first screen is not dominated by one class.
    buckets: dict[tuple[int, str, int], list[dict[str, object]]] = defaultdict(list)
    for record in records:
        buckets[
            (
                int(record["automatic_point_quality_risk_decile"]),
                str(record["split"]),
                int(record["class_index"]),
            )
        ].append(record)
    ordered: list[dict[str, object]] = []
    keys = sorted(buckets, key=lambda value: (-value[0], value[1], value[2]))
    while any(buckets[key] for key in keys):
        for key in keys:
            if buckets[key]:
                ordered.append(buckets[key].pop(0))
    records[:] = ordered
    for index, record in enumerate(records, start=1):
        record["review_order"] = index


def write_metrics_csv(path: Path, records: list[dict[str, object]]) -> None:
    rows = []
    for record in records:
        row = {
            "review_order": record["review_order"],
            "sample_key": record["sample_key"],
            "split": record["split"],
            "class_index": record["class_index"],
            "scientific_name": record["scientific_name"],
            "road_id": record["road_id"],
            "trajectory_id": record["trajectory_id"],
            "tree_id": record["tree_id"],
            "automatic_point_quality_risk_score": record["automatic_point_quality_risk_score"],
            "automatic_point_quality_risk_decile": record["automatic_point_quality_risk_decile"],
        }
        row.update(record["point_metrics"])
        row.pop("centroid_xyz", None)
        row.pop("pca_eigen_fraction", None)
        for modality, outcome in record["model_outcomes"].items():
            for field in ("prediction", "correct", "true_probability", "confidence", "entropy_normalized", "seed_agreement"):
                row[f"{modality}_{field}"] = outcome[field]
        rows.append(row)
    fields = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--image-suite", type=Path, default=DEFAULT_SUITES["image"])
    parser.add_argument("--point-suite", type=Path, default=DEFAULT_SUITES["point"])
    parser.add_argument("--fusion-suite", type=Path, default=DEFAULT_SUITES["fusion"])
    args = parser.parse_args()

    package_root = args.package_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = package_root / "manifest.json"
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    source_records = [row for row in source_manifest["records"] if row["split"] in {"val", "test"}]
    outcomes = {
        "image": load_ensemble_predictions(args.image_suite.resolve()),
        "point": load_ensemble_predictions(args.point_suite.resolve()),
        "fusion": load_ensemble_predictions(args.fusion_suite.resolve()),
    }

    records: list[dict[str, object]] = []
    for source in source_records:
        key = str(source["sample_key"])
        missing = [name for name, values in outcomes.items() if key not in values]
        if missing:
            raise ValueError(f"{key} missing prediction modalities: {missing}")
        point_path = package_root / str(source["point_path"])
        image_path = package_root / str(source["image_path"])
        record = {
            "sample_key": key,
            "split": str(source["split"]),
            "class_index": int(source["class_index"]),
            "scientific_name": str(source["scientific_name"]),
            "road_id": str(source["road_id"]),
            "trajectory_id": str(source["trajectory_id"]),
            "tree_id": int(source["tree_id"]),
            "image_path": str(image_path.resolve()),
            "point_path": str(point_path.resolve()),
            "image_sha256": sha256_file(image_path),
            "point_sha256": sha256_file(point_path),
            "point_metrics": point_metrics(point_path),
            "model_outcomes": {name: values[key] for name, values in outcomes.items()},
        }
        records.append(record)

    add_risk_scores(records)
    deterministic_review_order(records)
    class_counts = Counter(str(record["scientific_name"]) for record in records)
    split_counts = Counter(str(record["split"]) for record in records)
    road_counts = Counter(str(record["road_id"]) for record in records)
    manifest = {
        "format_version": 1,
        "stage": "D2f-real-point-quality-review",
        "status": "ready_for_human_review",
        "generated_at": timestamp(),
        "design": "full census of all model-unseen D1 validation and test trees; outcomes hidden during primary quality grading",
        "interpretation_limit": "Tree-ID-extracted reference point clouds, not automatic segmentation model outputs.",
        "package_root": str(package_root),
        "source_manifest": str(manifest_path.resolve()),
        "source_manifest_sha256": sha256_file(manifest_path),
        "sample_count": len(records),
        "split_counts": dict(sorted(split_counts.items())),
        "class_counts": dict(sorted(class_counts.items())),
        "road_count": len(road_counts),
        "review_protocol": {
            "completeness": ["complete", "slight_loss", "moderate_loss", "severe_loss", "unjudgeable"],
            "purity": ["clean", "slight_contamination", "moderate_contamination", "severe_contamination", "unjudgeable"],
            "overall_usability": ["pass", "caution", "fail"],
            "confidence": ["high", "medium", "low"],
            "issue_types": [
                "crown_truncated", "trunk_missing", "one_side_missing", "sparse",
                "fragmented", "neighbor_tree", "ground_or_background", "outliers", "other"
            ],
        },
        "records": records,
    }
    manifest_output = output_root / "manifest.json"
    atomic_json(manifest_output, manifest)
    write_metrics_csv(output_root / "automatic_metrics_and_hidden_outcomes.csv", records)
    review_state = {
        "schema_version": 1,
        "stage": "D2f-real-point-quality-review",
        "reviewer": "user",
        "source_manifest": str(manifest_output.resolve()),
        "source_manifest_sha256": sha256_file(manifest_output),
        "created_at": timestamp(),
        "updated_at": timestamp(),
        "reviews": {},
    }
    review_path = output_root / "review_state.json"
    if review_path.exists():
        existing = json.loads(review_path.read_text(encoding="utf-8-sig"))
        if existing.get("source_manifest_sha256") != review_state["source_manifest_sha256"]:
            raise ValueError("existing review_state.json belongs to a different manifest")
    else:
        atomic_json(review_path, review_state)

    unique_keys = {str(record["sample_key"]) for record in records}
    validation = {
        "status": "passed" if len(records) == 241 and len(unique_keys) == 241 else "failed",
        "generated_at": timestamp(),
        "record_count": len(records),
        "unique_sample_count": len(unique_keys),
        "split_counts": dict(sorted(split_counts.items())),
        "class_counts": dict(sorted(class_counts.items())),
        "all_three_modality_outcomes_present": all(
            set(record["model_outcomes"]) == {"image", "point", "fusion"} for record in records
        ),
        "all_assets_exist": all(
            Path(str(record["image_path"])).is_file() and Path(str(record["point_path"])).is_file()
            for record in records
        ),
        "all_point_arrays_valid": all(
            int(record["point_metrics"]["actual_sampled_unique_point_count"]) > 0 for record in records
        ),
        "prediction_outcomes_hidden_by_review_ui": True,
        "source_is_tree_id_extracted_not_automatic_segmentation": True,
        "manifest_sha256": sha256_file(manifest_output),
    }
    atomic_json(output_root / "validation.json", validation)
    print(json.dumps(validation, ensure_ascii=False, indent=2))
    return 0 if validation["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
