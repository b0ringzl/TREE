"""Evaluate D1 fusion checkpoints on an exposure-paired cohort."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
import uuid
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


SANDBOX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SANDBOX_ROOT / "src"))
sys.path.insert(0, str(SANDBOX_ROOT / "scripts"))

from cache_d1_fusion_features import (  # noqa: E402
    build_models,
    image_transform,
    ptv2_forward_features,
)
from train_b4_tscmdl import make_model  # noqa: E402
from tscmdl_whu import knn_query_packed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--evaluation-manifest", type=Path)
    parser.add_argument("--image-point-comparison", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def atomic_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def exposure_group(reasons: list[str]) -> str:
    if "excessive_dark_pixels" in reasons:
        return "dark"
    if "excessive_bright_pixels" in reasons:
        return "bright"
    return "other"


def finite_probabilities(values: np.ndarray, sample_key: str) -> None:
    if values.ndim != 1 or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError(f"Invalid probabilities: {sample_key}")
    if not math.isclose(float(values.sum()), 1.0, rel_tol=0.0, abs_tol=1e-5):
        raise ValueError(f"Probabilities do not sum to one: {sample_key}")


def sample_std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) >= 2 else 0.0


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot summarize an empty cohort")

    def one(group: list[dict[str, Any]]) -> dict[str, Any]:
        correct = sum(int(row["correct"]) for row in group)
        return {
            "sample_count": len(group),
            "correct_count": correct,
            "accuracy": correct / len(group),
            "mean_true_class_probability": statistics.fmean(
                float(row["true_class_probability"]) for row in group
            ),
            "mean_confidence": statistics.fmean(
                float(row["confidence"]) for row in group
            ),
        }

    by_exposure: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_species: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_exposure[str(row["exposure_group"])].append(row)
        by_species[str(row["true_species"])].append(row)
    return {
        **one(rows),
        "by_exposure": {
            name: one(group) for name, group in sorted(by_exposure.items())
        },
        "by_species": {
            name: one(group) for name, group in sorted(by_species.items())
        },
    }


def main() -> None:
    args = parse_args()
    suite_root = args.suite_root.resolve()
    dataset_root = args.dataset_root.resolve()
    comparison_path = args.image_point_comparison.resolve()
    output_dir = args.output_dir.resolve()
    protocol_path = suite_root / "protocol.json"
    holdout_path = (
        args.evaluation_manifest.resolve()
        if args.evaluation_manifest
        else dataset_root / "holdout" / "manifest.json"
    )
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    holdout = json.loads(holdout_path.read_text(encoding="utf-8"))
    prior_comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    seeds = [int(value) for value in protocol["seeds"]]
    class_names = [str(value) for value in protocol["class_order"]]
    holdout_classes = [
        str(item["scientific_name"])
        for item in sorted(holdout["classes"], key=lambda item: item["class_index"])
    ]
    if class_names != holdout_classes or not holdout["records"]:
        raise ValueError("Holdout contract differs from the frozen fusion protocol")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(
        "cpu"
        if args.device == "cpu"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    transform = image_transform()
    records: list[dict[str, Any]] = []
    image_tensors: list[torch.Tensor] = []
    point_tensors: list[torch.Tensor] = []
    reference_tensors: list[torch.Tensor] = []
    verified_images: dict[str, str] = {}
    verified_points: dict[str, str] = {}
    for item in holdout["records"]:
        sample_key = str(item["sample_key"])
        image_path = (holdout_path.parent / str(item["image_path"])).resolve()
        point_path = (holdout_path.parent / str(item["point_path"])).resolve()
        image_hash = sha256_file(image_path)
        point_hash = sha256_file(point_path)
        if image_hash != str(item["packaged_image_sha256"]):
            raise ValueError(f"Image hash mismatch: {sample_key}")
        if point_hash != str(item["packaged_point_sha256"]):
            raise ValueError(f"Point hash mismatch: {sample_key}")
        with Image.open(image_path) as image:
            image_tensor = transform(image.convert("RGB"))
        with np.load(point_path, allow_pickle=False) as archive:
            points = np.asarray(archive["points_xyz"], dtype=np.float32)
            label = int(archive["class_index"])
        if points.shape != (8192, 3) or not np.isfinite(points).all():
            raise ValueError(f"Invalid point tensor: {sample_key}")
        if label != int(item["class_index"]):
            raise ValueError(f"Point label mismatch: {sample_key}")
        point_tensor = torch.from_numpy(points).to(device)
        reference = knn_query_packed(
            point_tensor,
            torch.tensor([8192], dtype=torch.long, device=device),
            8,
        )
        if tuple(reference.shape) != (8192, 8):
            raise ValueError(f"Unexpected kNN shape: {sample_key}")
        records.append(item)
        image_tensors.append(image_tensor)
        point_tensors.append(point_tensor.cpu())
        reference_tensors.append(reference.cpu())
        verified_images[sample_key] = image_hash
        verified_points[sample_key] = point_hash

    images = torch.stack(image_tensors).to(device)
    points = torch.stack(point_tensors).to(device)
    references = torch.stack(reference_tensors).to(device)
    long_rows: list[dict[str, Any]] = []
    probabilities_by_seed: dict[int, np.ndarray] = {}
    final_metrics_by_seed: dict[int, dict[str, Any]] = {}
    input_hashes: dict[str, dict[str, str]] = {}
    for seed in seeds:
        run_dir = suite_root / "runs" / f"seed_{seed}"
        config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
        final = json.loads((run_dir / "final_metrics.json").read_text(encoding="utf-8"))
        fusion_path = run_dir / "best.pt"
        fusion_checkpoint = torch.load(
            fusion_path, map_location="cpu", weights_only=False
        )
        source_paths = {
            name: Path(path)
            for name, path in config["feature_cache"]["source_checkpoint_paths"].items()
        }
        source_hashes = {name: sha256_file(path) for name, path in source_paths.items()}
        if source_hashes != config["feature_cache"]["source_checkpoint_sha256"]:
            raise ValueError(f"Source checkpoint hash mismatch for seed {seed}")
        if int(fusion_checkpoint["epoch"]) != int(final["best_epoch"]):
            raise ValueError(f"Fusion best epoch mismatch for seed {seed}")
        if list(fusion_checkpoint["class_names"]) != class_names:
            raise ValueError(f"Fusion class order mismatch for seed {seed}")

        point_model, image_model, checkpoint_load = build_models(
            tuple(class_names), source_paths["point_best"], source_paths["image_best"]
        )
        point_model.to(device).eval()
        image_model.to(device).eval()
        image_model.to(memory_format=torch.channels_last)
        architecture = config["fusion_architecture"]
        fusion_model = make_model(
            len(class_names),
            float(architecture["dropout"]),
            str(architecture["normalization"]),
            tuple(int(value) for value in architecture["classifier"][:-1]),
            str(architecture["modality"]),
            int(architecture["point_raw"]),
            int(architecture["image_raw"]),
            int(architecture["modal_projection"]),
        ).to(device)
        fusion_model.load_state_dict(fusion_checkpoint["model_state"], strict=True)
        fusion_model.eval()
        with torch.inference_mode(), torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            point_features = ptv2_forward_features(point_model, points, references)
            image_features = image_model(images.to(memory_format=torch.channels_last))
        point_features = point_features.float()
        image_features = image_features.float()
        if point_features.shape != (
            len(records), int(checkpoint_load["point_feature_dim"])
        ):
            raise ValueError(f"Point feature shape mismatch for seed {seed}")
        if image_features.shape != (
            len(records), int(checkpoint_load["image_feature_dim"])
        ):
            raise ValueError(f"Image feature shape mismatch for seed {seed}")
        with torch.inference_mode(), torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            logits = fusion_model.forward_from_features(
                point_features, image_features, "fusion"
            )
        probabilities = torch.softmax(logits.float(), dim=1).cpu().numpy()
        probabilities_by_seed[seed] = probabilities
        final_metrics_by_seed[seed] = final
        input_hashes[str(seed)] = {
            "fusion_best": sha256_file(fusion_path),
            "point_best": source_hashes["point_best"],
            "image_best": source_hashes["image_best"],
        }
        for item, probability in zip(records, probabilities, strict=True):
            sample_key = str(item["sample_key"])
            finite_probabilities(probability, sample_key)
            true_class = int(item["class_index"])
            predicted_class = int(np.argmax(probability))
            row: dict[str, Any] = {
                "seed": seed,
                "best_epoch": int(final["best_epoch"]),
                "sample_key": sample_key,
                "tree_id": int(item["tree_id"]),
                "road_id": str(item["road_id"]),
                "trajectory_id": str(item["trajectory_id"]),
                "exposure_group": exposure_group(list(item["automatic_risk_reasons"])),
                "automatic_risk_reasons": ";".join(item["automatic_risk_reasons"]),
                "true_class": true_class,
                "true_species": class_names[true_class],
                "predicted_class": predicted_class,
                "predicted_species": class_names[predicted_class],
                "correct": int(predicted_class == true_class),
                "confidence": float(probability[predicted_class]),
                "true_class_probability": float(probability[true_class]),
            }
            for class_index, value in enumerate(probability):
                row[f"probability_{class_index}"] = float(value)
            long_rows.append(row)
        del point_model, image_model, fusion_model, fusion_checkpoint
        if device.type == "cuda":
            torch.cuda.empty_cache()

    recommended_seed = max(
        seeds,
        key=lambda seed: (
            float(final_metrics_by_seed[seed]["best_validation_score"]["macro_f1"]),
            float(final_metrics_by_seed[seed]["best_validation_score"]["accuracy"]),
        ),
    )
    ensemble_rows: list[dict[str, Any]] = []
    tree_rows: list[dict[str, Any]] = []
    for index, item in enumerate(records):
        sample_key = str(item["sample_key"])
        seed_rows = [row for row in long_rows if row["sample_key"] == sample_key]
        mean_probability = np.mean(
            np.stack([probabilities_by_seed[seed][index] for seed in seeds]), axis=0
        )
        finite_probabilities(mean_probability, sample_key)
        true_class = int(item["class_index"])
        ensemble_class = int(np.argmax(mean_probability))
        ensemble = {
            "sample_key": sample_key,
            "tree_id": int(item["tree_id"]),
            "exposure_group": exposure_group(list(item["automatic_risk_reasons"])),
            "true_species": class_names[true_class],
            "predicted_species": class_names[ensemble_class],
            "correct": int(ensemble_class == true_class),
            "confidence": float(mean_probability[ensemble_class]),
            "true_class_probability": float(mean_probability[true_class]),
        }
        ensemble_rows.append(ensemble)
        predictions = [str(row["predicted_species"]) for row in seed_rows]
        tree = {
            **ensemble,
            "correct_seed_count": sum(int(row["correct"]) for row in seed_rows),
            "seed_accuracy": statistics.fmean(float(row["correct"]) for row in seed_rows),
            "unique_seed_predictions": len(set(predictions)),
            "all_seeds_agree": int(len(set(predictions)) == 1),
            "recommended_seed": recommended_seed,
        }
        for seed in seeds:
            seed_row = next(row for row in seed_rows if int(row["seed"]) == seed)
            tree[f"seed_{seed}_prediction"] = seed_row["predicted_species"]
            tree[f"seed_{seed}_correct"] = seed_row["correct"]
            tree[f"seed_{seed}_confidence"] = seed_row["confidence"]
        tree_rows.append(tree)

    per_seed = {
        str(seed): summarize([row for row in long_rows if int(row["seed"]) == seed])
        for seed in seeds
    }
    accuracies = [float(per_seed[str(seed)]["accuracy"]) for seed in seeds]
    metrics = {
        "format_version": 1,
        "status": "complete",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "scope": holdout.get(
            "scope", "frozen-feature fusion exposure-probe descriptive evaluation"
        ),
        "interpretation_limit": holdout.get(
            "interpretation_limit",
            "Exposure-enriched cohort; fusion differences are descriptive, not causal.",
        ),
        "device": str(device),
        "class_names": class_names,
        "seeds": seeds,
        "recommended_seed_by_validation_macro_f1": recommended_seed,
        "per_seed": per_seed,
        "three_seed_accuracy": {
            "mean": statistics.fmean(accuracies),
            "sample_std": sample_std(accuracies),
        },
        "mean_probability_ensemble": summarize(ensemble_rows),
        "tree_count": len(records),
        "prediction_row_count": len(long_rows),
        "all_seed_agreement_tree_count": sum(
            int(row["all_seeds_agree"]) for row in tree_rows
        ),
    }

    prior_by_key = {str(row["sample_key"]): row for row in prior_comparison["rows"]}
    joined_rows: list[dict[str, Any]] = []
    for fusion in ensemble_rows:
        prior = prior_by_key[fusion["sample_key"]]
        row = {
            "sample_key": fusion["sample_key"],
            "tree_id": fusion["tree_id"],
            "exposure_group": fusion["exposure_group"],
            "true_species": fusion["true_species"],
            "image_prediction": prior["image_prediction"],
            "image_correct": int(prior["image_correct"]),
            "image_confidence": float(prior["image_confidence"]),
            "point_prediction": prior["point_prediction"],
            "point_correct": int(prior["point_correct"]),
            "point_confidence": float(prior["point_confidence"]),
            "fusion_prediction": fusion["predicted_species"],
            "fusion_correct": int(fusion["correct"]),
            "fusion_confidence": float(fusion["confidence"]),
            "fusion_true_class_probability": float(fusion["true_class_probability"]),
        }
        row["fusion_vs_image"] = (
            "same_correctness"
            if row["fusion_correct"] == row["image_correct"]
            else ("fusion_corrected_image" if row["fusion_correct"] else "fusion_degraded_image")
        )
        row["fusion_vs_point"] = (
            "same_correctness"
            if row["fusion_correct"] == row["point_correct"]
            else ("fusion_corrected_point" if row["fusion_correct"] else "fusion_degraded_point")
        )
        joined_rows.append(row)
    joined = {
        "format_version": 1,
        "status": "image_point_fusion_complete",
        "generated_at": metrics["generated_at"],
        "tree_count": len(joined_rows),
        "accuracy": {
            "image": statistics.fmean(float(row["image_correct"]) for row in joined_rows),
            "point": statistics.fmean(float(row["point_correct"]) for row in joined_rows),
            "fusion": statistics.fmean(float(row["fusion_correct"]) for row in joined_rows),
        },
        "fusion_vs_image_counts": dict(Counter(row["fusion_vs_image"] for row in joined_rows)),
        "fusion_vs_point_counts": dict(Counter(row["fusion_vs_point"] for row in joined_rows)),
        "interpretation_limit": metrics["interpretation_limit"],
        "rows": joined_rows,
    }

    probability_fields = [f"probability_{index}" for index in range(len(class_names))]
    long_fields = [
        "seed", "best_epoch", "sample_key", "tree_id", "road_id", "trajectory_id",
        "exposure_group", "automatic_risk_reasons", "true_class", "true_species",
        "predicted_class", "predicted_species", "correct", "confidence",
        "true_class_probability", *probability_fields,
    ]
    tree_fields = list(tree_rows[0])
    joined_fields = list(joined_rows[0])
    predictions_path = output_dir / "predictions_long.csv"
    tree_path = output_dir / "per_tree_summary.csv"
    metrics_path = output_dir / "metrics.json"
    joined_csv_path = output_dir / "three_modality_comparison.csv"
    joined_json_path = output_dir / "three_modality_comparison.json"
    atomic_csv(predictions_path, long_rows, long_fields)
    atomic_csv(tree_path, tree_rows, tree_fields)
    atomic_json(metrics_path, metrics)
    atomic_csv(joined_csv_path, joined_rows, joined_fields)
    atomic_json(joined_json_path, joined)
    manifest = {
        "format_version": 1,
        "status": "complete",
        "generated_at": metrics["generated_at"],
        "inputs": {
            "protocol": {"path": str(protocol_path), "sha256": sha256_file(protocol_path)},
            "holdout_manifest": {"path": str(holdout_path), "sha256": sha256_file(holdout_path)},
            "image_point_comparison": {"path": str(comparison_path), "sha256": sha256_file(comparison_path)},
            "checkpoints": input_hashes,
            "images_verified": verified_images,
            "points_verified": verified_points,
        },
        "implementation": {
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "device": str(device),
            "knn": "exact Euclidean 8-neighbour graph recomputed per holdout tree",
        },
        "outputs": {
            path.name: sha256_file(path)
            for path in (
                predictions_path, tree_path, metrics_path, joined_csv_path, joined_json_path
            )
        },
    }
    atomic_json(output_dir / "evaluation_manifest.json", manifest)
    print(json.dumps({"metrics": metrics, "comparison": joined}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
