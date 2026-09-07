"""Evaluate validation-selected D1 PTv2 checkpoints on an exposure cohort."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


SANDBOX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SANDBOX_ROOT / "src"))

from tscmdl_whu import PointTransformerV2Classifier, knn_query_packed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--evaluation-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
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
    if values.ndim != 1 or not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError(f"Invalid probability vector: {sample_key}")
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
    output_dir = args.output_dir.resolve()
    protocol_path = suite_root / "protocol.json"
    holdout_path = (
        args.evaluation_manifest.resolve()
        if args.evaluation_manifest
        else dataset_root / "holdout" / "manifest.json"
    )
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    holdout = json.loads(holdout_path.read_text(encoding="utf-8"))
    seeds = [int(seed) for seed in protocol["seeds"]]
    class_names = [str(value) for value in protocol["class_order"]]
    holdout_classes = [
        str(item["scientific_name"])
        for item in sorted(holdout["classes"], key=lambda item: item["class_index"])
    ]
    if class_names != holdout_classes or not holdout["records"]:
        raise ValueError("Exposure holdout contract differs from the frozen protocol")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    records: list[dict[str, Any]] = []
    point_tensors: list[torch.Tensor] = []
    reference_tensors: list[torch.Tensor] = []
    for item in holdout["records"]:
        point_path = (holdout_path.parent / str(item["point_path"])).resolve()
        if sha256_file(point_path) != str(item["packaged_point_sha256"]):
            raise ValueError(f"Packaged point hash mismatch: {item['sample_key']}")
        with np.load(point_path, allow_pickle=False) as archive:
            points = np.asarray(archive["points_xyz"], dtype=np.float32)
            label = int(archive["class_index"])
        if points.shape != (8192, 3) or not np.isfinite(points).all():
            raise ValueError(f"Invalid point tensor: {item['sample_key']}")
        if label != int(item["class_index"]):
            raise ValueError(f"Point label mismatch: {item['sample_key']}")
        tensor = torch.from_numpy(points).to(device)
        offset = torch.tensor([8192], dtype=torch.long, device=device)
        reference = knn_query_packed(tensor, offset, 8)
        if tuple(reference.shape) != (8192, 8):
            raise ValueError(f"Unexpected kNN shape: {item['sample_key']}")
        if int(reference.min()) < 0 or int(reference.max()) >= 8192:
            raise ValueError(f"kNN index outside point range: {item['sample_key']}")
        records.append({**item, "resolved_point_path": str(point_path)})
        point_tensors.append(tensor.cpu())
        reference_tensors.append(reference.to(device="cpu", dtype=torch.long))
    points_batch = torch.stack(point_tensors).to(device)
    references_batch = torch.stack(reference_tensors).to(device)

    long_rows: list[dict[str, Any]] = []
    probabilities_by_seed: dict[int, np.ndarray] = {}
    finals: dict[int, dict[str, Any]] = {}
    checkpoint_hashes: dict[str, str] = {}
    for seed in seeds:
        run_dir = suite_root / "runs" / f"seed_{seed}"
        final = json.loads((run_dir / "final_metrics.json").read_text(encoding="utf-8"))
        checkpoint_path = run_dir / "best.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if final["status"] != "complete" or list(final["class_names"]) != class_names:
            raise ValueError(f"Invalid final metrics contract for seed {seed}")
        if int(checkpoint["epoch"]) != int(final["best_epoch"]):
            raise ValueError(f"Best checkpoint epoch mismatch for seed {seed}")
        if list(checkpoint["class_names"]) != class_names:
            raise ValueError(f"Checkpoint class order mismatch for seed {seed}")
        model_config = dict(checkpoint["model_config"])
        if int(model_config.pop("num_classes")) != len(class_names):
            raise ValueError(f"Checkpoint model class count mismatch for seed {seed}")
        model = PointTransformerV2Classifier(len(class_names), **model_config).to(device)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        model.eval()
        use_amp = bool(checkpoint.get("args", {}).get("amp", False)) and device.type == "cuda"
        with torch.inference_mode():
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=use_amp
            ):
                logits = model(points_batch, references_batch)
            probabilities = torch.softmax(logits.float(), dim=1).cpu().numpy()
        probabilities_by_seed[seed] = probabilities
        finals[seed] = final
        checkpoint_hashes[str(seed)] = sha256_file(checkpoint_path)
        for item, probability in zip(records, probabilities, strict=True):
            sample_key = str(item["sample_key"])
            finite_probabilities(probability, sample_key)
            true_class = int(item["class_index"])
            predicted_class = int(np.argmax(probability))
            sorted_probability = np.sort(probability)
            entropy = -float(
                np.sum(probability * np.log(np.clip(probability, 1e-12, 1.0)))
            )
            row: dict[str, Any] = {
                "seed": seed,
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": checkpoint_hashes[str(seed)],
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
                "top1_top2_margin": float(sorted_probability[-1] - sorted_probability[-2]),
                "entropy_nats": entropy,
                "normalized_entropy": entropy / math.log(len(class_names)),
            }
            for index, value in enumerate(probability):
                row[f"probability_{index}"] = float(value)
            long_rows.append(row)
        del model, checkpoint
        if device.type == "cuda":
            torch.cuda.empty_cache()

    recommended_seed = max(
        seeds,
        key=lambda seed: (
            float(finals[seed]["best_validation_score"]["macro_f1"]),
            float(finals[seed]["best_validation_score"]["accuracy"]),
        ),
    )
    tree_rows: list[dict[str, Any]] = []
    ensemble_rows: list[dict[str, Any]] = []
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
        row = {
            "sample_key": sample_key,
            "tree_id": int(item["tree_id"]),
            "road_id": str(item["road_id"]),
            "trajectory_id": str(item["trajectory_id"]),
            "exposure_group": ensemble["exposure_group"],
            "automatic_risk_reasons": ";".join(item["automatic_risk_reasons"]),
            "true_class": true_class,
            "true_species": class_names[true_class],
            "correct_seed_count": sum(int(value["correct"]) for value in seed_rows),
            "seed_accuracy": statistics.fmean(float(value["correct"]) for value in seed_rows),
            "unique_seed_predictions": len(set(predictions)),
            "all_seeds_agree": int(len(set(predictions)) == 1),
            "mean_true_class_probability": statistics.fmean(
                float(value["true_class_probability"]) for value in seed_rows
            ),
            "std_true_class_probability": sample_std(
                [float(value["true_class_probability"]) for value in seed_rows]
            ),
            "mean_confidence": statistics.fmean(
                float(value["confidence"]) for value in seed_rows
            ),
            "std_confidence": sample_std(
                [float(value["confidence"]) for value in seed_rows]
            ),
            "ensemble_predicted_species": class_names[ensemble_class],
            "ensemble_correct": int(ensemble_class == true_class),
            "ensemble_confidence": float(mean_probability[ensemble_class]),
            "ensemble_true_class_probability": float(mean_probability[true_class]),
            "recommended_seed": recommended_seed,
        }
        for seed in seeds:
            value = next(value for value in seed_rows if int(value["seed"]) == seed)
            row[f"seed_{seed}_prediction"] = value["predicted_species"]
            row[f"seed_{seed}_correct"] = value["correct"]
            row[f"seed_{seed}_confidence"] = value["confidence"]
            row[f"seed_{seed}_true_probability"] = value["true_class_probability"]
        tree_rows.append(row)

    seed_summaries = {
        str(seed): summarize([row for row in long_rows if int(row["seed"]) == seed])
        for seed in seeds
    }
    seed_accuracies = [float(seed_summaries[str(seed)]["accuracy"]) for seed in seeds]
    metrics = {
        "format_version": 1,
        "status": "complete",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "scope": holdout.get(
            "scope", "point-cloud-only exposure-probe descriptive evaluation"
        ),
        "interpretation_limit": holdout.get(
            "interpretation_limit",
            "Exposure labels describe paired images; point predictions are a reference.",
        ),
        "device": str(device),
        "class_names": class_names,
        "seeds": seeds,
        "recommended_seed_by_validation_macro_f1": recommended_seed,
        "per_seed": seed_summaries,
        "three_seed_accuracy": {
            "mean": statistics.fmean(seed_accuracies),
            "sample_std": sample_std(seed_accuracies),
        },
        "mean_probability_ensemble": summarize(ensemble_rows),
        "tree_count": len(records),
        "prediction_row_count": len(long_rows),
        "all_seed_agreement_tree_count": sum(
            int(row["all_seeds_agree"]) for row in tree_rows
        ),
    }
    probability_fields = [f"probability_{index}" for index in range(len(class_names))]
    long_fields = [
        "seed", "checkpoint", "checkpoint_sha256", "best_epoch", "sample_key",
        "tree_id", "road_id", "trajectory_id", "exposure_group",
        "automatic_risk_reasons", "true_class", "true_species", "predicted_class",
        "predicted_species", "correct", "confidence", "true_class_probability",
        "top1_top2_margin", "entropy_nats", "normalized_entropy", *probability_fields,
    ]
    tree_fields = [
        "sample_key", "tree_id", "road_id", "trajectory_id", "exposure_group",
        "automatic_risk_reasons", "true_class", "true_species", "correct_seed_count",
        "seed_accuracy", "unique_seed_predictions", "all_seeds_agree",
        "mean_true_class_probability", "std_true_class_probability", "mean_confidence",
        "std_confidence", "ensemble_predicted_species", "ensemble_correct",
        "ensemble_confidence", "ensemble_true_class_probability", "recommended_seed",
    ]
    for seed in seeds:
        tree_fields.extend(
            [f"seed_{seed}_prediction", f"seed_{seed}_correct",
             f"seed_{seed}_confidence", f"seed_{seed}_true_probability"]
        )
    predictions_path = output_dir / "predictions_long.csv"
    tree_path = output_dir / "per_tree_summary.csv"
    metrics_path = output_dir / "metrics.json"
    atomic_csv(predictions_path, long_rows, long_fields)
    atomic_csv(tree_path, tree_rows, tree_fields)
    atomic_json(metrics_path, metrics)
    manifest = {
        "format_version": 1,
        "status": "complete",
        "generated_at": metrics["generated_at"],
        "inputs": {
            "protocol": {"path": str(protocol_path), "sha256": sha256_file(protocol_path)},
            "holdout_manifest": {"path": str(holdout_path), "sha256": sha256_file(holdout_path)},
            "checkpoints": checkpoint_hashes,
            "points_verified": {
                str(item["sample_key"]): str(item["packaged_point_sha256"])
                for item in records
            },
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
            "predictions_long.csv": sha256_file(predictions_path),
            "per_tree_summary.csv": sha256_file(tree_path),
            "metrics.json": sha256_file(metrics_path),
        },
    }
    atomic_json(output_dir / "evaluation_manifest.json", manifest)
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
