"""Validate a completed B4a frozen-fusion smoke run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch


REQUIRED_FILES = (
    "last.pt",
    "run_config.json",
    "training_history.json",
    "final_metrics.json",
    "smoke_result.json",
    "train.log",
)
EXPECTED_FEATURE_SHAPES = {
    "point_raw": [2, 1024],
    "image_raw": [2, 2048],
    "point_normalized": [2, 1024],
    "image_normalized": [2, 1024],
    "fused": [2, 2048],
}
EXPECTED_SPLIT_SIZES = {"train": 420, "val": 90, "test": 90}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{path}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_history(history: list[dict[str, object]]) -> bool:
    for row in history:
        for value in row.values():
            if isinstance(value, (int, float)) and not math.isfinite(float(value)):
                return False
    return True


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    missing = [name for name in REQUIRED_FILES if not (run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing B4a artifacts: {missing}")
    temporary_files = sorted(path.name for path in run_dir.glob("*.tmp"))
    if temporary_files:
        raise ValueError(f"Temporary files remain: {temporary_files}")

    config = read_json(run_dir / "run_config.json")
    history = read_json(run_dir / "training_history.json")
    final = read_json(run_dir / "final_metrics.json")
    smoke = read_json(run_dir / "smoke_result.json")
    if not all(isinstance(item, dict) for item in (config, final, smoke)):
        raise ValueError("B4a JSON artifacts have unexpected top-level types")
    if not isinstance(history, list) or not history:
        raise ValueError("B4a history is empty or invalid")
    if not finite_history(history):
        raise ValueError("B4a history contains non-finite values")

    run_type = "b4a_frozen_fusion_smoke"
    if any(item.get("run_type") != run_type for item in (config, final, smoke)):
        raise ValueError("B4a run type is inconsistent")
    if final.get("status") != "complete" or smoke.get("status") != "passed":
        raise ValueError("B4a smoke did not complete and pass")

    configured_steps = int(config["args"]["steps"])
    if configured_steps < 2 or len(history) != configured_steps:
        raise ValueError("B4a history length does not match configured steps")
    if [int(row["epoch"]) for row in history] != list(
        range(1, configured_steps + 1)
    ):
        raise ValueError("B4a history steps are not contiguous")
    if int(final["epochs_completed"]) != configured_steps:
        raise ValueError("B4a final step count is inconsistent")

    class_names = [str(value) for value in config["class_names"]]
    if len(class_names) != 3 or final["class_names"] != class_names:
        raise ValueError("B4a class names are inconsistent")
    split_sizes = {key: int(value) for key, value in config["split_sizes"].items()}
    if split_sizes != EXPECTED_SPLIT_SIZES:
        raise ValueError(f"Unexpected split sizes: {split_sizes}")
    alignment_sizes = {
        split: int(item["sample_count"])
        for split, item in smoke["dataset_alignment"].items()
    }
    if alignment_sizes != EXPECTED_SPLIT_SIZES:
        raise ValueError(f"Dataset alignment mismatch: {alignment_sizes}")

    checkpoint_load = smoke["checkpoint_load"]
    required_load_checks = (
        bool(checkpoint_load["b2_strict_load"]),
        bool(checkpoint_load["b3_strict_load"]),
        int(checkpoint_load["point_feature_dim"]) == 1024,
        int(checkpoint_load["image_feature_dim"]) == 2048,
        int(checkpoint_load["modal_feature_dim"]) == 1024,
        int(checkpoint_load["fused_feature_dim"]) == 2048,
    )
    if not all(required_load_checks):
        raise ValueError("A source checkpoint or feature dimension check failed")
    if smoke["feature_shapes"] != EXPECTED_FEATURE_SHAPES:
        raise ValueError(f"Unexpected feature shapes: {smoke['feature_shapes']}")

    required_flags = (
        "backbones_frozen",
        "backbone_gradient_free",
        "fusion_gradient_finite",
        "checkpoint_restore_verified",
        "metrics_are_smoke_only",
    )
    failed_flags = [name for name in required_flags if smoke.get(name) is not True]
    if failed_flags:
        raise ValueError(f"B4a validation flags failed: {failed_flags}")
    if int(smoke["successful_optimizer_steps"]) != configured_steps:
        raise ValueError("Not every B4a optimizer step succeeded")
    if int(smoke["skipped_optimizer_steps"]) != 0:
        raise ValueError("B4a had skipped optimizer steps")

    checkpoint = torch.load(
        run_dir / "last.pt", map_location="cpu", weights_only=False
    )
    if checkpoint.get("run_type") != run_type:
        raise ValueError("last.pt run type mismatch")
    if int(checkpoint["epoch"]) != configured_steps:
        raise ValueError("last.pt step mismatch")
    if list(checkpoint["class_names"]) != class_names:
        raise ValueError("last.pt class names mismatch")
    if not checkpoint["model_state"] or not checkpoint["optimizer_state"]:
        raise ValueError("last.pt has an empty model or optimizer state")
    if len(checkpoint["history"]) != configured_steps:
        raise ValueError("last.pt history length mismatch")

    source_paths = {
        "b2_best": Path(config["args"]["b2_checkpoint"]),
        "b3_best": Path(config["args"]["b3_checkpoint"]),
    }
    source_hashes = {
        name: sha256_file(path) for name, path in source_paths.items()
    }
    if source_hashes != smoke["source_checkpoint_sha256"]:
        raise ValueError("Source checkpoint hash mismatch")

    result = {
        "status": "passed",
        "run_dir": str(run_dir),
        "run_type": run_type,
        "steps": configured_steps,
        "class_names": class_names,
        "split_sizes": split_sizes,
        "feature_shapes": smoke["feature_shapes"],
        "checkpoint_load": checkpoint_load,
        "parameter_counts": smoke["parameter_counts"],
        "checks": {
            name: bool(smoke[name]) for name in required_flags
        },
        "optimizer_steps": {
            "successful": int(smoke["successful_optimizer_steps"]),
            "skipped": int(smoke["skipped_optimizer_steps"]),
        },
        "memory_mib": {
            "peak_allocated": float(smoke["peak_allocated_mib"]),
            "peak_reserved": float(smoke["peak_reserved_mib"]),
        },
        "elapsed_seconds": float(smoke["elapsed_seconds"]),
        "source_checkpoint_sha256": source_hashes,
        "last_checkpoint": {
            "bytes": (run_dir / "last.pt").stat().st_size,
            "sha256": sha256_file(run_dir / "last.pt"),
        },
        "temporary_files": temporary_files,
        "note": "Validation metrics are one-batch smoke checks, not performance results.",
    }
    atomic_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
