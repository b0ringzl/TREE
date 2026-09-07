"""Aggregate C2a dataset-view, cache, GPU-smoke, and test evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-view-validation", type=Path, required=True)
    parser.add_argument("--cache-validation", type=Path, required=True)
    parser.add_argument("--smoke-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--unit-test-count", type=int, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{path}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def artifact(path: Path) -> dict[str, object]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def main() -> None:
    args = parse_args()
    if args.unit_test_count <= 0:
        raise ValueError("unit-test-count must be positive")
    view_path = args.dataset_view_validation.resolve()
    cache_path = args.cache_validation.resolve()
    smoke_path = args.smoke_result.resolve()
    view = read_json(view_path)
    cache = read_json(cache_path)
    smoke = read_json(smoke_path)
    for name, value in (
        ("dataset view", view),
        ("cache", cache),
        ("GPU smoke", smoke),
    ):
        if value.get("status") != "passed":
            raise ValueError(f"C2a {name} validation did not pass")

    if int(view["summary"]["sample_count"]) != 17134:
        raise ValueError("C2a dataset view must contain 17,134 samples")
    if int(view["summary"]["class_count"]) != 19:
        raise ValueError("C2a dataset view must contain 19 classes")
    if int(cache["sample_count"]) != 17134:
        raise ValueError("C2a cache must contain 17,134 samples")
    if int(cache["resume_evidence"]["partial_run_count"]) < 1:
        raise ValueError("C2a cache did not prove partial-run recovery")
    if smoke.get("formal_training_started") is not False:
        raise ValueError("C2a unexpectedly started formal training")
    if smoke.get("checkpoint_written") is not False:
        raise ValueError("C2a unexpectedly wrote a model checkpoint")
    if smoke.get("official_test_split_loaded") is not False:
        raise ValueError("C2a smoke unexpectedly loaded the official test split")
    if smoke["train_batch"].get("optimizer_step_succeeded") is not True:
        raise ValueError("C2a optimizer-step smoke did not succeed")

    cache_dir = Path(str(cache["cache_dir"])).resolve()
    cache_index = cache_dir / "index.json"
    cache_checkpoint = cache_dir / "checkpoint.json"
    cache_progress = cache_dir / "progress.json"
    training_artifacts = sorted(cache_dir.parent.rglob("*.pt"))
    if training_artifacts:
        raise ValueError(f"Unexpected C2a model artifacts: {training_artifacts}")

    checks = {
        "dataset_view_validation_passed": True,
        "physical_assets_not_duplicated": (
            view["summary"].get("physical_assets_copied") is False
        ),
        "full_cache_validation_passed": True,
        "all_cache_indices_in_range": all(
            int(split["minimum_index"]) >= 0
            and int(split["maximum_index"]) < 8192
            for split in cache["split_results"].values()
        ),
        "partial_run_and_resume_proved": True,
        "gpu_forward_backward_passed": True,
        "optimizer_step_succeeded": True,
        "validation_forward_passed": True,
        "official_test_split_not_loaded_by_smoke": True,
        "formal_training_not_started": True,
        "model_checkpoint_not_written": True,
        "unit_tests_passed": args.unit_test_count,
    }
    if not all(
        value is True or (key == "unit_tests_passed" and int(value) > 0)
        for key, value in checks.items()
    ):
        raise ValueError(f"C2a checks did not all pass: {checks}")

    result = {
        "stage": "C2a",
        "status": "passed",
        "validated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "scope": "19-class PTv2 cache, loading, and one-batch GPU smoke",
        "summary": {
            "sample_count": int(cache["sample_count"]),
            "class_count": int(smoke["class_count"]),
            "split_sizes": view["summary"]["split_histogram"],
            "cache_bytes": int(cache["cache_bytes"]),
            "cache_run_count": int(cache["resume_evidence"]["run_count"]),
            "cache_elapsed_seconds": sum(
                float(event["elapsed_seconds"])
                for event in read_json(cache_checkpoint)["run_events"]
            ),
            "batch_size": int(smoke["batch_size"]),
            "train_loss": float(smoke["train_batch"]["loss"]),
            "validation_loss": float(smoke["validation_batch"]["loss"]),
            "train_peak_allocated_mib": float(
                smoke["train_batch"]["peak_allocated_mib"]
            ),
            "train_peak_reserved_mib": float(
                smoke["train_batch"]["peak_reserved_mib"]
            ),
            "gpu": smoke["gpu"]["name"],
            "unit_test_count": args.unit_test_count,
        },
        "checks": checks,
        "artifacts": {
            "dataset_view_validation": artifact(view_path),
            "cache_validation": artifact(cache_path),
            "cache_index": artifact(cache_index),
            "cache_checkpoint": artifact(cache_checkpoint),
            "cache_progress": artifact(cache_progress),
            "gpu_smoke_result": artifact(smoke_path),
        },
    }
    atomic_json(args.output.resolve(), result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
