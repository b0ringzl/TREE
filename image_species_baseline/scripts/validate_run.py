#!/usr/bin/env python3
"""Validate provenance, split isolation, checkpoints, and outputs of a run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    manifest_path = run_dir / "dataset" / "manifest.csv"
    summary_path = run_dir / "dataset" / "dataset_summary.json"
    config_path = run_dir / "run_config.json"
    final_path = run_dir / "final_metrics.json"
    best_path = run_dir / "best.pt"
    source_path = run_dir / "source_model_copy" / "d1_seed_20260728_best_4class.pt"
    required = [manifest_path, summary_path, config_path, final_path, best_path, source_path]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing run artifacts: {missing}")

    rows = csv_rows(manifest_path)
    summary = read_json(summary_path)
    config = read_json(config_path)
    final = read_json(final_path)
    errors: list[str] = []
    warnings: list[str] = []

    manifest_hash = sha256_file(manifest_path)
    if manifest_hash != summary["manifest"]["sha256"]:
        errors.append("dataset summary manifest SHA-256 mismatch")
    if manifest_hash != config["manifest_sha256"]:
        errors.append("run config manifest SHA-256 mismatch")
    if summary.get("checks", {}).get("exact_content_leakage") is not False:
        errors.append("dataset summary did not pass exact-content leakage check")

    class_names = tuple(final.get("class_names", []))
    if len(class_names) != 10 or len(set(class_names)) != 10:
        errors.append("final output does not contain ten unique classes")
    split_counts = Counter(row["split"] for row in rows)
    if dict(split_counts) != final.get("split_sizes"):
        errors.append(f"manifest/final split size mismatch: {dict(split_counts)}")

    group_splits: defaultdict[str, set[str]] = defaultdict(set)
    content_splits: defaultdict[str, set[str]] = defaultdict(set)
    for row in rows:
        group_splits[row["group_id"]].add(row["split"])
        content_splits[sha256_file(PROJECT_ROOT / row["image_path"])].add(row["split"])
        if row["split"] in {"val", "test"} and row["source"] != "hk_streetview":
            errors.append(f"non-HK holdout row: {row['sample_id']}")
    leaking_groups = [group for group, splits in group_splits.items() if len(splits) > 1]
    leaking_content = [value for value, splits in content_splits.items() if len(splits) > 1]
    if leaking_groups:
        errors.append(f"{len(leaking_groups)} group IDs cross splits")
    if leaking_content:
        errors.append(f"{len(leaking_content)} exact image hashes cross splits")

    best = torch.load(best_path, map_location="cpu", weights_only=False)
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    if tuple(best.get("class_names", [])) != class_names:
        errors.append("best checkpoint class order differs from final output")
    if list(best["model_state"]["fc.weight"].shape) != [10, 2048]:
        errors.append("best checkpoint does not have a 10x2048 head")
    if list(source["model_state"]["fc.weight"].shape) != [4, 2048]:
        errors.append("copied source checkpoint does not have a 4x2048 head")
    source_hash = sha256_file(source_path)
    if source_hash != config["transfer"]["source_sha256"]:
        errors.append("copied source checkpoint SHA-256 mismatch")
    if config["transfer"].get("loaded_backbone_keys") != 318:
        errors.append("unexpected number of transferred backbone keys")
    if not final.get("test_evaluated_after_training"):
        errors.append("final output does not assert post-selection test evaluation")

    for split in ("val", "test"):
        predictions = csv_rows(run_dir / f"predictions_{split}.csv")
        if len(predictions) != split_counts[split]:
            errors.append(f"{split} prediction row count mismatch")
    if final.get("status") != "complete" or final.get("epochs_completed") != 60:
        errors.append("formal 60-epoch run is not complete")

    test_supports = {
        row["scientific_name"]: int(row["support"])
        for row in final["metrics"]["test_tree_group"]["per_class"]
    }
    small = {name: count for name, count in test_supports.items() if count < 3}
    if small:
        warnings.append(f"test tree support below 3: {small}")
    zero_f1 = [
        row["scientific_name"]
        for row in final["metrics"]["test_tree_group"]["per_class"]
        if float(row["f1"]) == 0.0
    ]
    if zero_f1:
        warnings.append(f"zero test tree-group F1: {zero_f1}")

    acceptance = {
        "status": "passed_with_warnings" if not errors and warnings else "passed" if not errors else "failed",
        "run_dir": str(run_dir),
        "errors": errors,
        "warnings": warnings,
        "checks": {
            "manifest_sha256": manifest_hash,
            "source_checkpoint_sha256": source_hash,
            "best_checkpoint_sha256": sha256_file(best_path),
            "ten_class_head": list(best["model_state"]["fc.weight"].shape),
            "transferred_backbone_keys": config["transfer"]["loaded_backbone_keys"],
            "group_leakage_count": len(leaking_groups),
            "exact_content_leakage_count": len(leaking_content),
            "holdout_is_hk_only": True,
            "prediction_counts": {split: split_counts[split] for split in ("val", "test")},
            "test_evaluated_after_training": final["test_evaluated_after_training"],
        },
        "headline_metrics": {
            "test_tree_group_accuracy": final["metrics"]["test_tree_group"]["accuracy"],
            "test_tree_group_macro_f1": final["metrics"]["test_tree_group"]["macro_f1"],
            "test_crop_accuracy": final["metrics"]["test_crop"]["accuracy"],
            "test_crop_macro_f1": final["metrics"]["test_crop"]["macro_f1"],
        },
    }
    output_path = run_dir / "acceptance.json"
    output_path.write_text(
        json.dumps(acceptance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(acceptance, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
