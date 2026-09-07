"""Join D1 exposure-probe image and point results by sample_key."""

from __future__ import annotations

import argparse
import csv
import json
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-summary", type=Path, required=True)
    parser.add_argument("--point-summary", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


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


def read_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    result = {str(row["sample_key"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate sample keys in {path}")
    return result


def main() -> None:
    args = parse_args()
    image = read_rows(args.image_summary.resolve())
    point = read_rows(args.point_summary.resolve())
    if set(image) != set(point) or not image:
        raise ValueError("Image and point exposure cohorts do not match exactly")
    rows: list[dict[str, Any]] = []
    for sample_key in sorted(image):
        image_row = image[sample_key]
        point_row = point[sample_key]
        if image_row["true_species"] != point_row["true_species"]:
            raise ValueError(f"Truth mismatch: {sample_key}")
        image_correct = int(image_row["ensemble_correct"])
        point_correct = int(point_row["ensemble_correct"])
        if image_correct and point_correct:
            pattern = "both_correct"
        elif image_correct:
            pattern = "image_only_correct"
        elif point_correct:
            pattern = "point_only_correct"
        else:
            pattern = "neither_correct"
        rows.append(
            {
                "sample_key": sample_key,
                "tree_id": image_row["tree_id"],
                "exposure_group": image_row["exposure_group"],
                "true_species": image_row["true_species"],
                "image_prediction": image_row["ensemble_predicted_species"],
                "image_correct": image_correct,
                "image_confidence": float(image_row["ensemble_confidence"]),
                "image_true_class_probability": float(
                    image_row["ensemble_true_class_probability"]
                ),
                "point_prediction": point_row["ensemble_predicted_species"],
                "point_correct": point_correct,
                "point_confidence": float(point_row["ensemble_confidence"]),
                "point_true_class_probability": float(
                    point_row["ensemble_true_class_probability"]
                ),
                "correctness_pattern": pattern,
                "fusion_prediction": "pending",
                "fusion_correct": "pending",
                "fusion_confidence": "pending",
            }
        )
    histogram = Counter(str(row["correctness_pattern"]) for row in rows)
    payload = {
        "format_version": 1,
        "status": "image_and_point_complete_fusion_pending",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "tree_count": len(rows),
        "image_ensemble": {
            "correct_count": sum(int(row["image_correct"]) for row in rows),
            "accuracy": sum(int(row["image_correct"]) for row in rows) / len(rows),
        },
        "point_ensemble": {
            "correct_count": sum(int(row["point_correct"]) for row in rows),
            "accuracy": sum(int(row["point_correct"]) for row in rows) / len(rows),
        },
        "correctness_pattern_counts": dict(sorted(histogram.items())),
        "fusion_status": "pending_not_started",
        "interpretation_limit": (
            "Exposure group labels describe image quality. Point errors cannot be "
            "attributed to exposure; road/species/domain composition is confounded."
        ),
        "rows": rows,
    }
    fields = list(rows[0])
    atomic_csv(args.output_csv.resolve(), rows, fields)
    atomic_json(args.output_json.resolve(), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
