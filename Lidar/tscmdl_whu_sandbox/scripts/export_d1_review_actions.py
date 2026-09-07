"""Export non-destructive D1 review action lists from the user-owned checkpoint."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_D1_ROOT = (
    PROJECT_ROOT
    / "lidar data"
    / "whu"
    / "derived"
    / "tscmdl"
    / "d1_four_class_image_quality"
)
VALID_RATINGS = {"pass", "borderline", "reject"}
VALID_ACTIONS = {
    "keep",
    "exclude_image",
    "retry_alternate_view",
    "manual_recrop",
    "exclude_sample",
}
EXPECTED_CLASSES = {
    "Cinnamomum camphora",
    "Lagerstroemia indica",
    "Magnolia grandiflora",
    "Other",
}

FIELDS = [
    "sample_key",
    "review_scope",
    "model_class_index",
    "model_class_name",
    "source_class_index",
    "source_scientific_name",
    "model_split",
    "benchmark_split",
    "road_id",
    "trajectory_id",
    "tree_id",
    "rating",
    "action",
    "reasons",
    "note",
    "reviewed_at",
    "image_path",
    "point_path",
    "image_sha256",
    "automatic_quality_tier",
    "automatic_risk_score",
    "automatic_risk_reasons",
    "crop_visible_fraction",
    "panorama_visible_fraction",
    "view_candidate_count",
    "camera_distance_m",
]


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def read_json_snapshot(path: Path) -> tuple[dict[str, object], str]:
    payload = path.read_bytes()
    value = json.loads(payload.decode("utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value, sha256_bytes(payload)


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def review_row(record: dict[str, object], review: dict[str, object] | None) -> dict[str, object]:
    review = review or {}
    return {
        "sample_key": record["sample_key"],
        "review_scope": record.get("review_scope", ""),
        "model_class_index": record.get("model_class_index", ""),
        "model_class_name": record.get("model_class_name", ""),
        "source_class_index": record.get("source_class_index", ""),
        "source_scientific_name": record.get("source_scientific_name", ""),
        "model_split": record.get("model_split", ""),
        "benchmark_split": record.get("benchmark_split", ""),
        "road_id": record.get("road_id", ""),
        "trajectory_id": record.get("trajectory_id", ""),
        "tree_id": record.get("tree_id", ""),
        "rating": review.get("rating", ""),
        "action": review.get("action", ""),
        "reasons": "|".join(str(item) for item in review.get("reasons", [])),
        "note": review.get("note", ""),
        "reviewed_at": review.get("reviewed_at", ""),
        "image_path": record.get("image_path", ""),
        "point_path": record.get("point_path", ""),
        "image_sha256": record.get("image_sha256", ""),
        "automatic_quality_tier": record.get("automatic_quality_tier", ""),
        "automatic_risk_score": record.get("automatic_risk_score", ""),
        "automatic_risk_reasons": "|".join(
            str(item) for item in record.get("automatic_risk_reasons", [])
        ),
        "crop_visible_fraction": record.get("crop_visible_fraction", ""),
        "panorama_visible_fraction": record.get("panorama_visible_fraction", ""),
        "view_candidate_count": record.get("view_candidate_count", ""),
        "camera_distance_m": record.get("camera_distance_m", ""),
    }


def nested_counts(rows: list[dict[str, object]], outer: str, inner: str) -> dict[str, dict[str, int]]:
    values: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        values[str(row.get(outer, ""))][str(row.get(inner, ""))] += 1
    return {
        key: dict(sorted(counter.items()))
        for key, counter in sorted(values.items())
    }


def validate_and_prepare(
    manifest_path: Path,
    review_path: Path,
) -> tuple[dict[str, object], dict[str, list[dict[str, object]]]]:
    manifest, manifest_sha256 = read_json_snapshot(manifest_path)
    state, review_sha256 = read_json_snapshot(review_path)

    if state.get("reviewer") != "user":
        raise ValueError("D1 checkpoint is not marked as user-owned")
    if str(state.get("source_manifest_sha256", "")).lower() != manifest_sha256:
        raise ValueError("D1 checkpoint is bound to a different review manifest")

    records = manifest.get("records")
    reviews = state.get("reviews")
    if not isinstance(records, list) or not isinstance(reviews, dict):
        raise ValueError("Invalid D1 manifest or review checkpoint structure")

    record_by_key: dict[str, dict[str, object]] = {}
    for record in records:
        if not isinstance(record, dict) or not record.get("sample_key"):
            raise ValueError("Invalid D1 manifest record")
        key = str(record["sample_key"])
        if key in record_by_key:
            raise ValueError(f"Duplicate D1 sample key: {key}")
        record_by_key[key] = record

    manifest_classes = {
        str(record.get("model_class_name", "")) for record in record_by_key.values()
    }
    if manifest_classes != EXPECTED_CLASSES:
        raise ValueError(f"Unexpected D1 model classes: {sorted(manifest_classes)}")

    invalid_keys = sorted(set(reviews) - set(record_by_key))
    if invalid_keys:
        raise ValueError(f"Reviews reference unknown samples: {invalid_keys[:3]}")

    for key, review in reviews.items():
        if not isinstance(review, dict):
            raise ValueError(f"Invalid review entry: {key}")
        if review.get("rating") not in VALID_RATINGS:
            raise ValueError(f"Invalid rating for {key}: {review.get('rating')}")
        if review.get("action") not in VALID_ACTIONS:
            raise ValueError(f"Invalid action for {key}: {review.get('action')}")

    evaluation_keys = {
        key
        for key, record in record_by_key.items()
        if record.get("review_scope") == "evaluation"
    }
    training_risk_keys = {
        key
        for key, record in record_by_key.items()
        if record.get("review_scope") == "training_risk"
    }
    missing_evaluation = sorted(evaluation_keys - set(reviews))
    if missing_evaluation:
        raise ValueError(
            f"Evaluation review is incomplete: {len(missing_evaluation)} samples remain"
        )

    evaluation_rows = sorted(
        [review_row(record_by_key[key], dict(reviews[key])) for key in evaluation_keys],
        key=lambda row: str(row["sample_key"]),
    )
    training_reviewed = sorted(
        [
            review_row(record_by_key[key], dict(reviews[key]))
            for key in training_risk_keys & set(reviews)
        ],
        key=lambda row: str(row["sample_key"]),
    )
    training_pending = sorted(
        [review_row(record_by_key[key], None) for key in training_risk_keys - set(reviews)],
        key=lambda row: str(row["sample_key"]),
    )

    action_rows = {
        action: [row for row in evaluation_rows if row["action"] == action]
        for action in sorted(VALID_ACTIONS)
    }
    training_action_rows = {
        action: [row for row in training_reviewed if row["action"] == action]
        for action in sorted(VALID_ACTIONS)
    }
    outputs = {
        "evaluation_reviewed": evaluation_rows,
        "evaluation_keep": action_rows["keep"],
        "evaluation_exclude_image": action_rows["exclude_image"],
        "evaluation_retry_alternate_view": action_rows["retry_alternate_view"],
        "evaluation_manual_recrop": action_rows["manual_recrop"],
        "evaluation_exclude_sample": action_rows["exclude_sample"],
        "evaluation_rework_queue": [
            row
            for row in evaluation_rows
            if row["action"] in {"retry_alternate_view", "manual_recrop"}
        ],
        "evaluation_multimodal_quarantine": [
            row for row in evaluation_rows if row["action"] != "keep"
        ],
        "training_risk_reviewed": training_reviewed,
        "training_risk_keep": training_action_rows["keep"],
        "training_risk_exclude_image": training_action_rows["exclude_image"],
        "training_risk_retry_alternate_view": training_action_rows[
            "retry_alternate_view"
        ],
        "training_risk_manual_recrop": training_action_rows["manual_recrop"],
        "training_risk_exclude_sample": training_action_rows["exclude_sample"],
        "training_risk_rework_queue": [
            row
            for row in training_reviewed
            if row["action"] in {"retry_alternate_view", "manual_recrop"}
        ],
        "training_risk_multimodal_quarantine": [
            row for row in training_reviewed if row["action"] != "keep"
        ],
        "training_risk_pending": training_pending,
    }
    ratings = Counter(str(row["rating"]) for row in evaluation_rows)
    actions = Counter(str(row["action"]) for row in evaluation_rows)
    training_ratings = Counter(str(row["rating"]) for row in training_reviewed)
    training_actions = Counter(str(row["action"]) for row in training_reviewed)
    reasons = Counter(
        reason
        for review in reviews.values()
        if isinstance(review, dict)
        for reason in review.get("reasons", [])
    )
    summary: dict[str, object] = {
        "schema_version": 1,
        "stage": "D1-four-class-image-review-actions",
        "generated_at": timestamp(),
        "policy": "non-destructive; no source image, point cloud, manifest, or user review is modified",
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": manifest_sha256,
        "source_user_review": str(review_path),
        "source_user_review_sha256": review_sha256,
        "manifest_record_count": len(records),
        "evaluation_total": len(evaluation_keys),
        "evaluation_reviewed": len(evaluation_rows),
        "evaluation_remaining": len(missing_evaluation),
        "training_risk_total": len(training_risk_keys),
        "training_risk_reviewed": len(training_risk_keys & set(reviews)),
        "training_risk_remaining": len(training_pending),
        "rating_counts": dict(sorted(ratings.items())),
        "action_counts": dict(sorted(actions.items())),
        "training_risk_rating_counts": dict(sorted(training_ratings.items())),
        "training_risk_action_counts": dict(sorted(training_actions.items())),
        "reason_counts": dict(sorted(reasons.items())),
        "class_action_counts": nested_counts(
            evaluation_rows, "model_class_name", "action"
        ),
        "split_action_counts": nested_counts(
            evaluation_rows, "model_split", "action"
        ),
        "training_risk_class_action_counts": nested_counts(
            training_reviewed, "model_class_name", "action"
        ),
        "output_counts": {
            name: len(rows) for name, rows in sorted(outputs.items())
        },
        "next_gate": (
            "complete the remaining training-risk human review"
            if training_pending
            else "resolve training-risk retry/recrop actions and re-review replacements"
            if outputs["training_risk_rework_queue"]
            else "construct the locked four-class image training package"
        ),
    }
    return summary, outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d1-root", type=Path, default=DEFAULT_D1_ROOT)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    d1_root = args.d1_root.resolve()
    manifest_path = (
        args.manifest.resolve()
        if args.manifest is not None
        else d1_root / "review_manifest.json"
    )
    review_path = (
        args.review.resolve()
        if args.review is not None
        else d1_root / "user_visual_review.json"
    )
    summary, outputs = validate_and_prepare(manifest_path, review_path)
    if args.check_only:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else d1_root
        / "review_actions"
        / datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    output_paths: dict[str, str] = {}
    for name, rows in outputs.items():
        path = output_dir / f"{name}.csv"
        write_csv(path, rows)
        output_paths[name] = str(path)
    summary["output_dir"] = str(output_dir)
    summary["outputs"] = output_paths
    write_json(output_dir / "review_action_summary.json", summary)

    current_review_sha256 = sha256_bytes(review_path.read_bytes())
    if current_review_sha256 != summary["source_user_review_sha256"]:
        raise RuntimeError("D1 review checkpoint changed while action lists were exported")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
