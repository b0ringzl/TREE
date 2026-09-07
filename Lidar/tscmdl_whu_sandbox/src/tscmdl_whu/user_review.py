"""User-owned visual review state and non-destructive action exports."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from .export_assets import overlay_points
from .panorama import CameraPose, ProjectionCrop


VALID_RATINGS = {"pass", "borderline", "reject"}
VALID_ACTIONS = {
    "keep",
    "exclude_image",
    "retry_alternate_view",
    "manual_recrop",
    "exclude_sample",
}
REVIEW_SCOPE_QUALITY = "quality_sample"
REVIEW_SCOPE_ALL = "all_samples"


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{path}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    deadline = time.monotonic() + 10.0
    delay = 0.02
    while True:
        try:
            temporary.replace(path)
            return
        except OSError as error:
            retryable = os.name == "nt" and getattr(error, "winerror", None) in {
                5,
                32,
            }
            if not retryable or time.monotonic() >= deadline:
                raise
            time.sleep(delay)
            delay = min(delay * 1.6, 0.5)


def atomic_csv(path: Path, rows: Iterable[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{path}.tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def load_manifest(path: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
        raise ValueError(f"Invalid dataset manifest: {path}")
    records = [dict(record) for record in payload["records"]]
    keys = [str(record.get("sample_key", "")) for record in records]
    if not all(keys) or len(keys) != len(set(keys)):
        raise ValueError("Manifest sample keys are missing or duplicated")
    return payload, records


def discover_manifest(dataset_root: Path) -> Path:
    candidates = [dataset_root / "shared_manifest.json", dataset_root / "manifest.json"]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"No supported manifest under {dataset_root}")


def quality_scope_keys(records: Iterable[dict[str, object]]) -> list[str]:
    return [
        str(record["sample_key"])
        for record in records
        if str(record.get("quality_preview_path", "")).strip()
    ]


def scope_keys(records: list[dict[str, object]], scope: str) -> list[str]:
    if scope == REVIEW_SCOPE_QUALITY:
        return quality_scope_keys(records)
    if scope == REVIEW_SCOPE_ALL:
        return [str(record["sample_key"]) for record in records]
    raise ValueError(f"Unsupported review scope: {scope}")


def initial_review_state(dataset_root: Path, manifest_path: Path) -> dict[str, object]:
    now = timestamp()
    return {
        "schema_version": 1,
        "stage": "user_visual_review",
        "reviewer": "user",
        "dataset_root": str(dataset_root.resolve()),
        "source_manifest": str(manifest_path.resolve()),
        "source_manifest_sha256": sha256_file(manifest_path),
        "created_at": now,
        "updated_at": now,
        "scope_status": {},
        "reviews": {},
    }


def load_or_create_review_state(
    path: Path, dataset_root: Path, manifest_path: Path
) -> dict[str, object]:
    expected_hash = sha256_file(manifest_path)
    if path.is_file():
        state = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(state, dict):
            raise ValueError(f"Invalid user review file: {path}")
        if state.get("reviewer") != "user":
            raise ValueError("The formal review file must be owned by the user")
        if state.get("source_manifest_sha256") != expected_hash:
            raise ValueError("Dataset manifest changed after the user review started")
        state.setdefault("scope_status", {})
        state.setdefault("reviews", {})
        return state
    state = initial_review_state(dataset_root, manifest_path)
    atomic_json(path, state)
    return state


def validate_review_payload(payload: dict[str, object]) -> dict[str, object]:
    sample_key = str(payload.get("sample_key", "")).strip()
    rating = str(payload.get("rating", "")).strip()
    action = str(payload.get("action", "")).strip()
    if not sample_key:
        raise ValueError("sample_key is required")
    if rating not in VALID_RATINGS:
        raise ValueError(f"Unsupported rating: {rating}")
    if action not in VALID_ACTIONS:
        raise ValueError(f"Unsupported action: {action}")
    if rating == "pass" and action != "keep":
        raise ValueError("A passing sample must use the keep action")
    if rating == "reject" and action == "keep":
        raise ValueError("A rejected sample must have a corrective or exclusion action")
    reasons = payload.get("reasons", [])
    if not isinstance(reasons, list):
        raise ValueError("reasons must be a list")
    return {
        "sample_key": sample_key,
        "rating": rating,
        "action": action,
        "reasons": sorted({str(reason).strip() for reason in reasons if str(reason).strip()}),
        "note": str(payload.get("note", "")).strip(),
        "reviewed_at": timestamp(),
        "reviewer": "user",
    }


def save_user_review(
    state_path: Path,
    state: dict[str, object],
    payload: dict[str, object],
    valid_keys: set[str],
) -> dict[str, object]:
    review = validate_review_payload(payload)
    sample_key = str(review["sample_key"])
    if sample_key not in valid_keys:
        raise KeyError(f"Unknown sample key: {sample_key}")
    reviews = state.setdefault("reviews", {})
    if not isinstance(reviews, dict):
        raise ValueError("Invalid reviews object")
    reviews[sample_key] = review
    state["updated_at"] = timestamp()
    atomic_json(state_path, state)
    return review


def complete_scope(
    state_path: Path,
    state: dict[str, object],
    records: list[dict[str, object]],
    scope: str,
) -> dict[str, object]:
    keys = scope_keys(records, scope)
    reviews = state.get("reviews", {})
    if not isinstance(reviews, dict):
        raise ValueError("Invalid reviews object")
    missing = [key for key in keys if key not in reviews]
    scope_state = {
        "status": "complete" if not missing else "in_progress",
        "total": len(keys),
        "reviewed": len(keys) - len(missing),
        "remaining": len(missing),
        "completed_at": timestamp() if not missing else None,
    }
    statuses = state.setdefault("scope_status", {})
    if not isinstance(statuses, dict):
        raise ValueError("Invalid scope_status object")
    statuses[scope] = scope_state
    state["updated_at"] = timestamp()
    atomic_json(state_path, state)
    return scope_state


def inspect_user_review_state(
    path: Path,
    manifest_path: Path,
    records: list[dict[str, object]],
    scope: str,
) -> dict[str, object]:
    keys = scope_keys(records, scope)
    total = len(keys)
    if not path.is_file():
        return {
            "status": "pending",
            "scope": scope,
            "reviewed": 0,
            "total": total,
            "remaining": total,
            "review_file": str(path),
        }
    try:
        state = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as error:
        return {
            "status": "invalid",
            "scope": scope,
            "reviewed": 0,
            "total": total,
            "remaining": total,
            "review_file": str(path),
            "error": str(error),
        }
    if not isinstance(state, dict):
        state = {}
    errors = []
    if state.get("reviewer") != "user":
        errors.append("reviewer must be user")
    if state.get("source_manifest_sha256") != sha256_file(manifest_path):
        errors.append("source manifest hash mismatch")
    reviews = state.get("reviews", {})
    if not isinstance(reviews, dict):
        reviews = {}
        errors.append("reviews must be an object")
    reviewed = 0
    for key in keys:
        review = reviews.get(key)
        if not isinstance(review, dict):
            continue
        rating = str(review.get("rating", ""))
        action = str(review.get("action", ""))
        valid = (
            review.get("reviewer") == "user"
            and rating in VALID_RATINGS
            and action in VALID_ACTIONS
            and not (rating == "pass" and action != "keep")
            and not (rating == "reject" and action == "keep")
        )
        if valid:
            reviewed += 1
        else:
            errors.append(f"invalid review: {key}")
    scope_status = state.get("scope_status", {})
    declared = (
        scope_status.get(scope, {})
        if isinstance(scope_status, dict)
        else {}
    )
    explicitly_complete = (
        isinstance(declared, dict)
        and declared.get("status") == "complete"
        and int(declared.get("total", -1)) == total
        and int(declared.get("reviewed", -1)) == total
        and int(declared.get("remaining", -1)) == 0
    )
    if errors:
        status = "invalid"
    elif reviewed == total and explicitly_complete:
        status = "complete"
    elif reviewed:
        status = "in_progress"
    else:
        status = "pending"
    result: dict[str, object] = {
        "status": status,
        "scope": scope,
        "reviewed": reviewed,
        "total": total,
        "remaining": total - reviewed,
        "explicitly_completed": explicitly_complete,
        "review_file": str(path),
    }
    if errors:
        result["errors"] = errors
    return result


def refresh_validation_review_status(
    validation_path: Path,
    review_path: Path,
    review_status: dict[str, object],
) -> None:
    if not validation_path.is_file():
        return
    validation = json.loads(validation_path.read_text(encoding="utf-8-sig"))
    if not isinstance(validation, dict):
        raise ValueError(f"Invalid validation file: {validation_path}")
    status = validation.setdefault("review_status", {})
    if not isinstance(status, dict):
        raise ValueError("validation review_status must be an object")
    status["user_visual_review"] = review_status
    status.pop("user_decision", None)
    summary = validation.setdefault("summary", {})
    if not isinstance(summary, dict):
        raise ValueError("validation summary must be an object")
    summary["user_visual_review_complete"] = review_status.get("status") == "complete"
    artifacts = validation.setdefault("artifacts", {})
    if not isinstance(artifacts, dict):
        raise ValueError("validation artifacts must be an object")
    if review_path.is_file():
        artifacts["user_visual_review"] = str(review_path)
    validation["review_status_refreshed_at"] = timestamp()
    atomic_json(validation_path, validation)


def _pose(record: dict[str, object]) -> CameraPose:
    raw = dict(record["camera_pose"])
    return CameraPose(
        image_name=str(raw["image_name"]),
        position=tuple(float(value) for value in raw["position"]),
        roll_deg=float(raw["roll_deg"]),
        pitch_deg=float(raw["pitch_deg"]),
        heading_deg=float(raw["heading_deg"]),
    )


def _crop(record: dict[str, object]) -> ProjectionCrop:
    raw = dict(record["crop"])
    return ProjectionCrop(**{key: float(value) for key, value in raw.items()})


def render_record_image(
    dataset_root: Path, record: dict[str, object], mode: str
) -> tuple[bytes, str]:
    image_path = (dataset_root / str(record["image_path"])).resolve()
    if mode == "crop":
        return image_path.read_bytes(), "image/jpeg"
    preview = str(record.get("quality_preview_path", "")).strip()
    if preview:
        preview_path = (dataset_root / preview).resolve()
        if preview_path.is_file():
            return preview_path.read_bytes(), "image/jpeg"
    point_path = (dataset_root / str(record["point_path"])).resolve()
    with np.load(point_path, allow_pickle=False) as archive:
        normalized = np.asarray(archive["points_xyz"], dtype=np.float64)
        centroid = np.asarray(archive["centroid_xyz"], dtype=np.float64)
        scale = float(archive["scale"])
    points = normalized * scale + centroid
    with Image.open(image_path) as image:
        output = overlay_points(
            image.convert("RGB"),
            points,
            _pose(record),
            _crop(record),
            int(record["panorama_size"][0]),
            int(record["panorama_size"][1]),
        )
    stream = io.BytesIO()
    output.save(stream, format="JPEG", quality=92)
    return stream.getvalue(), "image/jpeg"


def review_row(
    record: dict[str, object], review: dict[str, object] | None
) -> dict[str, object]:
    review = review or {}
    return {
        "sample_key": record["sample_key"],
        "class_index": record["class_index"],
        "scientific_name": record["scientific_name"],
        "split": record.get("benchmark_split", record.get("split", "")),
        "road_id": record["road_id"],
        "trajectory_id": record["trajectory_id"],
        "tree_id": record["tree_id"],
        "review_status": "reviewed" if review else "unreviewed",
        "rating": review.get("rating", ""),
        "action": review.get("action", ""),
        "reasons": "|".join(review.get("reasons", [])),
        "note": review.get("note", ""),
        "reviewed_at": review.get("reviewed_at", ""),
        "point_path": record["point_path"],
        "image_path": record["image_path"],
    }


def export_action_lists(
    output_dir: Path,
    records: list[dict[str, object]],
    state: dict[str, object],
) -> dict[str, object]:
    record_by_key = {str(record["sample_key"]): record for record in records}
    reviews = state.get("reviews", {})
    if not isinstance(reviews, dict):
        raise ValueError("Invalid reviews object")
    rows = [
        review_row(record_by_key[key], dict(review))
        for key, review in reviews.items()
        if key in record_by_key and isinstance(review, dict)
    ]
    rows.sort(key=lambda row: str(row["sample_key"]))
    all_rows = [
        review_row(
            record,
            dict(reviews[str(record["sample_key"])])
            if isinstance(reviews.get(str(record["sample_key"])), dict)
            else None,
        )
        for record in records
    ]
    all_rows.sort(key=lambda row: str(row["sample_key"]))
    accepted = [row for row in rows if row["action"] == "keep"]
    point_usable = [row for row in rows if row["action"] != "exclude_sample"]
    image_excluded = [row for row in rows if row["action"] == "exclude_image"]
    rework = [
        row
        for row in rows
        if row["action"] in {"retry_alternate_view", "manual_recrop"}
    ]
    rejected = [row for row in rows if row["action"] == "exclude_sample"]
    multimodal_blocking_actions = {
        "exclude_image",
        "retry_alternate_view",
        "manual_recrop",
        "exclude_sample",
    }
    multimodal_manifest = [
        row for row in all_rows if row["action"] not in multimodal_blocking_actions
    ]
    point_manifest = [
        row for row in all_rows if row["action"] != "exclude_sample"
    ]
    multimodal_quarantine = [
        row for row in rows if row["action"] in multimodal_blocking_actions
    ]
    fields = [
        "sample_key",
        "class_index",
        "scientific_name",
        "split",
        "road_id",
        "trajectory_id",
        "tree_id",
        "review_status",
        "rating",
        "action",
        "reasons",
        "note",
        "reviewed_at",
        "point_path",
        "image_path",
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "reviewed_samples": (rows, "reviewed_samples.csv"),
        "accepted_multimodal": (accepted, "accepted_multimodal.csv"),
        "point_only_usable": (point_usable, "point_only_usable.csv"),
        "image_excluded": (image_excluded, "image_excluded.csv"),
        "rework_queue": (rework, "rework_queue.csv"),
        "excluded_samples": (rejected, "excluded_samples.csv"),
        "multimodal_training_manifest": (
            multimodal_manifest,
            "multimodal_training_manifest.csv",
        ),
        "point_training_manifest": (
            point_manifest,
            "point_training_manifest.csv",
        ),
        "multimodal_quarantine": (
            multimodal_quarantine,
            "multimodal_quarantine.csv",
        ),
    }
    paths = {}
    for name, (content, filename) in outputs.items():
        path = output_dir / filename
        atomic_csv(path, content, fields)
        paths[name] = str(path)
    summary = {
        "schema_version": 1,
        "generated_at": timestamp(),
        "policy": "non-destructive; no source assets are deleted, moved, or overwritten",
        "reviewed_count": len(rows),
        "accepted_multimodal_count": len(accepted),
        "point_only_usable_count": len(point_usable),
        "image_excluded_count": len(image_excluded),
        "rework_queue_count": len(rework),
        "excluded_sample_count": len(rejected),
        "multimodal_training_count": len(multimodal_manifest),
        "point_training_count": len(point_manifest),
        "multimodal_quarantine_count": len(multimodal_quarantine),
        "unreviewed_count": len(records) - len(rows),
        "outputs": paths,
    }
    atomic_json(output_dir / "review_action_summary.json", summary)
    return summary
