"""Build the audited four-class image subset and its human-review queue."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = False

TARGET_CLASSES = (
    "Cinnamomum camphora",
    "Lagerstroemia indica",
    "Magnolia grandiflora",
)
MODEL_CLASSES = (*TARGET_CLASSES, "Other")
MODEL_CLASS_INDEX = {name: index for index, name in enumerate(MODEL_CLASSES)}
VALID_KEEP_ACTIONS = {"keep"}
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DERIVED_ROOT = PROJECT_ROOT / "lidar data" / "whu" / "derived" / "tscmdl"
SHARED_DATASET_ROOT = DERIVED_ROOT / "c1_full_shared_dataset"


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    preferred = [
        "sample_key",
        "model_split",
        "model_class_index",
        "model_class_name",
        "source_scientific_name",
        "road_id",
        "trajectory_id",
        "tree_id",
        "automatic_quality_tier",
        "automatic_risk_score",
        "automatic_risk_reasons",
        "review_required",
        "review_scope",
        "review_status",
    ]
    fields = [field for field in preferred if field in fields] + [
        field for field in fields if field not in preferred
    ]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=SHARED_DATASET_ROOT,
    )
    parser.add_argument(
        "--eligible-manifest",
        type=Path,
        default=SHARED_DATASET_ROOT / "user_review_outputs" / "multimodal_training_manifest.csv",
    )
    parser.add_argument(
        "--existing-review",
        type=Path,
        default=SHARED_DATASET_ROOT / "user_visual_review.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DERIVED_ROOT / "d1_four_class_image_quality",
    )
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--other-train-per-source", type=int, default=160)
    parser.add_argument("--other-val-gold", type=int, default=30)
    parser.add_argument("--test-gold-per-class", type=int, default=50)
    parser.add_argument("--other-test-per-source", type=int, default=80)
    parser.add_argument("--training-risk-per-class", type=int, default=30)
    parser.add_argument("--skip-hash-recheck", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def load_records(dataset_root: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    path = dataset_root / "shared_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records", [])
    if not isinstance(records, list):
        raise ValueError("shared_manifest.json has no records list")
    keys = [str(record.get("sample_key", "")) for record in records]
    if not all(keys) or len(keys) != len(set(keys)):
        raise ValueError("Source sample keys are missing or duplicated")
    return payload, [dict(record) for record in records]


def eligible_keys(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    keys = {str(row["sample_key"]) for row in rows}
    if len(keys) != len(rows):
        raise ValueError("Eligible manifest contains duplicated sample keys")
    return keys


def stable_seed(seed: int, *parts: object) -> int:
    raw = "|".join([str(seed), *(str(part) for part in parts)])
    return int(hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16], 16)


def road_round_robin(
    records: list[dict[str, object]], count: int, seed: int, token: str
) -> list[dict[str, object]]:
    if count >= len(records):
        return sorted(records, key=lambda item: str(item["sample_key"]))
    buckets: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        buckets[str(record["road_id"])].append(record)
    for road, values in buckets.items():
        random.Random(stable_seed(seed, token, road)).shuffle(values)
    roads = sorted(buckets)
    random.Random(stable_seed(seed, token, "road-order")).shuffle(roads)
    selected: list[dict[str, object]] = []
    while roads and len(selected) < count:
        next_roads: list[str] = []
        for road in roads:
            values = buckets[road]
            if values and len(selected) < count:
                selected.append(values.pop())
            if values:
                next_roads.append(road)
        roads = next_roads
    return selected


def strata_round_robin(
    records: list[dict[str, object]], count: int, seed: int, token: str
) -> list[dict[str, object]]:
    if count >= len(records):
        return sorted(records, key=lambda item: str(item["sample_key"]))
    buckets: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for record in records:
        key = (str(record["source_scientific_name"]), str(record["road_id"]))
        buckets[key].append(record)
    for key, values in buckets.items():
        random.Random(stable_seed(seed, token, *key)).shuffle(values)
    strata = sorted(buckets)
    random.Random(stable_seed(seed, token, "strata-order")).shuffle(strata)
    selected: list[dict[str, object]] = []
    while strata and len(selected) < count:
        next_strata: list[tuple[str, str]] = []
        for key in strata:
            values = buckets[key]
            if values and len(selected) < count:
                selected.append(values.pop())
            if values:
                next_strata.append(key)
        strata = next_strata
    return selected


def prepare_record(source: dict[str, object]) -> dict[str, object]:
    record = dict(source)
    source_name = str(record["scientific_name"])
    model_name = source_name if source_name in TARGET_CLASSES else "Other"
    record["source_class_index"] = int(record["class_index"])
    record["source_scientific_name"] = source_name
    record["class_index"] = MODEL_CLASS_INDEX[model_name]
    record["scientific_name"] = model_name
    record["model_class_index"] = MODEL_CLASS_INDEX[model_name]
    record["model_class_name"] = model_name
    record["source_split"] = str(record["benchmark_split"])
    record["taxonomy_source"] = "WHU-STree official benchmark label"
    record["review_status"] = "unreviewed"
    record["rating"] = ""
    record["action"] = ""
    record["review_scope"] = ""
    record["review_required"] = False
    return record


def select_candidates(
    records: list[dict[str, object]], args: argparse.Namespace
) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for source in records:
        record = prepare_record(source)
        grouped[(str(record["model_class_name"]), str(record["source_split"]))].append(
            record
        )

    for target in TARGET_CLASSES:
        train = grouped[(target, "train")]
        for record in train:
            record["model_split"] = "train"
        selected.extend(train)

        val = grouped[(target, "val")]
        for record in val:
            record["model_split"] = "val"
            record["review_required"] = True
            record["review_scope"] = "evaluation"
        selected.extend(val)

        test = grouped[(target, "test")]
        gold = {
            str(record["sample_key"])
            for record in road_round_robin(
                test, args.test_gold_per_class, args.seed, f"test-gold-{target}"
            )
        }
        for record in test:
            if str(record["sample_key"]) in gold:
                record["model_split"] = "test"
                record["review_required"] = True
                record["review_scope"] = "evaluation"
            else:
                record["model_split"] = "test_silver"
        selected.extend(test)

    other_train = grouped[("Other", "train")]
    by_source: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in other_train:
        by_source[str(record["source_scientific_name"])].append(record)
    for source_name, values in sorted(by_source.items()):
        chosen = road_round_robin(
            values,
            args.other_train_per_source,
            args.seed,
            f"other-train-{source_name}",
        )
        for record in chosen:
            record["model_split"] = "train"
        selected.extend(chosen)

    other_val = strata_round_robin(
        grouped[("Other", "val")],
        args.other_val_gold,
        args.seed,
        "other-val-gold",
    )
    for record in other_val:
        record["model_split"] = "val"
        record["review_required"] = True
        record["review_scope"] = "evaluation"
    selected.extend(other_val)

    other_test_all = grouped[("Other", "test")]
    by_source = defaultdict(list)
    for record in other_test_all:
        by_source[str(record["source_scientific_name"])].append(record)
    other_test: list[dict[str, object]] = []
    for source_name, values in sorted(by_source.items()):
        other_test.extend(
            road_round_robin(
                values,
                args.other_test_per_source,
                args.seed,
                f"other-test-{source_name}",
            )
        )
    gold = {
        str(record["sample_key"])
        for record in strata_round_robin(
            other_test,
            args.test_gold_per_class,
            args.seed,
            "other-test-gold",
        )
    }
    for record in other_test:
        if str(record["sample_key"]) in gold:
            record["model_split"] = "test"
            record["review_required"] = True
            record["review_scope"] = "evaluation"
        else:
            record["model_split"] = "test_silver"
    selected.extend(other_test)

    keys = [str(record["sample_key"]) for record in selected]
    if len(keys) != len(set(keys)):
        raise ValueError("Candidate selection produced duplicated sample keys")
    return sorted(selected, key=lambda item: str(item["sample_key"]))


def image_metrics(path: Path) -> dict[str, object]:
    with Image.open(path) as image:
        image.load()
        width, height = image.size
        grayscale = image.convert("L")
        grayscale.thumbnail((256, 256), Image.Resampling.BILINEAR)
        array = np.asarray(grayscale, dtype=np.float32)
    p01, p05, p50, p95, p99 = np.percentile(array, [1, 5, 50, 95, 99])
    dx = np.abs(np.diff(array, axis=1)).mean() if array.shape[1] > 1 else 0.0
    dy = np.abs(np.diff(array, axis=0)).mean() if array.shape[0] > 1 else 0.0
    return {
        "decoded": True,
        "width": int(width),
        "height": int(height),
        "mean_luminance": round(float(array.mean()), 4),
        "luminance_std": round(float(array.std()), 4),
        "p01": round(float(p01), 4),
        "p05": round(float(p05), 4),
        "p50": round(float(p50), 4),
        "p95": round(float(p95), 4),
        "p99": round(float(p99), 4),
        "dynamic_range_90": round(float(p95 - p05), 4),
        "edge_energy": round(float(dx + dy), 4),
        "dark_ratio": round(float((array <= 15).mean()), 6),
        "bright_ratio": round(float((array >= 240).mean()), 6),
    }


def audit_images(
    dataset_root: Path,
    records: list[dict[str, object]],
    skip_hash_recheck: bool,
    workers: int,
    progress_path: Path,
) -> tuple[list[dict[str, object]], dict[str, list[str]]]:
    def audit_one(record: dict[str, object]) -> dict[str, object]:
        image_path = dataset_root / str(record["image_path"])
        row: dict[str, object] = {"sample_key": str(record["sample_key"])}
        try:
            row.update(image_metrics(image_path))
            row["file_bytes"] = image_path.stat().st_size
            actual_hash = (
                str(record["image_sha256"])
                if skip_hash_recheck
                else sha256_file(image_path)
            )
            row["sha256"] = actual_hash
            row["hash_matches_source"] = actual_hash == str(record["image_sha256"])
        except Exception as error:  # noqa: BLE001
            row.update(
                {"decoded": False, "error": str(error), "hash_matches_source": False}
            )
        return row

    metrics: list[dict[str, object]] = []
    hashes: dict[str, list[str]] = defaultdict(list)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        results = executor.map(audit_one, records, chunksize=16)
        for index, row in enumerate(results, start=1):
            if row.get("sha256"):
                hashes[str(row["sha256"])].append(str(row["sample_key"]))
            metrics.append(row)
            if index % 100 == 0 or index == len(records):
                write_json(
                    progress_path,
                    {
                        "stage": "automatic_image_audit",
                        "status": "running" if index < len(records) else "complete",
                        "completed": index,
                        "total": len(records),
                        "percent": round(index * 100.0 / len(records), 2),
                        "updated_at": timestamp(),
                    },
                )
                print(f"Automatic image audit: {index}/{len(records)}", flush=True)
    duplicates = {digest: keys for digest, keys in hashes.items() if len(keys) > 1}
    return metrics, duplicates


def assign_quality(
    records: list[dict[str, object]], metrics: list[dict[str, object]]
) -> None:
    by_key = {str(row["sample_key"]): row for row in metrics}
    valid_edges = sorted(
        float(row["edge_energy"])
        for row in metrics
        if bool(row.get("decoded")) and "edge_energy" in row
    )
    blur_threshold = float(np.percentile(valid_edges, 3)) if valid_edges else math.inf
    for record in records:
        row = by_key[str(record["sample_key"])]
        reasons: list[str] = []
        score = 0.0
        if not bool(row.get("decoded")):
            reasons.append("decode_failure")
            score += 100.0
        if not bool(row.get("hash_matches_source")):
            reasons.append("hash_mismatch")
            score += 100.0
        if (int(row.get("width", 0)), int(row.get("height", 0))) != (768, 512):
            reasons.append("unexpected_dimensions")
            score += 50.0
        edge = float(row.get("edge_energy", 0.0))
        if edge <= blur_threshold:
            reasons.append("low_edge_energy")
            score += max(0.0, blur_threshold - edge) + 10.0
        dynamic_range = float(row.get("dynamic_range_90", 0.0))
        if dynamic_range < 45.0:
            reasons.append("low_dynamic_range")
            score += (45.0 - dynamic_range) / 2.0
        dark_ratio = float(row.get("dark_ratio", 0.0))
        bright_ratio = float(row.get("bright_ratio", 0.0))
        if dark_ratio > 0.35:
            reasons.append("excessive_dark_pixels")
            score += (dark_ratio - 0.35) * 40.0
        if bright_ratio > 0.35:
            reasons.append("excessive_bright_pixels")
            score += (bright_ratio - 0.35) * 40.0
        crop_fraction = float(record.get("crop_visible_fraction", 0.0))
        if crop_fraction < 0.995:
            reasons.append("crop_near_panorama_boundary")
            score += (0.995 - crop_fraction) * 1000.0
        camera_distance = float(record.get("camera_distance_m", 0.0))
        if camera_distance > 30.0:
            reasons.append("distant_camera")
            score += min(20.0, (camera_distance - 30.0) / 2.0)
        hard_failure = any(
            reason in {"decode_failure", "hash_mismatch", "unexpected_dimensions"}
            for reason in reasons
        )
        record["automatic_quality_tier"] = (
            "blocked" if hard_failure else "risk" if reasons else "passed"
        )
        record["automatic_risk_score"] = round(score, 4)
        record["automatic_risk_reasons"] = reasons
        record["automatic_metrics"] = row


def add_training_risk_review(
    records: list[dict[str, object]], count_per_class: int, seed: int
) -> None:
    for class_name in MODEL_CLASSES:
        candidates = [
            record
            for record in records
            if record["model_split"] == "train"
            and record["model_class_name"] == class_name
            and record["automatic_quality_tier"] != "blocked"
        ]
        candidates.sort(
            key=lambda item: (
                -float(item["automatic_risk_score"]),
                stable_seed(seed, "risk", item["sample_key"]),
            )
        )
        for record in candidates[:count_per_class]:
            record["review_required"] = True
            record["review_scope"] = "training_risk"


def inherit_reviews(
    records: list[dict[str, object]], existing_review_path: Path
) -> dict[str, dict[str, object]]:
    if not existing_review_path.is_file():
        return {}
    payload = json.loads(existing_review_path.read_text(encoding="utf-8-sig"))
    reviews = payload.get("reviews", {})
    if not isinstance(reviews, dict):
        return {}
    inherited: dict[str, dict[str, object]] = {}
    by_key = {str(record["sample_key"]): record for record in records}
    for key, raw in reviews.items():
        if key not in by_key or not isinstance(raw, dict):
            continue
        if raw.get("rating") == "pass" and raw.get("action") in VALID_KEEP_ACTIONS:
            review = dict(raw)
            review["inherited_from"] = str(existing_review_path)
            inherited[key] = review
            by_key[key]["review_status"] = "reviewed"
            by_key[key]["rating"] = "pass"
            by_key[key]["action"] = "keep"
    return inherited


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_manifest, source_records = load_records(dataset_root)
    allowed = eligible_keys(args.eligible_manifest.resolve())
    eligible = [record for record in source_records if str(record["sample_key"]) in allowed]
    if len(eligible) != len(allowed):
        raise ValueError("Some eligible sample keys are absent from the source manifest")

    records = select_candidates(eligible, args)
    metrics, duplicates = audit_images(
        dataset_root,
        records,
        args.skip_hash_recheck,
        args.workers,
        output_dir / "progress.json",
    )
    assign_quality(records, metrics)
    add_training_risk_review(records, args.training_risk_per_class, args.seed)
    inherited = inherit_reviews(records, args.existing_review.resolve())

    review_records = [record for record in records if bool(record["review_required"])]
    review_keys = {str(record["sample_key"]) for record in review_records}
    inherited = {key: value for key, value in inherited.items() if key in review_keys}
    for record in review_records:
        record["quality_preview_path"] = str(
            record.get("quality_preview_path", "") or "__dynamic_overlay__"
        )

    classes = [
        {"class_index": index, "scientific_name": name}
        for index, name in enumerate(MODEL_CLASSES)
    ]
    candidate_manifest = {
        "format_version": 1,
        "stage": "D1-four-class-image-quality-plan",
        "status": "awaiting_user_review",
        "generated_at": timestamp(),
        "source_dataset_root": str(dataset_root),
        "source_manifest": str(dataset_root / "shared_manifest.json"),
        "source_manifest_sha256": sha256_file(dataset_root / "shared_manifest.json"),
        "eligible_manifest": str(args.eligible_manifest.resolve()),
        "selection_seed": args.seed,
        "classes": classes,
        "records": records,
    }
    review_manifest = {
        "format_version": 1,
        "stage": "D1-four-class-image-human-review",
        "status": "awaiting_user_review",
        "generated_at": timestamp(),
        "source_dataset_root": str(dataset_root),
        "classes": classes,
        "records": review_records,
    }
    candidate_path = output_dir / "candidate_manifest.json"
    review_path = output_dir / "review_manifest.json"
    write_json(candidate_path, candidate_manifest)
    write_json(review_path, review_manifest)

    review_state = {
        "schema_version": 1,
        "stage": "D1-four-class-image-human-review",
        "reviewer": "user",
        "dataset_root": str(dataset_root),
        "source_manifest": str(review_path),
        "source_manifest_sha256": sha256_file(review_path),
        "created_at": timestamp(),
        "updated_at": timestamp(),
        "scope_status": {},
        "reviews": inherited,
    }
    write_json(output_dir / "user_visual_review.json", review_state)

    metric_by_key = {str(row["sample_key"]): row for row in metrics}
    quality_rows = []
    for record in records:
        row = {
            "sample_key": record["sample_key"],
            "model_split": record["model_split"],
            "model_class_index": record["model_class_index"],
            "model_class_name": record["model_class_name"],
            "source_scientific_name": record["source_scientific_name"],
            "road_id": record["road_id"],
            "trajectory_id": record["trajectory_id"],
            "tree_id": record["tree_id"],
            "automatic_quality_tier": record["automatic_quality_tier"],
            "automatic_risk_score": record["automatic_risk_score"],
            "automatic_risk_reasons": "|".join(record["automatic_risk_reasons"]),
            "review_required": record["review_required"],
            "review_scope": record["review_scope"],
            "review_status": record["review_status"],
        }
        row.update(metric_by_key[str(record["sample_key"])])
        quality_rows.append(row)
    write_csv(output_dir / "automatic_quality_audit.csv", quality_rows)

    split_hist = Counter(str(record["model_split"]) for record in records)
    class_split_hist = Counter(
        (str(record["model_class_name"]), str(record["model_split"]))
        for record in records
    )
    tier_hist = Counter(str(record["automatic_quality_tier"]) for record in records)
    review_scope_hist = Counter(
        str(record["review_scope"])
        for record in review_records
        if str(record["review_scope"])
    )
    duplicate_cross_split = []
    by_key = {str(record["sample_key"]): record for record in records}
    for digest, keys in duplicates.items():
        splits = sorted({str(by_key[key]["model_split"]) for key in keys})
        if len(splits) > 1:
            duplicate_cross_split.append({"sha256": digest, "keys": keys, "splits": splits})
    blocked = [
        str(record["sample_key"])
        for record in records
        if record["automatic_quality_tier"] == "blocked"
    ]
    plan = {
        "stage": "D1",
        "status": "awaiting_user_review" if review_records else "ready",
        "generated_at": timestamp(),
        "policy": {
            "target_classes": list(TARGET_CLASSES),
            "target_samples": "all eligible train/val/test samples",
            "other_train_per_source_class": args.other_train_per_source,
            "other_val_gold_total": args.other_val_gold,
            "other_test_per_source_class": args.other_test_per_source,
            "test_gold_per_model_class": args.test_gold_per_class,
            "training_risk_review_per_model_class": args.training_risk_per_class,
            "evaluation_rule": "Only user-approved val/test samples enter formal metrics",
            "silver_rule": "Unreviewed train and test_silver samples remain explicitly marked",
        },
        "counts": {
            "eligible_source_samples": len(eligible),
            "candidate_samples": len(records),
            "split_histogram": dict(sorted(split_hist.items())),
            "class_split_histogram": {
                f"{class_name}|{split}": count
                for (class_name, split), count in sorted(class_split_hist.items())
            },
            "automatic_quality_tier_histogram": dict(sorted(tier_hist.items())),
            "review_required": len(review_records),
            "review_scope_histogram": dict(sorted(review_scope_hist.items())),
            "inherited_user_pass_reviews": len(inherited),
            "remaining_user_reviews": len(review_records) - len(inherited),
            "duplicate_image_groups": len(duplicates),
            "cross_split_duplicate_groups": len(duplicate_cross_split),
            "blocked_samples": len(blocked),
        },
        "checks": {
            "candidate_keys_unique": len(records)
            == len({str(record["sample_key"]) for record in records}),
            "all_target_samples_retained": all(
                sum(
                    1
                    for record in records
                    if record["source_scientific_name"] == target
                )
                == sum(
                    1
                    for record in eligible
                    if record["scientific_name"] == target
                )
                for target in TARGET_CLASSES
            ),
            "no_cross_split_image_duplicates": not duplicate_cross_split,
            "no_blocked_samples": not blocked,
            "source_asset_hashes_match": all(
                bool(row.get("hash_matches_source")) for row in metrics
            ),
        },
        "blocked_sample_keys": blocked,
        "cross_split_duplicates": duplicate_cross_split,
        "artifacts": {
            "candidate_manifest": str(candidate_path),
            "review_manifest": str(review_path),
            "review_state": str(output_dir / "user_visual_review.json"),
            "automatic_quality_audit": str(output_dir / "automatic_quality_audit.csv"),
        },
    }
    write_json(output_dir / "quality_plan.json", plan)
    print(json.dumps(plan, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
