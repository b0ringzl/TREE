"""Perform full integrity, format, split, and visual-artifact validation for C1."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


SANDBOX_ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(SANDBOX_ROOT / "src"))

from tscmdl_whu.export_assets import (  # noqa: E402
    atomic_json,
    directory_bytes,
    export_timestamp,
    sha256_file,
    sha256_json,
)
from tscmdl_whu.user_review import (  # noqa: E402
    REVIEW_SCOPE_QUALITY,
    inspect_user_review_state,
)


EXPECTED_SAMPLE_COUNT = 17134
EXPECTED_TRAJECTORY_COUNT = 110
EXPECTED_CLASS_COUNT = 19
EXPECTED_ROAD_CLASS_COUNT = 16
EXPECTED_POINT_COUNT = 8192
EXPECTED_IMAGE_SIZE = (768, 512)
EXPECTED_BENCHMARK_SPLITS = {"train": 13116, "val": 570, "test": 3448}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--progress-every", type=int, default=250)
    return parser.parse_args()


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def render_contact_sheet(
    dataset_root: Path,
    records: list[dict[str, object]],
    output_path: Path,
) -> None:
    preview_records = sorted(
        [record for record in records if "quality_preview_path" in record],
        key=lambda record: (
            int(record["benchmark_label_id"]),
            str(record["sample_key"]),
        ),
    )
    columns = 5
    rows = (len(preview_records) + columns - 1) // columns
    cell_width = 300
    cell_height = 240
    image_height = 190
    sheet = Image.new(
        "RGB",
        (columns * cell_width, rows * cell_height),
        (246, 247, 249),
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, record in enumerate(preview_records):
        column = index % columns
        row = index // columns
        left = column * cell_width
        top = row * cell_height
        preview_path = dataset_root / str(record["quality_preview_path"])
        with Image.open(preview_path) as source:
            preview = source.convert("RGB")
        preview.thumbnail(
            (cell_width - 12, image_height - 10), Image.Resampling.LANCZOS
        )
        image_left = left + (cell_width - preview.width) // 2
        sheet.paste(preview, (image_left, top + 6))
        draw.rectangle(
            (left, top, left + cell_width - 1, top + cell_height - 1),
            outline=(190, 196, 204),
            width=1,
        )
        label = int(record["benchmark_label_id"])
        name = str(record["scientific_name"])
        draw.text(
            (left + 8, top + image_height + 2),
            f"{label:02d}  {name[:35]}",
            fill=(25, 32, 42),
            font=font,
        )
        draw.text(
            (left + 8, top + image_height + 20),
            (
                f'{record["sample_key"]}  '
                f'{record["road_id"]}/{record["trajectory_id"]}'
            ),
            fill=(75, 85, 99),
            font=font,
        )
    temporary = Path(f"{output_path}.tmp")
    sheet.save(temporary, format="JPEG", quality=92)
    temporary.replace(output_path)


def write_validation_progress(
    path: Path,
    *,
    checked: int,
    total: int,
    current: str,
    started: float,
    status: str,
) -> None:
    elapsed = time.perf_counter() - started
    eta = elapsed / checked * (total - checked) if checked else None
    atomic_json(
        path,
        {
            "stage": "C1-validation",
            "status": status,
            "updated_at": export_timestamp(),
            "checked_samples": checked,
            "total_samples": total,
            "progress_fraction": checked / total,
            "current_sample": current,
            "elapsed_seconds": round(elapsed, 3),
            "eta_seconds": round(eta, 3) if eta is not None else None,
        },
    )


def main() -> None:
    args = parse_args()
    if args.progress_every <= 0:
        raise ValueError("progress-every must be positive")
    dataset_root = args.dataset_root.resolve()
    output_path = (
        args.output.resolve()
        if args.output is not None
        else dataset_root / "validation.json"
    )
    progress_path = dataset_root / "validation_progress.json"
    plan_path = dataset_root / "export_plan.json"
    checkpoint_path = dataset_root / "export_checkpoint.json"
    manifest_path = dataset_root / "shared_manifest.json"
    summary_path = dataset_root / "summary.json"
    shared_csv_path = dataset_root / "shared_manifest.csv"
    benchmark_csv_path = dataset_root / "benchmark_19_manifest.csv"
    road_csv_path = dataset_root / "road_domain_16_manifest.csv"
    classes_19_path = dataset_root / "classes_19.json"
    classes_16_path = dataset_root / "classes_16.json"
    required = (
        plan_path,
        checkpoint_path,
        manifest_path,
        summary_path,
        shared_csv_path,
        benchmark_csv_path,
        road_csv_path,
        classes_19_path,
        classes_16_path,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    started = time.perf_counter()
    plan = read_json(plan_path)
    checkpoint = read_json(checkpoint_path)
    manifest = read_json(manifest_path)
    summary = read_json(summary_path)
    if not all(
        isinstance(item, dict) for item in (plan, checkpoint, manifest, summary)
    ):
        raise ValueError("Invalid C1 JSON artifact")
    plan_without_hash = {
        key: value for key, value in plan.items() if key != "plan_sha256"
    }
    plan_hash = sha256_json(plan_without_hash)
    source_checks = {}
    for name, source in plan["sources"].items():
        path = Path(str(source["path"]))
        source_checks[name] = (
            path.is_file() and sha256_file(path) == str(source["sha256"])
        )

    records = list(manifest["records"])
    record_by_key = {str(record["sample_key"]): record for record in records}
    source_shared_rows = read_csv(
        Path(str(plan["sources"]["shared_assets"]["path"]))
    )
    source_benchmark_rows = read_csv(
        Path(str(plan["sources"]["benchmark_manifest"]["path"]))
    )
    source_road_rows = read_csv(
        Path(str(plan["sources"]["road_manifest"]["path"]))
    )
    source_shared_by_key = {
        row["sample_key"]: row for row in source_shared_rows
    }
    exported_shared_rows = read_csv(shared_csv_path)
    exported_benchmark_rows = read_csv(benchmark_csv_path)
    exported_road_rows = read_csv(road_csv_path)

    point_files = sorted((dataset_root / "assets" / "points").glob("*.npz"))
    image_files = sorted((dataset_root / "assets" / "images").glob("*.jpg"))
    class_histogram: Counter[int] = Counter()
    split_histogram: Counter[str] = Counter()
    trajectories = set()
    point_contract = True
    image_contract = True
    asset_hashes = True
    asset_metadata = True
    path_contract = True
    crop_visibility = True

    write_validation_progress(
        progress_path,
        checked=0,
        total=len(records),
        current="",
        started=started,
        status="running",
    )
    for index, record in enumerate(records, start=1):
        sample_key = str(record["sample_key"])
        source = source_shared_by_key.get(sample_key)
        if source is None:
            path_contract = False
            continue
        expected_point = f"assets/points/{sample_key}.npz"
        expected_image = f"assets/images/{sample_key}.jpg"
        point_relative = str(record["point_path"])
        image_relative = str(record["image_path"])
        path_contract &= (
            point_relative == expected_point == source["point_path"]
            and image_relative == expected_image == source["image_path"]
        )
        point_path = dataset_root / point_relative
        image_path = dataset_root / image_relative
        if not point_path.is_file() or not image_path.is_file():
            point_contract = image_contract = asset_hashes = asset_metadata = False
            continue
        asset_metadata &= (
            point_path.stat().st_size == int(record["point_bytes"])
            and image_path.stat().st_size == int(record["image_bytes"])
            and point_path.stat().st_mtime_ns == int(record["point_mtime_ns"])
            and image_path.stat().st_mtime_ns == int(record["image_mtime_ns"])
        )
        asset_hashes &= (
            sha256_file(point_path) == str(record["point_sha256"])
            and sha256_file(image_path) == str(record["image_sha256"])
        )
        try:
            with np.load(point_path, allow_pickle=False) as archive:
                points = archive["points_xyz"]
                point_contract &= (
                    points.shape == (EXPECTED_POINT_COUNT, 3)
                    and points.dtype == np.float32
                    and bool(np.isfinite(points).all())
                    and float(np.linalg.norm(points, axis=1).max()) <= 1.0001
                    and int(archive["class_index"]) == int(record["class_index"])
                    and int(archive["benchmark_label_id"])
                    == int(record["benchmark_label_id"])
                    and int(archive["raw_label_id"]) == int(record["raw_label_id"])
                    and int(archive["source_point_count"])
                    == int(record["source_point_count"])
                )
        except (KeyError, OSError, ValueError):
            point_contract = False
        try:
            with Image.open(image_path) as image:
                image_contract &= (
                    image.size == EXPECTED_IMAGE_SIZE and image.format == "JPEG"
                )
                image.verify()
        except (OSError, ValueError):
            image_contract = False
        class_histogram[int(record["benchmark_label_id"])] += 1
        split_histogram[str(record["benchmark_split"])] += 1
        trajectories.add(
            (str(record["road_id"]), str(record["trajectory_id"]))
        )
        crop_visibility &= float(record["crop_visible_fraction"]) >= 0.95

        if index % args.progress_every == 0 or index == len(records):
            write_validation_progress(
                progress_path,
                checked=index,
                total=len(records),
                current=sample_key,
                started=started,
                status="running",
            )
            elapsed = time.perf_counter() - started
            eta = elapsed / index * (len(records) - index)
            print(
                f"[{index:5d}/{len(records)}] {sample_key:<20} "
                f"elapsed={elapsed/60:5.1f}m eta={eta/60:5.1f}m",
                flush=True,
            )

    preview_records = [
        record for record in records if "quality_preview_path" in record
    ]
    preview_contract = len(preview_records) == 38
    for record in preview_records:
        preview_path = dataset_root / str(record["quality_preview_path"])
        try:
            with Image.open(preview_path) as image:
                preview_contract &= (
                    image.size == EXPECTED_IMAGE_SIZE and image.format == "JPEG"
                )
                image.verify()
        except (OSError, ValueError):
            preview_contract = False

    contact_sheet_path = dataset_root / "quality_contact_sheet.jpg"
    render_contact_sheet(dataset_root, records, contact_sheet_path)
    visual_review_path = dataset_root / "visual_review.json"
    visual_review = (
        read_json(visual_review_path) if visual_review_path.is_file() else None
    )
    codex_visual_precheck = None
    if isinstance(visual_review, dict):
        categories = visual_review.get("categories", {})
        codex_visual_precheck = (
            int(visual_review.get("reviewed_count", 0)) == 38
            and visual_review.get("review_type") == "codex_visual_precheck"
            and isinstance(categories, dict)
            and int(categories.get("severe_projection_or_target_failure", -1))
            == 0
            and str(visual_review.get("precheck_decision", "")).startswith(
                "passed"
            )
        )
    user_review_path = dataset_root / "user_visual_review.json"
    user_review_status = inspect_user_review_state(
        user_review_path,
        manifest_path,
        records,
        REVIEW_SCOPE_QUALITY,
    )

    checkpoint_shards = True
    for trajectory_key, entry in checkpoint["completed_trajectories"].items():
        shard_path = dataset_root / str(entry["shard_path"])
        checkpoint_shards &= (
            shard_path.is_file()
            and sha256_file(shard_path) == str(entry["shard_sha256"])
        )

    source_benchmark_keys = [row["sample_key"] for row in source_benchmark_rows]
    exported_benchmark_keys = [row["sample_key"] for row in exported_benchmark_rows]
    source_road_identity = sorted(
        (row["run_index"], row["sample_key"], row["split"])
        for row in source_road_rows
    )
    exported_road_identity = sorted(
        (row["run_index"], row["sample_key"], row["split"])
        for row in exported_road_rows
    )
    classes_19 = read_json(classes_19_path)
    classes_16 = read_json(classes_16_path)
    run_events = list(checkpoint["run_events"])
    checks = {
        "plan_integrity": (
            plan.get("plan_sha256") == plan_hash
            and manifest.get("source_plan_sha256") == plan_hash
            and checkpoint.get("source_plan_sha256") == plan_hash
        ),
        "source_files_unchanged": all(source_checks.values()),
        "checkpoint_and_manifest_complete": (
            checkpoint.get("status") == "complete"
            and manifest.get("status") == "complete"
        ),
        "trajectory_shards_intact": (
            checkpoint_shards
            and len(checkpoint["completed_trajectories"])
            == EXPECTED_TRAJECTORY_COUNT
        ),
        "shared_sample_keys_exact": (
            len(records) == EXPECTED_SAMPLE_COUNT
            and len(record_by_key) == EXPECTED_SAMPLE_COUNT
            and set(record_by_key) == set(source_shared_by_key)
            and len(exported_shared_rows) == EXPECTED_SAMPLE_COUNT
        ),
        "physical_asset_counts_exact": (
            len(point_files) == EXPECTED_SAMPLE_COUNT
            and len(image_files) == EXPECTED_SAMPLE_COUNT
        ),
        "all_19_classes_present": (
            sorted(class_histogram) == list(range(EXPECTED_CLASS_COUNT))
            and len(classes_19) == EXPECTED_CLASS_COUNT
        ),
        "all_110_trajectories_present": (
            len(trajectories) == EXPECTED_TRAJECTORY_COUNT
        ),
        "benchmark_manifest_exact": (
            source_benchmark_keys == exported_benchmark_keys
            and dict(split_histogram) == EXPECTED_BENCHMARK_SPLITS
        ),
        "road_domain_manifest_exact": (
            source_road_identity == exported_road_identity
            and len(exported_road_rows) == len(source_road_rows)
            and len(classes_16) == EXPECTED_ROAD_CLASS_COUNT
        ),
        "shared_paths_canonical": path_contract,
        "point_contract_valid": point_contract,
        "image_contract_valid": image_contract,
        "asset_hashes_match": asset_hashes,
        "asset_metadata_match_checkpoint": asset_metadata,
        "crop_visibility_acceptable": crop_visibility,
        "quality_previews_complete": preview_contract,
        "resume_history_present": (
            len(run_events) >= 2
            and any(event["status"] == "partial" for event in run_events)
            and run_events[-1]["status"] == "complete"
            and int(run_events[-1]["reused_sample_count"]) > 0
        ),
        "no_training_artifacts": not any(dataset_root.rglob("*.pt")),
    }
    if codex_visual_precheck is not None:
        checks["codex_visual_precheck_complete"] = codex_visual_precheck

    status = "passed" if all(checks.values()) else "failed"
    total_bytes = directory_bytes(dataset_root)
    validation = {
        "stage": "C1",
        "status": status,
        "generated_at": export_timestamp(),
        "checks": checks,
        "source_checks": source_checks,
        "review_status": {
            "codex_visual_precheck": (
                "complete" if codex_visual_precheck else "missing"
            ),
            "user_visual_review": user_review_status,
        },
        "summary": {
            "sample_count": len(records),
            "trajectory_count": len(trajectories),
            "class_count": len(class_histogram),
            "benchmark_split_histogram": dict(split_histogram),
            "road_manifest_row_count": len(exported_road_rows),
            "point_file_count": len(point_files),
            "image_file_count": len(image_files),
            "quality_preview_count": len(preview_records),
            "minimum_crop_visible_fraction": min(
                float(record["crop_visible_fraction"]) for record in records
            ),
            "shared_asset_bytes": sum(
                path.stat().st_size for path in point_files + image_files
            ),
            "output_directory_bytes": total_bytes,
            "validation_elapsed_seconds": round(time.perf_counter() - started, 3),
            "codex_visual_precheck_included": codex_visual_precheck is not None,
            "user_visual_review_complete": (
                user_review_status["status"] == "complete"
            ),
        },
        "artifacts": {
            "plan": str(plan_path),
            "checkpoint": str(checkpoint_path),
            "manifest": str(manifest_path),
            "shared_manifest_csv": str(shared_csv_path),
            "benchmark_manifest": str(benchmark_csv_path),
            "road_manifest": str(road_csv_path),
            "summary": str(summary_path),
            "classes_19": str(classes_19_path),
            "classes_16": str(classes_16_path),
            "contact_sheet": str(contact_sheet_path),
            "validation_progress": str(progress_path),
            **(
                {"visual_review": str(visual_review_path)}
                if visual_review_path.is_file()
                else {}
            ),
            **(
                {"user_visual_review": str(user_review_path)}
                if user_review_path.is_file()
                else {}
            ),
        },
    }
    atomic_json(output_path, validation)
    write_validation_progress(
        progress_path,
        checked=len(records),
        total=len(records),
        current="",
        started=started,
        status=status,
    )
    print(json.dumps(validation, ensure_ascii=False, indent=2), flush=True)
    if status != "passed":
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"C1 validation failed: {failed}")


if __name__ == "__main__":
    main()
