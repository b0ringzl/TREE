"""Validate the C1c shared-asset smoke export and render a visual contact sheet."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont


SANDBOX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SANDBOX_ROOT / "src"))

from tscmdl_whu.user_review import (  # noqa: E402
    REVIEW_SCOPE_QUALITY,
    inspect_user_review_state,
)


EXPECTED_CLASS_COUNT = 19
EXPECTED_SAMPLE_COUNT = 19
EXPECTED_TRAJECTORY_COUNT = 3
EXPECTED_POINT_COUNT = 8192
EXPECTED_IMAGE_SIZE = (768, 512)
MAX_OUTPUT_BYTES = 100 * 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{path}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def render_contact_sheet(
    smoke_root: Path,
    records: list[dict[str, object]],
    output_path: Path,
) -> None:
    columns = 5
    rows = (len(records) + columns - 1) // columns
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
    for index, record in enumerate(
        sorted(records, key=lambda item: int(item["benchmark_label_id"]))
    ):
        column = index % columns
        row = index // columns
        left = column * cell_width
        top = row * cell_height
        preview_path = smoke_root / str(record["quality_preview_path"])
        with Image.open(preview_path) as source:
            preview = source.convert("RGB")
        preview.thumbnail((cell_width - 12, image_height - 10), Image.Resampling.LANCZOS)
        image_left = left + (cell_width - preview.width) // 2
        image_top = top + 6
        sheet.paste(preview, (image_left, image_top))
        draw.rectangle(
            (left, top, left + cell_width - 1, top + cell_height - 1),
            outline=(190, 196, 204),
            width=1,
        )
        label = int(record["benchmark_label_id"])
        name = str(record["scientific_name"])
        track = f'{record["road_id"]}/{record["trajectory_id"]}'
        draw.text(
            (left + 8, top + image_height + 2),
            f"{label:02d}  {name[:35]}",
            fill=(25, 32, 42),
            font=font,
        )
        draw.text(
            (left + 8, top + image_height + 20),
            f'{record["sample_key"]}  track {track}',
            fill=(75, 85, 99),
            font=font,
        )
    temporary = Path(f"{output_path}.tmp")
    sheet.save(temporary, format="JPEG", quality=92)
    temporary.replace(output_path)


def main() -> None:
    args = parse_args()
    smoke_root = args.smoke_root.resolve()
    output_path = (
        args.output.resolve()
        if args.output is not None
        else smoke_root / "validation.json"
    )
    manifest_path = smoke_root / "manifest.json"
    checkpoint_path = smoke_root / "export_checkpoint.json"
    plan_path = smoke_root / "selection_plan.json"
    visual_review_path = smoke_root / "visual_review.json"
    for path in (manifest_path, checkpoint_path, plan_path, visual_review_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    manifest = read_json(manifest_path)
    checkpoint = read_json(checkpoint_path)
    selection_plan = read_json(plan_path)
    visual_review = read_json(visual_review_path)
    if not all(
        isinstance(item, dict)
        for item in (manifest, checkpoint, selection_plan, visual_review)
    ):
        raise ValueError("Invalid C1c JSON artifact")
    records = list(manifest["records"])
    user_review_path = smoke_root / "user_visual_review.json"
    user_review_status = inspect_user_review_state(
        user_review_path,
        manifest_path,
        records,
        REVIEW_SCOPE_QUALITY,
    )
    plan_without_hash = {
        key: value for key, value in selection_plan.items() if key != "plan_sha256"
    }
    plan_hash = sha256_json(plan_without_hash)
    source_checks = {}
    for name, source in selection_plan["sources"].items():
        source_path = Path(str(source["path"]))
        source_checks[name] = (
            source_path.is_file()
            and sha256_file(source_path) == str(source["sha256"])
        )

    shared_rows = read_csv(
        Path(str(selection_plan["sources"]["shared_assets"]["path"]))
    )
    benchmark_rows = read_csv(
        Path(str(selection_plan["sources"]["benchmark_manifest"]["path"]))
    )
    road_rows = read_csv(
        Path(str(selection_plan["sources"]["road_manifest"]["path"]))
    )
    shared_by_key = {row["sample_key"]: row for row in shared_rows}
    benchmark_by_key = {row["sample_key"]: row for row in benchmark_rows}
    road_by_key: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in road_rows:
        road_by_key[row["sample_key"]].append(row)

    point_contract = True
    image_contract = True
    hashes_match = True
    mtimes_match = True
    shared_paths_match = True
    road_membership_match = True
    preview_contract = True
    crop_visible = True
    point_paths: list[str] = []
    image_paths: list[str] = []

    for record in records:
        sample_key = str(record["sample_key"])
        point_relative = str(record["point_path"])
        image_relative = str(record["image_path"])
        point_paths.append(point_relative)
        image_paths.append(image_relative)
        point_path = smoke_root / point_relative
        image_path = smoke_root / image_relative
        preview_path = smoke_root / str(record["quality_preview_path"])
        if not point_path.is_file() or not image_path.is_file():
            point_contract = image_contract = False
            continue

        shared = shared_by_key[sample_key]
        benchmark = benchmark_by_key[sample_key]
        shared_paths_match &= (
            point_relative == shared["point_path"] == benchmark["point_path"]
            and image_relative == shared["image_path"] == benchmark["image_path"]
        )
        expected_membership = sorted(
            (
                int(item["run_index"]),
                item["split"],
                int(item["class_index"]),
                int(item["balanced_evaluation_selected"]),
            )
            for item in road_by_key.get(sample_key, [])
        )
        actual_membership = sorted(
            (
                int(item["run_index"]),
                str(item["split"]),
                int(item["class_index"]),
                int(item["balanced_evaluation_selected"]),
            )
            for item in record["road_membership"]
        )
        road_membership_match &= actual_membership == expected_membership

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
                image.load()
                image_contract &= (
                    image.size == EXPECTED_IMAGE_SIZE and image.format == "JPEG"
                )
            with Image.open(preview_path) as preview:
                preview.load()
                preview_contract &= (
                    preview.size == EXPECTED_IMAGE_SIZE and preview.format == "JPEG"
                )
        except (OSError, ValueError):
            image_contract = False
            preview_contract = False

        hashes_match &= (
            sha256_file(point_path) == str(record["point_sha256"])
            and sha256_file(image_path) == str(record["image_sha256"])
        )
        mtimes_match &= (
            point_path.stat().st_mtime_ns == int(record["point_mtime_ns"])
            and image_path.stat().st_mtime_ns == int(record["image_mtime_ns"])
        )
        crop_visible &= float(record["crop_visible_fraction"]) >= 0.95

    contact_sheet_path = smoke_root / "smoke_contact_sheet.jpg"
    render_contact_sheet(smoke_root, records, contact_sheet_path)
    events = list(checkpoint["run_events"])
    labels = [int(record["benchmark_label_id"]) for record in records]
    trajectories = {
        (str(record["road_id"]), str(record["trajectory_id"])) for record in records
    }
    point_files = list((smoke_root / "assets" / "points").glob("*.npz"))
    image_files = list((smoke_root / "assets" / "images").glob("*.jpg"))

    checks = {
        "selection_plan_integrity": (
            selection_plan.get("plan_sha256") == plan_hash
            and manifest.get("source_plan_sha256") == plan_hash
            and checkpoint.get("source_plan_sha256") == plan_hash
        ),
        "source_files_unchanged": all(source_checks.values()),
        "checkpoint_complete": (
            checkpoint.get("status") == "complete"
            and manifest.get("status") == "complete"
        ),
        "all_19_classes_once": (
            len(records) == EXPECTED_SAMPLE_COUNT
            and sorted(labels) == list(range(EXPECTED_CLASS_COUNT))
        ),
        "three_development_trajectories": (
            len(trajectories) == EXPECTED_TRAJECTORY_COUNT
            and all(
                int(shared_by_key[str(record["sample_key"])]["is_official_test"])
                == 0
                for record in records
            )
        ),
        "resume_reuse_verified": (
            len(events) >= 2
            and events[0]["status"] == "partial"
            and events[-1]["status"] == "complete"
            and int(events[-1]["verified_reused_asset_count"]) > 0
            and mtimes_match
        ),
        "assets_written_once": (
            len(point_files) == EXPECTED_SAMPLE_COUNT
            and len(image_files) == EXPECTED_SAMPLE_COUNT
            and len(point_paths) == len(set(point_paths))
            and len(image_paths) == len(set(image_paths))
        ),
        "shared_paths_match_c1b": shared_paths_match,
        "road_membership_matches_c1b": road_membership_match,
        "point_contract_valid": point_contract,
        "image_contract_valid": image_contract,
        "asset_hashes_match": hashes_match,
        "quality_previews_complete": (
            preview_contract
            and sum(
                1
                for record in records
                if (smoke_root / str(record["quality_preview_path"])).is_file()
            )
            == EXPECTED_SAMPLE_COUNT
        ),
        "crop_visibility_acceptable": crop_visible,
        "codex_visual_precheck_complete": (
            int(visual_review.get("reviewed_count", 0)) == EXPECTED_SAMPLE_COUNT
            and visual_review.get("review_type") == "codex_visual_precheck"
            and int(
                visual_review.get("categories", {}).get(
                    "severe_projection_or_target_failure", -1
                )
            )
            == 0
            and str(visual_review.get("precheck_decision", "")).startswith(
                "passed"
            )
        ),
        "disk_monitoring_recorded": (
            all(
                int(event["free_disk_before_bytes"]) > 0
                and int(event["free_disk_after_bytes"]) > 0
                for event in events
            )
            and int(manifest["summary"]["output_directory_bytes"]) > 0
        ),
        "smoke_output_under_100_mib": directory_bytes(smoke_root)
        < MAX_OUTPUT_BYTES,
        "full_export_not_started": (
            len(records) == EXPECTED_SAMPLE_COUNT
            and len(point_files) == EXPECTED_SAMPLE_COUNT
        ),
        "training_not_started": not any(smoke_root.rglob("*.pt")),
    }
    status = "passed" if all(checks.values()) else "failed"
    validation = {
        "stage": "C1c",
        "status": status,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "checks": checks,
        "source_checks": source_checks,
        "review_status": {
            "codex_visual_precheck": "complete",
            "user_visual_review": user_review_status,
        },
        "summary": {
            "sample_count": len(records),
            "class_count": len(set(labels)),
            "trajectory_count": len(trajectories),
            "resume_run_count": len(events),
            "reused_asset_count": int(events[-1]["verified_reused_asset_count"]),
            "minimum_crop_visible_fraction": min(
                float(record["crop_visible_fraction"]) for record in records
            ),
            "shared_asset_bytes": sum(
                path.stat().st_size for path in point_files + image_files
            ),
            "output_directory_bytes": directory_bytes(smoke_root),
        },
        "artifacts": {
            "selection_plan": str(plan_path),
            "checkpoint": str(checkpoint_path),
            "manifest": str(manifest_path),
            "manifest_csv": str(smoke_root / "manifest.csv"),
            "summary": str(smoke_root / "summary.json"),
            "contact_sheet": str(contact_sheet_path),
            "visual_review": str(visual_review_path),
            **(
                {"user_visual_review": str(user_review_path)}
                if user_review_path.is_file()
                else {}
            ),
            "progress": str(smoke_root / "progress.json"),
        },
    }
    atomic_json(output_path, validation)
    print(json.dumps(validation, ensure_ascii=False, indent=2), flush=True)
    if status != "passed":
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"C1c validation failed: {failed}")


if __name__ == "__main__":
    main()
