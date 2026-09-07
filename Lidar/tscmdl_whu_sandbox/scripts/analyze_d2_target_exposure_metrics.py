"""Compute target-aware exposure metrics from projected single-tree points."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


SANDBOX_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DERIVED_ROOT = PROJECT_ROOT / "lidar data" / "whu" / "derived" / "tscmdl"
DEFAULT_ROOT = (
    DERIVED_ROOT
    / "d2_exposure_stratified_evaluation"
    / "20260819_visual_confirmation_v1"
)
sys.path.insert(0, str(SANDBOX_ROOT / "src"))

from tscmdl_whu.panorama import (  # noqa: E402
    CameraPose,
    ProjectionCrop,
    project_equirectangular,
)


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def write_json(path: Path, value: object) -> None:
    temporary = Path(f"{path}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def pose(record: dict[str, object]) -> CameraPose:
    value = dict(record["camera_pose"])
    return CameraPose(
        image_name=str(value["image_name"]),
        position=tuple(float(item) for item in value["position"]),
        roll_deg=float(value["roll_deg"]),
        pitch_deg=float(value["pitch_deg"]),
        heading_deg=float(value["heading_deg"]),
    )


def crop(record: dict[str, object]) -> ProjectionCrop:
    value = dict(record["crop"])
    return ProjectionCrop(**{key: float(item) for key, item in value.items()})


def ratio_metrics(values: np.ndarray, prefix: str) -> dict[str, object]:
    values = np.asarray(values, dtype=np.float32)
    p05, p50, p95 = np.percentile(values, [5, 50, 95])
    return {
        f"{prefix}_pixel_count": int(values.size),
        f"{prefix}_mean_luminance": round(float(values.mean()), 4),
        f"{prefix}_luminance_std": round(float(values.std()), 4),
        f"{prefix}_p05": round(float(p05), 4),
        f"{prefix}_p50": round(float(p50), 4),
        f"{prefix}_p95": round(float(p95), 4),
        f"{prefix}_dark_ratio": round(float((values <= 15).mean()), 6),
        f"{prefix}_bright_ratio": round(float((values >= 240).mean()), 6),
        f"{prefix}_visible_ratio": round(
            float(((values > 15) & (values < 240)).mean()), 6
        ),
    }


def masked_edge_energy(gray: np.ndarray, mask: np.ndarray) -> float:
    values = gray.astype(np.float32)
    horizontal_mask = mask[:, 1:] & mask[:, :-1]
    vertical_mask = mask[1:, :] & mask[:-1, :]
    horizontal = np.abs(np.diff(values, axis=1))[horizontal_mask]
    vertical = np.abs(np.diff(values, axis=0))[vertical_mask]
    horizontal_mean = float(horizontal.mean()) if horizontal.size else 0.0
    vertical_mean = float(vertical.mean()) if vertical.size else 0.0
    return round(horizontal_mean + vertical_mean, 4)


def analyze_one(dataset_root: Path, record: dict[str, object]) -> dict[str, object]:
    image_path = dataset_root / str(record["image_path"])
    point_path = dataset_root / str(record["point_path"])
    with np.load(point_path, allow_pickle=False) as archive:
        normalized = np.asarray(archive["points_xyz"], dtype=np.float64)
        centroid = np.asarray(archive["centroid_xyz"], dtype=np.float64)
        scale = float(archive["scale"])
    points = normalized * scale + centroid
    panorama_width, panorama_height = (int(value) for value in record["panorama_size"])
    projection_crop = crop(record)
    u, v, _ = project_equirectangular(
        points,
        pose(record),
        panorama_width,
        panorama_height,
        apply_tilt=True,
    )
    x = np.mod(u - projection_crop.left_unwrapped_px, panorama_width)
    y = v - projection_crop.top_px
    visible = (
        (x >= 0)
        & (x < projection_crop.width_px)
        & (y >= 0)
        & (y < projection_crop.height_px)
    )
    with Image.open(image_path) as image:
        gray_image = image.convert("L")
        gray = np.asarray(gray_image, dtype=np.uint8)
    x_pixels = np.rint(
        x[visible] * gray.shape[1] / projection_crop.width_px
    ).astype(np.int64)
    y_pixels = np.rint(
        y[visible] * gray.shape[0] / projection_crop.height_px
    ).astype(np.int64)
    x_pixels = np.clip(x_pixels, 0, gray.shape[1] - 1)
    y_pixels = np.clip(y_pixels, 0, gray.shape[0] - 1)
    flat = np.unique(y_pixels * gray.shape[1] + x_pixels)
    y_unique, x_unique = np.divmod(flat, gray.shape[1])

    reasons = {str(value) for value in record.get("automatic_risk_reasons", [])}
    inferred_group = (
        "dark"
        if "excessive_dark_pixels" in reasons
        else "bright"
        if "excessive_bright_pixels" in reasons
        else ""
    )
    result: dict[str, object] = {
        "sample_key": str(record["sample_key"]),
        "automatic_group_v1": str(record.get("d2_exposure_group", inferred_group)),
        "source_scientific_name": str(record["source_scientific_name"]),
        "model_class_name": str(record["model_class_name"]),
        "road_id": str(record["road_id"]),
        "trajectory_id": str(record["trajectory_id"]),
        "visible_projected_point_fraction": round(float(visible.mean()), 6),
        "unique_projected_pixel_count": int(flat.size),
    }
    result.update(ratio_metrics(gray[y_unique, x_unique], "target_point"))

    base_mask = np.zeros(gray.shape, dtype=np.uint8)
    base_mask[y_unique, x_unique] = 255
    for radius in (2, 5):
        mask = np.asarray(
            Image.fromarray(base_mask).filter(ImageFilter.MaxFilter(radius * 2 + 1))
        ) > 0
        result[f"target_r{radius}_coverage"] = round(float(mask.mean()), 6)
        result.update(ratio_metrics(gray[mask], f"target_r{radius}"))
        result[f"target_r{radius}_edge_energy"] = masked_edge_energy(gray, mask)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=DEFAULT_ROOT / "visualization_manifest.json"
    )
    parser.add_argument(
        "--confirmation",
        type=Path,
        default=DEFAULT_ROOT / "user_exposure_confirmation.json",
    )
    parser.add_argument("--dataset-root", type=Path, default=DERIVED_ROOT / "c1_full_shared_dataset")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--exposure-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.resolve().read_text(encoding="utf-8"))
    records = [dict(value) for value in manifest["records"]]
    if args.exposure_only:
        records = [
            record
            for record in records
            if {
                "excessive_dark_pixels",
                "excessive_bright_pixels",
            }
            & {str(value) for value in record.get("automatic_risk_reasons", [])}
        ]
    dataset_root = args.dataset_root.resolve()
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        rows = list(executor.map(lambda item: analyze_one(dataset_root, item), records))
    confirmation_path = args.confirmation.resolve()
    confirmations = {}
    if confirmation_path.is_file():
        state = json.loads(confirmation_path.read_text(encoding="utf-8-sig"))
        confirmations = state.get("confirmations", {})
    for row in rows:
        confirmation = confirmations.get(str(row["sample_key"]), {})
        row["user_status"] = str(confirmation.get("status", ""))
        row["user_actual_group"] = str(confirmation.get("actual_group", ""))

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "target_exposure_metrics.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "stage": "D2a-target-aware-exposure-metrics",
        "status": "complete",
        "generated_at": timestamp(),
        "sample_count": len(rows),
        "confirmed_count": sum(bool(row["user_status"]) for row in rows),
        "metric_policy": {
            "target_point": "luminance at unique projected single-tree point pixels",
            "target_r2": "projected point mask dilated by 2 image pixels",
            "target_r5": "projected point mask dilated by 5 image pixels",
            "dark_pixel": "luminance <= 15",
            "bright_pixel": "luminance >= 240",
        },
        "rows": rows,
        "csv": str(csv_path),
    }
    json_path = output_dir / "target_exposure_metrics.json"
    write_json(json_path, summary)
    print(
        json.dumps(
            {
                "status": "complete",
                "sample_count": len(rows),
                "confirmed_count": summary["confirmed_count"],
                "csv": str(csv_path),
                "json": str(json_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
