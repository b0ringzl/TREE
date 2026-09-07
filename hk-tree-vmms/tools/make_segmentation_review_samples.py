#!/usr/bin/env python
"""Create a small original-panorama vs prediction review set from held-out test frames."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = PROJECT_ROOT / "derived" / "training_runs" / "combined_tree_segmentation"
ACTIVE_RUN_FILE = RUNS_ROOT / "active_run.json"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def latest_acceptance(run_dir: Path) -> Path:
    candidates = sorted(
        (path for path in run_dir.glob("test_acceptance_*") if (path / "per_image_metrics.csv").is_file()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("No completed test acceptance directory was found")
    return candidates[0]


def load_test_manifest(dataset_dir: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with (dataset_dir / "manifest.csv").open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row.get("split") == "test":
                rows[Path(row["image"]).stem] = row
    return rows


def choose_samples(metrics_path: Path) -> list[dict[str, str]]:
    with metrics_path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["source_dataset"]].append(row)

    selected: list[dict[str, str]] = []
    for route in ("hewentian", "jianshazui", "stubbs_road"):
        route_rows = sorted(grouped[route], key=lambda row: float(row["union_iou"]))
        difficult = dict(route_rows[0])
        representative = dict(route_rows[-1])
        difficult["selection_reason"] = "challenging"
        representative["selection_reason"] = "strong target overlap"
        selected.extend([representative, difficult])
    return selected


def resize_panorama(image: np.ndarray, width: int = 900) -> np.ndarray:
    height = max(1, round(image.shape[0] * width / image.shape[1]))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def labelled_panel(image: np.ndarray, title: str) -> np.ndarray:
    header_height = 44
    panel = np.zeros((image.shape[0] + header_height, image.shape[1], 3), dtype=np.uint8)
    panel[header_height:] = image
    cv2.putText(panel, title, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    return panel


def main() -> int:
    from ultralytics import YOLO

    marker = read_json(ACTIVE_RUN_FILE)
    run_dir = Path(marker["run_dir"])
    best_weight = run_dir / "weights" / "best.pt"
    data_yaml = Path(marker["data"])
    dataset_dir = data_yaml.parent
    acceptance_dir = latest_acceptance(run_dir)
    output_dir = acceptance_dir / "review_samples_conf050"
    output_dir.mkdir(parents=True, exist_ok=True)

    selected = choose_samples(acceptance_dir / "per_image_metrics.csv")
    manifest = load_test_manifest(dataset_dir)
    image_paths = [Path(manifest[row["image_stem"]]["image"]) for row in selected]
    model = YOLO(str(best_weight))
    results = list(
        model.predict(
            source=[str(path) for path in image_paths],
            imgsz=1024,
            conf=0.50,
            iou=0.70,
            device="0",
            stream=True,
            retina_masks=False,
            verbose=False,
        )
    )

    overview_tiles: list[np.ndarray] = []
    review_rows: list[dict[str, Any]] = []
    for row, source_path, result in zip(selected, image_paths, results):
        original = cv2.imread(str(source_path))
        if original is None:
            raise FileNotFoundError(source_path)
        prediction = result.plot(labels=False, boxes=True, masks=True, conf=True)
        original_small = resize_panorama(original)
        prediction_small = resize_panorama(prediction)
        left = labelled_panel(original_small, "Original panorama")
        prediction_count = len(result.boxes) if result.boxes is not None else 0
        right = labelled_panel(prediction_small, f"YOLO11m-seg | conf >= 0.50 | masks: {prediction_count}")
        comparison = np.hstack([left, right])
        route_title = f"{row['source_dataset']} | {row['image_stem']} | {row['selection_reason']}"
        title_bar = np.zeros((48, comparison.shape[1], 3), dtype=np.uint8)
        cv2.putText(title_bar, route_title, (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)
        comparison = np.vstack([title_bar, comparison])
        output_path = output_dir / f"{row['source_dataset']}__{row['selection_reason'].replace(' ', '_')}__{row['image_stem']}.jpg"
        cv2.imwrite(str(output_path), comparison, [cv2.IMWRITE_JPEG_QUALITY, 91])

        tile_width = 900
        tile_height = max(1, round(comparison.shape[0] * tile_width / comparison.shape[1]))
        overview_tiles.append(cv2.resize(comparison, (tile_width, tile_height), interpolation=cv2.INTER_AREA))
        review_rows.append(
            {
                "route": row["source_dataset"],
                "image_stem": row["image_stem"],
                "selection_reason": row["selection_reason"],
                "source_panorama": str(source_path),
                "comparison_image": str(output_path),
                "predicted_masks_at_conf050": prediction_count,
            }
        )

    target_height = max(tile.shape[0] for tile in overview_tiles)
    normalized_tiles: list[np.ndarray] = []
    for tile in overview_tiles:
        if tile.shape[0] < target_height:
            padding = np.zeros((target_height - tile.shape[0], tile.shape[1], 3), dtype=np.uint8)
            tile = np.vstack([tile, padding])
        normalized_tiles.append(tile)
    overview = np.vstack(
        [np.hstack(normalized_tiles[index:index + 2]) for index in range(0, len(normalized_tiles), 2)]
    )
    cv2.imwrite(str(output_dir / "six_frame_overview.jpg"), overview, [cv2.IMWRITE_JPEG_QUALITY, 90])

    with (output_dir / "review_samples.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(review_rows[0].keys()))
        writer.writeheader()
        writer.writerows(review_rows)
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
