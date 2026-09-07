#!/usr/bin/env python
"""Evaluate the best merged-route YOLO tree segmentation model on the held-out test split."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = PROJECT_ROOT / "derived" / "training_runs" / "combined_tree_segmentation"
ACTIVE_RUN_FILE = RUNS_ROOT / "active_run.json"


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def locate_run(explicit_run: Path | None) -> tuple[Path, Path, Path]:
    if explicit_run:
        run_dir = explicit_run.resolve()
        marker = {}
    else:
        marker = read_json(ACTIVE_RUN_FILE)
        run_dir = Path(marker.get("run_dir", ""))
        if not run_dir.is_dir():
            candidates = sorted(
                (path for path in RUNS_ROOT.glob("yolo11*_seg_*") if (path / "weights" / "best.pt").is_file()),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            if not candidates:
                raise FileNotFoundError("No completed segmentation run with best.pt was found")
            run_dir = candidates[0]

    best_weight = run_dir / "weights" / "best.pt"
    if not best_weight.is_file():
        raise FileNotFoundError(best_weight)

    data_value = marker.get("data") if marker else None
    data_yaml = Path(data_value) if data_value else Path()
    if not data_yaml.is_file():
        args_yaml = run_dir / "args.yaml"
        if args_yaml.is_file():
            import yaml

            args_data = yaml.safe_load(args_yaml.read_text(encoding="utf-8")) or {}
            data_yaml = Path(str(args_data.get("data", "")))
    if not data_yaml.is_file():
        raise FileNotFoundError("Unable to locate the data.yaml used by this run")
    return run_dir, best_weight, data_yaml


def load_manifest(dataset_dir: Path) -> dict[str, dict[str, str]]:
    manifest_path = dataset_dir / "manifest.csv"
    rows: dict[str, dict[str, str]] = {}
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row.get("split") != "test":
                continue
            rows[Path(row["image"]).stem] = row
    return rows


def load_yolo_polygons(label_path: Path) -> list[np.ndarray]:
    polygons: list[np.ndarray] = []
    if not label_path.is_file():
        return polygons
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        fields = raw_line.split()
        if len(fields) < 7:
            continue
        coords = np.asarray([float(value) for value in fields[1:]], dtype=np.float32)
        if coords.size % 2:
            continue
        polygon = coords.reshape(-1, 2)
        if len(polygon) >= 3:
            polygons.append(polygon)
    return polygons


def rasterize(polygons: list[np.ndarray], height: int, width: int) -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    scale = np.asarray([width - 1, height - 1], dtype=np.float32)
    for polygon in polygons:
        points = np.clip(np.rint(polygon * scale), [0, 0], [width - 1, height - 1]).astype(np.int32)
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [points], 1)
        masks.append(mask.astype(bool))
    return masks


def pairwise_iou(gt_masks: list[np.ndarray], pred_masks: list[np.ndarray]) -> np.ndarray:
    matrix = np.zeros((len(gt_masks), len(pred_masks)), dtype=np.float32)
    for gt_index, gt_mask in enumerate(gt_masks):
        for pred_index, pred_mask in enumerate(pred_masks):
            intersection = np.logical_and(gt_mask, pred_mask).sum()
            union = np.logical_or(gt_mask, pred_mask).sum()
            matrix[gt_index, pred_index] = float(intersection / union) if union else 0.0
    return matrix


def greedy_matches(iou_matrix: np.ndarray, threshold: float = 0.5) -> tuple[int, float]:
    candidates = [
        (float(iou_matrix[gt_index, pred_index]), gt_index, pred_index)
        for gt_index in range(iou_matrix.shape[0])
        for pred_index in range(iou_matrix.shape[1])
    ]
    candidates.sort(reverse=True)
    used_gt: set[int] = set()
    used_pred: set[int] = set()
    accepted: list[float] = []
    for iou, gt_index, pred_index in candidates:
        if iou < threshold:
            break
        if gt_index in used_gt or pred_index in used_pred:
            continue
        used_gt.add(gt_index)
        used_pred.add(pred_index)
        accepted.append(iou)
    return len(accepted), float(np.mean(accepted)) if accepted else 0.0


def union_scores(gt_masks: list[np.ndarray], pred_masks: list[np.ndarray], shape: tuple[int, int]) -> tuple[float, float, float]:
    gt_union = np.zeros(shape, dtype=bool)
    pred_union = np.zeros(shape, dtype=bool)
    for mask in gt_masks:
        gt_union |= mask
    for mask in pred_masks:
        pred_union |= mask
    true_positive = np.logical_and(gt_union, pred_union).sum()
    union = np.logical_or(gt_union, pred_union).sum()
    predicted = pred_union.sum()
    actual = gt_union.sum()
    iou = float(true_positive / union) if union else 1.0
    precision = float(true_positive / predicted) if predicted else 0.0
    recall = float(true_positive / actual) if actual else 0.0
    return iou, precision, recall


def draw_preview(result: Any, gt_polygons: list[np.ndarray], output_path: Path, metrics: dict[str, Any]) -> None:
    canvas = result.plot(labels=False, boxes=True, masks=True, conf=True)
    height, width = canvas.shape[:2]
    scale = np.asarray([width - 1, height - 1], dtype=np.float32)
    for polygon in gt_polygons:
        points = np.clip(np.rint(polygon * scale), [0, 0], [width - 1, height - 1]).astype(np.int32)
        cv2.polylines(canvas, [points], True, (0, 255, 0), max(3, width // 1000), cv2.LINE_AA)
    title = (
        f"GT green | pred mask | IoU={metrics['union_iou']:.3f} "
        f"GT={metrics['gt_instances']} pred={metrics['pred_instances']}"
    )
    cv2.rectangle(canvas, (0, 0), (min(width, 1120), 54), (0, 0, 0), -1)
    cv2.putText(canvas, title, (14, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
    max_width = 1600
    if width > max_width:
        target_height = max(1, round(height * max_width / width))
        canvas = cv2.resize(canvas, (max_width, target_height), interpolation=cv2.INTER_AREA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])


def make_contact_sheet(rows: list[dict[str, Any]], preview_dir: Path, output_path: Path) -> None:
    selected = sorted(rows, key=lambda row: row["union_iou"])[:12]
    tile_width, tile_height, header = 600, 300, 42
    tiles: list[np.ndarray] = []
    for row in selected:
        image = cv2.imread(str(preview_dir / f"{row['image_stem']}.jpg"))
        if image is None:
            continue
        image = cv2.resize(image, (tile_width, tile_height), interpolation=cv2.INTER_AREA)
        tile = np.zeros((tile_height + header, tile_width, 3), dtype=np.uint8)
        tile[header:] = image
        label = f"{row['source_dataset']} | {row['image_stem']} | IoU {row['union_iou']:.3f}"
        cv2.putText(tile, label, (9, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
    if not tiles:
        return
    blank = np.zeros_like(tiles[0])
    while len(tiles) < 12:
        tiles.append(blank.copy())
    sheet = np.vstack([np.hstack(tiles[index:index + 3]) for index in range(0, 12, 3)])
    cv2.imwrite(str(output_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=5)
    parser.add_argument("--conf", type=float, default=0.25, help="Threshold used for operational previews")
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    from ultralytics import YOLO

    run_dir, best_weight, data_yaml = locate_run(args.run)
    dataset_dir = data_yaml.parent
    test_images_dir = dataset_dir / "images" / "test"
    test_labels_dir = dataset_dir / "labels" / "test"
    manifest = load_manifest(dataset_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = f"test_acceptance_{timestamp}"
    output_dir = run_dir / output_name
    preview_dir = output_dir / "prediction_previews"
    output_dir.mkdir(parents=True, exist_ok=False)

    model = YOLO(str(best_weight))
    metrics = model.val(
        data=str(data_yaml),
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=8,
        plots=True,
        project=str(run_dir),
        name=output_name,
        exist_ok=True,
        verbose=True,
    )

    image_paths = sorted(test_images_dir.glob("*"))
    per_image: list[dict[str, Any]] = []
    predictions = model.predict(
        source=[str(path) for path in image_paths],
        imgsz=args.imgsz,
        conf=args.conf,
        iou=0.7,
        device=args.device,
        stream=True,
        verbose=False,
        retina_masks=False,
    )
    for result in predictions:
        image_path = Path(result.path)
        stem = image_path.stem
        meta = manifest.get(stem, {})
        label_path = test_labels_dir / f"{stem}.txt"
        gt_polygons = load_yolo_polygons(label_path)
        pred_polygons = [np.asarray(poly, dtype=np.float32) for poly in (result.masks.xyn if result.masks else [])]

        original_height, original_width = result.orig_shape
        raster_width = 1024
        raster_height = max(64, round(raster_width * original_height / original_width))
        gt_masks = rasterize(gt_polygons, raster_height, raster_width)
        pred_masks = rasterize(pred_polygons, raster_height, raster_width)
        iou_matrix = pairwise_iou(gt_masks, pred_masks)
        match_count, mean_match_iou = greedy_matches(iou_matrix)
        union_iou, union_precision, union_recall = union_scores(
            gt_masks, pred_masks, (raster_height, raster_width)
        )
        confidences = result.boxes.conf.detach().cpu().numpy().tolist() if result.boxes else []
        row: dict[str, Any] = {
            "image_stem": stem,
            "source_dataset": meta.get("source_dataset", "unknown"),
            "stream_id": meta.get("stream_id", ""),
            "frame_key": meta.get("frame_key", ""),
            "route_block_id": meta.get("route_block_id", ""),
            "gt_instances": len(gt_polygons),
            "pred_instances": len(pred_polygons),
            "matched_at_iou50": match_count,
            "instance_precision_at_iou50": match_count / len(pred_polygons) if pred_polygons else 0.0,
            "instance_recall_at_iou50": match_count / len(gt_polygons) if gt_polygons else 0.0,
            "mean_matched_iou": mean_match_iou,
            "union_iou": union_iou,
            "union_precision": union_precision,
            "union_recall": union_recall,
            "mean_confidence": float(np.mean(confidences)) if confidences else 0.0,
            "exposure_ev": safe_float(meta.get("exposure_ev", 0.0)),
            "contrast": safe_float(meta.get("contrast", 1.0)),
        }
        per_image.append(row)
        draw_preview(result, gt_polygons, preview_dir / f"{stem}.jpg", row)

    per_image_path = output_dir / "per_image_metrics.csv"
    with per_image_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_image[0].keys()))
        writer.writeheader()
        writer.writerows(per_image)

    route_summary: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in per_image:
        grouped[row["source_dataset"]].append(row)
    for route, rows in sorted(grouped.items()):
        gt_total = sum(row["gt_instances"] for row in rows)
        pred_total = sum(row["pred_instances"] for row in rows)
        match_total = sum(row["matched_at_iou50"] for row in rows)
        route_summary[route] = {
            "frames": len(rows),
            "gt_instances": gt_total,
            "pred_instances_at_conf": pred_total,
            "matched_at_iou50": match_total,
            "instance_precision_at_iou50": match_total / pred_total if pred_total else 0.0,
            "instance_recall_at_iou50": match_total / gt_total if gt_total else 0.0,
            "mean_union_iou": float(np.mean([row["union_iou"] for row in rows])),
            "mean_union_precision": float(np.mean([row["union_precision"] for row in rows])),
            "mean_union_recall": float(np.mean([row["union_recall"] for row in rows])),
        }

    gt_total = sum(row["gt_instances"] for row in per_image)
    pred_total = sum(row["pred_instances"] for row in per_image)
    match_total = sum(row["matched_at_iou50"] for row in per_image)
    f1_curve = np.asarray(metrics.seg.f1_curve, dtype=np.float64)
    confidence_axis = np.asarray(metrics.seg.px, dtype=np.float64)
    mean_f1_curve = np.nanmean(f1_curve, axis=0) if f1_curve.ndim > 1 else f1_curve
    best_f1_index = int(np.nanargmax(mean_f1_curve))
    best_f1 = float(mean_f1_curve[best_f1_index])
    best_f1_confidence = float(confidence_axis[best_f1_index])
    summary = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "best_weight": str(best_weight),
        "data_yaml": str(data_yaml),
        "split": "test",
        "acceptance_decision": "research_baseline_passed_application_not_yet_passed",
        "annotation_scope_warning": "target-oriented frame labels are not yet proven exhaustive for all visible trees",
        "test_frames": len(per_image),
        "test_instances": gt_total,
        "evaluation_imgsz": args.imgsz,
        "operational_preview_confidence": args.conf,
        "ultralytics": {
            "box_precision": float(metrics.box.mp),
            "box_recall": float(metrics.box.mr),
            "box_map50": float(metrics.box.map50),
            "box_map50_95": float(metrics.box.map),
            "mask_precision": float(metrics.seg.mp),
            "mask_recall": float(metrics.seg.mr),
            "mask_map50": float(metrics.seg.map50),
            "mask_map50_95": float(metrics.seg.map),
            "mask_best_f1": best_f1,
            "mask_best_f1_confidence": best_f1_confidence,
            "speed_ms_per_image": {key: float(value) for key, value in metrics.speed.items()},
        },
        "operational_at_confidence": {
            "pred_instances": pred_total,
            "matched_at_iou50": match_total,
            "instance_precision_at_iou50": match_total / pred_total if pred_total else 0.0,
            "instance_recall_at_iou50": match_total / gt_total if gt_total else 0.0,
            "mean_union_iou": float(np.mean([row["union_iou"] for row in per_image])),
            "mean_union_precision": float(np.mean([row["union_precision"] for row in per_image])),
            "mean_union_recall": float(np.mean([row["union_recall"] for row in per_image])),
        },
        "by_route": route_summary,
    }
    (output_dir / "acceptance_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    make_contact_sheet(per_image, preview_dir, output_dir / "lowest_iou_contact_sheet.jpg")

    report_lines = [
        "# Three-route tree segmentation test acceptance",
        "",
        "**Research baseline: passed. Application deployment: not yet passed.**",
        "",
        "The test is route-block independent, but the current target-oriented labels are not yet proven exhaustive for every visible tree. Legitimate unlabelled trees may therefore be counted as false positives; application precision requires an exhaustively reviewed subset or ignore regions.",
        "",
        f"- Model: `{best_weight}`",
        f"- Held-out test frames/instances: {len(per_image)} / {gt_total}",
        f"- Mask precision / recall: {metrics.seg.mp:.4f} / {metrics.seg.mr:.4f}",
        f"- Mask mAP50 / mAP50-95: {metrics.seg.map50:.4f} / {metrics.seg.map:.4f}",
        f"- Box mAP50 / mAP50-95: {metrics.box.map50:.4f} / {metrics.box.map:.4f}",
        f"- Best mask F1 / confidence: {best_f1:.4f} / {best_f1_confidence:.3f}",
        f"- Operational threshold: confidence >= {args.conf:.2f}",
        f"- Instance precision / recall at IoU 0.50: {summary['operational_at_confidence']['instance_precision_at_iou50']:.4f} / {summary['operational_at_confidence']['instance_recall_at_iou50']:.4f}",
        f"- Mean foreground union IoU: {summary['operational_at_confidence']['mean_union_iou']:.4f}",
        "",
        "## Route breakdown",
        "",
        "| Route | Frames | GT trees | Pred trees | P@0.5 | R@0.5 | Mean union IoU |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for route, values in route_summary.items():
        report_lines.append(
            f"| {route} | {values['frames']} | {values['gt_instances']} | "
            f"{values['pred_instances_at_conf']} | {values['instance_precision_at_iou50']:.4f} | "
            f"{values['instance_recall_at_iou50']:.4f} | {values['mean_union_iou']:.4f} |"
        )
    report_lines.extend(
        [
            "",
            "Green outlines in prediction previews are human ground truth; colored masks are model predictions.",
            "The Ultralytics mAP metrics are the formal threshold-swept test metrics. Per-image operational metrics use the fixed preview confidence threshold and are intended for deployment inspection, not as a replacement for mAP.",
        ]
    )
    (output_dir / "ACCEPTANCE_REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Acceptance artifacts: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
