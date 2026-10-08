#!/usr/bin/env python3
"""Predict tree instance masks for one image and print normalized polygons as JSON."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_CACHE = PROJECT_ROOT / "models" / "_cache"
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT))
os.environ.setdefault("TORCH_HOME", str(PROJECT_CACHE / "torch"))
os.environ.setdefault("HF_HOME", str(PROJECT_CACHE / "huggingface"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_CACHE / "xdg"))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_CACHE / "matplotlib"))
os.environ.setdefault("TMP", str(PROJECT_CACHE / "tmp"))
os.environ.setdefault("TEMP", str(PROJECT_CACHE / "tmp"))
os.environ.setdefault("WINDIR", os.environ.get("SystemRoot", r"C:\Windows"))
os.environ.setdefault("SystemRoot", os.environ.get("WINDIR", r"C:\Windows"))
for cache_dir in {
    Path(os.environ["TORCH_HOME"]),
    Path(os.environ["HF_HOME"]),
    Path(os.environ["XDG_CACHE_HOME"]),
    Path(os.environ["MPLCONFIGDIR"]),
    Path(os.environ["TMP"]),
}:
    cache_dir.mkdir(parents=True, exist_ok=True)

from ultralytics import YOLO


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def box_payload(box, width: int, height: int) -> dict:
    x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
    x1 = max(0.0, min(width, x1))
    x2 = max(0.0, min(width, x2))
    y1 = max(0.0, min(height, y1))
    y2 = max(0.0, min(height, y2))
    cls = int(box.cls[0].item()) if box.cls is not None else 0
    conf = float(box.conf[0].item()) if box.conf is not None else 0.0
    return {
        "class_id": cls,
        "x_center": ((x1 + x2) / 2) / width,
        "y_center": ((y1 + y2) / 2) / height,
        "width": max(0.0, (x2 - x1) / width),
        "height": max(0.0, (y2 - y1) / height),
        "confidence": conf,
    }


def normalized_polygon(points, width: int, height: int, stride: int) -> list[list[float]]:
    if points is None:
        return []
    coords = points.tolist() if hasattr(points, "tolist") else points
    if len(coords) < 3:
        return []
    sampled = coords[::stride] if stride > 1 else coords
    if len(sampled) < 3:
        sampled = coords
    return [[clamp(float(x) / width), clamp(float(y) / height)] for x, y in sampled]


def main() -> int:
    parser = argparse.ArgumentParser(description="Predict normalized YOLO segmentation polygons.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.12)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--max-masks", type=int, default=8)
    parser.add_argument("--polygon-stride", type=int, default=2)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    model = YOLO(str(args.model.resolve()))
    results = model.predict(
        source=str(args.image.resolve()),
        imgsz=args.image_size,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_masks,
        device=args.device,
        retina_masks=True,
        verbose=False,
    )
    candidates = []
    for result in results:
        height, width = result.orig_shape[:2]
        boxes = list(result.boxes) if result.boxes is not None else []
        polygons = result.masks.xy if result.masks is not None else []
        for index, box in enumerate(boxes):
            payload = box_payload(box, width, height)
            if payload["width"] <= 0 or payload["height"] <= 0:
                continue
            polygon = normalized_polygon(
                polygons[index] if index < len(polygons) else None,
                width,
                height,
                args.polygon_stride,
            )
            candidates.append({
                "box": payload,
                "polygon": polygon,
                "confidence": payload["confidence"],
                "source": "yolo_seg",
            })
    candidates.sort(key=lambda item: item["confidence"], reverse=True)
    print(json.dumps({"image": str(args.image.resolve()), "candidates": candidates[: args.max_masks]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
