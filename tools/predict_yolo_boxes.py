#!/usr/bin/env python3
"""Predict YOLO boxes for one image and print normalized boxes as JSON."""

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


def main() -> int:
    parser = argparse.ArgumentParser(description="Predict normalized YOLO boxes.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--max-boxes", type=int, default=3)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    model = YOLO(str(args.model.resolve()))
    results = model.predict(
        source=str(args.image.resolve()),
        imgsz=args.image_size,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_boxes,
        device=args.device,
        verbose=False,
    )
    boxes = []
    for result in results:
        if result.boxes is None:
            continue
        height, width = result.orig_shape[:2]
        for box in result.boxes:
            x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
            x1 = max(0.0, min(width, x1))
            x2 = max(0.0, min(width, x2))
            y1 = max(0.0, min(height, y1))
            y2 = max(0.0, min(height, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            cls = int(box.cls[0].item()) if box.cls is not None else 0
            conf = float(box.conf[0].item()) if box.conf is not None else 0.0
            boxes.append({
                "class_id": cls,
                "x_center": ((x1 + x2) / 2) / width,
                "y_center": ((y1 + y2) / 2) / height,
                "width": (x2 - x1) / width,
                "height": (y2 - y1) / height,
                "confidence": conf,
            })
    boxes.sort(key=lambda item: item["confidence"], reverse=True)
    print(json.dumps({"image": str(args.image.resolve()), "boxes": boxes[: args.max_boxes]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
