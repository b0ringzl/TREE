#!/usr/bin/env python3
"""Predict one SAM-assisted polygon from a normalized positive point."""

from __future__ import annotations

import argparse
import contextlib
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

import cv2
import numpy as np
import torch
from PIL import Image


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def inference_context(device: str):
    if device.startswith("cuda"):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def polygon_from_mask(mask, width: int, height: int, stride: int) -> tuple[list[list[float]], dict | None]:
    mask_np = mask.cpu().numpy() if torch.is_tensor(mask) else mask
    mask_np = np.squeeze(mask_np)
    mask_uint8 = (mask_np > 0.5).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return [], None

    largest = max(contours, key=cv2.contourArea)
    epsilon = 0.002 * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, epsilon, True).reshape(-1, 2)
    if len(approx) < 3:
        return [], None
    sampled = approx[::stride] if stride > 1 else approx
    if len(sampled) < 3:
        sampled = approx

    x, y, w, h = cv2.boundingRect(largest)
    polygon = [[clamp(float(px) / width), clamp(float(py) / height)] for px, py in sampled]
    box = {
        "class_id": 0,
        "x_center": clamp((x + w / 2) / width),
        "y_center": clamp((y + h / 2) / height),
        "width": clamp(w / width),
        "height": clamp(h / height),
    }
    return polygon, box


def predict_sam2(args, image_np, point_xy, device: str):
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    model = build_sam2(args.config, ckpt_path=str(args.model.resolve()), device=device)
    predictor = SAM2ImagePredictor(model)
    with torch.inference_mode(), inference_context(device):
        predictor.set_image(image_np)
        return predictor.predict(
            point_coords=np.array([point_xy], dtype=np.float32),
            point_labels=np.array([1]),
            multimask_output=True,
        )


def predict_sam3(args, image, point_xy, device: str):
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    model = build_sam3_image_model(checkpoint_path=str(args.model.resolve()), enable_inst_interactivity=True)
    model.to(device)
    processor = Sam3Processor(model)
    with torch.inference_mode(), inference_context(device):
        state = processor.set_image(image)
        return model.predict_inst(
            inference_state=state,
            point_coords=np.array([point_xy], dtype=np.float32),
            point_labels=np.array([1]),
            multimask_output=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Predict a normalized SAM polygon from one click point.")
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--model-type", choices=["sam2", "sam3"], required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--x", type=float, required=True, help="Normalized x coordinate")
    parser.add_argument("--y", type=float, required=True, help="Normalized y coordinate")
    parser.add_argument("--polygon-stride", type=int, default=1)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    image = Image.open(args.image).convert("RGB")
    width, height = image.size
    image_np = np.array(image)
    point_xy = (clamp(args.x) * width, clamp(args.y) * height)

    if args.model_type == "sam2":
        if not args.config:
            raise ValueError("SAM2 requires --config")
        masks, scores, _ = predict_sam2(args, image_np, point_xy, device)
    else:
        masks, scores, _ = predict_sam3(args, image, point_xy, device)

    candidates = []
    if len(scores) > 0:
        score_values = scores.cpu().numpy() if torch.is_tensor(scores) else np.asarray(scores)
        best_index = int(np.argmax(score_values))
        polygon, box = polygon_from_mask(masks[best_index], width, height, max(1, args.polygon_stride))
        if polygon and box:
            confidence = float(score_values[best_index])
            candidates.append(
                {
                    "box": {**box, "confidence": confidence},
                    "polygon": polygon,
                    "confidence": confidence,
                    "source": args.model_type,
                    "model_key": args.model_key,
                }
            )

    print(json.dumps({"image": str(args.image.resolve()), "candidates": candidates}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
