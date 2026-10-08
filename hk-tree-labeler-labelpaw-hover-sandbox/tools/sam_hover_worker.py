#!/usr/bin/env python3
"""Long-lived SAM point predictor for LabelPaw-style hover previews."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
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


def emit(payload: dict) -> None:
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def polygon_from_mask(mask, width: int, height: int) -> tuple[list[list[float]], dict | None]:
    mask_np = mask.cpu().numpy() if torch.is_tensor(mask) else mask
    mask_np = np.squeeze(mask_np)
    mask_uint8 = (mask_np > 0.5).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return [], None
    largest = max(contours, key=cv2.contourArea)
    approx = cv2.approxPolyDP(largest, 0.002 * cv2.arcLength(largest, True), True).reshape(-1, 2)
    if len(approx) < 3:
        return [], None
    x, y, w, h = cv2.boundingRect(largest)
    return (
        [[clamp(float(px) / width), clamp(float(py) / height)] for px, py in approx],
        {
            "class_id": 0,
            "x_center": clamp((x + w / 2) / width),
            "y_center": clamp((y + h / 2) / height),
            "width": clamp(w / width),
            "height": clamp(h / height),
        },
    )


class SamHoverPredictor:
    def __init__(self, args) -> None:
        self.args = args
        self.device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
        if self.device == "auto":
            self.device = "cpu"
        self.image_path = ""
        self.width = 0
        self.height = 0
        self.predictor = None
        self.processor = None
        self.inference_state = None
        self._load_model()

    def _load_model(self) -> None:
        if self.args.model_type == "sam2":
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            if not self.args.config:
                raise ValueError("SAM2 hover worker requires --config")
            model = build_sam2(self.args.config, ckpt_path=str(self.args.model.resolve()), device=self.device)
            self.predictor = SAM2ImagePredictor(model)
            return

        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        model = build_sam3_image_model(
            checkpoint_path=str(self.args.model.resolve()),
            enable_inst_interactivity=True,
        )
        model.to(self.device)
        self.processor = Sam3Processor(model)

    def set_image_if_needed(self, image_path: str) -> None:
        if image_path == self.image_path:
            return
        image = Image.open(image_path).convert("RGB")
        self.width, self.height = image.size
        with torch.inference_mode(), inference_context(self.device):
            if self.args.model_type == "sam2":
                self.predictor.set_image(np.array(image))
            else:
                self.inference_state = self.processor.set_image(image)
        self.image_path = image_path

    def predict(self, image_path: str, x: float, y: float) -> list[dict]:
        self.set_image_if_needed(image_path)
        point_xy = (clamp(x) * self.width, clamp(y) * self.height)
        with torch.inference_mode(), inference_context(self.device):
            if self.args.model_type == "sam2":
                masks, scores, _ = self.predictor.predict(
                    point_coords=np.array([point_xy], dtype=np.float32),
                    point_labels=np.array([1]),
                    multimask_output=True,
                )
            else:
                masks, scores, _ = self.processor.model.predict_inst(
                    inference_state=self.inference_state,
                    point_coords=np.array([point_xy], dtype=np.float32),
                    point_labels=np.array([1]),
                    multimask_output=True,
                )
        if len(scores) == 0:
            return []
        score_values = scores.cpu().numpy() if torch.is_tensor(scores) else np.asarray(scores)
        best_index = int(np.argmax(score_values))
        polygon, box = polygon_from_mask(masks[best_index], self.width, self.height)
        if not polygon or not box:
            return []
        confidence = float(score_values[best_index])
        return [
            {
                "box": {**box, "confidence": confidence},
                "polygon": polygon,
                "confidence": confidence,
                "source": f"{self.args.model_type}_hover",
                "model_key": self.args.model_key,
            }
        ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a long-lived SAM hover predictor.")
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--model-type", choices=["sam2", "sam3"], required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    predictor = SamHoverPredictor(args)
    emit({"status": "ready", "model_key": args.model_key, "model_type": args.model_type})
    for line in sys.stdin:
        request = None
        try:
            request = json.loads(line)
            image_path = request["image"]
            candidates = predictor.predict(image_path, float(request["x"]), float(request["y"]))
            emit(
                {
                    "id": request.get("id"),
                    "candidates": candidates,
                    "cached_image": predictor.image_path == image_path,
                }
            )
        except Exception as exc:
            emit({"id": request.get("id") if request else None, "candidates": [], "error": str(exc)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
