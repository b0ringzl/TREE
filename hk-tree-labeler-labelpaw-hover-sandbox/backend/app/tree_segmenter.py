from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from .config import (
    ROOT_DIR,
    TEMP_DIR,
    YOLO_PYTHON,
    YOLO_TREE_SEG_CONF,
    YOLO_TREE_SEG_IMAGE_SIZE,
    YOLO_TREE_SEG_MAX_MASKS,
    YOLO_TREE_SEG_MODEL,
    assert_inside_root,
)
from .prebox_predictor import predict_preboxes


def _resolve_temp_image(image: str) -> Path:
    if not image.startswith("/temp/"):
        raise ValueError("Only /temp images can be segmented")
    relative = image.removeprefix("/temp/").replace("/", "\\")
    path = assert_inside_root(TEMP_DIR / relative)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image}")
    return path


def predict_tree_segments(image: str) -> dict:
    model_path = Path(YOLO_TREE_SEG_MODEL)
    if not model_path.exists():
        fallback = predict_preboxes(image)
        return {
            "image": image,
            "candidates": [
                {
                    "box": box,
                    "polygon": [],
                    "confidence": box.get("confidence", 0.0),
                    "source": "box_fallback",
                }
                for box in fallback.get("boxes", [])
            ],
            "model_ready": False,
            "fallback_model_ready": fallback.get("model_ready", False),
        }

    image_path = _resolve_temp_image(image)
    script = ROOT_DIR / "tools" / "predict_yolo_segments.py"
    command = [
        YOLO_PYTHON,
        str(script),
        "--model",
        str(model_path),
        "--image",
        str(image_path),
        "--image-size",
        str(YOLO_TREE_SEG_IMAGE_SIZE),
        "--conf",
        str(YOLO_TREE_SEG_CONF),
        "--max-masks",
        str(YOLO_TREE_SEG_MAX_MASKS),
    ]
    env = os.environ.copy()
    env.update({
        "YOLO_CONFIG_DIR": str(ROOT_DIR),
        "TORCH_HOME": str(ROOT_DIR / "models" / "_cache" / "torch"),
        "MPLCONFIGDIR": str(ROOT_DIR / "models" / "_cache" / "matplotlib"),
        "TMP": str(ROOT_DIR / "models" / "_cache" / "tmp"),
        "TEMP": str(ROOT_DIR / "models" / "_cache" / "tmp"),
    })
    completed = subprocess.run(
        command,
        cwd=str(ROOT_DIR),
        env=env,
        text=True,
        capture_output=True,
        timeout=90,
        check=False,
    )
    if completed.returncode != 0:
        return {
            "image": image,
            "candidates": [],
            "model_ready": True,
            "error": completed.stderr[-1000:] or completed.stdout[-1000:],
        }
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    return {
        "image": image,
        "candidates": payload.get("candidates", []),
        "model_ready": True,
    }
