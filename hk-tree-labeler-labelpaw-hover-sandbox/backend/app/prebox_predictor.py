from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from .config import (
    ROOT_DIR,
    TEMP_DIR,
    YOLO_PREBOX_CONF,
    YOLO_PREBOX_IMAGE_SIZE,
    YOLO_PREBOX_MODEL,
    YOLO_PYTHON,
    assert_inside_root,
)


def _resolve_temp_image(image: str) -> Path:
    if not image.startswith("/temp/"):
        raise ValueError("Only /temp images can be predicted")
    relative = image.removeprefix("/temp/").replace("/", "\\")
    path = assert_inside_root(TEMP_DIR / relative)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image}")
    return path


def predict_preboxes(image: str) -> dict:
    model_path = Path(YOLO_PREBOX_MODEL)
    if not model_path.exists():
        return {"image": image, "boxes": [], "model_ready": False}

    image_path = _resolve_temp_image(image)
    script = ROOT_DIR / "tools" / "predict_yolo_boxes.py"
    command = [
        YOLO_PYTHON,
        str(script),
        "--model",
        str(model_path),
        "--image",
        str(image_path),
        "--image-size",
        str(YOLO_PREBOX_IMAGE_SIZE),
        "--conf",
        str(YOLO_PREBOX_CONF),
        "--max-boxes",
        "3",
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
        timeout=60,
        check=False,
    )
    if completed.returncode != 0:
        return {
            "image": image,
            "boxes": [],
            "model_ready": True,
            "error": completed.stderr[-1000:] or completed.stdout[-1000:],
        }
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    return {
        "image": image,
        "boxes": payload.get("boxes", []),
        "model_ready": True,
    }
