"""Optional, inference-only SAM proposal for a user-clicked tree.

The returned contour is a draft. The browser requires an explicit preview and
confirmation before using the native-resolution polygon crop for recognition.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile
import threading
from pathlib import Path

from PIL import Image

from . import tree_detection


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "predict_sam_polygon.py"
WEIGHT = Path(os.getenv("SPACESCAN_SAM_WEIGHT", str(ROOT / "LabelPaw-web-images" / "weights" / "sam_weights" / "sam2.1_hiera_tiny.pt")))
PYTHON = Path(os.getenv("SAM_PYTHON", str(Path(os.getenv("LOCALAPPDATA", "")) / "miniconda3" / "envs" / "yolo" / "python.exe")))
CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"
LOCK = threading.Lock()
MAX_EDGE = 1600


def status() -> dict:
    return {
        "ready": SCRIPT.is_file() and WEIGHT.is_file() and PYTHON.is_file(),
        "model": "SAM 2.1 Tiny",
        "manual_fallback": True,
    }


def _valid_polygon(candidate: dict) -> list[list[float]]:
    polygon = candidate.get("polygon")
    if not isinstance(polygon, list) or not 3 <= len(polygon) <= 2000:
        raise RuntimeError("SAM did not return a usable tree outline.")
    points = []
    for pair in polygon:
        if not isinstance(pair, list) or len(pair) != 2:
            raise RuntimeError("SAM returned invalid coordinates.")
        x, y = float(pair[0]), float(pair[1])
        if not (math.isfinite(x) and math.isfinite(y) and 0 <= x <= 1 and 0 <= y <= 1):
            raise RuntimeError("SAM returned invalid coordinates.")
        points.append([x, y])
    return points


def segment(data: bytes, x: float, y: float, projection: str = "auto") -> dict:
    if not (math.isfinite(x) and math.isfinite(y) and 0 <= x <= 1 and 0 <= y <= 1):
        raise ValueError("Click a point inside the image. / 請在影像內點選目標。")
    if not status()["ready"]:
        raise RuntimeError("SAM is not installed in this Demo. Use manual polygon crop. / 此 Demo 未安裝 SAM，請使用手動多邊形裁剪。")
    if not LOCK.acquire(blocking=False):
        raise BlockingIOError("SAM is busy; retry shortly. / SAM 正在處理另一張影像，請稍後重試。")
    try:
        image = tree_detection.decode(data)
        if tree_detection.panorama.resolve_projection(image.size, projection)["kind"] == "equirectangular":
            raise ValueError("For a 2:1 panorama, use perspective-corrected tree selection first. / 2:1 全景請先使用透視校正選樹。")
        if image.width * image.height < 16 * 16:
            raise ValueError("Image is too small for tree selection. / 影像過小，無法選樹。")
        image.thumbnail((MAX_EDGE, MAX_EDGE), Image.Resampling.LANCZOS)
        with tempfile.TemporaryDirectory(prefix="spacescan-sam-") as temporary:
            path = Path(temporary) / "input.png"
            image.save(path, format="PNG")
            env = os.environ.copy()
            env["PYTHONPATH"] = str(ROOT / "LabelPaw-web-images") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
            command = [str(PYTHON), str(SCRIPT), "--model-key", WEIGHT.stem,
                       "--model-type", "sam2", "--model", str(WEIGHT),
                       "--config", CONFIG, "--image", str(path),
                       "--x", str(x), "--y", str(y), "--polygon-stride", "2"]
            try:
                run = subprocess.run(command, cwd=str(ROOT), env=env, capture_output=True,
                                     text=True, timeout=180, check=False)
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError("SAM timed out; try a smaller photo or draw manually. / SAM 逾時，請縮小影像或手動描繪。") from exc
        if run.returncode:
            raise RuntimeError("SAM could not segment this image; try another point or draw manually. / SAM 無法分割此影像，請重新點選或手動描繪。")
        try:
            payload = json.loads(run.stdout.strip().splitlines()[-1])
        except (IndexError, ValueError) as exc:
            raise RuntimeError("SAM returned no outline; try another point or draw manually. / SAM 未回傳輪廓，請重新點選或手動描繪。") from exc
        candidate = (payload.get("candidates") or [None])[0]
        if not candidate:
            raise RuntimeError("No usable outline at this point. Try the tree crown or draw manually. / 此處未取得可用輪廓，請點選樹冠或手動描繪。")
        return {"polygon": _valid_polygon(candidate), "model": "SAM 2.1 Tiny",
                "proposal_only": True, "user_confirmation_required": True}
    finally:
        LOCK.release()
