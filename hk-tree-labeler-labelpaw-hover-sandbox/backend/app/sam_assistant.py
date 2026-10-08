from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from .config import ROOT_DIR, SAM_PYTHON, SAM_SEGMENT_TIMEOUT_SEC, SAM_WEIGHTS_DIR, TEMP_DIR, assert_inside_root


def _size_label(path: Path) -> str:
    size_mb = path.stat().st_size // (1024 * 1024)
    if size_mb < 1024:
        return f"{size_mb} MB"
    return f"{size_mb / 1024:.1f} GB"


def _sam2_config(filename: str) -> str | None:
    lower = filename.lower()
    if "tiny" in lower or "_t.pt" in lower:
        return "configs/sam2.1/sam2.1_hiera_t.yaml"
    if "small" in lower or "_s.pt" in lower:
        return "configs/sam2.1/sam2.1_hiera_s.yaml"
    if "base_plus" in lower or "_b+.pt" in lower or "base" in lower:
        return "configs/sam2.1/sam2.1_hiera_b+.yaml"
    if "large" in lower or "_l.pt" in lower:
        return "configs/sam2.1/sam2.1_hiera_l.yaml"
    return None


def discover_sam_models(weights_dir: Path = SAM_WEIGHTS_DIR) -> list[dict]:
    if not weights_dir.exists():
        return []

    models: list[dict] = []
    for path in sorted(weights_dir.glob("*.pt"), key=lambda item: item.name.lower()):
        key = path.stem
        lower = path.name.lower()
        if "sam3" in lower:
            models.append(
                {
                    "key": key,
                    "display_name": "SAM 3" if lower == "sam3.pt" else f"SAM 3 ({key})",
                    "type": "sam3",
                    "weight": str(path),
                    "config": None,
                    "supports_text": True,
                    "size_label": _size_label(path),
                }
            )
            continue

        models.append(
            {
                "key": key,
                "display_name": key.replace("_", " ").title(),
                "type": "sam2",
                "weight": str(path),
                "config": _sam2_config(path.name),
                "supports_text": False,
                "size_label": _size_label(path),
            }
        )
    return models


def _resolve_temp_image(image: str) -> Path:
    if not image.startswith("/temp/"):
        raise ValueError("Only /temp images can be segmented")
    relative = image.removeprefix("/temp/").replace("/", "\\")
    path = assert_inside_root(TEMP_DIR / relative)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image}")
    return path


def _find_model(model_key: str, models: list[dict]) -> dict | None:
    for model in models:
        if model.get("key") == model_key:
            return model
    return None


def predict_sam_polygon(
    image_url: str,
    model_key: str,
    point: tuple[float, float],
    models: list[dict] | None = None,
    image_path: Path | None = None,
) -> dict:
    available_models = discover_sam_models() if models is None else models
    model = _find_model(model_key, available_models)
    if not model:
        return {
            "image": image_url,
            "candidates": [],
            "model_ready": False,
            "error": f"SAM model is not available: {model_key}",
        }

    resolved_image = image_path or _resolve_temp_image(image_url)
    script = ROOT_DIR / "tools" / "predict_sam_polygon.py"
    command = [
        SAM_PYTHON,
        str(script),
        "--model-key",
        str(model["key"]),
        "--model-type",
        str(model["type"]),
        "--model",
        str(model["weight"]),
        "--image",
        str(resolved_image),
        "--x",
        str(point[0]),
        "--y",
        str(point[1]),
    ]
    if model.get("config"):
        command.extend(["--config", str(model["config"])])

    env = os.environ.copy()
    cache_dir = ROOT_DIR / "models" / "_cache"
    env.update(
        {
            "YOLO_CONFIG_DIR": str(ROOT_DIR),
            "TORCH_HOME": str(cache_dir / "torch"),
            "HF_HOME": str(cache_dir / "huggingface"),
            "XDG_CACHE_HOME": str(cache_dir / "xdg"),
            "MPLCONFIGDIR": str(cache_dir / "matplotlib"),
            "TMP": str(cache_dir / "tmp"),
            "TEMP": str(cache_dir / "tmp"),
        }
    )
    labelpaw_path = str(ROOT_DIR / "LabelPaw-web-images")
    env["PYTHONPATH"] = labelpaw_path + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    completed = subprocess.run(
        command,
        cwd=str(ROOT_DIR),
        env=env,
        text=True,
        capture_output=True,
        timeout=SAM_SEGMENT_TIMEOUT_SEC,
        check=False,
    )
    if completed.returncode != 0:
        return {
            "image": image_url,
            "candidates": [],
            "model_ready": True,
            "error": completed.stderr[-1200:] or completed.stdout[-1200:],
        }

    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    return {
        "image": image_url,
        "candidates": payload.get("candidates", []),
        "model_ready": True,
        "model": model,
    }
