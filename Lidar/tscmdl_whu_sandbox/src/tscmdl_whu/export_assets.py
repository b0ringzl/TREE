"""Generic atomic asset-export helpers shared by full-dataset scripts."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .panorama import project_equirectangular


def export_timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{path}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{path}.tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
    temporary.replace(path)


def atomic_jpeg(path: Path, image: Image.Image, *, quality: int = 92) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{path}.tmp")
    image.save(temporary, format="JPEG", quality=quality)
    temporary.replace(path)


def directory_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def wrapped_crop(
    image: Image.Image, left: float, top: float, width: float, height: float
) -> Image.Image:
    image_width, image_height = image.size
    crop_width = max(1, min(int(round(width)), image_width))
    crop_height = max(1, min(int(round(height)), image_height))
    top_int = max(0, min(int(round(top)), image_height - crop_height))
    left_mod = int(round(left)) % image_width
    if left_mod + crop_width <= image_width:
        return image.crop(
            (left_mod, top_int, left_mod + crop_width, top_int + crop_height)
        )
    first_width = image_width - left_mod
    output = Image.new(image.mode, (crop_width, crop_height))
    output.paste(
        image.crop((left_mod, top_int, image_width, top_int + crop_height)),
        (0, 0),
    )
    output.paste(
        image.crop((0, top_int, crop_width - first_width, top_int + crop_height)),
        (first_width, 0),
    )
    return output


def crop_visible_fraction(
    u: np.ndarray, v: np.ndarray, crop: object, panorama_width: int
) -> float:
    x = np.mod(u - crop.left_unwrapped_px, panorama_width)
    y = v - crop.top_px
    visible = (
        (x >= 0)
        & (x < crop.width_px)
        & (y >= 0)
        & (y < crop.height_px)
    )
    return float(np.mean(visible))


def overlay_points(
    crop_image: Image.Image,
    sampled_points: np.ndarray,
    pose: object,
    crop: object,
    panorama_width: int,
    panorama_height: int,
) -> Image.Image:
    output = crop_image.convert("RGBA")
    layer = Image.new("RGBA", output.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    u, v, _ = project_equirectangular(
        sampled_points,
        pose,
        panorama_width,
        panorama_height,
        apply_tilt=True,
    )
    x = np.mod(u - crop.left_unwrapped_px, panorama_width)
    y = v - crop.top_px
    visible = (
        (x >= 0)
        & (x < crop.width_px)
        & (y >= 0)
        & (y < crop.height_px)
    )
    x = x[visible] * output.width / crop.width_px
    y = y[visible] * output.height / crop.height_px
    for px, py in zip(x[::2], y[::2]):
        draw.ellipse(
            (px - 1.5, py - 1.5, px + 1.5, py + 1.5),
            fill=(255, 35, 20, 175),
        )
    return Image.alpha_composite(output, layer).convert("RGB")
