from __future__ import annotations

import os
from pathlib import Path


ROOT_DIR = Path(os.getenv("TREE_ROOT", r"D:\TREE")).resolve()
SPECIES_DIR = (ROOT_DIR / "选取树种").resolve()
TEMP_DIR = (ROOT_DIR / "temp").resolve()
DATASET_DIR = (ROOT_DIR / "dataset").resolve()

GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")
STATIC_IMAGE_SIZE = os.getenv("STREETVIEW_IMAGE_SIZE", "640x640")
MAX_CONCURRENT_DOWNLOADS = int(os.getenv("MAX_CONCURRENT_DOWNLOADS", "8"))


def ensure_runtime_dirs() -> None:
    for path in (ROOT_DIR, SPECIES_DIR, TEMP_DIR, DATASET_DIR):
        path.mkdir(parents=True, exist_ok=True)


def assert_inside_root(path: Path) -> Path:
    resolved = path.resolve()
    if resolved != ROOT_DIR and ROOT_DIR not in resolved.parents:
        raise ValueError(f"Path escapes TREE root: {resolved}")
    return resolved


def safe_name(value: str) -> str:
    allowed = []
    for ch in value.strip():
        if ch.isalnum() or ch in (" ", "_", "-", ".", "'", "(", ")"):
            allowed.append(ch)
        else:
            allowed.append("_")
    return "".join(allowed).strip(" .") or "unnamed"
