from __future__ import annotations

import os
from pathlib import Path


ROOT_DIR = Path(os.getenv("TREE_ROOT", r"D:\TREE")).resolve()
SPECIES_DIR = (ROOT_DIR / "选取树种").resolve()
TEMP_DIR = (ROOT_DIR / "temp").resolve()
DATASET_DIR = (ROOT_DIR / "dataset").resolve()
PROJECT_DIR = ROOT_DIR / "hk-tree-labeler"
RUNTIME_DIR = PROJECT_DIR / ".runtime"
GOOGLE_MAPS_API_KEY_FILE = Path(
    os.getenv("GOOGLE_MAPS_API_KEY_FILE", str(RUNTIME_DIR / "google_maps_api_key.txt"))
).resolve()


def load_google_maps_api_key() -> str:
    env_value = os.getenv("GOOGLE_MAPS_API_KEY", "").strip()
    if env_value:
        return env_value
    try:
        return GOOGLE_MAPS_API_KEY_FILE.read_text(encoding="utf-8-sig").strip()
    except FileNotFoundError:
        return ""


def google_maps_api_key_source() -> str:
    if os.getenv("GOOGLE_MAPS_API_KEY", "").strip():
        return "environment"
    return "file" if load_google_maps_api_key() else "missing"


GOOGLE_MAPS_API_KEY = load_google_maps_api_key()
GOOGLE_MAPS_API_KEY_SOURCE = google_maps_api_key_source()
STATIC_IMAGE_SIZE = os.getenv("STREETVIEW_IMAGE_SIZE", "640x640")
MAX_CONCURRENT_DOWNLOADS = int(os.getenv("MAX_CONCURRENT_DOWNLOADS", "8"))
NEAR_REJECT_RADIUS_M = float(os.getenv("NEAR_REJECT_RADIUS_M", "35"))
YOLO_PYTHON = os.getenv("YOLO_PYTHON", r"C:\Users\57680\.conda\envs\yolo\python.exe")
YOLO_PREBOX_MODEL = os.getenv(
    "YOLO_PREBOX_MODEL",
    str(ROOT_DIR / "models" / "yolo_runs" / "inat_prebox_full_v1" / "weights" / "best.pt"),
)
YOLO_PREBOX_CONF = float(os.getenv("YOLO_PREBOX_CONF", "0.01"))
YOLO_PREBOX_IMAGE_SIZE = int(os.getenv("YOLO_PREBOX_IMAGE_SIZE", "384"))
YOLO_TREE_SEG_MODEL = os.getenv(
    "YOLO_TREE_SEG_MODEL",
    str(ROOT_DIR / "models" / "yolo_runs" / "urban_tree_seg_v1" / "weights" / "best.pt"),
)
YOLO_TREE_SEG_CONF = float(os.getenv("YOLO_TREE_SEG_CONF", "0.12"))
YOLO_TREE_SEG_IMAGE_SIZE = int(os.getenv("YOLO_TREE_SEG_IMAGE_SIZE", "640"))
YOLO_TREE_SEG_MAX_MASKS = int(os.getenv("YOLO_TREE_SEG_MAX_MASKS", "8"))


def ensure_runtime_dirs() -> None:
    for path in (ROOT_DIR, SPECIES_DIR, TEMP_DIR, DATASET_DIR, RUNTIME_DIR):
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
