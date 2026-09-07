from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TREE_ROOT = PROJECT_ROOT.parent
RECEIVE_ROOT = TREE_ROOT.parent

VMMS_ROOT = Path(os.getenv("VMMS_ROOT", RECEIVE_ROOT / "vmms")).resolve()
COORDINATE_DIR = Path(
    os.getenv("VMMS_COORDINATE_DIR", PROJECT_ROOT / "outputs" / "coordinates")
).resolve()
APP_ROOT = PROJECT_ROOT / "annotation_app"
DERIVED_ROOT = Path(
    os.getenv("VMMS_LABELER_DERIVED_ROOT", PROJECT_ROOT / "derived" / "hewentian_labeler")
).resolve()
VIEW_CACHE_DIR = DERIVED_ROOT / "view_cache"
ANNOTATION_DIR = DERIVED_ROOT / "annotations"
RUNTIME_DIR = DERIVED_ROOT / "runtime"
FRAME_LABELER_ROOT = Path(
    os.getenv("VMMS_FRAME_LABELER_ROOT", PROJECT_ROOT / "derived" / "hewentian_frame_labeler")
).resolve()
FRAME_PREVIEW_DIR = FRAME_LABELER_ROOT / "preview_cache"
FRAME_ANNOTATION_DIR = FRAME_LABELER_ROOT / "annotations"
FRAME_RUNTIME_DIR = FRAME_LABELER_ROOT / "runtime"

FRAME_COORDINATES = COORDINATE_DIR / "frame_coordinates.csv"
NEARBY_TREES = COORDINATE_DIR / "nearby_trees.csv"
HEWENTIAN_STREAM_ID = os.getenv("VMMS_TREE_STREAM_ID", "hewentian_pano").strip()
FRAME_STREAM_IDS = tuple(
    value.strip()
    for value in os.getenv("VMMS_FRAME_STREAM_IDS", HEWENTIAN_STREAM_ID).split(",")
    if value.strip()
)
FRAME_LABELER_DATASET = os.getenv(
    "VMMS_FRAME_LABELER_DATASET", "hewentian"
).strip()
FRAME_LABELER_TITLE = os.getenv(
    "VMMS_FRAME_LABELER_TITLE",
    {
        "hewentian": "何文田全景影像逐帧标注",
        "jianshazui": "尖沙咀全景影像逐帧标注",
        "stubbs_road": "司徒拔道全景影像逐帧标注",
    }.get(FRAME_LABELER_DATASET, "香港全景影像逐帧标注"),
).strip()
FRAME_MIN_ROUTE_DISTANCE_M_BY_STREAM = {}
for _item in os.getenv("VMMS_FRAME_MIN_ROUTE_DISTANCE_M_BY_STREAM", "").split(","):
    if not _item.strip():
        continue
    _stream_id, _distance = _item.split(":", 1)
    FRAME_MIN_ROUTE_DISTANCE_M_BY_STREAM[_stream_id.strip()] = float(_distance)
FRAME_SEED_CLASSES = Path(
    os.getenv(
        "VMMS_FRAME_SEED_CLASSES",
        PROJECT_ROOT
        / "derived"
        / "hewentian_frame_labeler"
        / "annotations"
        / "classes.json",
    )
).resolve()
TREE_DISTANCE_LIMIT_M = float(os.getenv("VMMS_TREE_DISTANCE_LIMIT_M", "50"))
VIEW_MAX_RANGE_M = float(os.getenv("VMMS_VIEW_MAX_RANGE_M", "45"))
VIEW_MIN_SEPARATION_M = float(os.getenv("VMMS_VIEW_MIN_SEPARATION_M", "5"))
CROP_FOV_DEG = float(os.getenv("VMMS_CROP_FOV_DEG", "112.5"))
PANORAMA_X_SIGN = float(os.getenv("VMMS_PANORAMA_X_SIGN", "1"))
PANORAMA_YAW_OFFSET_DEG = float(os.getenv("VMMS_PANORAMA_YAW_OFFSET_DEG", "0"))
CROP_OUTPUT_SIZE = int(os.getenv("VMMS_CROP_OUTPUT_SIZE", "1024"))


def ensure_dirs() -> None:
    for path in (
        DERIVED_ROOT,
        VIEW_CACHE_DIR,
        ANNOTATION_DIR,
        RUNTIME_DIR,
        FRAME_LABELER_ROOT,
        FRAME_PREVIEW_DIR,
        FRAME_ANNOTATION_DIR,
        FRAME_RUNTIME_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def assert_inside(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    root = root.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"Path escapes allowed root {root}: {resolved}")
    return resolved


def safe_name(value: str) -> str:
    allowed = []
    for character in value.strip():
        if character.isalnum() or character in (" ", "_", "-", ".", "'", "(", ")"):
            allowed.append(character)
        else:
            allowed.append("_")
    return "".join(allowed).strip(" .") or "unnamed"
