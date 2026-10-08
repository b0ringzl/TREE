"""Inference-only VMMS corridor pilot for geo-indexed equirectangular panoramas.

The input panoramas remain in their source folders. Jobs store only compact
previews, selected tree crops and JSON results under the demo runtime folder.
Camera coordinates are never presented as measured tree coordinates.
"""
from __future__ import annotations

import bisect
import csv
import hashlib
import io
import json
import math
import os
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from scipy.optimize import linear_sum_assignment

from . import corridor_identity, panorama, presentation_groups, tree_detection
from .species_names import bilingual_names


PROJECT = Path(__file__).resolve().parents[1]
VMMS_ROOT = PROJECT.parent / "vmms"
COORDINATES = PROJECT / "hk-tree-vmms/outputs/coordinates/frame_coordinates.csv"
SURVEY_ID = "2025-02-10_jianshazui"
ALLOWED_STREAMS = {"jianshazui_pano_1", "jianshazui_pano_2"}
MAX_LENGTH_M = 200.0
MAX_TRACKS = 24
MAX_VIEWS_PER_TRACK = 3
MAX_FOLDER_IMAGES = 10000
MAX_FOLDER_SAMPLES = 300
ONLINE_TRACK_MAX_GAP = 2
FRAGMENT_RECONNECT_MAX_GAP = 8
FRAGMENT_RECONNECT_MAX_SPAN = 45
FRAGMENT_RECONNECT_MAX_PARTS = 3
FRAGMENT_RECONNECT_SIMILARITY = .90
TRACKING_REVISION = "corridor-geometry-gallery-v4-20261003"
PANORAMA_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
LOW_CONFIDENCE_BOX_THRESHOLD = .30
LOW_CONFIDENCE_BOX = {"outline": (250, 204, 21), "label": (250, 204, 21), "text": (18, 37, 29)}
STANDARD_CONFIDENCE_BOX = {"outline": (187, 235, 132), "label": (18, 63, 49), "text": "white"}
_PREVIEW_FONT = None


@dataclass(frozen=True)
class RouteFrame:
    stream_id: str
    frame_id: int
    source: Path
    hk80_easting: float | None
    hk80_northing: float | None
    latitude: float | None
    longitude: float | None
    heading_deg: float


@dataclass(frozen=True)
class RouteSegment:
    segment_id: str
    stream_id: str
    frames: tuple[RouteFrame, ...]
    distances: tuple[float, ...]
    source_label: str = "Coordinate-indexed panorama stream"
    repaired_coordinate_frames: tuple[int, ...] = ()

    @property
    def length_m(self) -> float:
        return self.distances[-1] if self.distances else 0.0


_CATALOG_LOCK = threading.Lock()
_SEGMENTS: dict[str, RouteSegment] | None = None


def _step(left: RouteFrame, right: RouteFrame) -> float:
    return math.hypot(right.hk80_easting - left.hk80_easting,
                      right.hk80_northing - left.hk80_northing)


def _cumulative(frames: list[RouteFrame]) -> list[float]:
    distances = [0.0]
    for left, right in zip(frames, frames[1:]):
        distances.append(distances[-1] + _step(left, right))
    return distances


def _repair_isolated_spikes(frames: list[RouteFrame]) -> tuple[list[RouteFrame], list[int]]:
    """Interpolate a single bad row only when its neighbours remain locally continuous."""
    repaired = list(frames)
    changed: list[int] = []
    for index in range(1, len(frames) - 1):
        left, current, right = repaired[index - 1], repaired[index], repaired[index + 1]
        if _step(left, current) <= 3 or _step(current, right) <= 3 or _step(left, right) > 2:
            continue
        # Circular mean avoids a false 180-degree heading at the 0/360 seam.
        headings = np.radians([left.heading_deg, right.heading_deg])
        heading = math.degrees(math.atan2(float(np.sin(headings).mean()),
                                          float(np.cos(headings).mean()))) % 360
        repaired[index] = replace(
            current,
            hk80_easting=(left.hk80_easting + right.hk80_easting) / 2,
            hk80_northing=(left.hk80_northing + right.hk80_northing) / 2,
            latitude=(left.latitude + right.latitude) / 2,
            longitude=(left.longitude + right.longitude) / 2,
            heading_deg=heading,
        )
        changed.append(current.frame_id)
    return repaired, changed


def _finite(value: str, default: float = 0.0) -> float:
    number = float(value or default)
    if not math.isfinite(number):
        raise ValueError("Non-finite coordinate")
    return number


def _load_segments() -> dict[str, RouteSegment]:
    global _SEGMENTS
    with _CATALOG_LOCK:
        if _SEGMENTS is not None:
            return _SEGMENTS
        if not COORDINATES.is_file() or not VMMS_ROOT.is_dir():
            raise RuntimeError("VMMS route index or image root is unavailable")
        grouped: dict[str, list[RouteFrame]] = {key: [] for key in ALLOWED_STREAMS}
        with COORDINATES.open("r", encoding="utf-8-sig", newline="") as source:
            for row in csv.DictReader(source):
                if row.get("survey_id") != SURVEY_ID or row.get("stream_id") not in grouped:
                    continue
                try:
                    image = (VMMS_ROOT / row["source_image_relpath"]).resolve()
                    if not image.is_relative_to(VMMS_ROOT.resolve()) or not image.is_file():
                        continue
                    grouped[row["stream_id"]].append(RouteFrame(
                        stream_id=row["stream_id"], frame_id=int(row["frame_id"]), source=image,
                        hk80_easting=_finite(row["hk80_easting"]),
                        hk80_northing=_finite(row["hk80_northing"]),
                        latitude=_finite(row["wgs84_latitude"]),
                        longitude=_finite(row["wgs84_longitude"]),
                        heading_deg=_finite(row.get("heading_deg", "0")),
                    ))
                except (KeyError, ValueError, OSError):
                    continue
        segments: dict[str, RouteSegment] = {}
        for stream_id, frames in grouped.items():
            frames.sort(key=lambda row: row.frame_id)
            runs: list[list[RouteFrame]] = []
            current: list[RouteFrame] = []
            for frame in frames:
                if current:
                    previous = current[-1]
                    step = math.hypot(frame.hk80_easting - previous.hk80_easting,
                                      frame.hk80_northing - previous.hk80_northing)
                    # Split positioning jumps; zero-distance captures remain valid.
                    if step > 3.0:
                        if len(current) >= 10:
                            runs.append(current)
                        current = []
                current.append(frame)
            if len(current) >= 10:
                runs.append(current)
            for run in runs:
                cumulative = _cumulative(run)
                if cumulative[-1] < 50:
                    continue
                key = f"{stream_id}:{run[0].frame_id}-{run[-1].frame_id}"
                segments[key] = RouteSegment(key, stream_id, tuple(run), tuple(cumulative))
        # User-selected folder: a contiguous copy of Pano-1 frames 1514–1951.
        # Three rows in the coordinate export have a truncated .100 second value;
        # repair only those isolated spikes whose surrounding frames are continuous.
        selected_folder = VMMS_ROOT / "20250210-jianshazui/ladybug/1"
        if selected_folder.is_dir():
            files: dict[int, Path] = {}
            for path in selected_folder.glob("*.jpg"):
                match = re.search(r"_Panoramic_(\d{6})_", path.name)
                if match:
                    files[int(match.group(1))] = path.resolve()
            coordinates_by_id = {frame.frame_id: frame for frame in grouped["jianshazui_pano_1"]}
            folder_frames = [replace(coordinates_by_id[frame_id],
                                     stream_id="jianshazui_folder_1", source=files[frame_id])
                             for frame_id in sorted(files) if frame_id in coordinates_by_id]
            if len(folder_frames) == len(files) and len(folder_frames) >= 10:
                folder_frames, repaired_ids = _repair_isolated_spikes(folder_frames)
                distances = _cumulative(folder_frames)
                if max((_step(left, right) for left, right in zip(folder_frames, folder_frames[1:])),
                       default=0) <= 3:
                    key = f"jianshazui_folder_1:{folder_frames[0].frame_id}-{folder_frames[-1].frame_id}"
                    segments[key] = RouteSegment(
                        key, "jianshazui_folder_1", tuple(folder_frames), tuple(distances),
                        source_label=str(selected_folder),
                        repaired_coordinate_frames=tuple(repaired_ids),
                    )
        if not segments:
            raise RuntimeError("No continuous VMMS route segment is available")
        _SEGMENTS = segments
        return segments


def catalog() -> dict:
    segments = sorted(_load_segments().values(), key=lambda item: item.length_m, reverse=True)
    eligible = [item for item in segments if item.length_m >= MAX_LENGTH_M]
    return {
        "survey_id": SURVEY_ID,
        "title": "Tsim Sha Tsui VMMS / 尖沙咀 VMMS",
        "coordinate_system": "HK80 + WGS84 camera trajectory",
        "default_segment_id": eligible[0].segment_id if eligible else segments[0].segment_id,
        "segments": [{
            "id": item.segment_id, "stream_id": item.stream_id,
            "start_frame": item.frames[0].frame_id, "end_frame": item.frames[-1].frame_id,
            "source_frames": len(item.frames), "continuous_length_m": round(item.length_m, 1),
            "max_test_length_m": round(min(MAX_LENGTH_M, item.length_m), 1),
            "source_label": item.source_label,
            "repaired_coordinate_frames": list(item.repaired_coordinate_frames),
        } for item in segments],
        "sampling_note": "Distance is measured along consecutive HK80 camera positions after splitting jumps above 3 m.",
    }


def sample_route(segment_id: str, offset_m: float, length_m: float,
                 spacing_m: float) -> list[tuple[float, RouteFrame]]:
    segment = _load_segments().get(segment_id)
    if segment is None:
        raise ValueError("Unknown route segment / 未知軌跡分段")
    if not 1 <= spacing_m <= 20:
        raise ValueError("Spacing must be 1–20 m / 採樣間距須為 1–20 米")
    if not 10 <= length_m <= MAX_LENGTH_M:
        raise ValueError("Test length must be 10–200 m / 測試長度須為 10–200 米")
    if offset_m < 0 or offset_m + length_m > segment.length_m + .25:
        raise ValueError("Selected test range exceeds the continuous route / 測試範圍超出連續軌跡")
    targets = [offset_m + index * spacing_m
               for index in range(math.floor(length_m / spacing_m) + 1)]
    if targets[-1] < offset_m + length_m - .25:
        targets.append(offset_m + length_m)
    samples: list[tuple[float, RouteFrame]] = []
    used: set[int] = set()
    for target in targets:
        right = bisect.bisect_left(segment.distances, target)
        choices = [index for index in (right - 1, right) if 0 <= index < len(segment.frames)]
        index = min(choices, key=lambda value: abs(segment.distances[value] - target))
        if index in used:
            continue
        used.add(index)
        samples.append((segment.distances[index] - offset_m, segment.frames[index]))
    return samples


def _natural_key(path: Path) -> tuple:
    """Sort VMMS captures by their numeric filename components."""
    return tuple(int(part) if part.isdigit() else part.casefold()
                 for part in re.split(r"(\d+)", path.name))


def _frame_id(path: Path, fallback: int) -> int:
    match = re.search(r"_Panoramic_(\d{6})_", path.name, re.IGNORECASE)
    if not match:
        numbers = re.findall(r"\d+", path.stem)
        match = re.match(r"(\d+)", numbers[-1]) if numbers else None
    return int(match.group(1)) if match else fallback


def inspect_panorama_folder(folder_path: str | Path) -> dict:
    """Inspect one flat VMMS image folder without copying its source files."""
    if not str(folder_path).strip():
        raise ValueError("Choose a VMMS panorama folder / 請選擇 VMMS 全景影像文件夾")
    folder = Path(folder_path).expanduser().resolve()
    if not folder.is_dir():
        raise ValueError("Folder does not exist / 文件夾不存在")
    candidates = sorted((path for path in folder.iterdir()
                         if path.is_file() and path.suffix.casefold() in PANORAMA_EXTENSIONS),
                        key=_natural_key)
    if len(candidates) > MAX_FOLDER_IMAGES:
        raise ValueError(f"Folder contains more than {MAX_FOLDER_IMAGES} supported images / 文件夾影像過多")
    valid: list[dict] = []
    rejected: list[dict] = []
    total_bytes = 0
    for index, path in enumerate(candidates, 1):
        try:
            with Image.open(path) as image:
                width, height = image.size
                image.verify()
            ratio = width / max(1, height)
            if width < 512 or height < 256 or not 1.96 <= ratio <= 2.04:
                rejected.append({"name": path.name, "reason": "not_2_to_1", "width": width,
                                 "height": height})
                continue
            size = path.stat().st_size
            total_bytes += size
            valid.append({"path": path, "name": path.name, "width": width, "height": height,
                          "bytes": size, "frame_id": _frame_id(path, index)})
        except (OSError, ValueError, Image.UnidentifiedImageError):
            rejected.append({"name": path.name, "reason": "unreadable"})
    if len(valid) < 2:
        raise ValueError("At least two readable 2:1 VMMS panoramas are required / 至少需要兩張可讀取的 2:1 VMMS 全景")
    dimensions: dict[str, int] = {}
    for row in valid:
        key = f'{row["width"]}×{row["height"]}'
        dimensions[key] = dimensions.get(key, 0) + 1
    return {
        "folder": str(folder), "name": folder.name,
        "valid_image_count": len(valid), "rejected_image_count": len(rejected),
        "non_image_count": max(0, sum(1 for path in folder.iterdir() if path.is_file()) - len(candidates)),
        "total_bytes": total_bytes, "first_file": valid[0]["name"],
        "last_file": valid[-1]["name"], "dimensions": dimensions,
        "rejected_examples": rejected[:8], "standard": "VMMS 2:1 equirectangular panorama",
        "_images": valid,
    }


def public_folder_inspection(folder_path: str | Path) -> dict:
    inspection = inspect_panorama_folder(folder_path)
    return {key: value for key, value in inspection.items() if not key.startswith("_")} | {
        "image_names": [row["name"] for row in inspection["_images"]],
    }


def choose_panorama_folder(initial_folder: str | None = None) -> str:
    """Open the native local-folder picker used by this localhost demo."""
    if os.name != "nt":
        raise RuntimeError("Native folder selection is only available on Windows")
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        initial = str(Path(initial_folder).resolve()) if initial_folder and Path(initial_folder).is_dir() else None
        return filedialog.askdirectory(
            parent=root, title="Select VMMS panorama folder / 選擇 VMMS 全景文件夾",
            initialdir=initial, mustexist=True,
        ) or ""
    finally:
        root.destroy()


def sample_panorama_folder(folder_path: str | Path, start_index: int = 0,
                           frame_step: int = 1, max_frames: int = 300
                           ) -> tuple[list[tuple[float, RouteFrame]], dict]:
    inspection = inspect_panorama_folder(folder_path)
    images = inspection["_images"]
    if not 0 <= start_index < len(images):
        raise ValueError("Start position is outside the folder / 起始位置超出文件夾範圍")
    if not 1 <= frame_step <= 50:
        raise ValueError("Frame step must be 1–50 / 幀間隔須為 1–50")
    if not 2 <= max_frames <= MAX_FOLDER_SAMPLES:
        raise ValueError(f"Sample count must be 2–{MAX_FOLDER_SAMPLES} / 採樣數須為 2–{MAX_FOLDER_SAMPLES}")
    selected = images[start_index::frame_step][:max_frames]
    if len(selected) < 2:
        raise ValueError("Selected range contains fewer than two panoramas / 所選範圍少於兩張全景")
    samples: list[tuple[float, RouteFrame]] = []
    for offset, row in enumerate(selected):
        source_index = start_index + offset * frame_step
        frame = RouteFrame(
            stream_id=f"folder:{inspection['name']}", frame_id=int(row["frame_id"]),
            source=row["path"], hk80_easting=None, hk80_northing=None,
            latitude=None, longitude=None, heading_deg=0.0,
        )
        samples.append((float(source_index), frame))
    return samples, {key: value for key, value in inspection.items() if not key.startswith("_")} | {
        "image_names": [row["name"] for row in images],
    }


def _jpeg(image: Image.Image, path: Path, edge: int | None = None, quality: int = 88) -> None:
    output = image.copy()
    if edge:
        output.thumbnail((edge, edge), Image.Resampling.LANCZOS)
    output.save(path, "JPEG", quality=quality, optimize=True)


def _normalise_angle(value: float) -> float:
    return (value + 180) % 360 - 180


def _angle_distance(left: float, right: float) -> float:
    return abs(_normalise_angle(left - right))


def _embedding(engine: tree_detection.Engine, crops: list[Image.Image]) -> np.ndarray:
    if not crops:
        return np.empty((0, 512), dtype=np.float32)
    rows = []
    for start in range(0, len(crops), 24):
        tensors = [engine.preprocess(ImageOps.pad(crop, (224, 224), color=(124, 116, 104)))
                   for crop in crops[start:start + 24]]
        with engine.torch.inference_mode():
            features = engine.gate.encode_image(engine.torch.stack(tensors).to(engine.device).float()).float()
            features = engine.normalized_features(features).cpu().numpy().astype(np.float32)
        rows.append(features)
    return np.concatenate(rows, axis=0)


def _detect_frame(image: Image.Image, engine: tree_detection.Engine) -> list[dict]:
    specs = [panorama.view_spec(image.size, yaw, 15, hfov=90, vfov=105, max_edge=1024)
             for yaw in range(-180, 180, 45)]
    views = [panorama.render(image, spec, max_edge=1024) for spec in specs]
    predictions = engine.yolo.predict(
        views, imgsz=1024, conf=tree_detection.THRESHOLD, iou=.5,
        device=engine.device, classes=engine.tree_classes, agnostic_nms=True,
        max_det=30, verbose=False,
    )
    rows: list[dict] = []
    for view, spec, result in zip(views, specs, predictions):
        found = [{
            "box": [value / divisor for value, divisor in zip(box, [view.width, view.height] * 2)],
            "score": float(score), "detector": "YOLO11m-seg",
        } for box, score in zip(result.boxes.xyxy.cpu().tolist(),
                                result.boxes.conf.cpu().tolist())]
        found = tree_detection.clean_boxes(sorted(found, key=lambda row: -row["score"]),
                                             view.width, view.height)[:10]
        crops = [view.crop(tree_detection.crop_rect(row["box"], view.size)) for row in found]
        checks: list[dict] = []
        for start in range(0, len(crops), 12):
            checks.extend(engine.background_checks(crops[start:start + 12]))
        for row, crop, check in zip(found, crops, checks):
            visibility = tree_detection.visibility_p90(crop)
            evidence = corridor_identity.crop_evidence(crop)
            if not _usable_tree_crop(check) or visibility < 24 or not evidence["usable"]:
                continue
            spherical = panorama.sphere_box(row["box"], spec)
            box_area = tree_detection.area(row["box"])
            rows.append({
                **row, "box": spherical, "view_box": row["box"], "view_spec": spec,
                "crop": crop, "visibility_p90": visibility,
                "background_minus_tree": check["background_minus_tree"],
                "crop_evidence": evidence,
                "relative_yaw": _normalise_angle((((spherical[0] + spherical[2]) / 2) % 1 - .5) * 360),
                "quality": float(row["score"] * (.7 + min(.3, box_area * 2))),
            })
    ordered = sorted(rows, key=lambda row: -row["quality"])
    unique: list[dict] = []
    for row in ordered:
        if not any(panorama.circular_iou(row["box"], accepted["box"]) > .45 for accepted in unique):
            unique.append(row)
    unique = unique[:8]
    vectors = _embedding(engine, [row["crop"] for row in unique])
    for row, vector in zip(unique, vectors):
        row["embedding"] = vector
    return unique


def _usable_tree_crop(check: dict) -> bool:
    # The full padded crop must favour a tree. A strongly non-tree centre
    # (e.g. vehicle hardware with a few leaves around it) must not be rescued
    # by background greenery at the crop's edge.
    margins = check["background_minus_tree"]
    return bool(check["keep"] and margins[0] < 0 and margins[1] < .015)


def _draw_panorama_preview(image: Image.Image, rows: list[dict], path: Path) -> None:
    preview = image.copy()
    preview.thumbnail((1000, 500), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(preview)
    for index, row in enumerate(rows, 1):
        x0, y0, x1, y1 = row["box"]
        y0, y1 = max(0, y0) * preview.height, min(1, y1) * preview.height
        intervals = []
        for shift in (-1, 0, 1):
            left, right = (x0 + shift) * preview.width, (x1 + shift) * preview.width
            if right >= 0 and left <= preview.width:
                intervals.append((max(0, left), min(preview.width, right)))
        for left, right in intervals:
            draw.rectangle((left, y0, right, y1), outline=(187, 235, 132), width=3)
            draw.rectangle((left, max(0, y0 - 20), min(preview.width, left + 54), y0), fill=(18, 63, 49))
            draw.text((left + 5, max(1, y0 - 18)), f"T{index}", fill="white")
    _jpeg(preview, path, quality=84)


def _preview_style(score: float) -> dict:
    return LOW_CONFIDENCE_BOX if score < LOW_CONFIDENCE_BOX_THRESHOLD else STANDARD_CONFIDENCE_BOX


def _preview_font():
    global _PREVIEW_FONT
    if _PREVIEW_FONT is not None:
        return _PREVIEW_FONT
    for candidate in (Path("C:/Windows/Fonts/msjh.ttc"), Path("C:/Windows/Fonts/msyh.ttc")):
        if candidate.is_file():
            _PREVIEW_FONT = ImageFont.truetype(str(candidate), 14)
            return _PREVIEW_FONT
    _PREVIEW_FONT = ImageFont.load_default()
    return _PREVIEW_FONT


def _draw_labelled_panorama_preview(image: Image.Image, rows: list[dict], path: Path) -> None:
    """Render stable tree IDs with species and scores after multi-view classification."""
    preview = image.copy()
    preview.thumbnail((1000, 500), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(preview)
    font = _preview_font()
    for row in rows:
        x0, y0, x1, y1 = row["box"]
        y0, y1 = max(0, y0) * preview.height, min(1, y1) * preview.height
        top = row["top"]
        score = float(top["score"])
        style = _preview_style(score)
        species = top.get("chinese_name") or top.get("scientific_name") or top.get("species")
        label = f'T{row["track_id"]} · {species} · {score * 100:.1f}%'
        try:
            bounds = draw.textbbox((0, 0), label, font=font)
        except UnicodeEncodeError:
            species = top.get("scientific_name") or top.get("species")
            label = f'T{row["track_id"]} · {species} · {score * 100:.1f}%'
            bounds = draw.textbbox((0, 0), label, font=font)
        label_width = min(360, bounds[2] - bounds[0] + 12)
        label_height = 24
        for shift in (-1, 0, 1):
            left, right = (x0 + shift) * preview.width, (x1 + shift) * preview.width
            if right < 0 or left > preview.width:
                continue
            left, right = max(0, left), min(preview.width, right)
            draw.rectangle((left, y0, right, y1), outline=style["outline"], width=4)
            # Keep the complete label inside the preview when a wrapped
            # panorama box touches either horizontal edge.
            label_left = min(max(0, left), max(0, preview.width - label_width))
            label_right = min(preview.width, label_left + label_width)
            label_top = max(0, y0 - label_height)
            draw.rectangle((label_left, label_top, label_right, y0), fill=style["label"])
            draw.text((label_left + 6, label_top + 3), label, fill=style["text"], font=font)
    _jpeg(preview, path, quality=86)


def _predicted_bearing(track: dict, sample_index: int) -> float:
    observations = track["observations"]
    if len(observations) < 2:
        return float(track["bearing"])
    previous, latest = observations[-2], observations[-1]
    elapsed = latest["sample_index"] - previous["sample_index"]
    if elapsed <= 0:
        return float(track["bearing"])
    velocity = _normalise_angle(latest["bearing"] - previous["bearing"]) / elapsed
    velocity = max(-12.0, min(12.0, velocity))
    return _normalise_angle(latest["bearing"] + velocity * (sample_index - latest["sample_index"]))


def _append_observation(track: dict, observation: dict) -> None:
    count = len(track["observations"])
    track["observations"].append(observation)
    vector = track["embedding"] * count + observation["embedding"]
    track["embedding"] = vector / max(1e-9, float(np.linalg.norm(vector)))
    track["bearing"] = observation["bearing"]
    track["last_sample"] = observation["sample_index"]


def _spatially_same_target(left: dict, right: dict, angle_limit: float = 10) -> bool:
    if _angle_distance(left["bearing"], right["bearing"]) > angle_limit:
        return False
    left_height = max(1e-6, left["box"][3] - left["box"][1])
    right_height = max(1e-6, right["box"][3] - right["box"][1])
    vertical = max(0.0, min(left["box"][3], right["box"][3]) -
                   max(left["box"][1], right["box"][1])) / min(left_height, right_height)
    # Vertical overlap alone also describes two neighbouring trees; require
    # actual overlap of their spherical boxes before collapsing their IDs.
    return panorama.circular_iou(left["box"], right["box"]) >= .05 and vertical >= .35


def _overlapping_fragments_compatible(left: dict, right: dict) -> bool:
    left_by_sample = {row["sample_index"]: row for row in left["observations"]}
    right_by_sample = {row["sample_index"]: row for row in right["observations"]}
    shared = sorted(set(left_by_sample) & set(right_by_sample))
    if shared:
        matches = sum(_spatially_same_target(left_by_sample[sample], right_by_sample[sample]) and
                      panorama.circular_iou(left_by_sample[sample]["box"],
                                            right_by_sample[sample]["box"]) >= .55 and
                      corridor_identity.views_compatible(left_by_sample[sample], right_by_sample[sample])
                      for sample in shared)
        return matches >= 2 and matches / len(shared) >= .75
    adjacent_matches = 0
    for left_row in left["observations"]:
        for right_row in right["observations"]:
            if abs(left_row["sample_index"] - right_row["sample_index"]) <= 1 and \
                    _spatially_same_target(left_row, right_row, angle_limit=5) and \
                    panorama.circular_iou(left_row["box"], right_row["box"]) >= .55 and \
                    corridor_identity.views_compatible(left_row, right_row):
                adjacent_matches += 1
                break
    return adjacent_matches >= 2


def _merge_track_pair(left: dict, right: dict) -> dict:
    by_sample: dict[int, dict] = {}
    for observation in left["observations"] + right["observations"]:
        sample = observation["sample_index"]
        if sample not in by_sample or observation["quality"] > by_sample[sample]["quality"]:
            by_sample[sample] = observation
    observations = sorted(by_sample.values(), key=lambda row: row["sample_index"])
    vectors = np.stack([row["embedding"] for row in observations])
    vector = vectors.mean(axis=0)
    vector /= max(1e-9, float(np.linalg.norm(vector)))
    return {
        "id": min(left["id"], right["id"]), "observations": observations,
        "last_sample": observations[-1]["sample_index"],
        "bearing": observations[-1]["bearing"], "embedding": vector,
        "merged_fragment_count": (left.get("merged_fragment_count", 0) +
                                  right.get("merged_fragment_count", 0) + 1),
        "source_ids": sorted(set(left.get("source_ids", [left["id"]]) +
                                 right.get("source_ids", [right["id"]]))),
    }


def _reconnect_track_fragments(tracks: list[dict]) -> list[dict]:
    working = list(tracks)
    while True:
        best: tuple[float, int, int] | None = None
        for left_index, left in enumerate(working):
            for right_index in range(left_index + 1, len(working)):
                right = working[right_index]
                # Complete-link appearance compatibility, not two contaminated
                # running averages which happen to describe generic greenery.
                if any(not corridor_identity.views_compatible(a, b)
                       for a in left["observations"] for b in right["observations"]):
                    continue
                similarity = float(left["embedding"] @ right["embedding"])
                if similarity < FRAGMENT_RECONNECT_SIMILARITY:
                    continue
                left_first = left["observations"][0]["sample_index"]
                left_last = left["observations"][-1]["sample_index"]
                right_first = right["observations"][0]["sample_index"]
                right_last = right["observations"][-1]["sample_index"]
                overlaps = left_first <= right_last and right_first <= left_last
                if overlaps:
                    if similarity < .92 or not _overlapping_fragments_compatible(left, right):
                        continue
                    combined_first = min(left_first, right_first)
                    combined_last = max(left_last, right_last)
                    parts = (left.get("merged_fragment_count", 0) + 1 +
                             right.get("merged_fragment_count", 0) + 1)
                    if combined_last - combined_first > FRAGMENT_RECONNECT_MAX_SPAN or \
                            parts > FRAGMENT_RECONNECT_MAX_PARTS:
                        continue
                    angle = min(_angle_distance(a["bearing"], b["bearing"])
                                for a in left["observations"] for b in right["observations"]
                                if abs(a["sample_index"] - b["sample_index"]) <= 1)
                    score = similarity + .15 - angle / 300
                    if best is None or score > best[0]:
                        best = (score, left_index, right_index)
                    continue
                earlier, later = (left, right) if left_last < right_first else (right, left)
                earlier_first = earlier["observations"][0]["sample_index"]
                earlier_last = earlier["observations"][-1]["sample_index"]
                later_first = later["observations"][0]["sample_index"]
                later_last = later["observations"][-1]["sample_index"]
                gap = later_first - earlier_last
                if gap > FRAGMENT_RECONNECT_MAX_GAP:
                    continue
                if later_last - earlier_first > FRAGMENT_RECONNECT_MAX_SPAN:
                    continue
                parts = (earlier.get("merged_fragment_count", 0) + 1 +
                         later.get("merged_fragment_count", 0) + 1)
                if parts > FRAGMENT_RECONNECT_MAX_PARTS:
                    continue
                endpoint_similarity = float(
                    np.mean([row["embedding"] for row in earlier["observations"][-3:]], axis=0) @
                    np.mean([row["embedding"] for row in later["observations"][:3]], axis=0))
                if endpoint_similarity < .88:
                    continue
                predicted = _predicted_bearing(earlier, later_first)
                angle = _angle_distance(predicted, later["observations"][0]["bearing"])
                endpoint_a = earlier["observations"][-1]
                endpoint_b = later["observations"][0]
                ah = endpoint_a["box"][3] - endpoint_a["box"][1]
                bh = endpoint_b["box"][3] - endpoint_b["box"][1]
                vertical_delta = abs(sum(endpoint_a["box"][1::2]) -
                                     sum(endpoint_b["box"][1::2])) / 2
                # With no calibrated geometry through an occlusion, require
                # nearly stationary endpoints; do not bridge unrelated trees.
                if angle > 8 or _angle_distance(endpoint_a["bearing"], endpoint_b["bearing"]) > 8 or \
                        vertical_delta > max(.025, min(ah, bh) * .25) or \
                        panorama.circular_iou(endpoint_a["box"], endpoint_b["box"]) < .30:
                    continue
                score = similarity + endpoint_similarity * .25 - angle / 250 - gap / 180
                if best is None or score > best[0]:
                    best = (score, left_index, right_index)
        if best is None:
            return working
        _, left_index, right_index = best
        merged = _merge_track_pair(working[left_index], working[right_index])
        working = [track for index, track in enumerate(working)
                   if index not in {left_index, right_index}] + [merged]


def _optimal_pairs(pairs: list[tuple[float, int, int]], observation_count: int,
                   track_count: int) -> list[tuple[int, int]]:
    if not pairs:
        return []
    # One private zero-score dummy column per observation allows unmatched
    # detections. Invalid pairs can never beat a dummy assignment.
    costs = np.full((observation_count, track_count + observation_count), 1e6)
    costs[:, track_count:] = 0
    for value, observation_index, track_index in pairs:
        costs[observation_index, track_index] = -value
    rows, columns = linear_sum_assignment(costs)
    return [(int(row), int(column)) for row, column in zip(rows, columns)
            if column < track_count and costs[row, column] < 0]


def _tracking_snapshot(frames: list[dict]) -> list[dict]:
    return [{
        **{key: value for key, value in frame.items() if key != "candidates"},
        "candidates": [{
            **{key: value for key, value in row.items() if key not in {"crop", "embedding"}},
            "embedding": row["embedding"].tolist(), "crop_size": list(row["crop"].size),
        } for row in frame["candidates"]],
    } for frame in frames]


def _track_candidates(frames: list[dict], audit: dict | None = None) -> list[dict]:
    tracks: list[dict] = []
    history = {frame["sample_index"]: frame for frame in frames}
    audit = audit if audit is not None else {}
    for frame in frames:
        observations = []
        for candidate in frame["candidates"]:
            bearing = _normalise_angle(frame["heading_deg"] + candidate["relative_yaw"])
            observations.append({
                **candidate, "sample_index": frame["sample_index"], "chainage_m": frame["chainage_m"],
                "frame_id": frame["frame_id"], "bearing": bearing,
                "camera_latitude": frame["latitude"], "camera_longitude": frame["longitude"],
            })
        # Optimize the total score of all one-to-one assignments in this frame.
        pairs: list[tuple[float, int, int]] = []
        for observation_index, observation in enumerate(observations):
            for track_index, track in enumerate(tracks):
                gap = frame["sample_index"] - track["last_sample"]
                if gap <= 0 or gap > ONLINE_TRACK_MAX_GAP:
                    continue
                value, reason = corridor_identity.identity_match(track, observation, frame, history)
                if value is None:
                    audit[reason] = audit.get(reason, 0) + 1
                    continue
                pairs.append((value, observation_index, track_index))
        # A near tie is an unresolved identity, not a reason to pick arbitrarily.
        ambiguous_observations = set()
        ambiguous_tracks = set()
        for key_index, ambiguous in ((1, ambiguous_observations), (2, ambiguous_tracks)):
            grouped: dict[int, list[float]] = {}
            for pair in pairs:
                grouped.setdefault(pair[key_index], []).append(pair[0])
            for key, scores in grouped.items():
                scores.sort(reverse=True)
                if len(scores) > 1 and scores[0] - scores[1] < .035:
                    ambiguous.add(key)
        audit["ambiguous_matches_skipped"] = audit.get("ambiguous_matches_skipped", 0) + len(ambiguous_observations)
        pairs = [pair for pair in pairs if pair[1] not in ambiguous_observations and
                 pair[2] not in ambiguous_tracks]
        used_observations: set[int] = set()
        for observation_index, track_index in _optimal_pairs(pairs, len(observations), len(tracks)):
            _append_observation(tracks[track_index], observations[observation_index])
            used_observations.add(observation_index)
        for observation_index, observation in enumerate(observations):
            if observation_index in used_observations:
                continue
            source_id = len(tracks) + 1
            tracks.append({
                "id": source_id, "source_ids": [source_id], "observations": [observation],
                "last_sample": frame["sample_index"], "bearing": observation["bearing"],
                "embedding": observation["embedding"].copy(), "merged_fragment_count": 0,
            })
    tracks = _reconnect_track_fragments(tracks)
    # Stable tracks only: prevents a single low-value shrub/object hit becoming a result.
    stable = [track for track in tracks if len(track["observations"]) >= 2 and
              max(row["quality"] for row in track["observations"]) >= .45]
    stable.sort(key=lambda track: (-len(track["observations"]),
                                  -max(row["quality"] for row in track["observations"])))
    return stable


def _select_views(track: dict) -> list[dict]:
    chosen: list[dict] = []
    for row in sorted(track["observations"], key=lambda item: -item["quality"]):
        if not row.get("crop_evidence", {}).get("usable", True):
            continue
        if not all(corridor_identity.views_compatible(row, old) for old in chosen):
            continue
        if all(abs(row["chainage_m"] - old["chainage_m"]) >= 3 or
               _angle_distance(row["relative_yaw"], old["relative_yaw"]) >= 18 for old in chosen):
            chosen.append(row)
        if len(chosen) >= MAX_VIEWS_PER_TRACK:
            break
    return chosen


def _mean_distribution(results: list[dict]) -> dict[str, float]:
    labels = sorted({label for result in results for label in result["image_distribution"]})
    values = {label: float(np.mean([result["image_distribution"].get(label, 0.0)
                                    for result in results])) for label in labels}
    total = sum(values.values())
    if total <= 0 or not math.isfinite(total):
        raise RuntimeError("Invalid classifier distribution")
    return {label: value / total for label, value in values.items()}


class VmmsCorridorJobManager:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vmms-corridor")
        self.lock = threading.Lock()
        self.jobs: dict[str, dict] = {}

    def create(self, segment_id: str, offset_m: float, length_m: float,
               spacing_m: float, infer_fn) -> str:
        samples = sample_route(segment_id, offset_m, length_m, spacing_m)
        segment = _load_segments()[segment_id]
        request = {"source_mode": "catalog", "segment_id": segment_id,
                   "offset_m": offset_m, "length_m": length_m, "spacing_m": spacing_m}
        source = {
            "source_mode": "catalog", "survey_id": SURVEY_ID,
            "stream_id": segment.stream_id, "source_label": segment.source_label,
            "distance_unit": "m", "coordinate_system": "HK80 + WGS84 camera trajectory",
        }
        return self._create_job(samples, request, source, infer_fn)

    def create_folder(self, folder_path: str, start_index: int, frame_step: int,
                      max_frames: int, infer_fn) -> str:
        samples, inspection = sample_panorama_folder(
            folder_path, start_index=start_index, frame_step=frame_step, max_frames=max_frames)
        request = {
            "source_mode": "folder", "folder": inspection["folder"],
            "start_index": start_index, "frame_step": frame_step, "max_frames": max_frames,
        }
        source = {
            "source_mode": "folder", "survey_id": "user_selected_vmms_folder",
            "stream_id": f'folder:{inspection["name"]}', "source_label": inspection["folder"],
            "distance_unit": "frame", "coordinate_system": None,
            "folder_inspection": inspection,
        }
        return self._create_job(samples, request, source, infer_fn)

    def _create_job(self, samples: list[tuple[float, RouteFrame]], request: dict,
                    source: dict, infer_fn) -> str:
        job_id = uuid.uuid4().hex
        directory = self.root / job_id
        (directory / "frames").mkdir(parents=True)
        (directory / "views").mkdir()
        state = {
            "job_id": job_id, "status": "queued", "created_at": time.time(),
            "request": request, "source": source,
            "progress": {"percent": 0, "message": "Queued / 已排隊", "sampled_frames": 0},
        }
        with self.lock:
            self.jobs[job_id] = state
        self._write(job_id)
        self.pool.submit(self._run, job_id, samples, source, infer_fn)
        return job_id

    def _write(self, job_id: str) -> None:
        directory = self.root / job_id
        temporary = directory / "state.tmp"
        with self.lock:
            state = self.jobs[job_id]
            serial = json.loads(json.dumps(state, allow_nan=False))
        temporary.write_text(json.dumps(serial, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(directory / "state.json")

    def _update(self, job_id: str, **values) -> None:
        with self.lock:
            self.jobs[job_id].update(values)
        self._write(job_id)

    def _progress(self, job_id: str, percent: int, message: str, **values) -> None:
        progress = {"percent": percent, "message": message, **values}
        with self.lock:
            self.jobs[job_id]["progress"] = progress
        self._write(job_id)

    def _run(self, job_id: str, samples: list[tuple[float, RouteFrame]], source_info: dict,
             infer_fn) -> None:
        directory = self.root / job_id
        started = time.time()
        try:
            self._update(job_id, status="running", started_at=started)
            self._progress(job_id, 2, "Loading local detector / 載入本機偵測器")
            if not tree_detection.LOCK.acquire(timeout=180):
                raise RuntimeError("Tree detector remained busy")
            try:
                if tree_detection.ENGINE is None:
                    tree_detection.ENGINE = tree_detection.Engine()
                engine = tree_detection.ENGINE
            finally:
                tree_detection.LOCK.release()
            detected_frames: list[dict] = []
            previous_motion_image = None
            for index, (chainage, frame) in enumerate(samples):
                with Image.open(frame.source) as source:
                    image = ImageOps.exif_transpose(source).convert("RGB")
                if not 1.96 <= image.width / image.height <= 2.04:
                    raise RuntimeError(f"Frame {frame.frame_id} is not an uncropped 2:1 panorama")
                current_motion_image = corridor_identity.motion_image(image)
                motion = corridor_identity.motion_snapshot(previous_motion_image, current_motion_image)
                previous_motion_image = current_motion_image
                if not tree_detection.LOCK.acquire(timeout=180):
                    raise RuntimeError("Tree detector remained busy")
                try:
                    candidates = _detect_frame(image, engine)
                finally:
                    tree_detection.LOCK.release()
                preview_name = f"frame_{index:03d}.jpg"
                _draw_panorama_preview(image, candidates, directory / "frames" / preview_name)
                detected_frames.append({
                    "sample_index": index, "chainage_m": float(chainage), "frame_id": frame.frame_id,
                    "heading_deg": frame.heading_deg, "latitude": frame.latitude,
                    "longitude": frame.longitude, "candidates": candidates,
                    "motion": motion,
                    "preview_url": f"/api/vmms/corridor/jobs/{job_id}/frames/{preview_name}",
                })
                percent = 5 + round(55 * (index + 1) / max(1, len(samples)))
                self._progress(job_id, percent, "Detecting trees in perspective views / 透視展開並偵測樹木",
                               sampled_frames=index + 1, total_frames=len(samples),
                               detections=sum(len(row["candidates"]) for row in detected_frames))
            association_audit: dict = {}
            all_stable_tracks = _track_candidates(detected_frames, association_audit)
            # Keep enough diagnostic data to replay association without another
            # detector/model run, while leaving source panoramas in place.
            (directory / "tracking_diagnostics.json").write_text(json.dumps({
                "tracking_revision": TRACKING_REVISION,
                "association_gate_counts": association_audit,
                "frames": _tracking_snapshot(detected_frames),
                "stable_tracks": [{
                    "source_ids": track.get("source_ids", [track["id"]]),
                    "observations": [{key: row[key] for key in
                        ("sample_index", "frame_id", "box", "bearing", "quality")}
                        for row in track["observations"]],
                } for track in all_stable_tracks],
            }, ensure_ascii=False, allow_nan=False), encoding="utf-8")
            stable_tracks_before_cap = len(all_stable_tracks)
            tracks = all_stable_tracks[:MAX_TRACKS]
            low_value = sum(len(row["candidates"]) for row in detected_frames) - sum(
                len(track["observations"]) for track in all_stable_tracks)
            output_tracks = []
            for track_index, track in enumerate(tracks):
                selected = _select_views(track)
                if not selected:
                    raise RuntimeError("No usable, identity-consistent view in retained track")
                inferences = []
                view_rows = []
                for view_index, observation in enumerate(selected):
                    buffer = io.BytesIO()
                    observation["crop"].save(buffer, "JPEG", quality=94)
                    data = buffer.getvalue()
                    result = infer_fn(data)
                    inferences.append(result)
                    name = f"tree_{track_index + 1:02d}_{view_index + 1}.jpg"
                    _jpeg(observation["crop"], directory / "views" / name, edge=960, quality=90)
                    view_rows.append({
                        "url": f"/api/vmms/corridor/jobs/{job_id}/views/{name}",
                        "frame_id": observation["frame_id"],
                        "chainage_m": round(observation["chainage_m"], 1),
                        "detector_score": round(float(observation["score"]), 4),
                        "sha256": hashlib.sha256(data).hexdigest(),
                    })
                distribution = _mean_distribution(inferences)
                ranked = presentation_groups.top_grouped(distribution)[:3]
                top3 = [{"species": species, "score": float(score), **bilingual_names(species)}
                        for species, score in ranked]
                top_score = top3[0]["score"]
                output_tracks.append({
                    "id": track_index + 1,
                    "first_chainage_m": round(min(row["chainage_m"] for row in track["observations"]), 1),
                    "last_chainage_m": round(max(row["chainage_m"] for row in track["observations"]), 1),
                    "observations": len(track["observations"]), "view_count": len(view_rows),
                    "source_fragments": track.get("merged_fragment_count", 0) + 1,
                    "source_ids": track.get("source_ids", [track["id"]]),
                    "identity_check": {"method": "geometry-gated appearance gallery + complete-link view check",
                                       "pairwise_view_similarity_min": min(
                                           (float(a["embedding"] @ b["embedding"])
                                            for i, a in enumerate(selected) for b in selected[i + 1:]),
                                           default=None),
                                       "status": "consistent_selected_views" if len(selected) > 1 else "single_view_only"},
                    "top3": top3, "views": view_rows,
                    "determination": "undetermined" if top_score < .14 else "candidate",
                    "warnings": (["All species scores are below 14%; change the view or mark this tree undetermined. / 所有樹種分數均低於 14%，請更換觀察角度或標記為暫不能確定。"]
                                 if top_score < .14 else []) +
                                 (["Only one suitable view; no multi-view identity confirmation. / 僅有一張合適視角，未形成多視角身份核驗。"]
                                  if len(selected) == 1 else []),
                })
                self._progress(job_id, 62 + round(34 * (track_index + 1) / max(1, len(tracks))),
                               "Combining multi-view species evidence / 綜合多視角樹種證據",
                               sampled_frames=len(samples), total_frames=len(samples),
                               stable_tracks=len(tracks), classified_tracks=track_index + 1)
            labelled_by_sample: dict[int, list[dict]] = {}
            for track, public_track in zip(tracks, output_tracks):
                for observation in track["observations"]:
                    labelled_by_sample.setdefault(observation["sample_index"], []).append({
                        "box": observation["box"],
                        "track_id": public_track["id"],
                        "top": public_track["top3"][0],
                    })
            for index, (_, frame) in enumerate(samples):
                with Image.open(frame.source) as source:
                    image = ImageOps.exif_transpose(source).convert("RGB")
                _draw_labelled_panorama_preview(
                    image,
                    labelled_by_sample.get(index, []),
                    directory / "frames" / f"frame_{index:03d}.jpg",
                )
                detected_frames[index]["stable_candidate_count"] = len(labelled_by_sample.get(index, []))
            public_frames = [{key: value for key, value in row.items()
                              if key not in {"candidates", "stable_candidate_count", "motion"}}
                             | {"candidate_count": row.get("stable_candidate_count", 0)}
                             for row in detected_frames]
            request = self.jobs[job_id]["request"]
            folder_mode = source_info["source_mode"] == "folder"
            route = {
                **request, "source_mode": source_info["source_mode"],
                "stream_id": source_info["stream_id"], "source_label": source_info["source_label"],
                "distance_unit": source_info["distance_unit"],
                "start_frame": samples[0][1].frame_id, "end_frame": samples[-1][1].frame_id,
                "sample_count": len(samples),
            }
            if folder_mode:
                route.update({
                    "sampled_start_position": int(samples[0][0]) + 1,
                    "sampled_end_position": int(samples[-1][0]) + 1,
                    "actual_covered_frames": int(samples[-1][0] - samples[0][0]),
                    "folder_inspection": source_info["folder_inspection"],
                })
            else:
                route["actual_covered_m"] = round(samples[-1][0] - samples[0][0], 1)
            result = {
                "kind": "vmms_corridor_pilot", "survey_id": source_info["survey_id"],
                "created_at": time.time(), "elapsed_seconds": round(time.time() - started, 1),
                "route": route,
                "pipeline": {
                    "projection": "2:1 spherical panorama → eight overlapping pinhole views",
                    "detector": "YOLO11m-seg + CLIP background gate",
                    "tracking": "periodic sparse LK flow + hard spherical geometry gate + anchored appearance gallery + Hungarian assignment + conservative reconnect",
                    "reid_encoder": "Existing generic CLIP ViT-B/32; no tree-instance ReID training performed",
                    "tracking_revision": TRACKING_REVISION,
                    "classifier": "SSTv1-H120-DV, one vote per selected crop, probability mean",
                    "model_weights_changed": False,
                },
                "frames": public_frames, "tracks": output_tracks,
                "quality": {"stable_tracks": len(output_tracks),
                            "stable_tracks_before_display_cap": stable_tracks_before_cap,
                            "stable_tracks_omitted_by_display_cap": max(
                                0, stable_tracks_before_cap - MAX_TRACKS),
                            "track_fragments_reconnected": sum(
                                track.get("merged_fragment_count", 0) for track in all_stable_tracks),
                            "low_value_or_unstable_detections_rejected": max(0, low_value)},
                "limitations": ([
                    "Folder images are ordered by natural filename order. Without a coordinate table, frame spacing is not reported as metres. / 文件夾影像按文件名自然順序排列；沒有座標表時，不會把幀間隔當作米制距離。",
                ] if folder_mode else [
                    "HK80/WGS84 values are camera trajectory coordinates, not measured tree positions. / 座標是相機軌跡位置，不是量測所得的樹木位置。",
                ]) + [
                    f"Up to {MAX_TRACKS} stable candidates are classified, prioritised by observation count and image quality. / 最多識別 {MAX_TRACKS} 條穩定候選軌跡，優先處理觀察次數多、影像清晰的目標。",
                    "Tracks require at least two sampled observations; short or occluded appearances may be omitted. / 軌跡至少需兩個採樣視角，短暫或受遮擋的樹木可能被略過。",
                    "This run is a workflow result, not a labelled accuracy benchmark. / 本次結果是流程輸出，並非帶真值標註的準確率評測。",
                    "Species scores are model scores and still require review for high-stakes use. / 樹種分數為模型分數，高風險用途仍需人工核驗。",
                ],
            }
            (directory / "result.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
            self._update(job_id, status="complete", completed_at=time.time(), result=result,
                         progress={"percent": 100, "message": "Complete / 已完成",
                                   "sampled_frames": len(samples), "total_frames": len(samples),
                                   "stable_tracks": len(output_tracks),
                                   "classified_tracks": len(output_tracks)})
        except Exception as exc:
            import logging
            logging.getLogger(__name__).exception("VMMS corridor job failed")
            self._update(job_id, status="failed", completed_at=time.time(),
                         error=f"{type(exc).__name__}: {exc}",
                         progress={"percent": 100, "message": "Failed / 失敗"})

    def status(self, job_id: str) -> dict:
        with self.lock:
            if job_id in self.jobs:
                return json.loads(json.dumps(self.jobs[job_id], allow_nan=False))
        path = self.root / job_id / "state.json"
        if not path.is_file():
            raise KeyError(job_id)
        return json.loads(path.read_text(encoding="utf-8"))

    def asset(self, job_id: str, group: str, name: str) -> Path:
        if group not in {"frames", "views"} or not name.endswith(".jpg"):
            raise KeyError(name)
        root = (self.root / job_id / group).resolve()
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise KeyError(name)
        return path
