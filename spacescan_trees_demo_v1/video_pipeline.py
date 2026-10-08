"""Inference-only video tree tracking and multi-view aggregation.

The pipeline never trains or changes model weights. Uploaded source videos are
deleted after analysis; only small representative crops are retained temporarily
for the local result page.
"""
from __future__ import annotations

import hashlib
import math
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .species_names import bilingual_names
from . import presentation_groups


MAX_VIDEO_BYTES = 2 * 1024 * 1024 * 1024
MAX_VIDEO_LABEL = "2 GB"
MAX_DURATION_SECONDS = 300
MAX_TRACKS_TO_CLASSIFY = 40
MIN_IDENTIFIABLE_VIEWS = 2
LABEL_EVIDENCE_MARGIN_SECONDS = 0.50
PROFILES = {
    "fast": {"tracking_fps": 10.0, "target_fps": 3.0, "max_processed": 3000, "views": 4},
    "balanced": {"tracking_fps": 15.0, "target_fps": 5.0, "max_processed": 4500, "views": 5},
    "detail": {"tracking_fps": 20.0, "target_fps": 7.0, "max_processed": 6000, "views": 6},
}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".webm", ".mkv"}


def video_size_error():
    return f"Use a video up to {MAX_VIDEO_LABEL}. / 影片上限為 {MAX_VIDEO_LABEL}。"


def _orientation_degrees(value):
    """Return the nearest supported clockwise display rotation."""
    if not math.isfinite(float(value)):
        return 0
    return min((0, 90, 180, 270), key=lambda angle: abs(((float(value) - angle + 180) % 360) - 180))


def _rotate_frame(frame, degrees):
    """Apply container rotation metadata when the decoder cannot do it."""
    rotation = _orientation_degrees(degrees)
    if rotation == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if rotation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if rotation == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def _iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(1e-9, aa + bb - inter)


def _center_distance(a, b, width, height):
    ax, ay = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    bx, by = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return math.hypot((ax - bx) / max(1, width), (ay - by) / max(1, height))


def _evidence_window(first_second, last_second, views):
    """Limit a track-level species label to times supported by classified crops."""
    seconds = [float(view["second"]) for view in views]
    if not seconds:
        return float(first_second), float(last_second)
    return (max(float(first_second), min(seconds) - LABEL_EVIDENCE_MARGIN_SECONDS),
            min(float(last_second), max(seconds) + LABEL_EVIDENCE_MARGIN_SECONDS))


def _handoff_display_ids(raw_ids, fallback_ids, id_map, next_display_id):
    """Keep one public ID when BoT-SORT changes its temporary internal ID."""
    assigned = []
    used = set()
    next_display_id = max(next_display_id, max(fallback_ids, default=0) + 1)
    for raw_id, fallback_id in zip(raw_ids, fallback_ids):
        candidate = id_map.get(raw_id, fallback_id) if raw_id is not None else fallback_id
        if candidate in used:
            candidate = max(next_display_id, max(id_map.values(), default=0) + 1)
            next_display_id = candidate + 1
        if raw_id is not None:
            id_map[raw_id] = candidate
        used.add(candidate)
        assigned.append(candidate)
    return assigned, next_display_id


class FallbackTracker:
    """Small deterministic fallback used only when Ultralytics returns no IDs."""

    def __init__(self, max_age=10):
        self.max_age = max_age
        self.next_id = 1
        self.rows = {}

    def update(self, boxes, frame_index, width, height):
        assigned = []
        available = set(self.rows)
        for box in boxes:
            candidates = []
            for track_id in available:
                prior = self.rows[track_id]
                if frame_index - prior["frame"] > self.max_age:
                    continue
                overlap = _iou(box, prior["box"])
                distance = _center_distance(box, prior["box"], width, height)
                old_area = max(1.0, (prior["box"][2] - prior["box"][0]) * (prior["box"][3] - prior["box"][1]))
                new_area = max(1.0, (box[2] - box[0]) * (box[3] - box[1]))
                scale = new_area / old_area
                if (overlap >= 0.08 or distance <= 0.22) and 0.28 <= scale <= 3.6:
                    candidates.append((overlap - 0.7 * distance, track_id))
            if candidates:
                track_id = max(candidates)[1]
                available.remove(track_id)
            else:
                track_id = self.next_id
                self.next_id += 1
            self.rows[track_id] = {"box": box, "frame": frame_index}
            assigned.append(track_id)
        self.rows = {key: row for key, row in self.rows.items()
                     if frame_index - row["frame"] <= self.max_age}
        return assigned


def _difference_hash(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    tiny = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    return (tiny[:, 1:] > tiny[:, :-1]).reshape(-1)


def _hash_distance(a, b):
    return int(np.count_nonzero(a != b))


def _quality(crop, box, width, height, detector_score):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    p10, p90 = (float(v) for v in np.percentile(gray, [10, 90]))
    area_ratio = ((box[2] - box[0]) * (box[3] - box[1])) / max(1, width * height)
    sharp_component = min(1.0, sharpness / 300.0)
    area_component = min(1.0, area_ratio / 0.18)
    exposure_component = max(0.0, min(1.0, (p90 - 35.0) / 145.0))
    dynamic_component = max(0.0, min(1.0, (p90 - p10) / 100.0))
    blue, green, red = cv2.split(crop.astype(np.float32))
    vegetation = ((green > 45) & (green > red * 1.06) & (green > blue * 1.04)
                  & ((2 * green - red - blue) > 18))
    vegetation_fraction = float(vegetation.mean())
    crop_height, crop_width = vegetation.shape
    centre = vegetation[crop_height // 4:3 * crop_height // 4,
                        crop_width // 4:3 * crop_width // 4]
    vegetation_central_fraction = float(centre.mean()) if centre.size else 0.0
    vegetation_grid_coverage = sum(
        float(vegetation[y * crop_height // 4:(y + 1) * crop_height // 4,
                         x * crop_width // 4:(x + 1) * crop_width // 4].mean()) >= 0.035
        for y in range(4) for x in range(4)
    )
    value = (0.35 * float(detector_score) + 0.25 * sharp_component
             + 0.20 * area_component + 0.10 * exposure_component
             + 0.10 * dynamic_component)
    return value, {"sharpness": sharpness, "p10": p10, "p90": p90,
                   "area_ratio": area_ratio, "detector_score": float(detector_score),
                   "vegetation_fraction": vegetation_fraction,
                   "vegetation_central_fraction": vegetation_central_fraction,
                   "vegetation_grid_coverage": vegetation_grid_coverage}


def _crop(frame, box):
    height, width = frame.shape[:2]
    x0, y0, x1, y1 = box
    padx, pady = 0.05 * (x1 - x0), 0.05 * (y1 - y0)
    x0, y0 = max(0, int(x0 - padx)), max(0, int(y0 - pady))
    x1, y1 = min(width, int(x1 + padx)), min(height, int(y1 + pady))
    return frame[y0:y1, x0:x1].copy(), [x0, y0, x1, y1]


def _retain_view(views, view, limit):
    if not views:
        views.append(view)
        return
    distances = [_hash_distance(view["hash"], old["hash"]) for old in views]
    closest = int(np.argmin(distances))
    if distances[closest] < 8:
        if view["quality"] > views[closest]["quality"]:
            views[closest] = view
        return
    views.append(view)
    views.sort(key=lambda row: row["quality"], reverse=True)
    del views[limit:]


def _view_is_identifiable(view, width, height):
    """Reject ground cover, partial bottom-edge crops and obvious non-vegetation."""
    x0, y0, x1, y1 = view["box"]
    centre_y = (y0 + y1) / (2 * max(1, height))
    height_ratio = (y1 - y0) / max(1, height)
    width_ratio = (x1 - x0) / max(1, width)
    metrics = view.get("quality_metrics", {})
    vegetation_fraction = float(metrics.get("vegetation_fraction", 0.0))
    vegetation_central_fraction = float(metrics.get("vegetation_central_fraction", 0.0))
    vegetation_grid_coverage = int(metrics.get("vegetation_grid_coverage", 0))
    return (centre_y <= 0.80 and height_ratio >= 0.15 and width_ratio >= 0.12
            and vegetation_fraction >= 0.035 and vegetation_central_fraction >= 0.035
            and vegetation_grid_coverage >= 8)


def aggregate_predictions(predictions):
    """Quality-weighted probability mean; never inflates or recalibrates scores."""
    usable = [row for row in predictions if row.get("image_distribution")]
    if not usable:
        raise ValueError("No usable view prediction")
    labels = set(usable[0]["image_distribution"])
    if any(set(row["image_distribution"]) != labels for row in usable):
        raise ValueError("View label maps differ")
    weights = np.asarray([max(0.2, min(1.0, float(row.get("view_quality", 1.0))))
                          for row in usable], dtype=np.float64)
    weights /= weights.sum()
    distribution = {label: float(sum(weight * float(row["image_distribution"][label])
                                     for weight, row in zip(weights, usable)))
                    for label in labels}
    total = sum(distribution.values())
    if not math.isfinite(total) or total <= 0:
        raise ValueError("Invalid aggregate distribution")
    distribution = {label: score / total for label, score in distribution.items()}
    ranked = presentation_groups.top_grouped(distribution)[:3]
    top3 = [{"species": species, "score": float(score), **bilingual_names(species)}
            for species, score in ranked]
    view_winners = [presentation_groups.top_grouped(row["image_distribution"])[0][0]
                    for row in usable]
    consensus = sum(name == top3[0]["species"] for name in view_winners) / len(view_winners)
    return distribution, top3, consensus


def _tracks_are_duplicate(first, second, width, height):
    """Conservatively identify split IDs for one continuously visible tree."""
    if not first.get("top3") or not second.get("top3"):
        return False
    if first["top3"][0]["species"] != second["top3"][0]["species"]:
        return False
    if min(first["top3"][0]["score"], second["top3"][0]["score"]) < 0.75:
        return False
    if min(first.get("consensus", 0.0), second.get("consensus", 0.0)) < 0.75:
        return False
    overlap = min(first["last_second"], second["last_second"]) - max(first["first_second"], second["first_second"])
    shorter = min(first["last_second"] - first["first_second"],
                  second["last_second"] - second["first_second"])
    if overlap <= 0 or overlap / max(0.1, shorter) < 0.45:
        return False
    for one in first.get("trajectory", []):
        for two in second.get("trajectory", []):
            if abs(float(one["second"]) - float(two["second"])) > 0.55:
                continue
            a, b = one["box"], two["box"]
            horizontal_gap = max(0.0, max(a[0], b[0]) - min(a[2], b[2]))
            vertical_overlap = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
            smaller_height = max(1.0, min(a[3] - a[1], b[3] - b[1]))
            if horizontal_gap <= 0.12 * max(1, width) and vertical_overlap / smaller_height >= 0.25:
                return True
    return False


def _top3_from_distribution(distribution):
    ranked = presentation_groups.top_grouped(distribution)[:3]
    return [{"species": species, "score": float(score), **bilingual_names(species)}
            for species, score in ranked]


def _merge_duplicate_tracks(rows, width, height, view_limit):
    """Merge only high-confidence, simultaneous and spatially adjacent duplicates."""
    merged = []
    merge_count = 0
    for row in sorted(rows, key=lambda item: item["first_second"]):
        target = next((old for old in merged if _tracks_are_duplicate(old, row, width, height)), None)
        if target is None:
            row["merged_track_ids"] = [row["id"]]
            merged.append(row)
            continue
        first_weight = max(1, target["view_count"])
        second_weight = max(1, row["view_count"])
        labels = set(target["_distribution"]) & set(row["_distribution"])
        distribution = {
            label: (target["_distribution"][label] * first_weight
                    + row["_distribution"][label] * second_weight) / (first_weight + second_weight)
            for label in labels
        }
        total = sum(distribution.values())
        if total > 0:
            distribution = {label: value / total for label, value in distribution.items()}
            target["_distribution"] = distribution
            target["top3"] = _top3_from_distribution(distribution)
        target["id"] = min(target["id"], row["id"])
        target["merged_track_ids"] = sorted(set(target["merged_track_ids"] + [row["id"]]))
        target["first_second"] = min(target["first_second"], row["first_second"])
        target["last_second"] = max(target["last_second"], row["last_second"])
        target["label_first_second"] = min(target["label_first_second"], row["label_first_second"])
        target["label_last_second"] = max(target["label_last_second"], row["label_last_second"])
        target["detections"] = max(target["detections"], row["detections"])
        target["consensus"] = round(
            (target["consensus"] * first_weight + row["consensus"] * second_weight)
            / (first_weight + second_weight), 4)
        target["views"] = sorted(target["views"] + row["views"],
                                 key=lambda item: item["quality"], reverse=True)[:view_limit]
        target["view_count"] = len(target["views"])
        target["trajectory"] = sorted(target["trajectory"] + row["trajectory"],
                                      key=lambda item: item["second"])
        note = "Overlapping duplicate tracks were consolidated. / 已合併同一株樹的重疊軌跡。"
        if note not in target["warnings"]:
            target["warnings"].append(note)
        merge_count += 1
    return merged, merge_count


class VideoJobManager:
    def __init__(self, root: Path, detector_weight: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.detector_weight = Path(detector_weight)
        self.jobs = {}
        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sst-video")

    def _cleanup(self):
        cutoff = time.time() - 12 * 3600
        for job_id, job in list(self.jobs.items()):
            if job["created_epoch"] >= cutoff or job["status"] in {"queued", "running"}:
                continue
            directory = Path(job["_directory"])
            for child in directory.glob("*"):
                if child.is_file():
                    child.unlink(missing_ok=True)
            try:
                directory.rmdir()
            except OSError:
                pass
            self.jobs.pop(job_id, None)

    def reserve(self, filename):
        suffix = Path(filename or "video.mp4").suffix.lower()
        if suffix not in VIDEO_EXTENSIONS:
            raise ValueError("Use MP4, MOV, M4V, AVI, WEBM or MKV. / 請使用支援的影片格式。")
        with self.lock:
            self._cleanup()
            job_id = uuid.uuid4().hex
            directory = self.root / job_id
            directory.mkdir(parents=False, exist_ok=False)
            source = directory / ("source" + suffix)
            self.jobs[job_id] = {
                "id": job_id, "status": "uploading", "created_epoch": time.time(),
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "filename": Path(filename or "video").name,
                "progress": {"phase": "upload", "percent": 0, "message": "Uploading / 正在上傳"},
                "_directory": str(directory), "_source": str(source),
            }
        return job_id, source

    def discard(self, job_id):
        with self.lock:
            job = self.jobs.pop(job_id, None)
        if not job:
            return
        directory = Path(job["_directory"])
        for child in directory.glob("*"):
            if child.is_file():
                child.unlink(missing_ok=True)
        try:
            directory.rmdir()
        except OSError:
            pass

    def start(self, job_id, profile, classify_fn: Callable[[bytes], dict]):
        if profile not in PROFILES:
            raise ValueError("Invalid analysis profile")
        with self.lock:
            job = self.jobs[job_id]
            job["status"] = "queued"
            job["profile"] = profile
            job["progress"] = {"phase": "queued", "percent": 1,
                               "message": "Waiting for local GPU / 等待本機 GPU"}
        self.executor.submit(self._run, job_id, PROFILES[profile], classify_fn)

    def status(self, job_id):
        with self.lock:
            if job_id not in self.jobs:
                raise KeyError(job_id)
            job = self.jobs[job_id]
            return {key: value for key, value in job.items() if not key.startswith("_")}

    def view_path(self, job_id, name):
        if not name or Path(name).name != name:
            raise KeyError(name)
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                raise KeyError(job_id)
            path = (Path(job["_directory"]) / name).resolve()
            if not path.is_relative_to(Path(job["_directory"]).resolve()) or not path.is_file():
                raise KeyError(name)
            return path

    def _update(self, job_id, **values):
        with self.lock:
            if job_id in self.jobs:
                self.jobs[job_id].update(values)

    def _run(self, job_id, profile, classify_fn):
        source = None
        try:
            with self.lock:
                job = self.jobs[job_id]
                source = Path(job["_source"])
                directory = Path(job["_directory"])
            self._update(job_id, status="running",
                         progress={"phase": "opening", "percent": 2,
                                   "message": "Opening video / 正在開啟影片"})
            result = self._analyze(job_id, source, directory, profile, classify_fn)
            self._update(job_id, status="complete", result=result,
                         progress={"phase": "complete", "percent": 100,
                                   "message": "Analysis complete / 分析完成"})
        except Exception as exc:
            self._update(job_id, status="failed", error=str(exc)[:500],
                         progress={"phase": "failed", "percent": 100,
                                   "message": "Analysis failed / 分析失敗"})
        finally:
            if source is not None:
                source.unlink(missing_ok=True)

    def _analyze(self, job_id, source, directory, profile, classify_fn):
        if not self.detector_weight.is_file():
            raise RuntimeError("Tree detector checkpoint is missing")
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            raise ValueError("Video cannot be decoded. / 影片無法解碼。")
        encoded_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        encoded_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        orientation_property = getattr(cv2, "CAP_PROP_ORIENTATION_META", None)
        orientation_auto_property = getattr(cv2, "CAP_PROP_ORIENTATION_AUTO", None)
        orientation_degrees = _orientation_degrees(
            cap.get(orientation_property) if orientation_property is not None else 0
        )
        orientation_auto = False
        if orientation_auto_property is not None:
            cap.set(orientation_auto_property, 1)
            orientation_auto = cap.get(orientation_auto_property) >= 0.5
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        fps = fps if math.isfinite(fps) and 1 <= fps <= 240 else 25.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        first_ok, first_frame = cap.read()
        if not first_ok or first_frame is None or first_frame.size == 0:
            cap.release()
            raise ValueError("Video contains no decodable frame. / 影片沒有可解碼畫面。")
        if not orientation_auto:
            first_frame = _rotate_frame(first_frame, orientation_degrees)
        height, width = first_frame.shape[:2]
        if width < 64 or height < 64 or width * height > 40_000_000:
            cap.release()
            raise ValueError("Unsupported video dimensions. / 影片尺寸不受支援。")
        reported_duration = frame_count / fps if frame_count > 0 else None
        analysis_duration = min(MAX_DURATION_SECONDS, reported_duration or MAX_DURATION_SECONDS)
        tracking_fps = min(fps, profile.get("tracking_fps", profile["target_fps"]))
        stride = max(1, int(round(fps / tracking_fps)))
        collection_every = max(1, int(round(tracking_fps / profile["target_fps"])))
        planned = min(profile["max_processed"], max(1, int(analysis_duration * fps / stride)))

        from ultralytics import YOLO
        import torch

        model = YOLO(str(self.detector_weight))
        tree_classes = [index for index, name in model.names.items() if "tree" in name.lower()]
        if not tree_classes:
            cap.release()
            raise RuntimeError("Detector has no tree class")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        fallback = FallbackTracker(max_age=max(5, round(tracking_fps * 4)))
        id_map = {}
        next_display_id = 1
        tracks = {}
        processed = 0
        source_frame = -1
        pending_frame = first_frame
        tracker_backend = "IoU + centre fallback"
        ultralytics_ids_seen = False
        trajectory_every = max(1, round(tracking_fps / 5.0))
        max_trajectory_points = 1500

        try:
            while processed < profile["max_processed"]:
                if pending_frame is not None:
                    frame = pending_frame
                    pending_frame = None
                else:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if not orientation_auto:
                        frame = _rotate_frame(frame, orientation_degrees)
                source_frame += 1
                timestamp = source_frame / fps
                if timestamp > MAX_DURATION_SECONDS:
                    break
                if source_frame % stride:
                    continue
                processed += 1
                try:
                    detection = model.track(
                        frame, persist=True, tracker="botsort.yaml", imgsz=1024,
                        conf=0.15, iou=0.5, classes=tree_classes, agnostic_nms=True,
                        max_det=60, device=device, verbose=False,
                    )[0]
                    boxes = detection.boxes.xyxy.cpu().tolist() if detection.boxes is not None else []
                    scores = detection.boxes.conf.cpu().tolist() if detection.boxes is not None else []
                    raw_ids = (detection.boxes.id.int().cpu().tolist()
                               if detection.boxes is not None and detection.boxes.id is not None else None)
                except Exception:
                    tracker_backend = "IoU + centre fallback"
                    detection = model.predict(
                        frame, imgsz=1024, conf=0.25, iou=0.5, classes=tree_classes,
                        agnostic_nms=True, max_det=60, device=device, verbose=False,
                    )[0]
                    boxes = detection.boxes.xyxy.cpu().tolist() if detection.boxes is not None else []
                    scores = detection.boxes.conf.cpu().tolist() if detection.boxes is not None else []
                    raw_ids = None
                valid = []
                for item_index, (box, score) in enumerate(zip(boxes, scores)):
                    x0, y0, x1, y1 = [float(value) for value in box]
                    if x1 - x0 < 48 or y1 - y0 < 72:
                        continue
                    if (x1 - x0) * (y1 - y0) > 0.92 * width * height:
                        continue
                    valid.append(([x0, y0, x1, y1], float(score),
                                  raw_ids[item_index] if raw_ids is not None and item_index < len(raw_ids) else None))
                boxes = [row[0] for row in valid]
                scores = [row[1] for row in valid]
                raw_ids = [row[2] for row in valid]
                fallback_ids = fallback.update(boxes, processed, width, height)
                ultralytics_ids_seen = ultralytics_ids_seen or any(raw_id is not None for raw_id in raw_ids)
                local_ids, next_display_id = _handoff_display_ids(
                    raw_ids, fallback_ids, id_map, max(next_display_id, fallback.next_id)
                )
                tracker_backend = ("BoT-SORT + geometric handoff" if ultralytics_ids_seen
                                   else "IoU + centre fallback")

                for box, score, track_id in zip(boxes, scores, local_ids):
                    crop, expanded = _crop(frame, box)
                    if crop.size == 0:
                        continue
                    quality, metrics = _quality(crop, expanded, width, height, score)
                    if metrics["p90"] < 30 or metrics["area_ratio"] < 0.008:
                        continue
                    row = tracks.setdefault(track_id, {
                        "id": track_id, "first_second": timestamp, "last_second": timestamp,
                        "detections": 0, "views": [], "trajectory": [], "best_quality": 0.0,
                    })
                    row["last_second"] = timestamp
                    row["detections"] += 1
                    row["best_quality"] = max(row["best_quality"], quality)
                    if (len(row["trajectory"]) < max_trajectory_points
                            and (row["detections"] == 1 or row["detections"] % trajectory_every == 0)):
                        row["trajectory"].append({"second": round(timestamp, 2),
                                                  "box": [round(value, 1) for value in expanded]})
                    encoded_ok, encoded = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
                    if encoded_ok and (processed == 1 or processed % collection_every == 0):
                        _retain_view(row["views"], {
                            "bytes": encoded.tobytes(), "hash": _difference_hash(crop),
                            "quality": quality, "quality_metrics": metrics,
                            "second": timestamp, "box": expanded,
                        }, profile["views"] + 2)

                percent = min(70, 5 + int(65 * processed / max(1, planned)))
                if processed == 1 or processed % 5 == 0:
                    self._update(job_id, progress={
                        "phase": "tracking", "percent": percent,
                        "processed_frames": processed, "tracks_seen": len(tracks),
                        "message": f"Tracking trees · {processed} sampled frames / 正在追蹤樹木",
                    })
        finally:
            cap.release()

        candidates = []
        rejected_short = 0
        rejected_low_value = 0
        for row in tracks.values():
            if row["detections"] < 3 or not row["views"]:
                rejected_short += 1
                continue
            identifiable = [view for view in row["views"]
                            if _view_is_identifiable(view, width, height)]
            if len(identifiable) < MIN_IDENTIFIABLE_VIEWS:
                rejected_low_value += 1
                continue
            row["views"] = identifiable
            candidates.append(row)
        candidates.sort(key=lambda row: (row["detections"], row["best_quality"]), reverse=True)
        omitted = max(0, len(candidates) - MAX_TRACKS_TO_CLASSIFY)
        candidates = candidates[:MAX_TRACKS_TO_CLASSIFY]
        completed = []
        model_version = None
        for index, row in enumerate(candidates):
            self._update(job_id, progress={
                "phase": "classifying", "percent": 72 + int(25 * index / max(1, len(candidates))),
                "processed_frames": processed, "tracks_seen": len(tracks),
                "classified_tracks": index,
                "message": f"Recognising Tree {row['id']} / 正在識別第 {row['id']} 株樹",
            })
            predictions = []
            public_views = []
            selected_views = sorted(row["views"], key=lambda view: view["quality"], reverse=True)[:profile["views"]]
            for view_index, view in enumerate(selected_views, start=1):
                try:
                    prediction = classify_fn(view["bytes"])
                    prediction["view_quality"] = view["quality"]
                    predictions.append(prediction)
                    model_version = model_version or prediction.get("image_model_version")
                except Exception:
                    continue
                filename = f"tree_{row['id']:03d}_view_{view_index:02d}.jpg"
                (directory / filename).write_bytes(view["bytes"])
                public_views.append({
                    "url": f"/api/video/jobs/{job_id}/views/{filename}",
                    "second": round(view["second"], 2),
                    "quality": round(view["quality"], 4),
                    "box": [round(value, 1) for value in view["box"]],
                })
            if not predictions:
                continue
            distribution, top3, consensus = aggregate_predictions(predictions)
            label_first_second, label_last_second = _evidence_window(
                row["first_second"], row["last_second"], public_views)
            uncertain = top3[0]["score"] < 0.14
            warnings = []
            if uncertain:
                warnings.append("All candidates are weak; change the viewing angle or mark the species undetermined. / 候選分數偏低，請更換觀察角度或標記為暫不能確定。")
            if consensus < 0.6:
                warnings.append("Views disagree; review the representative crops before accepting the species. / 多視角意見不一致，採納前請核對代表切片。")
            if row["detections"] < 4:
                warnings.append("Short track; add a longer or steadier view. / 軌跡較短，建議補充較長或較穩定的視角。")
            completed.append({
                "id": row["id"], "first_second": round(row["first_second"], 2),
                "last_second": round(row["last_second"], 2), "detections": row["detections"],
                "label_first_second": round(label_first_second, 2),
                "label_last_second": round(label_last_second, 2),
                "view_count": len(predictions), "views": public_views, "top3": top3,
                "consensus": round(consensus, 4), "determination": "undetermined" if uncertain else "candidate",
                "warnings": warnings, "trajectory": row["trajectory"],
                "_distribution": distribution,
            })

        completed, duplicate_tracks_merged = _merge_duplicate_tracks(
            completed, width, height, profile["views"])
        completed.sort(key=lambda row: row["first_second"])
        for row in completed:
            row.pop("_distribution", None)
        return {
            "media": {"width": width, "height": height,
                      "encoded_width": encoded_width, "encoded_height": encoded_height,
                      "orientation_degrees": orientation_degrees,
                      "orientation_normalized": bool(orientation_degrees),
                      "fps": round(fps, 3),
                      "reported_frames": frame_count,
                      "reported_duration_seconds": round(reported_duration, 2) if reported_duration else None,
                      "analysed_duration_seconds": round(min(MAX_DURATION_SECONDS,
                                                               (source_frame + 1) / fps), 2)},
            "sampling": {"profile": next((name for name, value in PROFILES.items() if value == profile), "custom"),
                         "tracking_fps": round(tracking_fps, 3),
                         "target_fps": profile["target_fps"], "source_frame_stride": stride,
                         "processed_frames": processed},
            "tracker": {"name": tracker_backend, "persistent_ids": True,
                        "scope": "one continuous uploaded video"},
            "detector": {"name": "YOLO11m-seg", "weight_sha256": hashlib.sha256(self.detector_weight.read_bytes()).hexdigest()},
            "classifier": {"name": "SSTv1-H120-DV", "version": model_version,
                           "fusion": "quality-weighted arithmetic mean of complete per-view distributions",
                           "score_is_accuracy": False},
            "tracks": completed, "tracks_seen": len(tracks),
            "tracks_rejected": rejected_short + rejected_low_value,
            "tracks_rejected_short": rejected_short,
            "tracks_rejected_low_value": rejected_low_value,
            "duplicate_tracks_merged": duplicate_tracks_merged,
            "tracks_omitted_by_limit": omitted,
            "analysis_truncated": bool((reported_duration and reported_duration > MAX_DURATION_SECONDS)
                                       or processed >= profile["max_processed"]),
            "source_video_retained": False,
            "representative_crops_retention": "local runtime, automatically eligible for cleanup after 12 hours",
            "limitations": [
                "Track IDs express continuity inside this video, not a permanent physical-tree identity.",
                "Shot cuts, long occlusion and visually overlapping trees can still cause ID changes.",
                "Species overlays are shown only near representative views that were actually classified, so a later label is not carried backward through a contaminated track.",
                "Model scores are not measured accuracy; low scores and conflicting views remain undetermined.",
            ],
        }
