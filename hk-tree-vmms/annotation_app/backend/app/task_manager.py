from __future__ import annotations

import csv
import json
import math
import shutil
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from .image_preprocessing import (
    analyze_image_quality,
    apply_photo_recipe,
    recipe_changes_pixels,
    suggest_recipe,
)

from .config import (
    ANNOTATION_DIR,
    CROP_FOV_DEG,
    CROP_OUTPUT_SIZE,
    FRAME_COORDINATES,
    HEWENTIAN_STREAM_ID,
    NEARBY_TREES,
    PANORAMA_X_SIGN,
    PANORAMA_YAW_OFFSET_DEG,
    RUNTIME_DIR,
    TREE_DISTANCE_LIMIT_M,
    VIEW_CACHE_DIR,
    VIEW_MAX_RANGE_M,
    VIEW_MIN_SEPARATION_M,
    VMMS_ROOT,
    assert_inside,
    ensure_dirs,
    safe_name,
)
from .schemas import CheckpointRequest, ImageAnnotation, SubmitRequest


@dataclass(frozen=True)
class FrameRecord:
    frame_id: str
    seq_id: int
    utc_seconds: float
    utc_datetime: str
    hong_kong_datetime: str
    source_image_relpath: str
    easting: float
    northing: float
    latitude: float
    longitude: float
    heading_deg: float


@dataclass(frozen=True)
class TreeRecord:
    tree_id: str
    species: str
    source_dataset: str
    source_feature_index: str
    location_name: str
    easting: float
    northing: float
    latitude: float
    longitude: float
    distance_to_route_m: float


@dataclass
class ReviewSample:
    tree_id: str
    species: str
    images: list[str]
    candidates: list[dict[str, Any]]
    tree: dict[str, Any]
    warnings: list[str] = field(default_factory=list)
    draft_annotations: list[dict[str, Any]] = field(default_factory=list)


class TaskManager:
    def __init__(self) -> None:
        ensure_dirs()
        self.frames = self._load_frames()
        self.trees = self._load_trees()
        self.trees_by_id = {tree.tree_id: tree for tree in self.trees}
        self.review_state_path = RUNTIME_DIR / "review_state.json"
        self.review_state = self._load_review_state()
        self.checkpoint_path = RUNTIME_DIR / "checkpoints.json"
        self.checkpoints = self._load_checkpoints()
        self.species: str | None = None
        self.target_count = 0
        self.queue: list[TreeRecord] = []
        self.cursor = 0
        self.reviewed_count = 0
        self.rejected_count = 0
        self.failed_count = 0
        self.message = ""
        self.local_api_counts: dict[str, int] = {}
        self._lock = threading.RLock()

    def record_api_call(self, name: str) -> None:
        self.local_api_counts[name] = self.local_api_counts.get(name, 0) + 1

    def _load_frames(self) -> list[FrameRecord]:
        if not FRAME_COORDINATES.is_file():
            raise FileNotFoundError(FRAME_COORDINATES)
        records: list[FrameRecord] = []
        with FRAME_COORDINATES.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["stream_id"] != HEWENTIAN_STREAM_ID:
                    continue
                records.append(
                    FrameRecord(
                        frame_id=row["frame_id"],
                        seq_id=int(row["seq_id"]),
                        utc_seconds=float(row["utc_seconds"]),
                        utc_datetime=row["utc_datetime"],
                        hong_kong_datetime=row["hong_kong_datetime"],
                        source_image_relpath=row["source_image_relpath"],
                        easting=float(row["hk80_easting"]),
                        northing=float(row["hk80_northing"]),
                        latitude=float(row["wgs84_latitude"]),
                        longitude=float(row["wgs84_longitude"]),
                        heading_deg=float(row["heading_deg"]) % 360.0,
                    )
                )
        if not records:
            raise RuntimeError(f"No frames found for {HEWENTIAN_STREAM_ID}")
        return records

    def _load_trees(self) -> list[TreeRecord]:
        if not NEARBY_TREES.is_file():
            raise FileNotFoundError(NEARBY_TREES)
        records: list[TreeRecord] = []
        duplicate_counts: dict[str, int] = {}
        with NEARBY_TREES.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["nearest_route_id"] != HEWENTIAN_STREAM_ID:
                    continue
                if float(row["distance_to_route_m"]) > TREE_DISTANCE_LIMIT_M:
                    continue
                base_id = row["source_tree_id"].strip()
                duplicate_counts[base_id] = duplicate_counts.get(base_id, 0) + 1
                tree_id = (
                    base_id
                    if duplicate_counts[base_id] == 1
                    else f"{base_id}_{duplicate_counts[base_id]}"
                )
                records.append(
                    TreeRecord(
                        tree_id=tree_id,
                        species=row["species_name"].strip() or "Unknown 未知",
                        source_dataset=row["source_dataset"],
                        source_feature_index=row["source_feature_index"],
                        location_name=row["location_name"],
                        easting=float(row["hk80_easting"]),
                        northing=float(row["hk80_northing"]),
                        latitude=float(row["wgs84_latitude"]),
                        longitude=float(row["wgs84_longitude"]),
                        distance_to_route_m=float(row["distance_to_route_m"]),
                    )
                )
        return records

    def _load_review_state(self) -> dict[str, dict[str, Any]]:
        if not self.review_state_path.is_file():
            return {}
        return json.loads(self.review_state_path.read_text(encoding="utf-8"))

    def _save_review_state(self) -> None:
        temporary = self.review_state_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(self.review_state, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(self.review_state_path)

    def _save_preprocess_manifest(self) -> None:
        manifest_path = ANNOTATION_DIR / "preprocess_manifest.csv"
        temporary = manifest_path.with_suffix(".tmp")
        fieldnames = [
            "tree_id",
            "species",
            "review_status",
            "image",
            "keep",
            "visibility_status",
            "visibility_reason",
            "visibility_note",
            "recipe_version",
            "exposure_ev",
            "contrast",
            "preprocess_reason",
            "auto_suggested",
            "human_accepted",
            "quality_status",
            "p50_luminance",
            "dark_pixel_ratio",
            "bright_pixel_ratio",
        ]
        with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for tree_id, state in sorted(self.review_state.items()):
                for annotation in state.get("annotations", []):
                    preprocess = annotation.get("preprocess", {})
                    visibility = annotation.get("visibility", {})
                    quality = preprocess.get("quality_before", {})
                    writer.writerow(
                        {
                            "tree_id": tree_id,
                            "species": state.get("species", ""),
                            "review_status": state.get("status", ""),
                            "image": annotation.get("image", ""),
                            "keep": annotation.get("keep", False),
                            "visibility_status": visibility.get("status", "clear"),
                            "visibility_reason": visibility.get("reason", "none"),
                            "visibility_note": visibility.get("note", ""),
                            "recipe_version": preprocess.get("recipe_version", "photo_v1"),
                            "exposure_ev": preprocess.get("exposure_ev", 0.0),
                            "contrast": preprocess.get("contrast", 1.0),
                            "preprocess_reason": preprocess.get("reason", "normal"),
                            "auto_suggested": preprocess.get("auto_suggested", False),
                            "human_accepted": preprocess.get("human_accepted", False),
                            "quality_status": quality.get("quality_status", ""),
                            "p50_luminance": quality.get("p50_luminance", ""),
                            "dark_pixel_ratio": quality.get("dark_pixel_ratio", ""),
                            "bright_pixel_ratio": quality.get("bright_pixel_ratio", ""),
                        }
                    )
        temporary.replace(manifest_path)

    def _load_checkpoints(self) -> dict[str, dict[str, Any]]:
        if not self.checkpoint_path.is_file():
            return {}
        return json.loads(self.checkpoint_path.read_text(encoding="utf-8"))

    def _save_checkpoints(self) -> None:
        temporary = self.checkpoint_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(self.checkpoints, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(self.checkpoint_path)

    def list_species(self) -> list[str]:
        return sorted({tree.species for tree in self.trees}, key=str.casefold)

    def species_info(self, species: str) -> str:
        records = [tree for tree in self.trees if tree.species == species]
        completed = sum(
            self.review_state.get(tree.tree_id, {}).get("status") == "accepted"
            for tree in records
        )
        rejected = sum(
            self.review_state.get(tree.tree_id, {}).get("status") == "rejected"
            for tree in records
        )
        return (
            f"何文田 VMMS 轨迹 50 m 走廊\n"
            f"树种：{species}\n"
            f"树点：{len(records)}\n"
            f"已保存：{completed}\n"
            f"已拒绝：{rejected}\n"
            "来源：香港 CSDI / 公园树木清单与本地 Ladybug 全景影像"
        )

    def start(self, species: str, target_count: int) -> dict[str, Any]:
        with self._lock:
            records = [tree for tree in self.trees if tree.species == species]
            if not records:
                raise ValueError(f"Unknown He Wentian species: {species}")
            pending = [
                tree
                for tree in records
                if self.review_state.get(tree.tree_id, {}).get("status") not in {"accepted", "rejected"}
            ]
            checkpoint = self.checkpoints.get(species)
            resume_tree_id = checkpoint.get("tree_id") if checkpoint else None
            if resume_tree_id:
                pending.sort(key=lambda tree: 0 if tree.tree_id == resume_tree_id else 1)
            self.species = species
            self.target_count = min(target_count, len(pending))
            self.queue = pending[: self.target_count]
            self.cursor = 0
            self.reviewed_count = 0
            self.rejected_count = 0
            self.failed_count = 0
            self.message = (
                f"已载入 {len(self.queue)} 棵待标注树。"
                if self.queue
                else "该树种当前没有未处理树点。"
            )
            return {
                "status": "started" if self.queue else "completed",
                "species": species,
                "target_count": self.target_count,
                "records": len(records),
                "pending_records": len(pending),
                "route_id": HEWENTIAN_STREAM_ID,
                "source": "local_vmms",
                "resume_tree_id": resume_tree_id,
            }

    def status(self) -> dict[str, Any]:
        return {
            "species": self.species,
            "target_count": self.target_count,
            "queued_count": len(self.queue),
            "reviewed_count": self.reviewed_count,
            "rejected_count": self.rejected_count,
            "failed_count": self.failed_count,
            "remaining_count": max(0, len(self.queue) - self.cursor),
            "status": "running" if self.cursor < len(self.queue) else "completed",
            "message": self.message,
            "api_calls": dict(self.local_api_counts),
            "api_counts": dict(self.local_api_counts),
            "api_error_counts": {},
            "source_mode": "local_vmms_hewentian",
            "source_frame_count": len(self.frames),
            "source_tree_count": len(self.trees),
            "source_species_count": len(self.list_species()),
            "checkpoint_tree_id": self.checkpoints.get(self.species or "", {}).get("tree_id"),
        }

    @staticmethod
    def _bearing(dx: float, dy: float) -> float:
        return (math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0

    @staticmethod
    def _signed_angle(angle: float) -> float:
        return (angle + 180.0) % 360.0 - 180.0

    def _candidate_views(self, tree: TreeRecord) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        for frame in self.frames:
            dx = tree.easting - frame.easting
            dy = tree.northing - frame.northing
            distance = math.hypot(dx, dy)
            if distance > VIEW_MAX_RANGE_M:
                continue
            target_bearing = self._bearing(dx, dy)
            relative_bearing = self._signed_angle(target_bearing - frame.heading_deg)
            candidates.append(
                {
                    "frame": frame,
                    "distance_m": distance,
                    "target_bearing_deg": target_bearing,
                    "relative_bearing_deg": relative_bearing,
                }
            )
        if not candidates:
            raise RuntimeError(f"No panorama frame within {VIEW_MAX_RANGE_M:g} m of {tree.tree_id}")

        nearest = min(candidates, key=lambda item: item["distance_m"])
        side = 1.0 if nearest["relative_bearing_deg"] >= 0 else -1.0
        desired_angles = [35.0 * side, 90.0 * side, 145.0 * side]
        selected: list[dict[str, Any]] = []
        for desired in desired_angles:
            ranked = sorted(
                candidates,
                key=lambda item: (
                    abs(self._signed_angle(item["relative_bearing_deg"] - desired))
                    + 0.12 * item["distance_m"]
                ),
            )
            choice = next(
                (
                    item
                    for item in ranked
                    if all(
                        math.hypot(
                            item["frame"].easting - previous["frame"].easting,
                            item["frame"].northing - previous["frame"].northing,
                        )
                        >= VIEW_MIN_SEPARATION_M
                        for previous in selected
                    )
                ),
                None,
            )
            if choice is None:
                choice = next(
                    (item for item in ranked if item["frame"].frame_id not in {p["frame"].frame_id for p in selected}),
                    ranked[0],
                )
            selected.append(choice)
        return selected

    def _crop_view(
        self,
        tree: TreeRecord,
        candidate: dict[str, Any],
        view_index: int,
    ) -> tuple[str, dict[str, Any]]:
        frame: FrameRecord = candidate["frame"]
        source = assert_inside(VMMS_ROOT / frame.source_image_relpath, VMMS_ROOT)
        if not source.is_file():
            raise FileNotFoundError(source)
        tree_cache = assert_inside(VIEW_CACHE_DIR / safe_name(tree.tree_id), VIEW_CACHE_DIR)
        tree_cache.mkdir(parents=True, exist_ok=True)
        filename = f"view_{view_index + 1}_frame_{frame.frame_id}.jpg"
        output = assert_inside(tree_cache / filename, VIEW_CACHE_DIR)

        with Image.open(source) as panorama:
            width, height = panorama.size
            relative = float(candidate["relative_bearing_deg"])
            center_fraction = 0.5 + (
                PANORAMA_X_SIGN * relative + PANORAMA_YAW_OFFSET_DEG
            ) / 360.0
            center_x = (center_fraction % 1.0) * width
            crop_size = min(int(round(width * CROP_FOV_DEG / 360.0)), int(height * 0.75))
            crop_size = max(512, crop_size)
            top = max(0, min(height - crop_size, int(round(height * 0.45 - crop_size / 2))))
            left = int(round(center_x - crop_size / 2))
            right = left + crop_size
            crop = Image.new("RGB", (crop_size, crop_size))
            if left < 0:
                first_width = -left
                crop.paste(panorama.crop((width + left, top, width, top + crop_size)), (0, 0))
                crop.paste(panorama.crop((0, top, right, top + crop_size)), (first_width, 0))
            elif right > width:
                first_width = width - left
                crop.paste(panorama.crop((left, top, width, top + crop_size)), (0, 0))
                crop.paste(panorama.crop((0, top, right - width, top + crop_size)), (first_width, 0))
            else:
                crop.paste(panorama.crop((left, top, right, top + crop_size)), (0, 0))
            if crop.size != (CROP_OUTPUT_SIZE, CROP_OUTPUT_SIZE):
                crop = crop.resize((CROP_OUTPUT_SIZE, CROP_OUTPUT_SIZE), Image.Resampling.LANCZOS)
            if not output.exists():
                crop.save(output, format="JPEG", quality=92, optimize=True)

        metadata = {
            "filename": filename,
            "frame_id": frame.frame_id,
            "seq_id": frame.seq_id,
            "source_panorama_relpath": frame.source_image_relpath,
            "utc_datetime": frame.utc_datetime,
            "hong_kong_datetime": frame.hong_kong_datetime,
            "lon": frame.longitude,
            "lat": frame.latitude,
            "camera_hk80_easting": frame.easting,
            "camera_hk80_northing": frame.northing,
            "camera_heading_deg": frame.heading_deg,
            "heading": candidate["target_bearing_deg"],
            "relative_bearing_deg": candidate["relative_bearing_deg"],
            "pitch": 0.0,
            "fov": CROP_FOV_DEG,
            "distance_m": candidate["distance_m"],
            "date": frame.hong_kong_datetime,
            "crop": {
                "panorama_width": width,
                "panorama_height": height,
                "center_x_px": center_x,
                "top_px": top,
                "crop_size_px": crop_size,
                "output_size_px": CROP_OUTPUT_SIZE,
                "panorama_x_sign": PANORAMA_X_SIGN,
                "panorama_yaw_offset_deg": PANORAMA_YAW_OFFSET_DEG,
            },
        }
        return f"/views/{safe_name(tree.tree_id)}/{filename}", metadata

    def _prepare_sample(self, tree: TreeRecord) -> ReviewSample:
        selected = self._candidate_views(tree)
        images: list[str] = []
        candidates: list[dict[str, Any]] = []
        for index, candidate in enumerate(selected):
            image_url, metadata = self._crop_view(tree, candidate, index)
            images.append(image_url)
            candidates.append(metadata)
        tree_dict = asdict(tree)
        metadata = {
            "tree_id": tree.tree_id,
            "species": tree.species,
            "tree": tree_dict,
            "shots": candidates,
            "source_mode": "local_vmms_hewentian",
            "orientation_assumption": (
                "Ladybug panorama horizontal center is vehicle forward; "
                f"x_sign={PANORAMA_X_SIGN:g}; yaw_offset={PANORAMA_YAW_OFFSET_DEG:g} deg"
            ),
        }
        cache_metadata = assert_inside(
            VIEW_CACHE_DIR / safe_name(tree.tree_id) / "metadata.json", VIEW_CACHE_DIR
        )
        cache_metadata.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        warnings = [
            "当前裁剪使用车辆航向与全景中心对齐的初始假设，请确认目标树位于裁剪中心附近。",
            f"树点距采集轨迹最近距离约 {tree.distance_to_route_m:.1f} m。",
        ]
        checkpoint = self.checkpoints.get(tree.species, {})
        draft_annotations = (
            checkpoint.get("annotations", [])
            if checkpoint.get("tree_id") == tree.tree_id
            else []
        )
        return ReviewSample(
            tree.tree_id,
            tree.species,
            images,
            candidates,
            tree_dict,
            warnings,
            draft_annotations,
        )

    def next_sample(self) -> ReviewSample | None:
        with self._lock:
            while self.cursor < len(self.queue):
                tree = self.queue[self.cursor]
                self.cursor += 1
                try:
                    sample = self._prepare_sample(tree)
                    self.message = f"已载入 {tree.tree_id}。"
                    return sample
                except Exception as error:
                    self.failed_count += 1
                    self.message = f"跳过 {tree.tree_id}: {error}"
            self.message = "当前任务已经完成。"
            return None

    @staticmethod
    def _annotation_lines(annotation: Any) -> list[str]:
        lines: list[str] = []
        for polygon in annotation.polygons:
            values = [str(polygon.class_id)]
            for x, y in polygon.points:
                values.extend((f"{x:.8f}", f"{y:.8f}"))
            lines.append(" ".join(values))
        for box in annotation.boxes:
            lines.append(
                f"{box.class_id} {box.x_center:.8f} {box.y_center:.8f} "
                f"{box.width:.8f} {box.height:.8f}"
            )
        return lines

    @staticmethod
    def _resolve_view_image(image_url: str) -> Path:
        if not image_url.startswith("/views/"):
            raise ValueError(f"Unsupported image path: {image_url}")
        relative = image_url.removeprefix("/views/")
        source = assert_inside(VIEW_CACHE_DIR / relative, VIEW_CACHE_DIR)
        if not source.is_file():
            raise FileNotFoundError(source)
        return source

    def analyze_image(self, image_url: str) -> dict[str, Any]:
        source = self._resolve_view_image(image_url)
        with Image.open(source) as image:
            metrics = analyze_image_quality(image)
        return {
            "image": image_url,
            "analysis_scope": "target_center_roi",
            "metrics": metrics,
            "suggested_preprocess": suggest_recipe(metrics),
        }

    def submit(self, payload: SubmitRequest) -> dict[str, Any]:
        with self._lock:
            tree = self.trees_by_id.get(payload.tree_id)
            if tree is None:
                raise ValueError(f"Unknown tree: {payload.tree_id}")
            destination = assert_inside(
                ANNOTATION_DIR / safe_name(payload.species) / safe_name(payload.tree_id),
                ANNOTATION_DIR,
            )
            destination.mkdir(parents=True, exist_ok=True)
            saved_images: list[str] = []
            saved_enhanced_images: list[str] = []
            annotation_records: list[dict[str, Any]] = []
            for annotation in payload.annotations:
                source = self._resolve_view_image(annotation.image)
                record = annotation.model_dump()
                preprocess = annotation.preprocess
                with Image.open(source) as source_image:
                    source_rgb = source_image.convert("RGB")
                    quality_before = analyze_image_quality(source_rgb)
                    enhanced_image = apply_photo_recipe(
                        source_rgb,
                        exposure_ev=preprocess.exposure_ev,
                        contrast=preprocess.contrast,
                    )
                    quality_after = analyze_image_quality(enhanced_image)
                record["preprocess"]["quality_before"] = quality_before
                record["preprocess"]["quality_after"] = quality_after
                annotation_records.append(record)

                if not annotation.keep:
                    continue
                target_image = destination / source.name
                shutil.copy2(source, target_image)
                label_path = target_image.with_suffix(".txt")
                label_text = "\n".join(self._annotation_lines(annotation)) + (
                    "\n" if annotation.polygons or annotation.boxes else ""
                )
                label_path.write_text(label_text, encoding="utf-8")
                saved_images.append(target_image.name)

                if recipe_changes_pixels(preprocess.exposure_ev, preprocess.contrast):
                    enhanced_dir = destination / "enhanced"
                    enhanced_dir.mkdir(parents=True, exist_ok=True)
                    enhanced_name = f"{source.stem}_enhanced_photo_v1.jpg"
                    enhanced_path = enhanced_dir / enhanced_name
                    enhanced_image.save(
                        enhanced_path,
                        format="JPEG",
                        quality=92,
                        optimize=True,
                    )
                    enhanced_path.with_suffix(".txt").write_text(
                        label_text, encoding="utf-8"
                    )
                    saved_enhanced_images.append(f"enhanced/{enhanced_name}")
            cache_metadata_path = assert_inside(
                VIEW_CACHE_DIR / safe_name(tree.tree_id) / "metadata.json", VIEW_CACHE_DIR
            )
            cache_metadata = json.loads(cache_metadata_path.read_text(encoding="utf-8"))
            cache_metadata.update(
                {
                    "saved_at_utc": datetime.now(timezone.utc).isoformat(),
                    "saved_images": saved_images,
                    "saved_enhanced_images": saved_enhanced_images,
                    "annotations": annotation_records,
                    "annotation_coordinate_space": "derived_square_crop_normalized_0_1",
                    "preprocess_policy": {
                        "recipe_version": "photo_v1",
                        "scope": "per_image_non_destructive",
                        "training_split_group": "tree_id",
                        "original_is_preserved": True,
                    },
                }
            )
            (destination / "metadata.json").write_text(
                json.dumps(cache_metadata, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            self.review_state[tree.tree_id] = {
                "status": "accepted",
                "species": tree.species,
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                "saved_images": saved_images,
                "saved_enhanced_images": saved_enhanced_images,
                "annotations": annotation_records,
                "preprocess_recipe_version": "photo_v1",
            }
            self._save_review_state()
            self._save_preprocess_manifest()
            checkpoint = self.checkpoints.get(tree.species)
            if checkpoint and checkpoint.get("tree_id") == tree.tree_id:
                self.checkpoints.pop(tree.species, None)
                self._save_checkpoints()
            self.reviewed_count += 1
            return {
                "status": "saved",
                "tree_id": tree.tree_id,
                "saved_images": len(saved_images),
                "saved_enhanced_images": len(saved_enhanced_images),
                "destination": str(destination),
            }

    def checkpoint(self, payload: CheckpointRequest) -> dict[str, Any]:
        with self._lock:
            tree = self.trees_by_id.get(payload.tree_id)
            if tree is None:
                raise ValueError(f"Unknown tree: {payload.tree_id}")
            if tree.species != payload.species:
                raise ValueError(
                    f"Species mismatch for {payload.tree_id}: "
                    f"expected {tree.species}, received {payload.species}"
                )
            self.checkpoints[payload.species] = {
                "tree_id": payload.tree_id,
                "species": payload.species,
                "annotations": [item.model_dump() for item in payload.annotations],
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            self._save_checkpoints()
            self.message = f"已保存 {payload.tree_id} 的标注草稿，可切换树种。"
            return {
                "status": "checkpoint_saved",
                "tree_id": payload.tree_id,
                "species": payload.species,
                "annotation_views": len(payload.annotations),
            }

    def reject(
        self, tree_id: str, annotations: list[ImageAnnotation] | None = None
    ) -> dict[str, Any]:
        with self._lock:
            tree = self.trees_by_id.get(tree_id)
            if tree is None:
                raise ValueError(f"Unknown tree: {tree_id}")
            annotation_records: list[dict[str, Any]] = []
            for annotation in annotations or []:
                record = annotation.model_dump()
                source = self._resolve_view_image(annotation.image)
                with Image.open(source) as source_image:
                    source_rgb = source_image.convert("RGB")
                    record["preprocess"]["quality_before"] = analyze_image_quality(
                        source_rgb
                    )
                    record["preprocess"]["quality_after"] = analyze_image_quality(
                        apply_photo_recipe(
                            source_rgb,
                            exposure_ev=annotation.preprocess.exposure_ev,
                            contrast=annotation.preprocess.contrast,
                        )
                    )
                annotation_records.append(record)
            self.review_state[tree_id] = {
                "status": "rejected",
                "species": tree.species,
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                "annotations": annotation_records,
            }
            self._save_review_state()
            self._save_preprocess_manifest()
            checkpoint = self.checkpoints.get(tree.species)
            if checkpoint and checkpoint.get("tree_id") == tree.tree_id:
                self.checkpoints.pop(tree.species, None)
                self._save_checkpoints()
            self.rejected_count += 1
            return {"status": "rejected", "tree_id": tree_id}


manager = TaskManager()
