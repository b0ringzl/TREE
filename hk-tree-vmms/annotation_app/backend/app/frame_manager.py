from __future__ import annotations

import csv
import json
import math
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from .config import (
    FRAME_ANNOTATION_DIR,
    FRAME_COORDINATES,
    FRAME_LABELER_DATASET,
    FRAME_LABELER_TITLE,
    FRAME_MIN_ROUTE_DISTANCE_M_BY_STREAM,
    FRAME_PREVIEW_DIR,
    FRAME_RUNTIME_DIR,
    FRAME_SEED_CLASSES,
    FRAME_STREAM_IDS,
    NEARBY_TREES,
    VMMS_ROOT,
    assert_inside,
    ensure_dirs,
)
from .image_preprocessing import analyze_image_quality, apply_photo_recipe, suggest_recipe
from .schemas import FrameAnnotationRequest


FOXTAIL_PALM_LABEL = "Wodyetia bifurcata 狐尾椰子"
NORFOLK_ISLAND_PINE_LABEL = "Araucaria heterophylla 异叶南洋杉"
ALSTONIA_LABEL = "Alstonia scholaris 糖膠樹"
CRAPE_MYRTLE_LABEL = "Lagerstroemia indica 紫薇"
TRAVELLERS_PALM_LABEL = "Ravenala madagascariensis 旅人蕉"
BANYAN_LABEL = "榕树"
SPECIES_LABEL_ALIASES = {
    "Archontophoenix alexandrae 假檳榔": FOXTAIL_PALM_LABEL,
    "Archontophoenix alexandrae 假槟榔": FOXTAIL_PALM_LABEL,
    "Araucaria heterophylla异叶南阳杉": NORFOLK_ISLAND_PINE_LABEL,
    "Araucaria heterophylla 异叶南阳杉": NORFOLK_ISLAND_PINE_LABEL,
    "Araucaria heterophylla异叶南洋杉": NORFOLK_ISLAND_PINE_LABEL,
    "Alstonia scholaris": ALSTONIA_LABEL,
    "Lagerstroemia indica": CRAPE_MYRTLE_LABEL,
    "紫薇": CRAPE_MYRTLE_LABEL,
    "Ravenala madagascariensis": TRAVELLERS_PALM_LABEL,
    "旅人蕉": TRAVELLERS_PALM_LABEL,
    "Ficus benjamina": BANYAN_LABEL,
    "Ficus benjamina 垂葉榕": BANYAN_LABEL,
    "Ficus benjamina 垂叶榕": BANYAN_LABEL,
    "Ficus microcarpa": BANYAN_LABEL,
    "Ficus microcarpa 榕樹(細葉榕)": BANYAN_LABEL,
    "Ficus microcarpa 榕树(细叶榕)": BANYAN_LABEL,
    "Ficus microcarpa 細葉榕": BANYAN_LABEL,
    "Ficus microcarpa 细叶榕": BANYAN_LABEL,
    "垂葉榕": BANYAN_LABEL,
    "垂叶榕": BANYAN_LABEL,
    "細葉榕": BANYAN_LABEL,
    "细叶榕": BANYAN_LABEL,
    "榕樹": BANYAN_LABEL,
}


def normalize_species_name(value: str) -> str:
    """Normalize corrected human labels without rewriting source inventory data."""

    normalized = str(value).strip()
    if normalized.casefold().startswith(("ficus benjamina", "ficus microcarpa")):
        return BANYAN_LABEL
    if any(name in normalized for name in ("細葉榕", "细叶榕", "垂葉榕", "垂叶榕")):
        return BANYAN_LABEL
    return SPECIES_LABEL_ALIASES.get(normalized, normalized)


@dataclass(frozen=True)
class PanoramaFrame:
    frame_key: str
    stream_id: str
    frame_id: str
    seq_id: int
    utc_datetime: str
    hong_kong_datetime: str
    source_image_relpath: str
    easting: float
    northing: float
    latitude: float
    longitude: float
    heading_deg: float
    route_distance_m: float


class FrameAnnotationManager:
    """Image-first annotation workflow independent of inventory tree coordinates."""

    def __init__(self) -> None:
        ensure_dirs()
        self.frames = self._load_frames()
        self.frames_by_id = {frame.frame_key: frame for frame in self.frames}
        self.class_names_path = FRAME_ANNOTATION_DIR / "classes.json"
        self.species = self._load_species()
        self._save_classes()
        self.review_state_path = FRAME_RUNTIME_DIR / "frame_review_state.json"
        self.drafts_path = FRAME_RUNTIME_DIR / "frame_drafts.json"
        self.session_path = FRAME_RUNTIME_DIR / "session.json"
        self.review_state = self._load_json(self.review_state_path)
        self.drafts = self._load_json(self.drafts_path)
        session = self._load_json(self.session_path)
        self.spacing_m = float(session.get("spacing_m", 5.0))
        self.anchor_indices = self._build_anchor_indices(self.spacing_m)
        self.current_index = min(
            max(0, int(session.get("current_index", 0))),
            max(0, len(self.anchor_indices) - 1),
        )
        self._lock = threading.RLock()

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        if not path.is_file():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _save_json(path: Path, value: dict[str, Any]) -> None:
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(path)

    def _load_frames(self) -> list[PanoramaFrame]:
        grouped: dict[str, list[dict[str, str]]] = {
            stream_id: [] for stream_id in FRAME_STREAM_IDS
        }
        with FRAME_COORDINATES.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["stream_id"] in grouped:
                    grouped[row["stream_id"]].append(row)
        frames: list[PanoramaFrame] = []
        multi_stream = len(FRAME_STREAM_IDS) > 1
        for stream_id in FRAME_STREAM_IDS:
            rows = sorted(grouped.get(stream_id, []), key=lambda row: int(row["seq_id"]))
            route_distance = 0.0
            previous: tuple[float, float] | None = None
            for row in rows:
                point = (float(row["hk80_easting"]), float(row["hk80_northing"]))
                if previous is not None:
                    route_distance += math.dist(previous, point)
                previous = point
                frame_id = row["frame_id"]
                frames.append(
                    PanoramaFrame(
                        frame_key=(f"{stream_id}__{frame_id}" if multi_stream else frame_id),
                        stream_id=stream_id,
                        frame_id=frame_id,
                        seq_id=int(row["seq_id"]),
                        utc_datetime=row["utc_datetime"],
                        hong_kong_datetime=row["hong_kong_datetime"],
                        source_image_relpath=row["source_image_relpath"],
                        easting=point[0],
                        northing=point[1],
                        latitude=float(row["wgs84_latitude"]),
                        longitude=float(row["wgs84_longitude"]),
                        heading_deg=float(row["heading_deg"]) % 360.0,
                        route_distance_m=route_distance,
                    )
                )
        if not frames:
            raise RuntimeError(f"No frames found for {', '.join(FRAME_STREAM_IDS)}")
        return frames

    def _load_species(self) -> list[str]:
        names: list[str] = []
        seen: set[str] = set()

        def add_name(value: str) -> None:
            normalized = normalize_species_name(value)
            if normalized and normalized not in seen:
                names.append(normalized)
                seen.add(normalized)

        # Preserve the existing class order so numeric YOLO class IDs remain stable.
        if self.class_names_path.is_file():
            for name in json.loads(self.class_names_path.read_text(encoding="utf-8")):
                add_name(str(name))

        # Seed files define the shared numeric class-ID order across route projects.
        # Append missing seed entries in that exact order, including for existing projects.
        if FRAME_SEED_CLASSES.is_file():
            for name in json.loads(FRAME_SEED_CLASSES.read_text(encoding="utf-8")):
                add_name(str(name))

        discovered: set[str] = set()
        if NEARBY_TREES.is_file():
            with NEARBY_TREES.open("r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    name = normalize_species_name(row.get("species_name", ""))
                    if name:
                        discovered.add(name)
        for name in sorted(discovered, key=str.casefold):
            add_name(name)
        add_name("Unknown / 待定")
        return names

    def _save_classes(self) -> None:
        self.species = list(
            dict.fromkeys(normalize_species_name(name) for name in self.species if name)
        )
        self.class_names_path.write_text(
            json.dumps(self.species, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _build_anchor_indices(self, spacing_m: float) -> list[int]:
        indices: list[int] = []
        ordered_streams = list(dict.fromkeys(frame.stream_id for frame in self.frames))
        for stream_id in ordered_streams:
            minimum = FRAME_MIN_ROUTE_DISTANCE_M_BY_STREAM.get(stream_id, 0.0)
            eligible = [
                index
                for index, frame in enumerate(self.frames)
                if frame.stream_id == stream_id and frame.route_distance_m >= minimum
            ]
            if not eligible:
                continue
            if spacing_m <= 0:
                indices.extend(eligible)
                continue
            indices.append(eligible[0])
            last_distance = self.frames[eligible[0]].route_distance_m
            for index in eligible[1:]:
                frame = self.frames[index]
                if frame.route_distance_m - last_distance >= spacing_m:
                    indices.append(index)
                    last_distance = frame.route_distance_m
            if eligible[-1] not in indices:
                indices.append(eligible[-1])
        indices.sort()
        return indices

    def _save_session(self) -> None:
        self._save_json(
            self.session_path,
            {
                "spacing_m": self.spacing_m,
                "current_index": self.current_index,
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )

    def start(self, spacing_m: float) -> dict[str, Any]:
        with self._lock:
            previous_frame_id = None
            if self.anchor_indices:
                previous_frame_id = self.frames[
                    self.anchor_indices[self.current_index]
                ].frame_key
            self.spacing_m = float(spacing_m)
            self.anchor_indices = self._build_anchor_indices(self.spacing_m)
            matching_index = next(
                (
                    index
                    for index, source_index in enumerate(self.anchor_indices)
                    if self.frames[source_index].frame_key == previous_frame_id
                ),
                None,
            )
            if matching_index is not None:
                self.current_index = matching_index
            else:
                self.current_index = next(
                (
                    index
                    for index, source_index in enumerate(self.anchor_indices)
                    if self.frames[source_index].frame_key not in self.review_state
                ),
                    0,
                )
            self._save_session()
            return self.status()

    def status(self) -> dict[str, Any]:
        sampled_ids = {
            self.frames[source_index].frame_key for source_index in self.anchor_indices
        }
        counts = {"annotated": 0, "no_tree": 0, "unusable": 0}
        for frame_id, record in self.review_state.items():
            if frame_id in sampled_ids and record.get("frame_status") in counts:
                counts[record["frame_status"]] += 1
        valuable_frame_count = 0
        tree_label_count = 0
        for frame_id in sampled_ids:
            # The UI restores drafts before older submitted records, so progress uses
            # the same precedence and treats every frame with a polygon as valuable.
            current = self.drafts.get(frame_id) or self.review_state.get(frame_id) or {}
            label_count = len(current.get("labels") or [])
            if label_count:
                valuable_frame_count += 1
                tree_label_count += label_count
        return {
            "route_id": ",".join(FRAME_STREAM_IDS),
            "stream_ids": list(FRAME_STREAM_IDS),
            "labeler_title": FRAME_LABELER_TITLE,
            "dataset": FRAME_LABELER_DATASET,
            "raw_frame_count": len(self.frames),
            "sampled_frame_count": len(self.anchor_indices),
            "spacing_m": self.spacing_m,
            "minimum_route_distance_m_by_stream": FRAME_MIN_ROUTE_DISTANCE_M_BY_STREAM,
            "current_index": self.current_index,
            "reviewed_count": sum(counts.values()),
            "counts": counts,
            "valuable_frame_count": valuable_frame_count,
            "tree_label_count": tree_label_count,
            "unlabeled_frame_count": len(self.anchor_indices) - valuable_frame_count,
            "species_count": len(self.species),
            "route_length_m": round(
                sum(
                    max(
                        (frame.route_distance_m for frame in self.frames if frame.stream_id == stream_id),
                        default=0.0,
                    )
                    for stream_id in FRAME_STREAM_IDS
                ),
                3,
            ),
            "streams": [
                {
                    "stream_id": stream_id,
                    "raw_frame_count": sum(frame.stream_id == stream_id for frame in self.frames),
                    "sampled_frame_count": sum(
                        self.frames[index].stream_id == stream_id for index in self.anchor_indices
                    ),
                    "route_length_m": round(
                        max(
                            (frame.route_distance_m for frame in self.frames if frame.stream_id == stream_id),
                            default=0.0,
                        ),
                        3,
                    ),
                }
                for stream_id in FRAME_STREAM_IDS
            ],
        }

    def _source_path(self, frame: PanoramaFrame) -> Path:
        source = assert_inside(VMMS_ROOT / frame.source_image_relpath, VMMS_ROOT)
        if not source.is_file():
            raise FileNotFoundError(source)
        return source

    def preview_file(self, frame_id: str) -> Path:
        frame = self.frames_by_id.get(frame_id)
        if frame is None:
            raise ValueError(f"Unknown frame: {frame_id}")
        target = FRAME_PREVIEW_DIR / frame.stream_id / f"{frame.frame_id}.jpg"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_file():
            return target
        source = self._source_path(frame)
        with Image.open(source) as image:
            preview = image.convert("RGB")
            preview.thumbnail((2048, 1024), Image.Resampling.LANCZOS)
            temporary = target.with_suffix(".tmp")
            preview.save(temporary, format="JPEG", quality=90, optimize=True)
            temporary.replace(target)
        return target

    def item(self, index: int) -> dict[str, Any]:
        with self._lock:
            if not self.anchor_indices:
                raise RuntimeError("Frame task is empty")
            index = min(max(0, int(index)), len(self.anchor_indices) - 1)
            self.current_index = index
            self._save_session()
            frame = self.frames[self.anchor_indices[index]]
            return {
                "index": index,
                "total": len(self.anchor_indices),
                "frame": asdict(frame),
                "image": f"/api/frame/image/{frame.frame_key}",
                "route_block_id": f"{frame.stream_id}_{int(frame.route_distance_m // 100):04d}",
                "review": self.review_state.get(frame.frame_key),
                "draft": self.drafts.get(frame.frame_key),
                "next_unreviewed_index": self._next_unreviewed_index(index),
            }

    def _next_unreviewed_index(self, index: int) -> int | None:
        for candidate in range(index + 1, len(self.anchor_indices)):
            frame_key = self.frames[self.anchor_indices[candidate]].frame_key
            if frame_key not in self.review_state:
                return candidate
        return None

    def analyze(self, frame_id: str) -> dict[str, Any]:
        frame = self.frames_by_id.get(frame_id)
        if frame is None:
            raise ValueError(f"Unknown frame: {frame_id}")
        with Image.open(self.preview_file(frame_id)) as image:
            metrics = analyze_image_quality(image, (0.0, 0.15, 1.0, 0.82))
        return {
            "frame_id": frame_id,
            "analysis_scope": "panorama_center_roi",
            "metrics": metrics,
            "suggested_preprocess": suggest_recipe(metrics),
        }

    def save_draft(self, payload: FrameAnnotationRequest) -> dict[str, Any]:
        with self._lock:
            if payload.frame_id not in self.frames_by_id:
                raise ValueError(f"Unknown frame: {payload.frame_id}")
            draft = payload.model_dump()
            species_changed = False
            for label in draft.get("labels", []):
                label["species"] = normalize_species_name(label.get("species", ""))
                if label["species"] and label["species"] not in self.species:
                    self.species.append(label["species"])
                    species_changed = True
            if species_changed:
                self._save_classes()
            self.drafts[payload.frame_id] = {
                **draft,
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            self._save_json(self.drafts_path, self.drafts)
            return {"status": "draft_saved", "frame_id": payload.frame_id}

    @staticmethod
    def _bbox(points: list[tuple[float, float]]) -> dict[str, float]:
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        left, right = min(xs), max(xs)
        top, bottom = min(ys), max(ys)
        return {
            "x_center": (left + right) / 2,
            "y_center": (top + bottom) / 2,
            "width": right - left,
            "height": bottom - top,
        }

    def _class_id(self, species: str) -> int:
        species = normalize_species_name(species)
        if species not in self.species:
            self.species.append(species)
            self._save_classes()
        return self.species.index(species)

    def submit(self, payload: FrameAnnotationRequest) -> dict[str, Any]:
        with self._lock:
            frame = self.frames_by_id.get(payload.frame_id)
            if frame is None:
                raise ValueError(f"Unknown frame: {payload.frame_id}")
            if payload.frame_status == "annotated" and not payload.labels:
                raise ValueError("Annotated frames require at least one tree label")
            if payload.frame_status != "annotated" and payload.labels:
                raise ValueError("No-tree and unusable frames cannot contain tree labels")

            source = self._source_path(frame)
            with Image.open(source) as image:
                image_size = {"width": image.width, "height": image.height}
            with Image.open(self.preview_file(frame.frame_key)) as preview_image:
                preview_rgb = preview_image.convert("RGB")
                quality_before = analyze_image_quality(
                    preview_rgb, (0.0, 0.15, 1.0, 0.82)
                )
                quality_after = analyze_image_quality(
                    apply_photo_recipe(
                        preview_rgb,
                        exposure_ev=payload.preprocess.exposure_ev,
                        contrast=payload.preprocess.contrast,
                    ),
                    (0.0, 0.15, 1.0, 0.82),
                )

            labels: list[dict[str, Any]] = []
            yolo_lines: list[str] = []
            for label in payload.labels:
                species = normalize_species_name(label.species)
                class_id = self._class_id(species)
                label_record = {
                    **label.model_dump(),
                    "species": species,
                    "class_id": class_id,
                    "bbox": self._bbox(label.points),
                }
                labels.append(label_record)
                values = [str(class_id)]
                for x, y in label.points:
                    values.extend((f"{x:.8f}", f"{y:.8f}"))
                yolo_lines.append(" ".join(values))

            preprocess = payload.preprocess.model_dump()
            preprocess["quality_before"] = quality_before
            preprocess["quality_after"] = quality_after
            record = {
                "frame_id": frame.frame_id,
                "frame_key": frame.frame_key,
                "stream_id": frame.stream_id,
                "frame_status": payload.frame_status,
                "labels": labels,
                "preprocess": preprocess,
                "note": payload.note,
                "frame": asdict(frame),
                "image_size": image_size,
                "source_image": str(source),
                "route_block_id": f"{frame.stream_id}_{int(frame.route_distance_m // 100):04d}",
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            }

            records_dir = FRAME_ANNOTATION_DIR / "records" / frame.stream_id
            labels_dir = FRAME_ANNOTATION_DIR / "labels" / frame.stream_id
            records_dir.mkdir(parents=True, exist_ok=True)
            labels_dir.mkdir(parents=True, exist_ok=True)
            (records_dir / f"{frame.frame_id}.json").write_text(
                json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            label_path = labels_dir / f"{frame.frame_id}.txt"
            if payload.frame_status in {"annotated", "no_tree"}:
                label_path.write_text(
                    "\n".join(yolo_lines) + ("\n" if yolo_lines else ""),
                    encoding="utf-8",
                )
            elif label_path.exists():
                label_path.unlink()

            self.review_state[frame.frame_key] = record
            self._save_json(self.review_state_path, self.review_state)
            self.drafts.pop(frame.frame_key, None)
            self._save_json(self.drafts_path, self.drafts)
            self._save_manifest()
            return {
                "status": "saved",
                "frame_id": frame.frame_id,
                "frame_key": frame.frame_key,
                "stream_id": frame.stream_id,
                "frame_status": payload.frame_status,
                "label_count": len(labels),
                "next_unreviewed_index": self._next_unreviewed_index(self.current_index),
            }

    def _save_manifest(self) -> None:
        path = FRAME_ANNOTATION_DIR / "dataset_manifest.csv"
        temporary = path.with_suffix(".tmp")
        fieldnames = [
            "frame_id",
            "frame_key",
            "stream_id",
            "frame_status",
            "tree_count",
            "source_image",
            "label_file",
            "route_distance_m",
            "route_block_id",
            "hong_kong_datetime",
            "longitude",
            "latitude",
            "training_eligible",
        ]
        with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for frame_key, record in sorted(
                self.review_state.items(),
                key=lambda item: (
                    FRAME_STREAM_IDS.index(item[1]["frame"].get("stream_id", FRAME_STREAM_IDS[0])),
                    item[1]["frame"]["seq_id"],
                ),
            ):
                frame = record["frame"]
                status = record["frame_status"]
                writer.writerow(
                    {
                        "frame_id": frame["frame_id"],
                        "frame_key": frame_key,
                        "stream_id": frame.get("stream_id", FRAME_STREAM_IDS[0]),
                        "frame_status": status,
                        "tree_count": len(record.get("labels", [])),
                        "source_image": record["source_image"],
                        "label_file": (
                            str(
                                FRAME_ANNOTATION_DIR
                                / "labels"
                                / frame.get("stream_id", FRAME_STREAM_IDS[0])
                                / f"{frame['frame_id']}.txt"
                            )
                            if status in {"annotated", "no_tree"}
                            else ""
                        ),
                        "route_distance_m": frame["route_distance_m"],
                        "route_block_id": record["route_block_id"],
                        "hong_kong_datetime": frame["hong_kong_datetime"],
                        "longitude": frame["longitude"],
                        "latitude": frame["latitude"],
                        "training_eligible": status in {"annotated", "no_tree"},
                    }
                )
        temporary.replace(path)


frame_manager = FrameAnnotationManager()
