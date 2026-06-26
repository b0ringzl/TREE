from __future__ import annotations

import asyncio
import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import DATASET_DIR, MAX_CONCURRENT_DOWNLOADS, NEAR_REJECT_RADIUS_M, TEMP_DIR, assert_inside_root, safe_name
from .data_access import TreeRecord, load_tree_records
from .google_streetview import StreetViewClient
from .rejection_store import RejectionStore
from .schemas import SubmitRequest


@dataclass
class ReviewSample:
    tree_id: str
    species: str
    images: list[str]
    candidates: list[dict]
    tree: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass
class TaskState:
    species: str | None = None
    target_count: int = 0
    queued_count: int = 0
    reviewed_count: int = 0
    rejected_count: int = 0
    failed_count: int = 0
    status: str = "idle"
    message: str = ""
    producer: asyncio.Task | None = None
    queue: asyncio.Queue[ReviewSample] = field(default_factory=asyncio.Queue)
    pending_reject_deletions: dict[str, dict] = field(default_factory=dict)


class TaskManager:
    def __init__(self) -> None:
        self.state = TaskState()
        self.client = StreetViewClient()
        self.rejections = RejectionStore()
        self._lock = asyncio.Lock()
        self.local_api_counts: dict[str, int] = {}

    def record_api_call(self, name: str) -> None:
        self.local_api_counts[name] = self.local_api_counts.get(name, 0) + 1

    async def configure_api_key(self, api_key: str) -> dict:
        value = api_key.strip()
        if not value:
            raise RuntimeError("Google Maps API Key cannot be empty")
        self.client.api_key = value
        await self.client.validate_api_key()
        return {"status": "ok", "google_maps_api_key_available": True, "google_maps_api_key_source": "session"}

    async def start(self, species: str, target_count: int) -> dict:
        async with self._lock:
            await self.client.validate_api_key()
            if self.state.producer and not self.state.producer.done():
                self.state.producer.cancel()
            self.state = TaskState(species=species, target_count=target_count, status="running")
            records = load_tree_records(species)
            records, deprioritized_count = self._prioritize_records(species, records)
            self.state.producer = asyncio.create_task(self._produce(species, target_count, records))
            return {
                "status": "started",
                "species": species,
                "target_count": target_count,
                "records": len(records),
                "deprioritized_near_rejections": deprioritized_count,
                "near_reject_radius_m": NEAR_REJECT_RADIUS_M,
            }

    def _prioritize_records(self, species: str, records: list[TreeRecord]) -> tuple[list[TreeRecord], int]:
        ranked = []
        deprioritized_count = 0
        for index, record in enumerate(records):
            exact_rejected = self.rejections.is_tree_rejected(species, record)
            penalty = self.rejections.tree_proximity_penalty(species, record, NEAR_REJECT_RADIUS_M)
            nearby = bool(penalty["nearby"])
            if nearby and not exact_rejected:
                deprioritized_count += 1
            nearest = float(penalty["nearest_rejected_m"])
            ranked.append(
                (
                    exact_rejected,
                    nearby,
                    -nearest if nearest != float("inf") else 0.0,
                    index,
                    record,
                )
            )
        ranked.sort(key=lambda item: item[:4])
        return [item[4] for item in ranked], deprioritized_count

    def _sample_priority(self, species: str, sample: ReviewSample, index: int) -> tuple:
        exact_rejected = bool(sample.tree and self.rejections.is_tree_rejected(species, sample.tree))
        penalty = self.rejections.tree_proximity_penalty(species, sample.tree or {"lat": 0, "lon": 0}, NEAR_REJECT_RADIUS_M)
        nearest = float(penalty["nearest_rejected_m"])
        return (
            exact_rejected,
            bool(penalty["nearby"]),
            -nearest if nearest != float("inf") else 0.0,
            index,
        )

    def _reprioritize_pending(self) -> None:
        species = self.state.species
        if not species:
            return
        samples: list[ReviewSample] = []
        while True:
            try:
                samples.append(self.state.queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        ranked = []
        for index, sample in enumerate(samples):
            if sample.tree and self.rejections.is_tree_rejected(species, sample.tree):
                continue
            ranked.append((self._sample_priority(species, sample, index), sample))
        ranked.sort(key=lambda item: item[0])
        for _, sample in ranked:
            self.state.queue.put_nowait(sample)

    async def _produce(self, species: str, target_count: int, records: list[TreeRecord]) -> None:
        cursor = 0
        cursor_lock = asyncio.Lock()
        try:
            async def worker() -> None:
                nonlocal cursor
                while True:
                    async with cursor_lock:
                        if self.state.queued_count >= target_count or cursor >= len(records):
                            return
                        record = records[cursor]
                        cursor += 1
                    if self.rejections.is_tree_rejected(species, record):
                        continue
                    try:
                        sample = await self._prepare_sample(species, record)
                    except Exception as exc:
                        self.state.failed_count += 1
                        self.state.message = f"Skipped {record.tree_id}: {exc}"
                        continue
                    if sample:
                        async with cursor_lock:
                            if self.state.queued_count >= target_count:
                                overflow = assert_inside_root(TEMP_DIR / sample.tree_id)
                                if overflow.exists():
                                    shutil.rmtree(overflow)
                                return
                            await self.state.queue.put(sample)
                            self.state.queued_count += 1

            workers = [asyncio.create_task(worker()) for _ in range(MAX_CONCURRENT_DOWNLOADS)]
            await asyncio.gather(*workers)
            if self.state.queued_count < target_count:
                self.state.status = "completed"
                self.state.message = "Coordinate table exhausted before reaching target count."
            else:
                self.state.status = "paused"
                self.state.message = "Target queue size reached; waiting for review."
        except asyncio.CancelledError:
            self.state.status = "cancelled"
            raise
        except Exception as exc:
            self.state.status = "error"
            self.state.message = str(exc)

    async def _prepare_sample(self, species: str, record: TreeRecord) -> ReviewSample | None:
        candidates = await self.client.discover_candidates(record.lat, record.lon, record.height_m, limit=12)
        candidates = [item for item in candidates if not self.rejections.is_view_rejected(species, item)]
        if not candidates:
            return None
        shots = self.client.build_angle_shots(candidates, record.height_m)
        shots = [item for item in shots if not self.rejections.is_view_rejected(species, item)]
        if len(shots) < 3:
            return None

        tree_dir = assert_inside_root(TEMP_DIR / record.tree_id)
        if tree_dir.exists():
            shutil.rmtree(tree_dir)
        tree_dir.mkdir(parents=True, exist_ok=True)

        image_paths: list[str] = []
        for shot in shots:
            output = tree_dir / shot.filename
            await self.client.download_image(shot, output)
            image_paths.append(f"/temp/{record.tree_id}/{shot.filename}")

        metadata = {
            "tree_id": record.tree_id,
            "species": species,
            "tree": asdict(record),
            "streetview_candidates": [asdict(item) for item in candidates],
            "shots": [asdict(item) for item in shots],
            "warnings": self._shot_warnings([asdict(item) for item in shots]),
        }
        (tree_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        return ReviewSample(
            tree_id=record.tree_id,
            species=species,
            images=image_paths,
            candidates=[asdict(item) for item in shots],
            tree=asdict(record),
            warnings=metadata["warnings"],
        )

    def _shot_warnings(self, shots: list[dict]) -> list[str]:
        if len(shots) < 3:
            return ["Less than three feature views were generated."]

        headings = [float(item["heading"]) for item in shots]
        pitches = [float(item["pitch"]) for item in shots]
        fovs = [float(item["fov"]) for item in shots]
        distances = [float(item["distance_m"]) for item in shots]

        heading_spread = self._circular_heading_spread(headings)
        pitch_spread = max(pitches) - min(pitches)
        fov_spread = max(fovs) - min(fovs)
        distance_spread = max(distances) - min(distances)

        warnings: list[str] = []
        if heading_spread < 45:
            warnings.append(f"View angles are still close together ({heading_spread:.1f} deg minimum spread); obstruction risk is high.")
        if pitch_spread > 55:
            warnings.append(f"Pitch spread is {pitch_spread:.1f} deg; verify all views keep the same crown target.")
        if fov_spread > 45:
            warnings.append(f"FOV spread is {fov_spread:.1f} deg; views may frame the tree at very different scales.")
        if distance_spread > 18:
            warnings.append(f"Street View distance spread is {distance_spread:.1f} m; verify multi-view continuity.")
        return warnings

    def _circular_heading_spread(self, headings: list[float]) -> float:
        normalized = sorted(value % 360 for value in headings)
        if len(normalized) <= 1:
            return 0.0
        gaps = [
            (normalized[(index + 1) % len(normalized)] - normalized[index]) % 360
            for index in range(len(normalized))
        ]
        return 360 - max(gaps)

    def _annotation_label_lines(self, annotation) -> list[str]:
        lines: list[str] = []
        polygons = [(polygon.class_id, polygon.points) for polygon in annotation.polygons]
        if not polygons:
            for box in annotation.boxes:
                x1 = max(0.0, box.x_center - box.width / 2)
                y1 = max(0.0, box.y_center - box.height / 2)
                x2 = min(1.0, box.x_center + box.width / 2)
                y2 = min(1.0, box.y_center + box.height / 2)
                polygons.append((box.class_id, [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]))
        for class_id, points in polygons:
            coords = " ".join(f"{value:.6f}" for point in points for value in point)
            lines.append(f"{class_id} {coords}")
        return lines

    async def next_sample(self) -> ReviewSample | None:
        try:
            sample = self.state.queue.get_nowait()
        except asyncio.QueueEmpty:
            return None
        self._advance_pending_reject_deletions()
        return sample

    def status(self) -> dict:
        return {
            "species": self.state.species,
            "target_count": self.state.target_count,
            "queued_count": self.state.queued_count,
            "reviewed_count": self.state.reviewed_count,
            "rejected_count": self.state.rejected_count,
            "failed_count": self.state.failed_count,
            "pending": self.state.queue.qsize(),
            "pending_reject_deletions": len(self.state.pending_reject_deletions),
            "status": self.state.status,
            "message": self.state.message,
            "api_counts": {
                **self.client.api_counts,
                **{f"Local {key}": value for key, value in sorted(self.local_api_counts.items())},
            },
            "api_error_counts": self.client.api_error_counts,
            "rejection_db": self.rejections.stats(),
        }

    def submit(self, payload: SubmitRequest) -> dict:
        source = assert_inside_root(TEMP_DIR / safe_name(payload.tree_id))
        destination = assert_inside_root(DATASET_DIR / safe_name(payload.species) / safe_name(payload.tree_id))
        self._cancel_pending_reject_delete(payload.tree_id)
        self.rejections.unreject_tree(payload.species, payload.tree_id)
        if source.exists():
            if destination.exists():
                shutil.rmtree(destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source, destination)
        elif not destination.exists():
            raise FileNotFoundError(f"Sample not found in temp or dataset: {payload.tree_id}")

        metadata_path = destination / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        shots_by_filename = {shot.get("filename"): shot for shot in metadata.get("shots", [])}
        image_lookup = {Path(item.image).name: item for item in payload.annotations}
        kept_images = 0
        for image_file in destination.glob("*.jpg"):
            annotation = image_lookup.get(image_file.name)
            label_path = image_file.with_suffix(".txt")
            if annotation and not annotation.keep:
                shot = shots_by_filename.get(image_file.name)
                if shot:
                    self.rejections.reject_view(payload.species, payload.tree_id, shot, "dropped_view")
                image_file.unlink(missing_ok=True)
                label_path.unlink(missing_ok=True)
                continue
            kept_images += 1
            lines = []
            if annotation:
                lines = self._annotation_label_lines(annotation)
            label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        if kept_images == 0:
            self.rejections.reject_tree(
                payload.species,
                metadata.get("tree", {"tree_id": payload.tree_id, "lat": 0, "lon": 0}),
                metadata.get("shots", []),
                "all_views_dropped",
            )
            if destination.exists():
                shutil.rmtree(destination)
            if source.exists():
                shutil.rmtree(source)
            self.state.rejected_count += 1
            return {"status": "deleted", "tree_id": payload.tree_id, "reason": "all_views_dropped"}

        self.state.reviewed_count += 1
        return {"status": "saved", "path": str(destination)}

    def reject(self, tree_id: str) -> dict:
        source = assert_inside_root(TEMP_DIR / safe_name(tree_id))
        metadata = self._read_sample_metadata(source)
        dataset_sample = None
        dataset_metadata = None
        if self.state.species:
            dataset_sample = assert_inside_root(DATASET_DIR / safe_name(self.state.species) / safe_name(tree_id))
            dataset_metadata = self._read_sample_metadata(dataset_sample)
        if self.state.species and metadata:
            self.rejections.reject_tree(
                self.state.species,
                metadata.get("tree", {"tree_id": tree_id, "lat": 0, "lon": 0}),
                metadata.get("shots", []),
                "rejected_sample",
            )
        elif self.state.species and dataset_metadata:
            self.rejections.reject_tree(
                self.state.species,
                dataset_metadata.get("tree", {"tree_id": tree_id, "lat": 0, "lon": 0}),
                dataset_metadata.get("shots", []),
                "rejected_saved_sample",
            )
        self._schedule_reject_delete(tree_id)
        self._reprioritize_pending()
        self.state.rejected_count += 1
        return {"status": "pending_delete", "tree_id": tree_id, "delete_after_next_samples": 2}

    def _schedule_reject_delete(self, tree_id: str) -> None:
        self.state.pending_reject_deletions[safe_name(tree_id)] = {
            "tree_id": tree_id,
            "species": self.state.species,
            "remaining_next_samples": 2,
        }

    def _cancel_pending_reject_delete(self, tree_id: str) -> None:
        self.state.pending_reject_deletions.pop(safe_name(tree_id), None)

    def _advance_pending_reject_deletions(self) -> None:
        for key, item in list(self.state.pending_reject_deletions.items()):
            item["remaining_next_samples"] -= 1
            if item["remaining_next_samples"] <= 0:
                self._delete_sample_files(item["tree_id"], item.get("species"))
                self.state.pending_reject_deletions.pop(key, None)

    def _delete_sample_files(self, tree_id: str, species: str | None) -> None:
        temp_sample = assert_inside_root(TEMP_DIR / safe_name(tree_id))
        if temp_sample.exists():
            shutil.rmtree(temp_sample)
        if species:
            dataset_sample = assert_inside_root(DATASET_DIR / safe_name(species) / safe_name(tree_id))
            if dataset_sample.exists():
                shutil.rmtree(dataset_sample)

    def _read_sample_metadata(self, sample_dir: Path) -> dict | None:
        metadata_path = sample_dir / "metadata.json"
        if not metadata_path.exists():
            return None
        try:
            return json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None


manager = TaskManager()
