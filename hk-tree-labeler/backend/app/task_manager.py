from __future__ import annotations

import asyncio
import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import DATASET_DIR, MAX_CONCURRENT_DOWNLOADS, TEMP_DIR, assert_inside_root, safe_name
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
    warnings: list[str] = field(default_factory=list)


@dataclass
class TaskState:
    species: str | None = None
    target_count: int = 0
    queued_count: int = 0
    reviewed_count: int = 0
    rejected_count: int = 0
    status: str = "idle"
    message: str = ""
    producer: asyncio.Task | None = None
    queue: asyncio.Queue[ReviewSample] = field(default_factory=asyncio.Queue)


class TaskManager:
    def __init__(self) -> None:
        self.state = TaskState()
        self.client = StreetViewClient()
        self.rejections = RejectionStore()
        self._lock = asyncio.Lock()
        self.local_api_counts: dict[str, int] = {}

    def record_api_call(self, name: str) -> None:
        self.local_api_counts[name] = self.local_api_counts.get(name, 0) + 1

    async def start(self, species: str, target_count: int) -> dict:
        async with self._lock:
            await self.client.validate_api_key()
            if self.state.producer and not self.state.producer.done():
                self.state.producer.cancel()
            self.state = TaskState(species=species, target_count=target_count, status="running")
            records = load_tree_records(species)
            self.state.producer = asyncio.create_task(self._produce(species, target_count, records))
            return {"status": "started", "species": species, "target_count": target_count, "records": len(records)}

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
                    sample = await self._prepare_sample(species, record)
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
        candidates = await self.client.discover_nearest_three(record.lat, record.lon, record.height_m)
        candidates = [item for item in candidates if not self.rejections.is_view_rejected(species, item)]
        if not candidates:
            return None
        shots = self.client.build_feature_shots(candidates, record.height_m)
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
        if heading_spread > 65:
            warnings.append(f"Heading spread is {heading_spread:.1f} deg; verify all views target the same tree.")
        if pitch_spread > 55:
            warnings.append(f"Pitch spread is {pitch_spread:.1f} deg; verify crown/trunk alignment.")
        if fov_spread > 45:
            warnings.append(f"FOV spread is {fov_spread:.1f} deg; closeup and overview may frame different objects.")
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

    async def next_sample(self) -> ReviewSample | None:
        try:
            return self.state.queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    def status(self) -> dict:
        return {
            "species": self.state.species,
            "target_count": self.state.target_count,
            "queued_count": self.state.queued_count,
            "reviewed_count": self.state.reviewed_count,
            "rejected_count": self.state.rejected_count,
            "pending": self.state.queue.qsize(),
            "status": self.state.status,
            "message": self.state.message,
            "api_counts": {
                **self.client.api_counts,
                **{f"Local {key}": value for key, value in sorted(self.local_api_counts.items())},
            },
            "rejection_db": self.rejections.stats(),
        }

    def submit(self, payload: SubmitRequest) -> dict:
        source = assert_inside_root(TEMP_DIR / safe_name(payload.tree_id))
        if not source.exists():
            raise FileNotFoundError(f"Temp sample not found: {payload.tree_id}")

        destination = assert_inside_root(DATASET_DIR / safe_name(payload.species) / safe_name(payload.tree_id))
        if destination.exists():
            shutil.rmtree(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))

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
                for box in annotation.boxes:
                    lines.append(
                        f"{box.class_id} {box.x_center:.6f} {box.y_center:.6f} "
                        f"{box.width:.6f} {box.height:.6f}"
                    )
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
            self.state.rejected_count += 1
            return {"status": "deleted", "tree_id": payload.tree_id, "reason": "all_views_dropped"}

        self.state.reviewed_count += 1
        return {"status": "saved", "path": str(destination)}

    def reject(self, tree_id: str) -> dict:
        source = assert_inside_root(TEMP_DIR / safe_name(tree_id))
        metadata = self._read_sample_metadata(source)
        if self.state.species and metadata:
            self.rejections.reject_tree(
                self.state.species,
                metadata.get("tree", {"tree_id": tree_id, "lat": 0, "lon": 0}),
                metadata.get("shots", []),
                "rejected_sample",
            )
        if source.exists():
            shutil.rmtree(source)
        if self.state.species:
            dataset_sample = assert_inside_root(DATASET_DIR / safe_name(self.state.species) / safe_name(tree_id))
            dataset_metadata = self._read_sample_metadata(dataset_sample)
            if dataset_metadata:
                self.rejections.reject_tree(
                    self.state.species,
                    dataset_metadata.get("tree", {"tree_id": tree_id, "lat": 0, "lon": 0}),
                    dataset_metadata.get("shots", []),
                    "rejected_saved_sample",
                )
            if dataset_sample.exists():
                shutil.rmtree(dataset_sample)
        self.state.rejected_count += 1
        return {"status": "deleted", "tree_id": tree_id}

    def _read_sample_metadata(self, sample_dir: Path) -> dict | None:
        metadata_path = sample_dir / "metadata.json"
        if not metadata_path.exists():
            return None
        try:
            return json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None


manager = TaskManager()
