from __future__ import annotations

import json
import threading
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import ROOT_DIR, assert_inside_root, safe_name
from .data_access import TreeRecord


class RejectionStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = assert_inside_root(path or (ROOT_DIR / "hk-tree-labeler" / "backend" / "rejected_views.json"))
        self._lock = threading.Lock()

    def _empty(self) -> dict[str, Any]:
        return {"version": 1, "species": {}, "events": []}

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return self._empty()
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return self._empty()

    def _save(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        temp_path.replace(self.path)

    def _bucket(self, data: dict[str, Any], species: str) -> dict[str, Any]:
        species_key = safe_name(species)
        buckets = data.setdefault("species", {})
        return buckets.setdefault(species_key, {"trees": {}, "views": {}})

    def tree_key(self, record: TreeRecord | dict[str, Any]) -> str:
        if isinstance(record, TreeRecord):
            tree_id = record.tree_id
            lat = record.lat
            lon = record.lon
        else:
            tree_id = str(record.get("tree_id", "unknown"))
            lat = float(record.get("lat", 0))
            lon = float(record.get("lon", 0))
        return f"{safe_name(tree_id)}|{lat:.7f}|{lon:.7f}"

    def view_key(self, item: Any) -> str:
        if hasattr(item, "pano_id"):
            pano_id = getattr(item, "pano_id")
            lat = float(getattr(item, "lat"))
            lon = float(getattr(item, "lon"))
        else:
            pano_id = item.get("pano_id", "unknown")
            lat = float(item.get("lat", 0))
            lon = float(item.get("lon", 0))
        return f"{pano_id}|{lat:.7f}|{lon:.7f}"

    def is_tree_rejected(self, species: str, record: TreeRecord) -> bool:
        with self._lock:
            data = self._load()
            bucket = self._bucket(data, species)
            return self.tree_key(record) in bucket.get("trees", {})

    def is_view_rejected(self, species: str, item: Any) -> bool:
        with self._lock:
            data = self._load()
            bucket = self._bucket(data, species)
            return self.view_key(item) in bucket.get("views", {})

    def reject_tree(self, species: str, record: TreeRecord | dict[str, Any], shots: list[dict[str, Any]], reason: str) -> None:
        with self._lock:
            data = self._load()
            bucket = self._bucket(data, species)
            timestamp = datetime.now(timezone.utc).isoformat()
            tree_payload = asdict(record) if isinstance(record, TreeRecord) else record
            bucket["trees"][self.tree_key(record)] = {
                "reason": reason,
                "tree": tree_payload,
                "created_at": timestamp,
            }
            for shot in shots:
                bucket["views"][self.view_key(shot)] = {
                    "reason": reason,
                    "tree_id": tree_payload.get("tree_id", "unknown"),
                    "shot": shot,
                    "created_at": timestamp,
                }
            data.setdefault("events", []).append(
                {
                    "type": "reject_tree",
                    "species": species,
                    "tree_id": tree_payload.get("tree_id", "unknown"),
                    "reason": reason,
                    "created_at": timestamp,
                    "view_count": len(shots),
                }
            )
            self._save(data)

    def reject_view(self, species: str, tree_id: str, shot: dict[str, Any], reason: str) -> None:
        with self._lock:
            data = self._load()
            bucket = self._bucket(data, species)
            timestamp = datetime.now(timezone.utc).isoformat()
            bucket["views"][self.view_key(shot)] = {
                "reason": reason,
                "tree_id": tree_id,
                "shot": shot,
                "created_at": timestamp,
            }
            data.setdefault("events", []).append(
                {
                    "type": "reject_view",
                    "species": species,
                    "tree_id": tree_id,
                    "reason": reason,
                    "created_at": timestamp,
                    "view_key": self.view_key(shot),
                }
            )
            self._save(data)

    def stats(self) -> dict[str, int]:
        with self._lock:
            data = self._load()
            species_data = data.get("species", {})
            return {
                "species_count": len(species_data),
                "tree_rejections": sum(len(bucket.get("trees", {})) for bucket in species_data.values()),
                "view_rejections": sum(len(bucket.get("views", {})) for bucket in species_data.values()),
                "events": len(data.get("events", [])),
            }
