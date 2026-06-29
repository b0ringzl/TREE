from __future__ import annotations

import asyncio
import csv
import shutil
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from app.task_manager import ReviewSample, TaskManager
from app.tree_registry import lookup_tree_id, rebuild_tree_registry


class TreeRegistryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path("D:/TREE/codex_test_tmp/tree-registry")
        self.species_dir = self.root / "species"
        self.db_path = self.root / "tree_registry.sqlite"
        self.label_map = self.root / "merge.csv"
        shutil.rmtree(self.root, ignore_errors=True)
        self.species_dir.mkdir(parents=True)
        self.label_map.write_text(
            "original_species,label_species,note\n"
            "Old species,Canonical species,test merge\n",
            encoding="utf-8",
        )
        self._write_species_csv(
            "Old species",
            [
                {"tree_id": "TREE-001", "latitude": "22.3", "longitude": "114.1", "height": "8"},
                {"tree_id": "DUPLICATE", "latitude": "22.4", "longitude": "114.2", "height": "9"},
            ],
        )
        self._write_species_csv(
            "Another species",
            [
                {"tree_id": "DUPLICATE", "latitude": "22.5", "longitude": "114.3", "height": "10"},
            ],
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def _write_species_csv(self, species: str, rows: list[dict[str, str]]) -> None:
        folder = self.species_dir / species
        folder.mkdir(parents=True)
        path = folder / f"{species}.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            fieldnames = ["tree_id", "latitude", "longitude", "height"]
            for row in rows:
                for key in row:
                    if key not in fieldnames:
                        fieldnames.append(key)
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def test_rebuild_registry_applies_label_map_and_supports_global_id_lookup(self) -> None:
        summary = rebuild_tree_registry(self.db_path, self.species_dir, self.label_map)

        self.assertEqual(summary["records"], 3)
        matches = lookup_tree_id("TREE-001", self.db_path)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].tree_id, "TREE-001")
        self.assertEqual(matches[0].species, "Canonical species")
        self.assertEqual(matches[0].source_species, "Old species")
        self.assertEqual(matches[0].lat, 22.3)
        self.assertEqual(matches[0].lon, 114.1)

    def test_lookup_reports_all_species_when_tree_id_is_duplicated(self) -> None:
        rebuild_tree_registry(self.db_path, self.species_dir, self.label_map)

        matches = lookup_tree_id("duplicate", self.db_path)

        self.assertEqual([match.species for match in matches], ["Another species", "Canonical species"])
        self.assertTrue(all(match.registry_id for match in matches))

    def test_rebuild_registry_excludes_major_parks_rows_without_real_tree_ids(self) -> None:
        self._write_species_csv(
            "Park source species",
            [
                {
                    "dataset": "Trees_Major_Parks",
                    "tree_id": "Kowloon Walled City Park",
                    "latitude": "22.33237",
                    "longitude": "114.18968",
                    "height": "20",
                    "source_file": "Trees_Major_Parks_converted.geojson",
                }
            ],
        )

        summary = rebuild_tree_registry(self.db_path, self.species_dir, self.label_map)
        matches = lookup_tree_id("Kowloon Walled City Park", self.db_path)

        self.assertEqual(summary["records"], 3)
        self.assertEqual(matches, [])

    def test_task_manager_prepares_selected_registry_record(self) -> None:
        rebuild_tree_registry(self.db_path, self.species_dir, self.label_map)
        selected = lookup_tree_id("TREE-001", self.db_path)[0]
        manager = TaskManager()
        manager._prepare_sample = AsyncMock(
            return_value=ReviewSample(
                tree_id="TREE-001",
                species="Canonical species",
                images=["/temp/TREE-001/angle_1.jpg"],
                candidates=[],
                tree={"tree_id": "TREE-001", "lat": 22.3, "lon": 114.1, "height_m": 8},
            )
        )

        with patch.object(manager.client, "validate_api_key", AsyncMock()), patch("app.task_manager.TREE_REGISTRY_DB", self.db_path):
            result = asyncio.run(manager.lookup_tree_by_id("TREE-001", registry_id=selected.registry_id))

        self.assertEqual(result["sample"]["species"], "Canonical species")
        manager._prepare_sample.assert_awaited_once()
        species, record = manager._prepare_sample.await_args.args
        self.assertEqual(species, "Canonical species")
        self.assertEqual(record.tree_id, "TREE-001")
        self.assertEqual(record.lat, 22.3)


if __name__ == "__main__":
    unittest.main()
