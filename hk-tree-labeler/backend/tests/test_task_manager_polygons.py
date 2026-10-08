from __future__ import annotations

import json
import shutil
import asyncio
import unittest
from pathlib import Path
from unittest.mock import patch

from app.schemas import SubmitRequest
from app.rejection_store import RejectionStore
from app.task_manager import ReviewSample, TaskManager, TaskState


class TaskManagerPolygonSubmitTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path("D:/TREE/codex_test_tmp/polygon-submit")
        self.temp_dir = self.root / "temp"
        self.dataset_dir = self.root / "dataset"
        shutil.rmtree(self.root, ignore_errors=True)
        (self.temp_dir / "TREE001").mkdir(parents=True)
        (self.temp_dir / "TREE001" / "angle_1.jpg").write_bytes(b"fake image")
        (self.temp_dir / "TREE001" / "metadata.json").write_text(
            json.dumps({"tree": {"tree_id": "TREE001", "lat": 0, "lon": 0}, "shots": []}),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def test_submit_writes_one_yolo_segmentation_row_per_polygon(self) -> None:
        payload = SubmitRequest(
            tree_id="TREE001",
            species="Test species",
            annotations=[
                {
                    "image": "/temp/TREE001/angle_1.jpg",
                    "keep": True,
                    "polygons": [
                        {
                            "class_id": 0,
                            "points": [[0.1, 0.2], [0.3, 0.2], [0.2, 0.4]],
                        },
                        {
                            "class_id": 0,
                            "points": [[0.6, 0.6], [0.8, 0.6], [0.8, 0.9], [0.6, 0.9]],
                        },
                    ],
                }
            ],
        )

        manager = TaskManager()
        with patch("app.task_manager.TEMP_DIR", self.temp_dir), patch("app.task_manager.DATASET_DIR", self.dataset_dir):
            result = manager.submit(payload)

        self.assertEqual(result["status"], "saved")
        label_path = self.dataset_dir / "Test species" / "TREE001" / "angle_1.txt"
        self.assertEqual(
            label_path.read_text(encoding="utf-8").splitlines(),
            [
                "0 0.100000 0.200000 0.300000 0.200000 0.200000 0.400000",
                "0 0.600000 0.600000 0.800000 0.600000 0.800000 0.900000 0.600000 0.900000",
            ],
        )

    def test_reject_defers_physical_delete_until_second_next_sample(self) -> None:
        manager = TaskManager()
        manager.state = TaskState(species="Test species")
        manager.rejections = RejectionStore(self.root / "rejections.json")
        manager.state.queue.put_nowait(
            ReviewSample(tree_id="NEXT1", species="Test species", images=[], candidates=[], tree={"tree_id": "NEXT1", "lat": 1, "lon": 1})
        )
        manager.state.queue.put_nowait(
            ReviewSample(tree_id="NEXT2", species="Test species", images=[], candidates=[], tree={"tree_id": "NEXT2", "lat": 2, "lon": 2})
        )

        with patch("app.task_manager.TEMP_DIR", self.temp_dir), patch("app.task_manager.DATASET_DIR", self.dataset_dir):
            result = manager.reject("TREE001")
            self.assertEqual(result["status"], "pending_delete")
            self.assertTrue((self.temp_dir / "TREE001").exists())

            asyncio.run(manager.next_sample())
            self.assertTrue((self.temp_dir / "TREE001").exists())

            asyncio.run(manager.next_sample())
            self.assertFalse((self.temp_dir / "TREE001").exists())

    def test_reject_reprioritizes_pending_samples_near_rejected_tree(self) -> None:
        manager = TaskManager()
        manager.state = TaskState(species="Test species")
        manager.rejections = RejectionStore(self.root / "rejections.json")
        rejected_dir = self.temp_dir / "REJECTED"
        rejected_dir.mkdir(parents=True)
        (rejected_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "tree": {"tree_id": "REJECTED", "lat": 22.3000000, "lon": 114.1700000},
                    "shots": [],
                }
            ),
            encoding="utf-8",
        )
        nearby = ReviewSample(
            tree_id="NEAR",
            species="Test species",
            images=[],
            candidates=[],
            tree={"tree_id": "NEAR", "lat": 22.3000800, "lon": 114.1700000},
        )
        far = ReviewSample(
            tree_id="FAR",
            species="Test species",
            images=[],
            candidates=[],
            tree={"tree_id": "FAR", "lat": 22.3100000, "lon": 114.1800000},
        )
        manager.state.queue.put_nowait(nearby)
        manager.state.queue.put_nowait(far)

        with patch("app.task_manager.TEMP_DIR", self.temp_dir), patch("app.task_manager.DATASET_DIR", self.dataset_dir):
            manager.reject("REJECTED")

        next_sample = asyncio.run(manager.next_sample())
        self.assertEqual(next_sample.tree_id, "FAR")


if __name__ == "__main__":
    unittest.main()
