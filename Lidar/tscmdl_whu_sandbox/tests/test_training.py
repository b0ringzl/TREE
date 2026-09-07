from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image


SANDBOX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SANDBOX_ROOT / "src"))

from tscmdl_whu import (  # noqa: E402
    ImageManifestDataset,
    MultimodalManifestDataset,
    PointManifestDataset,
    classification_metrics,
    confusion_matrix,
)


class TrainingHelperTests(unittest.TestCase):
    def test_classification_metrics(self) -> None:
        matrix = confusion_matrix([0, 0, 1, 1, 2, 2], [0, 1, 1, 1, 0, 2], 3)
        np.testing.assert_array_equal(matrix, [[1, 1, 0], [0, 2, 0], [1, 0, 1]])
        metrics = classification_metrics(
            [0, 0, 1, 1, 2, 2], [0, 1, 1, 1, 0, 2], 3
        )
        self.assertAlmostEqual(metrics["accuracy"], 4 / 6)
        self.assertAlmostEqual(metrics["balanced_accuracy"], 2 / 3)
        self.assertEqual(len(metrics["per_class"]), 3)

    def test_manifest_dataset_loads_only_requested_split(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "points" / "train").mkdir(parents=True)
            (root / "points" / "val").mkdir(parents=True)
            records = []
            for split, label in (("train", 0), ("val", 1)):
                point_path = Path("points") / split / f"{split}.npz"
                np.savez(
                    root / point_path,
                    points_xyz=np.full((8192, 3), label, dtype=np.float32),
                    class_index=np.asarray(label, dtype=np.int64),
                )
                records.append(
                    {
                        "sample_key": split,
                        "split": split,
                        "class_index": label,
                        "point_path": point_path.as_posix(),
                    }
                )
            (root / "manifest.json").write_text(
                json.dumps({"records": records}), encoding="utf-8"
            )
            (root / "classes.json").write_text(
                json.dumps(
                    [
                        {"class_index": 0, "scientific_name": "zero"},
                        {"class_index": 1, "scientific_name": "one"},
                    ]
                ),
                encoding="utf-8",
            )
            dataset = PointManifestDataset(root, "train")
            points, label, index = dataset[0]
            self.assertEqual(len(dataset), 1)
            self.assertEqual(tuple(points.shape), (8192, 3))
            self.assertEqual((label, index), (0, 0))
            self.assertEqual(dataset.sample_keys, ("train",))

    def test_image_manifest_dataset_applies_transform(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = Path("images") / "train" / "sample.jpg"
            (root / image_path.parent).mkdir(parents=True)
            Image.new("RGB", (8, 6), color=(10, 20, 30)).save(root / image_path)
            (root / "manifest.json").write_text(
                json.dumps(
                    {
                        "image_size": [8, 6],
                        "records": [
                            {
                                "sample_key": "sample",
                                "split": "train",
                                "class_index": 0,
                                "image_path": image_path.as_posix(),
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (root / "classes.json").write_text(
                json.dumps([{"class_index": 0, "scientific_name": "zero"}]),
                encoding="utf-8",
            )

            def transform(image: Image.Image) -> torch.Tensor:
                array = np.asarray(image, dtype=np.float32).copy()
                return torch.from_numpy(array).permute(2, 0, 1) / 255.0

            dataset = ImageManifestDataset(root, "train", transform)
            image, label, index = dataset[0]
            self.assertEqual(tuple(image.shape), (3, 6, 8))
            self.assertEqual((label, index), (0, 0))
            self.assertEqual(dataset.sample_keys, ("sample",))

    def test_multimodal_manifest_dataset_preserves_pairing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            point_path = Path("points") / "train" / "sample.npz"
            image_path = Path("images") / "train" / "sample.jpg"
            (root / point_path.parent).mkdir(parents=True)
            (root / image_path.parent).mkdir(parents=True)
            np.savez(
                root / point_path,
                points_xyz=np.zeros((8192, 3), dtype=np.float32),
                class_index=np.asarray(0, dtype=np.int64),
            )
            Image.new("RGB", (8, 6), color=(10, 20, 30)).save(root / image_path)
            (root / "manifest.json").write_text(
                json.dumps(
                    {
                        "image_size": [8, 6],
                        "records": [
                            {
                                "sample_key": "sample",
                                "split": "train",
                                "class_index": 0,
                                "point_path": point_path.as_posix(),
                                "image_path": image_path.as_posix(),
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (root / "classes.json").write_text(
                json.dumps([{"class_index": 0, "scientific_name": "zero"}]),
                encoding="utf-8",
            )

            def transform(image: Image.Image) -> torch.Tensor:
                return torch.zeros((3, image.height, image.width))

            dataset = MultimodalManifestDataset(root, "train", transform)
            points, image, label, index = dataset[0]
            self.assertEqual(tuple(points.shape), (8192, 3))
            self.assertEqual(tuple(image.shape), (3, 6, 8))
            self.assertEqual((label, index), (0, 0))
            self.assertEqual(dataset.sample_keys, ("sample",))

if __name__ == "__main__":
    unittest.main()
