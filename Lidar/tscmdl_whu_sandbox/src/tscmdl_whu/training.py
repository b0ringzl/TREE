"""Dataset and metric helpers for WHU-STree classification training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class PointManifestDataset(Dataset):
    """Load one B1 split into RAM as fixed 8192-point tensors."""

    def __init__(self, dataset_root: str | Path, split: str) -> None:
        self.root = Path(dataset_root)
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split: {split}")
        self.split = split
        manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        self.records = sorted(
            (record for record in manifest["records"] if record["split"] == split),
            key=lambda record: str(record["sample_key"]),
        )
        if not self.records:
            raise ValueError(f"No records found for split {split}")

        classes = json.loads((self.root / "classes.json").read_text(encoding="utf-8"))
        classes = sorted(classes, key=lambda item: int(item["class_index"]))
        self.class_names = tuple(str(item["scientific_name"]) for item in classes)
        expected_indices = list(range(len(classes)))
        actual_indices = [int(item["class_index"]) for item in classes]
        if actual_indices != expected_indices:
            raise ValueError(f"Class indices are not contiguous: {actual_indices}")

        point_arrays = []
        labels = []
        for record in self.records:
            path = self.root / str(record["point_path"])
            with np.load(path, allow_pickle=False) as archive:
                points = archive["points_xyz"]
                label = int(archive["class_index"])
            if points.shape != (8192, 3) or points.dtype != np.float32:
                raise ValueError(f"Unexpected point tensor in {path}: {points.shape} {points.dtype}")
            if not np.isfinite(points).all():
                raise ValueError(f"Non-finite point tensor in {path}")
            if label != int(record["class_index"]) or not 0 <= label < len(classes):
                raise ValueError(f"Class mismatch in {path}")
            point_arrays.append(points)
            labels.append(label)

        self.points = np.stack(point_arrays).astype(np.float32, copy=False)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.sample_keys = tuple(str(record["sample_key"]) for record in self.records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        return torch.from_numpy(self.points[index]), int(self.labels[index]), index


class ImageManifestDataset(Dataset):
    """Load one B1 image split lazily while preserving manifest order."""

    def __init__(
        self,
        dataset_root: str | Path,
        split: str,
        transform: Callable[[Image.Image], torch.Tensor],
    ) -> None:
        self.root = Path(dataset_root)
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split: {split}")
        self.split = split
        self.transform = transform

        manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        self.records = sorted(
            (record for record in manifest["records"] if record["split"] == split),
            key=lambda record: str(record["sample_key"]),
        )
        if not self.records:
            raise ValueError(f"No records found for split {split}")

        classes = json.loads((self.root / "classes.json").read_text(encoding="utf-8"))
        classes = sorted(classes, key=lambda item: int(item["class_index"]))
        self.class_names = tuple(str(item["scientific_name"]) for item in classes)
        actual_indices = [int(item["class_index"]) for item in classes]
        if actual_indices != list(range(len(classes))):
            raise ValueError(f"Class indices are not contiguous: {actual_indices}")

        image_size = manifest.get("image_size")
        self.expected_image_size = (
            tuple(int(value) for value in image_size) if image_size is not None else None
        )
        self.image_paths: list[Path] = []
        labels = []
        sample_keys = []
        for record in self.records:
            path = self.root / str(record["image_path"])
            if not path.is_file():
                raise FileNotFoundError(path)
            label = int(record["class_index"])
            if not 0 <= label < len(classes):
                raise ValueError(f"Class index outside configured range: {path}")
            self.image_paths.append(path)
            labels.append(label)
            sample_keys.append(str(record["sample_key"]))

        self.labels = np.asarray(labels, dtype=np.int64)
        self.sample_keys = tuple(sample_keys)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        path = self.image_paths[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            if self.expected_image_size is not None and image.size != self.expected_image_size:
                raise ValueError(
                    f"Unexpected image size in {path}: "
                    f"{image.size} != {self.expected_image_size}"
                )
            tensor = self.transform(image)
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 3:
            raise ValueError(f"Transform returned an invalid tensor for {path}")
        return tensor, int(self.labels[index]), index


class MultimodalManifestDataset(Dataset):
    """Pair B1 point and image samples with strict manifest-order validation."""

    def __init__(
        self,
        dataset_root: str | Path,
        split: str,
        image_transform: Callable[[Image.Image], torch.Tensor],
    ) -> None:
        self.points = PointManifestDataset(dataset_root, split)
        self.images = ImageManifestDataset(dataset_root, split, image_transform)
        if self.points.sample_keys != self.images.sample_keys:
            raise ValueError(f"Point/image sample-key mismatch for split {split}")
        if self.points.class_names != self.images.class_names:
            raise ValueError(f"Point/image class-name mismatch for split {split}")
        if not np.array_equal(self.points.labels, self.images.labels):
            raise ValueError(f"Point/image label mismatch for split {split}")
        self.split = split
        self.class_names = self.points.class_names
        self.sample_keys = self.points.sample_keys
        self.labels = self.points.labels

    def __len__(self) -> int:
        return len(self.points)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        points, point_label, point_index = self.points[index]
        image, image_label, image_index = self.images[index]
        if point_label != image_label or point_index != image_index:
            raise ValueError(f"Point/image item mismatch at index {index}")
        return points, image, point_label, point_index


def confusion_matrix(
    y_true: Sequence[int], y_pred: Sequence[int], num_classes: int
) -> np.ndarray:
    truth = np.asarray(y_true, dtype=np.int64)
    predicted = np.asarray(y_pred, dtype=np.int64)
    if truth.shape != predicted.shape or truth.ndim != 1:
        raise ValueError("y_true and y_pred must be same-length one-dimensional arrays")
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    if len(truth) and (
        truth.min() < 0
        or predicted.min() < 0
        or truth.max() >= num_classes
        or predicted.max() >= num_classes
    ):
        raise ValueError("Class index is outside the configured range")
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(matrix, (truth, predicted), 1)
    return matrix


def classification_metrics(
    y_true: Sequence[int], y_pred: Sequence[int], num_classes: int
) -> dict[str, object]:
    matrix = confusion_matrix(y_true, y_pred, num_classes)
    true_counts = matrix.sum(axis=1)
    predicted_counts = matrix.sum(axis=0)
    diagonal = np.diag(matrix).astype(np.float64)
    precision = np.divide(
        diagonal,
        predicted_counts,
        out=np.zeros(num_classes, dtype=np.float64),
        where=predicted_counts != 0,
    )
    recall = np.divide(
        diagonal,
        true_counts,
        out=np.zeros(num_classes, dtype=np.float64),
        where=true_counts != 0,
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros(num_classes, dtype=np.float64),
        where=(precision + recall) != 0,
    )
    total = int(matrix.sum())
    return {
        "accuracy": float(diagonal.sum() / total) if total else 0.0,
        "balanced_accuracy": float(recall.mean()),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "confusion_matrix": matrix.tolist(),
        "per_class": [
            {
                "class_index": index,
                "support": int(true_counts[index]),
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
            }
            for index in range(num_classes)
        ],
    }
