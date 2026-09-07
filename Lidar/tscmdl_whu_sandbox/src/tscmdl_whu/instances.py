"""Chunked tree-instance statistics, extraction, sampling, and normalization."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .io import WHUFormatError, WHUTrajectoryReader


@dataclass(frozen=True)
class TreeInstanceStats:
    tree_id: int
    point_count: int
    label_counts: tuple[tuple[int, int], ...]
    min_xyz: tuple[float, float, float]
    max_xyz: tuple[float, float, float]

    @property
    def raw_label_id(self) -> int | None:
        if len(self.label_counts) != 1:
            return None
        return self.label_counts[0][0]

    @property
    def benchmark_label_id(self) -> int | None:
        return to_benchmark_label(self.raw_label_id)

    @property
    def is_label_consistent(self) -> bool:
        return len(self.label_counts) == 1

    @property
    def is_classification_valid(self) -> bool:
        return self.raw_label_id is not None and self.raw_label_id >= 0

    @property
    def dimensions(self) -> tuple[float, float, float]:
        return tuple(high - low for low, high in zip(self.min_xyz, self.max_xyz))

    def to_dict(self) -> dict[str, object]:
        return {
            "tree_id": self.tree_id,
            "point_count": self.point_count,
            "label_counts": {str(label): count for label, count in self.label_counts},
            "raw_label_id": self.raw_label_id,
            "benchmark_label_id": self.benchmark_label_id,
            "is_label_consistent": self.is_label_consistent,
            "is_classification_valid": self.is_classification_valid,
            "min_xyz": list(self.min_xyz),
            "max_xyz": list(self.max_xyz),
            "dimensions": list(self.dimensions),
        }


@dataclass(frozen=True)
class InstanceScanResult:
    total_point_count: int
    background_point_count: int
    background_label_counts: tuple[tuple[int, int], ...]
    instances: tuple[TreeInstanceStats, ...]

    @property
    def foreground_point_count(self) -> int:
        return self.total_point_count - self.background_point_count

    @property
    def classification_valid_count(self) -> int:
        return sum(instance.is_classification_valid for instance in self.instances)

    @property
    def unlabeled_count(self) -> int:
        return sum(instance.raw_label_id is not None and instance.raw_label_id < 0 for instance in self.instances)

    @property
    def inconsistent_count(self) -> int:
        return sum(not instance.is_label_consistent for instance in self.instances)

    def by_tree_id(self) -> dict[int, TreeInstanceStats]:
        return {instance.tree_id: instance for instance in self.instances}

    def to_dict(self) -> dict[str, object]:
        return {
            "total_point_count": self.total_point_count,
            "background_point_count": self.background_point_count,
            "foreground_point_count": self.foreground_point_count,
            "background_label_counts": {
                str(label): count for label, count in self.background_label_counts
            },
            "instance_count": len(self.instances),
            "classification_valid_count": self.classification_valid_count,
            "unlabeled_count": self.unlabeled_count,
            "inconsistent_count": self.inconsistent_count,
            "instances": [instance.to_dict() for instance in self.instances],
        }


@dataclass(frozen=True)
class TreeInstance:
    stats: TreeInstanceStats
    xyz: np.ndarray
    intensity: np.ndarray


@dataclass(frozen=True)
class NormalizationTransform:
    centroid: tuple[float, float, float]
    scale: float


def to_benchmark_label(raw_label_id: int | None) -> int | None:
    if raw_label_id is None or raw_label_id < 0:
        return None
    return raw_label_id if raw_label_id <= 17 else 18


def _integer_annotations(values: np.ndarray, field_name: str) -> np.ndarray:
    if np.issubdtype(values.dtype, np.floating):
        if not np.isfinite(values).all():
            raise WHUFormatError(f"Non-finite {field_name} annotation detected")
        rounded = np.rint(values)
        if not np.array_equal(values, rounded):
            raise WHUFormatError(f"Non-integer {field_name} annotation detected")
        return rounded.astype(np.int64)
    if not np.issubdtype(values.dtype, np.integer):
        raise WHUFormatError(f"Unsupported {field_name} dtype: {values.dtype}")
    return values.astype(np.int64, copy=False)


def scan_tree_instances(reader: WHUTrajectoryReader) -> InstanceScanResult:
    """Scan one trajectory chunk-wise and summarize every positive tree ID."""
    if reader.annotation_source == "none":
        raise WHUFormatError("Tree-instance scanning requires annotations")

    aggregates: dict[int, dict[str, object]] = {}
    background_count = 0
    background_labels: Counter[int] = Counter()

    for chunk in reader.iter_chunks():
        if chunk.tree is None or chunk.label is None:
            raise WHUFormatError("A chunk unexpectedly has no annotations")
        tree_ids = _integer_annotations(chunk.tree, "tree")
        labels = _integer_annotations(chunk.label, "label")
        if np.any(tree_ids < 0):
            raise WHUFormatError("Negative tree instance ID detected")
        if np.any(labels < np.iinfo(np.int16).min) or np.any(labels > np.iinfo(np.int16).max):
            raise WHUFormatError("Label value does not fit in int16")

        background_mask = tree_ids == 0
        background_count += int(np.count_nonzero(background_mask))
        if np.any(background_mask):
            values, counts = np.unique(labels[background_mask], return_counts=True)
            background_labels.update(
                {int(value): int(count) for value, count in zip(values, counts)}
            )

        foreground_mask = tree_ids > 0
        if not np.any(foreground_mask):
            continue
        foreground_trees = tree_ids[foreground_mask]
        foreground_labels = labels[foreground_mask]
        if int(foreground_trees.max()) >= 2**47:
            raise WHUFormatError("Tree ID is too large for packed label aggregation")

        unique_trees, inverse, point_counts = np.unique(
            foreground_trees, return_inverse=True, return_counts=True
        )
        local_min = np.full((len(unique_trees), 3), np.inf, dtype=np.float64)
        local_max = np.full((len(unique_trees), 3), -np.inf, dtype=np.float64)
        for axis, field_name in enumerate(("x", "y", "z")):
            coordinates = np.asarray(chunk.records[field_name][foreground_mask], dtype=np.float64)
            if not np.isfinite(coordinates).all():
                raise WHUFormatError(f"Non-finite {field_name} coordinate detected")
            np.minimum.at(local_min[:, axis], inverse, coordinates)
            np.maximum.at(local_max[:, axis], inverse, coordinates)

        packed = (foreground_trees << 16) | (foreground_labels & 0xFFFF)
        unique_pairs, pair_counts = np.unique(packed, return_counts=True)
        local_label_counts: dict[int, Counter[int]] = {}
        for packed_value, count in zip(unique_pairs, pair_counts):
            value = int(packed_value)
            tree_id = value >> 16
            raw_label = value & 0xFFFF
            label_id = raw_label if raw_label < 0x8000 else raw_label - 0x10000
            local_label_counts.setdefault(tree_id, Counter())[label_id] += int(count)

        for index, tree_value in enumerate(unique_trees):
            tree_id = int(tree_value)
            aggregate = aggregates.setdefault(
                tree_id,
                {
                    "point_count": 0,
                    "label_counts": Counter(),
                    "min_xyz": np.full(3, np.inf, dtype=np.float64),
                    "max_xyz": np.full(3, -np.inf, dtype=np.float64),
                },
            )
            aggregate["point_count"] = int(aggregate["point_count"]) + int(point_counts[index])
            aggregate["label_counts"].update(local_label_counts[tree_id])
            aggregate["min_xyz"] = np.minimum(aggregate["min_xyz"], local_min[index])
            aggregate["max_xyz"] = np.maximum(aggregate["max_xyz"], local_max[index])

    instances = []
    for tree_id in sorted(aggregates):
        aggregate = aggregates[tree_id]
        instances.append(
            TreeInstanceStats(
                tree_id=tree_id,
                point_count=int(aggregate["point_count"]),
                label_counts=tuple(sorted(aggregate["label_counts"].items())),
                min_xyz=tuple(float(value) for value in aggregate["min_xyz"]),
                max_xyz=tuple(float(value) for value in aggregate["max_xyz"]),
            )
        )

    return InstanceScanResult(
        total_point_count=len(reader),
        background_point_count=background_count,
        background_label_counts=tuple(sorted(background_labels.items())),
        instances=tuple(instances),
    )


def extract_tree_instances(
    reader: WHUTrajectoryReader,
    instance_stats: Iterable[TreeInstanceStats],
) -> tuple[TreeInstance, ...]:
    """Extract selected trees in one chunked pass, preallocating exact output sizes."""
    stats_list = list(instance_stats)
    selected = {stats.tree_id: stats for stats in stats_list}
    if not selected:
        raise ValueError("At least one tree instance must be selected")
    if len(selected) != len(stats_list):
        raise ValueError("Duplicate tree IDs were selected")
    if any(tree_id <= 0 for tree_id in selected):
        raise ValueError("Only positive tree IDs can be extracted")

    selected_ids = np.asarray(sorted(selected), dtype=np.int64)

    xyz = {
        tree_id: np.empty((stats.point_count, 3), dtype=np.float32)
        for tree_id, stats in selected.items()
    }
    intensity = {
        tree_id: np.empty(stats.point_count, dtype=np.float32)
        for tree_id, stats in selected.items()
    }
    cursors = {tree_id: 0 for tree_id in selected}

    for chunk in reader.iter_chunks():
        if chunk.tree is None or chunk.label is None:
            raise WHUFormatError("Tree extraction requires annotations")
        chunk_tree_ids = _integer_annotations(chunk.tree, "tree")
        chunk_labels = _integer_annotations(chunk.label, "label")

        positions = np.searchsorted(selected_ids, chunk_tree_ids)
        within = positions < len(selected_ids)
        matched = np.zeros(len(chunk_tree_ids), dtype=bool)
        matched[within] = selected_ids[positions[within]] == chunk_tree_ids[within]
        matched_indices = np.flatnonzero(matched)
        if len(matched_indices) == 0:
            continue

        matched_tree_ids = chunk_tree_ids[matched_indices]
        order = np.argsort(matched_tree_ids, kind="stable")
        source_indices = matched_indices[order]
        sorted_tree_ids = matched_tree_ids[order]
        tree_ids, starts, counts = np.unique(
            sorted_tree_ids, return_index=True, return_counts=True
        )

        for tree_value, group_start, count_value in zip(tree_ids, starts, counts):
            tree_id = int(tree_value)
            stats = selected[tree_id]
            count = int(count_value)
            indices = source_indices[group_start : group_start + count]
            stop = cursors[tree_id] + count
            if stop > stats.point_count:
                raise WHUFormatError(f"Tree {tree_id} exceeds its scanned point count")
            for axis, field_name in enumerate(("x", "y", "z")):
                xyz[tree_id][cursors[tree_id] : stop, axis] = chunk.records[field_name][indices]
            intensity[tree_id][cursors[tree_id] : stop] = chunk.records["intensity"][indices]

            expected_label = stats.raw_label_id
            labels = np.unique(chunk_labels[indices])
            if expected_label is None or len(labels) != 1 or int(labels[0]) != expected_label:
                raise WHUFormatError(f"Tree {tree_id} labels changed between scan and extraction")
            cursors[tree_id] = stop

    output = []
    for tree_id in sorted(selected):
        stats = selected[tree_id]
        if cursors[tree_id] != stats.point_count:
            raise WHUFormatError(
                f"Tree {tree_id} extraction count mismatch: {cursors[tree_id]} != {stats.point_count}"
            )
        output.append(TreeInstance(stats=stats, xyz=xyz[tree_id], intensity=intensity[tree_id]))
    return tuple(output)


def uniform_sample_indices(point_count: int, target_count: int, seed: int) -> np.ndarray:
    """Sample uniformly and deterministically, retaining every point when padding."""
    if point_count <= 0 or target_count <= 0:
        raise ValueError("point_count and target_count must be positive")
    rng = np.random.default_rng(seed)
    if point_count >= target_count:
        return rng.choice(point_count, size=target_count, replace=False).astype(np.int64)

    extra = rng.choice(point_count, size=target_count - point_count, replace=True)
    indices = np.concatenate((np.arange(point_count, dtype=np.int64), extra.astype(np.int64)))
    rng.shuffle(indices)
    return indices


def normalize_unit_sphere(
    points: np.ndarray,
) -> tuple[np.ndarray, NormalizationTransform]:
    """Center points at their centroid and scale their maximum radius to one."""
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError(f"Expected a non-empty N x 3 array, got {points.shape}")
    work = np.asarray(points, dtype=np.float64)
    if not np.isfinite(work).all():
        raise ValueError("Point coordinates must be finite")
    centroid = work.mean(axis=0)
    centered = work - centroid
    scale = float(np.linalg.norm(centered, axis=1).max())
    if scale <= np.finfo(np.float64).eps:
        raise ValueError("Cannot normalize a zero-radius point cloud")
    normalized = (centered / scale).astype(np.float32)
    transform = NormalizationTransform(
        centroid=tuple(float(value) for value in centroid),
        scale=scale,
    )
    return normalized, transform
