from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


SANDBOX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SANDBOX_ROOT / "src"))

from tscmdl_whu import (
    WHUFormatError,
    WHUTrajectoryReader,
    extract_tree_instances,
    normalize_unit_sphere,
    scan_tree_instances,
    to_benchmark_label,
    uniform_sample_indices,
)


POINT_DTYPE = np.dtype(
    [
        ("x", "<f8"),
        ("y", "<f8"),
        ("z", "<f8"),
        ("intensity", "<f8"),
        ("tree", "<f8"),
        ("label", "i1"),
    ]
)


def write_embedded_ply(path: Path, records: np.ndarray) -> None:
    header = "\n".join(
        [
            "ply",
            "format binary_little_endian 1.0",
            f"element vertex {len(records)}",
            "property float64 x",
            "property float64 y",
            "property float64 z",
            "property float64 intensity",
            "property float64 tree",
            "property int8 label",
            "end_header",
            "",
        ]
    ).encode("ascii")
    with path.open("wb") as stream:
        stream.write(header)
        stream.write(records.tobytes())


def make_records() -> np.ndarray:
    records = np.zeros(8, dtype=POINT_DTYPE)
    records["x"] = np.arange(8, dtype=np.float64)
    records["y"] = np.arange(8, dtype=np.float64) * 2
    records["z"] = np.arange(8, dtype=np.float64) * 3
    records["intensity"] = np.arange(8, dtype=np.float64) * 10
    records["tree"] = [0, 10, 10, 11, 11, 10, 12, 0]
    records["label"] = [-1, 1, 1, 29, 29, 1, -1, -1]
    return records


class TreeInstanceTests(unittest.TestCase):
    def test_scan_tracks_labels_bounds_and_benchmark_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "instances.ply"
            write_embedded_ply(path, make_records())
            with WHUTrajectoryReader(path, chunk_size=3) as reader:
                result = scan_tree_instances(reader)

            by_id = result.by_tree_id()
            self.assertEqual(result.background_point_count, 2)
            self.assertEqual(result.classification_valid_count, 2)
            self.assertEqual(result.unlabeled_count, 1)
            self.assertEqual(by_id[10].point_count, 3)
            self.assertEqual(by_id[10].raw_label_id, 1)
            self.assertEqual(by_id[10].min_xyz, (1.0, 2.0, 3.0))
            self.assertEqual(by_id[10].max_xyz, (5.0, 10.0, 15.0))
            self.assertEqual(by_id[11].raw_label_id, 29)
            self.assertEqual(by_id[11].benchmark_label_id, 18)
            self.assertFalse(by_id[12].is_classification_valid)
            self.assertEqual(to_benchmark_label(17), 17)
            self.assertEqual(to_benchmark_label(18), 18)

    def test_extract_selected_instances_across_chunk_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "instances.ply"
            write_embedded_ply(path, make_records())
            with WHUTrajectoryReader(path, chunk_size=3) as reader:
                scan = scan_tree_instances(reader)
                selected = [scan.by_tree_id()[10], scan.by_tree_id()[11]]
                instances = extract_tree_instances(reader, selected)

            self.assertEqual([instance.stats.tree_id for instance in instances], [10, 11])
            np.testing.assert_array_equal(instances[0].xyz[:, 0], [1, 2, 5])
            np.testing.assert_array_equal(instances[1].intensity, [30, 40])

    def test_inconsistent_labels_are_not_classification_valid(self) -> None:
        records = make_records()
        records["label"][2] = 2
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "inconsistent.ply"
            write_embedded_ply(path, records)
            with WHUTrajectoryReader(path, chunk_size=4) as reader:
                result = scan_tree_instances(reader)
            stats = result.by_tree_id()[10]
            self.assertFalse(stats.is_label_consistent)
            self.assertFalse(stats.is_classification_valid)
            self.assertEqual(result.inconsistent_count, 1)

    def test_non_integer_tree_ids_are_rejected(self) -> None:
        records = make_records()
        records["tree"][1] = 10.5
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "non_integer.ply"
            write_embedded_ply(path, records)
            with WHUTrajectoryReader(path, chunk_size=4) as reader:
                with self.assertRaisesRegex(WHUFormatError, "Non-integer tree"):
                    scan_tree_instances(reader)

    def test_sampling_is_deterministic_and_handles_both_size_cases(self) -> None:
        padded_a = uniform_sample_indices(5, 8, seed=42)
        padded_b = uniform_sample_indices(5, 8, seed=42)
        np.testing.assert_array_equal(padded_a, padded_b)
        self.assertEqual(set(range(5)), set(padded_a.tolist()))

        downsampled = uniform_sample_indices(10, 4, seed=42)
        self.assertEqual(len(np.unique(downsampled)), 4)

    def test_unit_sphere_normalization_records_reversible_transform(self) -> None:
        points = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0]], dtype=np.float32)
        normalized, transform = normalize_unit_sphere(points)
        np.testing.assert_allclose(normalized.mean(axis=0), 0, atol=1e-7)
        self.assertAlmostEqual(float(np.linalg.norm(normalized, axis=1).max()), 1.0, places=6)
        restored = normalized.astype(np.float64) * transform.scale + np.asarray(transform.centroid)
        np.testing.assert_allclose(restored, points, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
