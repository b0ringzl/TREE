from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


SANDBOX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SANDBOX_ROOT / "src"))

from tscmdl_whu import WHUFormatError, WHUTrajectoryReader, parse_ply_header


def write_ply(path: Path, records: np.ndarray, properties: list[tuple[str, str]]) -> None:
    header_lines = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {len(records)}",
        *(f"property {kind} {name}" for name, kind in properties),
        "end_header",
        "",
    ]
    with path.open("wb") as stream:
        stream.write("\n".join(header_lines).encode("ascii"))
        stream.write(records.tobytes())


class WHUTrajectoryReaderTests(unittest.TestCase):
    def test_embedded_annotations_and_partial_final_chunk(self) -> None:
        dtype = np.dtype(
            [
                ("x", "<f8"),
                ("y", "<f8"),
                ("z", "<f8"),
                ("intensity", "<f8"),
                ("tree", "<f8"),
                ("label", "i1"),
            ]
        )
        records = np.zeros(5, dtype=dtype)
        records["x"] = np.arange(5)
        records["tree"] = [0, 10, 10, 11, 11]
        records["label"] = [-1, 2, 2, 3, 3]

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "embedded.ply"
            write_ply(
                path,
                records,
                [
                    ("x", "float64"),
                    ("y", "float64"),
                    ("z", "float64"),
                    ("intensity", "float64"),
                    ("tree", "float64"),
                    ("label", "int8"),
                ],
            )
            with WHUTrajectoryReader(path, chunk_size=2) as reader:
                chunks = list(reader.iter_chunks())
                self.assertEqual(reader.annotation_source, "embedded")
                self.assertEqual([len(chunk) for chunk in chunks], [2, 2, 1])
                np.testing.assert_array_equal(chunks[-1].tree, [11])
                np.testing.assert_array_equal(chunks[-1].label, [3])
                np.testing.assert_array_equal(chunks[0].xyz()[:, 0], [0, 1])

    def test_external_reference_is_synchronized_by_row(self) -> None:
        dtype = np.dtype(
            [("x", "<f8"), ("y", "<f8"), ("z", "<f8"), ("intensity", "<f8")]
        )
        records = np.zeros(5, dtype=dtype)
        reference = np.array([[7, 1], [7, 1], [8, 4], [0, -1], [0, -1]], dtype=np.int16)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ply_path = root / "external.ply"
            reference_path = root / "external.npy"
            write_ply(
                ply_path,
                records,
                [
                    ("x", "float64"),
                    ("y", "float64"),
                    ("z", "float64"),
                    ("intensity", "float64"),
                ],
            )
            np.save(reference_path, reference)
            with WHUTrajectoryReader(
                ply_path, reference_path, chunk_size=3
            ) as reader:
                first, second = list(reader.iter_chunks())
                self.assertEqual(reader.annotation_source, "reference")
                np.testing.assert_array_equal(first.tree, [7, 7, 8])
                np.testing.assert_array_equal(first.label, [1, 1, 4])
                np.testing.assert_array_equal(second.label, [-1, -1])

    def test_reference_row_mismatch_is_rejected(self) -> None:
        dtype = np.dtype(
            [("x", "<f8"), ("y", "<f8"), ("z", "<f8"), ("intensity", "<f8")]
        )
        records = np.zeros(3, dtype=dtype)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ply_path = root / "mismatch.ply"
            reference_path = root / "mismatch.npy"
            write_ply(
                ply_path,
                records,
                [
                    ("x", "float64"),
                    ("y", "float64"),
                    ("z", "float64"),
                    ("intensity", "float64"),
                ],
            )
            np.save(reference_path, np.zeros((2, 2), dtype=np.int16))
            with self.assertRaisesRegex(WHUFormatError, "row mismatch"):
                WHUTrajectoryReader(ply_path, reference_path)

    def test_ascii_ply_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "ascii.ply"
            path.write_text(
                "ply\nformat ascii 1.0\nelement vertex 0\nproperty float x\nend_header\n",
                encoding="ascii",
            )
            with self.assertRaisesRegex(WHUFormatError, "binary_little_endian"):
                parse_ply_header(path)


if __name__ == "__main__":
    unittest.main()
