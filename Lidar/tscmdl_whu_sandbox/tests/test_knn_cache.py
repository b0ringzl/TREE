from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


SANDBOX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SANDBOX_ROOT / "src"))

from tscmdl_whu import MemmapReferenceCache  # noqa: E402


class MemmapReferenceCacheTests(unittest.TestCase):
    def test_index_select_matches_tensor_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "references.npy"
            values = np.arange(3 * 4 * 2, dtype=np.int32).reshape(3, 4, 2)
            np.save(path, values)
            cache = MemmapReferenceCache(
                path,
                ("a", "b", "c"),
                point_count=4,
                neighbours=2,
            )
            try:
                selected = cache.index_select(0, torch.tensor([2, 0]))
                self.assertEqual(tuple(selected.shape), (2, 4, 2))
                self.assertEqual(selected.dtype, torch.int32)
                self.assertTrue(np.array_equal(selected.numpy(), values[[2, 0]]))
            finally:
                cache.close()

    def test_invalid_dimension_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "references.npy"
            np.save(path, np.zeros((1, 4, 2), dtype=np.int32))
            cache = MemmapReferenceCache(
                path, ("a",), point_count=4, neighbours=2
            )
            try:
                with self.assertRaises(ValueError):
                    cache.index_select(1, torch.tensor([0]))
            finally:
                cache.close()


if __name__ == "__main__":
    unittest.main()
