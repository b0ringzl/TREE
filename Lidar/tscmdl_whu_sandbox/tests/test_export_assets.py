from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


SANDBOX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SANDBOX_ROOT / "src"))

from tscmdl_whu.export_assets import (  # noqa: E402
    atomic_jpeg,
    atomic_json,
    atomic_npz,
    sha256_file,
    wrapped_crop,
)


class ExportAssetHelperTests(unittest.TestCase):
    def test_wrapped_crop_crosses_panorama_seam(self) -> None:
        image = Image.new("RGB", (8, 2))
        pixels = image.load()
        for x in range(8):
            for y in range(2):
                pixels[x, y] = (x, 0, 0)
        crop = wrapped_crop(image, 6, 0, 4, 2)
        self.assertEqual(crop.size, (4, 2))
        self.assertEqual([crop.getpixel((x, 0))[0] for x in range(4)], [6, 7, 0, 1])

    def test_atomic_asset_writers_leave_complete_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            json_path = root / "nested" / "value.json"
            npz_path = root / "points.npz"
            image_path = root / "image.jpg"
            atomic_json(json_path, {"value": 7})
            atomic_npz(npz_path, points=np.ones((4, 3), dtype=np.float32))
            atomic_jpeg(image_path, Image.new("RGB", (16, 8), (10, 20, 30)))

            self.assertEqual(json.loads(json_path.read_text())["value"], 7)
            with np.load(npz_path, allow_pickle=False) as archive:
                self.assertEqual(archive["points"].shape, (4, 3))
            with Image.open(image_path) as image:
                self.assertEqual(image.size, (16, 8))
                image.verify()
            self.assertEqual(len(sha256_file(npz_path)), 64)
            self.assertFalse(any(root.rglob("*.tmp")))


if __name__ == "__main__":
    unittest.main()
