from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from app.image_preprocessing import (
    analyze_image_quality,
    apply_photo_recipe,
    recipe_changes_pixels,
    suggest_recipe,
)
import app.task_manager as task_manager_module


class ImagePreprocessingTests(unittest.TestCase):
    def test_dark_image_is_detected_and_suggestion_brightens_it(self) -> None:
        image = Image.new("RGB", (320, 240), (24, 24, 24))
        before = analyze_image_quality(image)
        recipe = suggest_recipe(before)
        enhanced = apply_photo_recipe(
            image,
            exposure_ev=recipe["exposure_ev"],
            contrast=recipe["contrast"],
        )
        after = analyze_image_quality(enhanced)

        self.assertEqual(before["quality_status"], "underexposed")
        self.assertGreater(recipe["exposure_ev"], 0)
        self.assertGreater(after["p50_luminance"], before["p50_luminance"])

    def test_bright_image_is_detected_and_suggestion_darkens_it(self) -> None:
        image = Image.new("RGB", (320, 240), (244, 244, 244))
        before = analyze_image_quality(image)
        recipe = suggest_recipe(before)
        enhanced = apply_photo_recipe(
            image,
            exposure_ev=recipe["exposure_ev"],
            contrast=recipe["contrast"],
        )
        after = analyze_image_quality(enhanced)

        self.assertEqual(before["quality_status"], "overexposed")
        self.assertLess(recipe["exposure_ev"], 0)
        self.assertLess(after["p50_luminance"], before["p50_luminance"])

    def test_default_recipe_is_non_destructive(self) -> None:
        image = Image.new("RGB", (64, 64), (80, 120, 160))
        result = apply_photo_recipe(image, exposure_ev=0.0, contrast=1.0)

        self.assertFalse(recipe_changes_pixels(0.0, 1.0))
        self.assertEqual(result.getpixel((20, 20)), image.getpixel((20, 20)))

    def test_preprocess_manifest_flattens_review_records(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            original_annotation_dir = task_manager_module.ANNOTATION_DIR
            task_manager_module.ANNOTATION_DIR = Path(temporary_directory)
            try:
                manager = object.__new__(task_manager_module.TaskManager)
                manager.review_state = {
                    "TREE-1": {
                        "status": "accepted",
                        "species": "Example species",
                        "annotations": [
                            {
                                "image": "/views/TREE-1/view_1.jpg",
                                "keep": True,
                                "visibility": {
                                    "status": "partial",
                                    "reason": "occluded",
                                    "note": "lower crown hidden",
                                },
                                "preprocess": {
                                    "recipe_version": "photo_v1",
                                    "exposure_ev": 0.6,
                                    "contrast": 1.1,
                                    "reason": "underexposed",
                                    "auto_suggested": True,
                                    "human_accepted": True,
                                    "quality_before": {
                                        "quality_status": "underexposed",
                                        "p50_luminance": 54.0,
                                        "dark_pixel_ratio": 0.4,
                                        "bright_pixel_ratio": 0.01,
                                    },
                                },
                            }
                        ],
                    }
                }
                manager._save_preprocess_manifest()
                with (Path(temporary_directory) / "preprocess_manifest.csv").open(
                    "r", encoding="utf-8-sig", newline=""
                ) as handle:
                    rows = list(csv.DictReader(handle))
            finally:
                task_manager_module.ANNOTATION_DIR = original_annotation_dir

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["tree_id"], "TREE-1")
        self.assertEqual(rows[0]["visibility_reason"], "occluded")
        self.assertEqual(rows[0]["exposure_ev"], "0.6")


if __name__ == "__main__":
    unittest.main()
