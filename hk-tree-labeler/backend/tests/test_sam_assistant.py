from pathlib import Path
import tempfile
import unittest

from app.sam_assistant import discover_sam_models, predict_sam_polygon


class SamAssistantTests(unittest.TestCase):
    def test_discover_sam_models_classifies_local_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            weights_dir = Path(tmp) / "sam_weights"
            weights_dir.mkdir()
            (weights_dir / "sam2.1_hiera_tiny.pt").write_bytes(b"tiny")
            (weights_dir / "sam3.pt").write_bytes(b"sam3")
            (weights_dir / "notes.txt").write_text("ignore", encoding="utf-8")

            models = discover_sam_models(weights_dir)

        self.assertEqual([model["key"] for model in models], ["sam2.1_hiera_tiny", "sam3"])
        self.assertEqual(models[0]["type"], "sam2")
        self.assertEqual(models[0]["config"], "configs/sam2.1/sam2.1_hiera_t.yaml")
        self.assertIs(models[0]["supports_text"], False)
        self.assertEqual(models[1]["type"], "sam3")
        self.assertIs(models[1]["supports_text"], True)

    def test_predict_sam_polygon_reports_missing_model_without_subprocess(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "sample.jpg"
            image.write_bytes(b"not-real-image")

            result = predict_sam_polygon(
                image_path=image,
                image_url="/temp/tree/sample.jpg",
                model_key="missing",
                point=(0.5, 0.5),
                models=[],
            )

        self.assertEqual(
            result,
            {
                "image": "/temp/tree/sample.jpg",
                "candidates": [],
                "model_ready": False,
                "error": "SAM model is not available: missing",
            },
        )


if __name__ == "__main__":
    unittest.main()
