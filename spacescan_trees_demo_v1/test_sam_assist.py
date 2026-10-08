import io
import json
import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from PIL import Image

from spacescan_trees_demo_v1 import sam_assist, server


def sample_image():
    output = io.BytesIO()
    Image.new("RGB", (320, 240), "green").save(output, format="PNG")
    return output.getvalue()


class SamAssistTests(unittest.TestCase):
    def test_status_is_exposed_without_local_paths(self):
        response = TestClient(server.app).get("/api/sam/status")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["model"], "SAM 2.1 Tiny")
        self.assertNotIn("weight", response.json())

    def test_click_returns_normalized_draft_not_classification(self):
        payload = {"candidates": [{"polygon": [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]]}]}
        run = Mock(returncode=0, stdout=json.dumps(payload), stderr="")
        with patch.object(sam_assist, "status", return_value={"ready": True}), \
             patch.object(sam_assist.subprocess, "run", return_value=run) as subprocess_run:
            response = TestClient(server.app).post(
                "/api/sam/segment",
                files={"image": ("tree.png", sample_image(), "image/png")},
                data={"x": "0.5", "y": "0.5"},
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["polygon"], payload["candidates"][0]["polygon"])
        self.assertTrue(response.json()["user_confirmation_required"])
        self.assertNotIn("species", response.json())
        self.assertIn("--x", subprocess_run.call_args.args[0])

    def test_invalid_click_and_panorama_are_rejected(self):
        with self.assertRaises(ValueError):
            sam_assist.segment(sample_image(), -0.1, 0.5)
        output = io.BytesIO()
        Image.new("RGB", (400, 200), "green").save(output, format="PNG")
        with patch.object(sam_assist, "status", return_value={"ready": True}):
            with self.assertRaisesRegex(ValueError, "panorama"):
                sam_assist.segment(output.getvalue(), 0.5, 0.5)
            with patch.object(sam_assist.subprocess, "run", return_value=Mock(
                returncode=0, stdout=json.dumps({"candidates": [{"polygon": [[0, 0], [1, 0], [1, 1]]}]}), stderr=""
            )):
                self.assertEqual(len(sam_assist.segment(output.getvalue(), 0.5, 0.5, "photo")["polygon"]), 3)

    def test_missing_model_keeps_manual_fallback(self):
        with patch.object(sam_assist, "status", return_value={"ready": False}):
            response = TestClient(server.app).post(
                "/api/sam/segment",
                files={"image": ("tree.png", sample_image(), "image/png")},
                data={"x": "0.5", "y": "0.5"},
            )
        self.assertEqual(response.status_code, 503)
        self.assertIn("manual", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
