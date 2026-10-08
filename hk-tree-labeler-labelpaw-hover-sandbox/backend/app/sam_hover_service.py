from __future__ import annotations

import json
import os
import subprocess
import threading
import time

from .config import PROJECT_TOOLS_DIR, ROOT_DIR, SAM_HOVER_IDLE_TIMEOUT_SEC, SAM_PYTHON, SAM_SEGMENT_TIMEOUT_SEC
from .sam_assistant import _resolve_temp_image, discover_sam_models


class SamHoverWorker:
    def __init__(self) -> None:
        self.process: subprocess.Popen[str] | None = None
        self.model_key = ""
        self.request_id = 0
        self.last_used = 0.0
        self.lock = threading.RLock()

    def predict(self, image_url: str, model_key: str, point: tuple[float, float]) -> dict:
        image_path = _resolve_temp_image(image_url)
        model = self._find_model(model_key)
        if not model:
            return {
                "image": image_url,
                "candidates": [],
                "model_ready": False,
                "error": f"SAM model is not available: {model_key}",
            }

        with self.lock:
            self._ensure_worker(model)
            assert self.process and self.process.stdin and self.process.stdout
            self.request_id += 1
            payload = {"id": self.request_id, "image": str(image_path), "x": point[0], "y": point[1]}
            self.process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self.process.stdin.flush()
            line = self.process.stdout.readline()
            self.last_used = time.monotonic()

        if not line:
            self.stop()
            return {
                "image": image_url,
                "candidates": [],
                "model_ready": True,
                "error": "SAM hover worker stopped before returning a result.",
            }

        result = json.loads(line)
        if result.get("error"):
            return {"image": image_url, "candidates": [], "model_ready": True, "error": result["error"]}
        return {
            "image": image_url,
            "candidates": result.get("candidates", []),
            "model_ready": True,
            "model": model,
            "worker_cached_image": result.get("cached_image"),
        }

    def cleanup_idle(self) -> None:
        if self.process and time.monotonic() - self.last_used > SAM_HOVER_IDLE_TIMEOUT_SEC:
            self.stop()

    def stop(self) -> None:
        with self.lock:
            if not self.process:
                return
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except Exception:
                self.process.kill()
            finally:
                self.process = None
                self.model_key = ""

    def _find_model(self, model_key: str) -> dict | None:
        for model in discover_sam_models():
            if model.get("key") == model_key:
                return model
        return None

    def _ensure_worker(self, model: dict) -> None:
        if self.process and self.model_key == model["key"] and self.process.poll() is None:
            return
        self.stop()

        command = [
            SAM_PYTHON,
            str(PROJECT_TOOLS_DIR / "sam_hover_worker.py"),
            "--model-key",
            str(model["key"]),
            "--model-type",
            str(model["type"]),
            "--model",
            str(model["weight"]),
        ]
        if model.get("config"):
            command.extend(["--config", str(model["config"])])

        env = os.environ.copy()
        cache_dir = ROOT_DIR / "models" / "_cache"
        env.update(
            {
                "YOLO_CONFIG_DIR": str(ROOT_DIR),
                "TORCH_HOME": str(cache_dir / "torch"),
                "HF_HOME": str(cache_dir / "huggingface"),
                "XDG_CACHE_HOME": str(cache_dir / "xdg"),
                "MPLCONFIGDIR": str(cache_dir / "matplotlib"),
                "TMP": str(cache_dir / "tmp"),
                "TEMP": str(cache_dir / "tmp"),
            }
        )
        labelpaw_path = str(ROOT_DIR / "LabelPaw-web-images")
        env["PYTHONPATH"] = labelpaw_path + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        self.process = subprocess.Popen(
            command,
            cwd=str(ROOT_DIR),
            env=env,
            text=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
        )
        self.model_key = model["key"]
        self.last_used = time.monotonic()
        self._wait_until_ready()

    def _wait_until_ready(self) -> None:
        assert self.process and self.process.stdout
        deadline = time.monotonic() + SAM_SEGMENT_TIMEOUT_SEC
        while time.monotonic() < deadline:
            line = self.process.stdout.readline()
            if not line:
                break
            payload = json.loads(line)
            if payload.get("status") == "ready":
                return
            if payload.get("error"):
                raise RuntimeError(payload["error"])
        raise RuntimeError("Timed out while loading SAM hover worker.")


hover_worker = SamHoverWorker()
