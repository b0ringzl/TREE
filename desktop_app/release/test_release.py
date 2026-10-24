from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from build_release import build, safe_relative, verify_parts


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ReleaseTests(unittest.TestCase):
    def test_rejects_path_traversal(self) -> None:
        for value in ("../outside", "/absolute", "C:/drive", "nested/../../outside"):
            with self.assertRaises(ValueError):
                safe_relative(value)

    def test_build_verify_and_install_small_fixture(self) -> None:
        with tempfile.TemporaryDirectory(prefix="spacescan-release-test-") as temp:
            root = Path(temp)
            stage, base, bundle, install = (root / name for name in
                                            ("stage", "base", "bundle", "install"))
            (base / "runtime").mkdir(parents=True)
            runtime_bytes = os.urandom(10_000)
            (base / "runtime/python.exe").write_bytes(runtime_bytes)
            app = stage / "app/spacescan_trees_demo_v1"
            app.mkdir(parents=True)
            release = app / "release.json"
            release.write_text('{"classifier_version":"fixture"}', encoding="utf-8")
            (stage / "SpaceScan.Trees.Desktop.exe").write_bytes(b"fixture host")
            (stage / "launch.py").write_text("print('fixture')", encoding="utf-8")
            (stage / "desktop-settings.json").write_text(
                '{"dropbox_app_key":"DO_NOT_DISTRIBUTE"}', encoding="utf-8"
            )
            (stage / "BUILD_INFO.json").write_text(
                '{"source":"C:/Users/private"}', encoding="utf-8"
            )
            (stage / "user_data").mkdir()
            (stage / "user_data/secret.txt").write_text("DO_NOT_DISTRIBUTE")
            (stage / "model_manifest.json").write_text(json.dumps([{
                "file": "app/spacescan_trees_demo_v1/release.json",
                "bytes": release.stat().st_size, "sha256": sha(release),
            }]), encoding="utf-8")
            version = "v2026.10.24.1"
            manifest_path = build(stage, base, bundle, version, "0" * 40, 1024)
            verify_parts(bundle, manifest_path)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            names = {row["path"] for row in manifest["files"]}
            self.assertIn("runtime/python.exe", names)
            self.assertNotIn("user_data/secret.txt", names)
            self.assertNotIn("package_manifest.json", names)
            self.assertGreater(len(manifest["parts"]), 1)

            script = Path(__file__).with_name("Install-SpaceScanTrees.ps1")
            preflight = subprocess.run([
                "powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass",
                "-File", str(script), "-BundleDir", str(bundle), "-VerifyOnly",
            ], capture_output=True, text=True, encoding="utf-8", errors="replace")
            self.assertEqual(preflight.returncode, 0, preflight.stdout + preflight.stderr)
            result = subprocess.run([
                "powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass",
                "-File", str(script),
                "-BundleDir", str(bundle), "-InstallRoot", str(install),
                "-NoShortcuts", "-SkipRuntimeCheck",
            ], capture_output=True, text=True, encoding="utf-8", errors="replace")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            installed = install / version
            self.assertEqual((installed / "runtime/python.exe").read_bytes(),
                             runtime_bytes)
            settings = json.loads((installed / "desktop-settings.json").read_text())
            self.assertEqual(settings["dropbox_app_key"], "")
            self.assertNotIn("C:/Users/private",
                             (installed / "BUILD_INFO.json").read_text())
            self.assertFalse((installed / "user_data/secret.txt").exists())
            self.assertTrue((installed / "install_receipt.json").is_file())


if __name__ == "__main__":
    unittest.main()
