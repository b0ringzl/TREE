from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from build_release import SourceFile, build, portable_overrides, safe_relative, verify_parts


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ReleaseTests(unittest.TestCase):
    def test_portable_overrides_rehash_manifest(self) -> None:
        with tempfile.TemporaryDirectory(prefix="spacescan-portable-test-") as temp:
            stage = Path(temp)
            paths = {
                "app/tools/demo_runtime_identity.py": (
                    "import json\nfrom pathlib import Path\n"
                    "expected_python = Path('C:/Users/private/python.exe').resolve()\n"
                ),
                "app/outputs/distillation/bioclip_probe_20260908_0227/protocol.json":
                    json.dumps({"assets_sha256": {"C:/Users/private/TREE/data/a": "hash"}}),
                "app/outputs/distillation/bioclip_probe_20260908_0227/selection_lock.json":
                    json.dumps({"backup": "C:/Users/private/TREE/outputs/backup"}),
            }
            rows = []
            for relative, content in paths.items():
                path = stage / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
                rows.append({"file": relative, "bytes": path.stat().st_size,
                             "sha256": sha(path)})
            manifest = stage / "model_manifest.json"
            manifest.write_text(json.dumps(rows), encoding="utf-8")
            (stage / "BUILD_INFO.json").write_text(
                '{"source":"C:/Users/private/TREE"}', encoding="utf-8")
            files = [SourceFile(row["file"], stage / row["file"],
                                row["bytes"], row["sha256"]) for row in rows]
            files.append(SourceFile("model_manifest.json", manifest,
                                    manifest.stat().st_size, sha(manifest)))
            overrides = portable_overrides(stage, files)
            self.assertIn(b"sys.executable", overrides["app/tools/demo_runtime_identity.py"])
            self.assertIn(b"app/data/a", overrides[next(p for p in paths if p.endswith("protocol.json"))])
            self.assertNotIn(b"C:/Users/private", b"".join(overrides.values()))
            rewritten = json.loads(overrides["model_manifest.json"])
            for row in rewritten:
                self.assertEqual(row["sha256"], hashlib.sha256(overrides[row["file"]]).hexdigest())

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
