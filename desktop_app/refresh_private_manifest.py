"""Refresh and verify the private desktop package after staged-only repairs."""
from __future__ import annotations

import json
from pathlib import Path

from build_private_package import APP, DEST, digest


def main() -> None:
    manifest = DEST / "model_manifest.json"
    entries = json.loads(manifest.read_text(encoding="utf-8"))
    paths = {row["file"] for row in entries}
    release = json.loads((APP / "spacescan_trees_demo_v1/release.json").read_text(encoding="utf-8"))
    paths.update("app/" + release["android_delivery"].replace("\\", "/") + "/" + name
                 for name in (release["android_apk"], release["android_model"], "安装与验收说明.md"))
    refreshed = []
    for relative in sorted(paths):
        path = DEST / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        refreshed.append({"file": relative, "bytes": path.stat().st_size, "sha256": digest(path)})
    manifest.write_text(json.dumps(refreshed, ensure_ascii=False, indent=2), encoding="utf-8")
    directory = APP / release["android_delivery"]
    for name, expected in ((release["android_apk"], release["android_apk_sha256"]),
                           (release["android_model"], release["android_model_sha256"])):
        if digest(directory / name) != expected:
            raise RuntimeError(f"Android deliverable checksum mismatch: {name}")
    print(f"MANIFEST_OK files={len(refreshed)}")


if __name__ == "__main__":
    main()
