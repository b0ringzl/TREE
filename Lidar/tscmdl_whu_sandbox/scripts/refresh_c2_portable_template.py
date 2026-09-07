"""Refresh portable launcher/template files and their transfer-manifest hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


SANDBOX_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = Path(f"{path}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    root = args.bundle_root.resolve()
    manifest_path = root / "transfer_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    template = SANDBOX_ROOT / "portable_c2_template"
    refreshed = []
    for source in sorted(item for item in template.rglob("*") if item.is_file()):
        if "__pycache__" in source.parts or source.suffix == ".pyc":
            continue
        relative = source.relative_to(template)
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        refreshed.append(relative.as_posix())

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = {
        str(entry["path"]): entry for entry in manifest["files"]
    }
    for relative in refreshed:
        path = root / relative
        entries[relative] = {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    ordered = [entries[key] for key in sorted(entries)]
    manifest["files"] = ordered
    manifest["file_count"] = len(ordered)
    manifest["total_bytes"] = sum(int(entry["bytes"]) for entry in ordered)
    atomic_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "status": "passed",
                "refreshed_files": refreshed,
                "file_count": len(ordered),
                "total_bytes": manifest["total_bytes"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
