"""Build a versioned, split, private Windows release from verified local inputs.

The portable Conda runtime is taken from the never-launched archived base, not
from an initialized installation (which can contain absolute relocation paths).
Application files are limited to the verified model_manifest.json allowlist.
No installed user_data, WebView2 profile, cloud keys or local VMMS path is packed.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import re
import shutil
import tarfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO

VERSION_PATTERN = re.compile(r"^v[0-9]{4}\.[0-9]{2}\.[0-9]{2}\.[0-9]+$")
EXCLUDED_ROOT = {
    "app", "runtime", "user_data", "SpaceScan.Trees.Desktop.exe.WebView2",
}
EXCLUDED_FILES = {"BUILD_INFO.json", "desktop-settings.json", "package_manifest.json"}


def digest(path: Path) -> str:
    check = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            check.update(block)
    return check.hexdigest()


def safe_relative(value: str) -> str:
    path = Path(value.replace("\\", "/"))
    parts = path.parts
    if (not parts or value.startswith(("/", "\\")) or path.is_absolute()
            or any(p in ("", ".", "..") for p in parts)):
        raise ValueError(f"Unsafe package path: {value!r}")
    if ":" in parts[0]:
        raise ValueError(f"Unsafe package path: {value!r}")
    return "/".join(parts)


@dataclass(frozen=True)
class SourceFile:
    relative: str
    path: Path
    size: int
    sha256: str


def from_file(relative: str, path: Path, expected: str | None = None) -> SourceFile:
    relative = safe_relative(relative)
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    actual = digest(path)
    if expected and actual.lower() != expected.lower():
        raise RuntimeError(f"Checksum mismatch: {relative}")
    return SourceFile(relative, path, size, actual)


def inventory(stage: Path, base: Path) -> list[SourceFile]:
    stage = stage.resolve(strict=True)
    base = base.resolve(strict=True)
    model_manifest = json.loads((stage / "model_manifest.json").read_text(encoding="utf-8"))
    sources: dict[str, SourceFile] = {}

    # The archived runtime has not been run or conda-unpacked in a machine-
    # specific installation directory.
    runtime = base / "runtime"
    if not (runtime / "python.exe").is_file():
        raise FileNotFoundError(runtime / "python.exe")
    for path in runtime.rglob("*"):
        if path.is_file():
            relative = "runtime/" + path.relative_to(runtime).as_posix()
            sources[relative] = from_file(relative, path)

    for row in model_manifest:
        relative = safe_relative(row["file"])
        if not relative.startswith("app/"):
            raise ValueError(f"Non-app model manifest entry: {relative}")
        source = from_file(relative, stage / relative, row["sha256"])
        if source.size != row["bytes"]:
            raise RuntimeError(f"Size mismatch: {relative}")
        sources[relative] = source

    for child in stage.iterdir():
        if child.name in EXCLUDED_ROOT:
            continue
        if child.is_file():
            paths = [child]
        elif child.is_dir():
            paths = (p for p in child.rglob("*") if p.is_file())
        else:
            continue
        for path in paths:
            relative = path.relative_to(stage).as_posix()
            if relative in EXCLUDED_FILES or path.suffix.lower() == ".log":
                continue
            sources[relative] = from_file(relative, path)

    required = (
        "SpaceScan.Trees.Desktop.exe", "launch.py", "model_manifest.json",
        "runtime/python.exe", "app/spacescan_trees_demo_v1/release.json",
    )
    for relative in required:
        if relative not in sources:
            raise RuntimeError(f"Release input missing: {relative}")
    return [sources[name] for name in sorted(sources)]


class SplitWriter:
    def __init__(self, output: Path, prefix: str, max_bytes: int) -> None:
        self.output = output
        self.prefix = prefix
        self.max_bytes = max_bytes
        self.parts: list[dict] = []
        self.current: BinaryIO | None = None
        self.hasher = hashlib.sha256()
        self.size = 0
        self.part_number = 0

    def _open(self) -> None:
        self.part_number += 1
        name = f"{self.prefix}.tar.gz.part{self.part_number:03d}"
        self.current = (self.output / name).open("xb")
        self.hasher = hashlib.sha256()
        self.size = 0

    def _finish(self) -> None:
        if self.current is None:
            return
        name = Path(self.current.name).name
        self.current.flush()
        os.fsync(self.current.fileno())
        self.current.close()
        self.parts.append({"name": name, "bytes": self.size,
                           "sha256": self.hasher.hexdigest()})
        self.current = None

    def write(self, data: bytes) -> int:
        view = memoryview(data)
        original = len(view)
        while view:
            if self.current is None:
                self._open()
            count = min(len(view), self.max_bytes - self.size)
            chunk = view[:count]
            assert self.current is not None
            self.current.write(chunk)
            self.hasher.update(chunk)
            self.size += count
            view = view[count:]
            if self.size == self.max_bytes:
                self._finish()
        return original

    def flush(self) -> None:
        if self.current is not None:
            self.current.flush()

    def close(self) -> None:
        self._finish()


def tar_info(name: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name=safe_relative(name))
    info.size = size
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mode = 0o644
    return info


def build(stage: Path, base: Path, output: Path, version: str,
          source_commit: str, part_bytes: int,
          source_snapshot_manifest: Path | None = None) -> Path:
    if not VERSION_PATTERN.fullmatch(version):
        raise ValueError("Version must look like v2026.10.24.1")
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValueError("Source commit must be a full Git SHA-1")
    if not 1024 <= part_bytes < 2 * 1024**3:
        raise ValueError("Part size must be at least 1 KiB and under 2 GiB")
    source_snapshot = None
    if source_snapshot_manifest is not None:
        snapshot = json.loads(source_snapshot_manifest.read_text(encoding="utf-8"))
        snapshot_zip = source_snapshot_manifest.with_name("source.zip")
        if digest(snapshot_zip) != snapshot["archive_sha256"]:
            raise RuntimeError("Source snapshot SHA-256 mismatch")
        source_snapshot = {
            "git_head": snapshot["git_head"],
            "archive_sha256": snapshot["archive_sha256"],
            "source_file_count": snapshot["source_file_count"],
        }
    files = inventory(stage, base)
    output.mkdir(parents=True, exist_ok=False)
    for name in ("Install-SpaceScanTrees.ps1", "Install-FromGitHub.ps1",
                 "Switch-SpaceScanTreesVersion.ps1"):
        shutil.copy2(Path(__file__).with_name(name), output / name)
    if source_snapshot_manifest is not None:
        shutil.copy2(source_snapshot_manifest, output / "source-snapshot-manifest.json")
        shutil.copy2(source_snapshot_manifest.with_name("source.zip"),
                     output / "source-snapshot.zip")
    build_info = json.dumps({
        "kind": "private-windows-desktop-release", "version": version,
        "source_commit": source_commit, "training_performed": False,
        "model_weights_changed": False, "external_vmms_folder_not_bundled": True,
    }, ensure_ascii=False, indent=2).encode("utf-8")
    settings = b'{"vmms_root":"","dropbox_app_key":"","onedrive_client_id":""}\n'
    generated = {"BUILD_INFO.json": build_info, "desktop-settings.json": settings}
    records = [{"path": row.relative, "bytes": row.size, "sha256": row.sha256}
               for row in files]
    records += [{"path": name, "bytes": len(data),
                 "sha256": hashlib.sha256(data).hexdigest()}
                for name, data in generated.items()]
    records.sort(key=lambda row: row["path"])
    prefix = f"SpaceScan-Trees-Desktop-{version}"
    writer = SplitWriter(output, prefix, part_bytes)
    try:
        with gzip.GzipFile(fileobj=writer, mode="wb", compresslevel=1,
                           mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w|",
                              format=tarfile.PAX_FORMAT) as archive:
                for row in files:
                    with row.path.open("rb") as stream:
                        archive.addfile(tar_info(row.relative, row.size), stream)
                for name, data in generated.items():
                    archive.addfile(tar_info(name, len(data)), io.BytesIO(data))
    finally:
        writer.close()
    manifest = {
        "format": 1, "product": "SpaceScan Trees Desktop", "version": version,
        "visibility": "private", "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": source_commit,
        "application_source_snapshot": source_snapshot,
        "file_count": len(records),
        "uncompressed_bytes": sum(row["bytes"] for row in records),
        "parts": writer.parts, "files": records,
    }
    manifest_path = output / f"{prefix}.release.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                             encoding="utf-8")
    verify_parts(output, manifest_path)
    return manifest_path


def verify_parts(output: Path, manifest_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["file_count"] != len(manifest["files"]):
        raise RuntimeError("Manifest file count mismatch")
    for part in manifest["parts"]:
        path = output / safe_relative(part["name"])
        if path.stat().st_size != part["bytes"] or digest(path) != part["sha256"]:
            raise RuntimeError(f"Release part mismatch: {part['name']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=Path)
    parser.add_argument("--base", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-snapshot-manifest", type=Path)
    parser.add_argument("--part-bytes", type=int, default=1_800_000_000)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--inventory-only", action="store_true")
    args = parser.parse_args()
    prefix = f"SpaceScan-Trees-Desktop-{args.version}"
    if args.verify_only:
        verify_parts(args.output, args.output / f"{prefix}.release.json")
        print(f"RELEASE_PARTS_OK {args.output}")
    elif args.inventory_only:
        if not args.stage or not args.base:
            parser.error("--stage and --base are required for inventory")
        files = inventory(args.stage, args.base)
        print(json.dumps({"file_count": len(files),
                          "uncompressed_bytes": sum(row.size for row in files),
                          "verified": True}))
    else:
        if not args.stage or not args.base:
            parser.error("--stage and --base are required when building")
        path = build(args.stage, args.base, args.output, args.version,
                     args.source_commit, args.part_bytes,
                     args.source_snapshot_manifest)
        print(f"RELEASE_BUNDLE_OK {path}")
