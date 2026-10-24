"""Create a recoverable, source-only snapshot without touching the Git index.

The snapshot is deliberately separate from the much larger model/data backup
required before training (see the repository's AGENTS.md). It captures the
desktop application sources that a release build consumes, including dirty
working-tree edits, while excluding credentials, generated jobs and weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

SOURCE_ROOTS = (
    "desktop_app",
    "spacescan_trees_demo_v1",
    "tree_species_teacher_demo",
    "tools",
)
EXCLUDED_DIRS = {
    ".git", ".venv", "__pycache__", "bin", "dist", "envs", "jobs",
    "local_snapshots", "logs", "models", "node_modules", "obj", "outputs",
    "repair_backups", "runs", "runtime", "user_data", "weights",
}
EXCLUDED_NAMES = {
    "desktop-settings.json", ".env", "credentials.json", "token.json",
}
ALLOWED_SUFFIXES = {
    ".bat", ".cmd", ".cs", ".csproj", ".css", ".html", ".ico",
    ".js", ".json", ".md", ".png", ".ps1", ".py", ".svg", ".xaml",
    ".xml", ".yaml", ".yml",
}
MAX_FILE_BYTES = 50 * 1024 * 1024


def sha256_stream(stream) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True,
        encoding="utf-8", errors="replace", check=True,
    )
    return result.stdout


def collect(repo: Path) -> tuple[list[tuple[str, Path]], list[str]]:
    included: list[tuple[str, Path]] = []
    skipped: list[str] = []
    for root_name in SOURCE_ROOTS:
        root = repo / root_name
        if not root.is_dir():
            skipped.append(f"{root_name}/ (not present)")
            continue
        for current, dirs, files in os.walk(root, followlinks=False):
            dirs[:] = sorted(d for d in dirs if d not in EXCLUDED_DIRS)
            for name in sorted(files):
                path = Path(current) / name
                relative = path.relative_to(repo).as_posix()
                if path.is_symlink() or name in EXCLUDED_NAMES:
                    skipped.append(relative)
                    continue
                if path.suffix.lower() not in ALLOWED_SUFFIXES:
                    continue
                if path.stat().st_size > MAX_FILE_BYTES:
                    skipped.append(relative + " (over 50 MiB)")
                    continue
                included.append((relative, path))
    for name in ("AGENTS.md", ".gitignore"):
        path = repo / name
        if path.is_file():
            included.append((name, path))
    return sorted(included), skipped


def create(repo: Path, output_root: Path, name: str | None) -> Path:
    repo = repo.resolve(strict=True)
    output_root = output_root.resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = output_root / (name or f"source-{stamp}")
    target.mkdir(parents=True, exist_ok=False)
    files, skipped = collect(repo)
    if not files:
        raise RuntimeError("No source files selected")
    records = []
    archive = target / "source.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6, allowZip64=True) as bundle:
        for relative, path in files:
            with path.open("rb") as source:
                digest = sha256_stream(source)
            bundle.write(path, arcname=relative)
            records.append({"path": relative, "bytes": path.stat().st_size,
                            "sha256": digest})
    with archive.open("rb") as stream:
        archive_sha256 = sha256_stream(stream)
    manifest = {
        "kind": "local-development-source-snapshot",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(repo),
        "git_head": git(repo, "rev-parse", "HEAD").strip(),
        "git_branch": git(repo, "branch", "--show-current").strip(),
        "source_file_count": len(records),
        "source_bytes": sum(row["bytes"] for row in records),
        "archive_sha256": archive_sha256,
        "files": records,
        "skipped": skipped,
        "scope_note": "Source only; model weights, data and deployed package require separate backups.",
    }
    (target / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (target / "git-status.txt").write_text(
        git(repo, "status", "--short", "--untracked-files=no"), encoding="utf-8"
    )
    verify(target)
    return target


def verify(target: Path) -> None:
    manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    archive = target / "source.zip"
    with archive.open("rb") as stream:
        if sha256_stream(stream) != manifest["archive_sha256"]:
            raise RuntimeError("Source ZIP checksum mismatch")
    with zipfile.ZipFile(archive) as bundle:
        if sorted(bundle.namelist()) != sorted(row["path"] for row in manifest["files"]):
            raise RuntimeError("Source ZIP file list mismatch")
        for row in manifest["files"]:
            with bundle.open(row["path"]) as stream:
                if sha256_stream(stream) != row["sha256"]:
                    raise RuntimeError(f"Source ZIP file mismatch: {row['path']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--name")
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify:
        verify(args.verify)
        print(f"SNAPSHOT_OK {args.verify}")
    else:
        print(f"SNAPSHOT_OK {create(args.repo, args.output_root, args.name)}")
