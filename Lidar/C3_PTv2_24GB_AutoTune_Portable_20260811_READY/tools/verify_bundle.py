"""Verify immutable portable-bundle files after USB transfer."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def replace_with_retry(temporary: Path, destination: Path) -> None:
    delay_seconds = 0.01
    deadline = time.monotonic() + 10.0
    while True:
        try:
            temporary.replace(destination)
            return
        except OSError as error:
            retryable = os.name == "nt" and getattr(error, "winerror", None) in {5, 32}
            if not retryable or time.monotonic() >= deadline:
                raise
            time.sleep(delay_seconds)
            delay_seconds = min(delay_seconds * 1.6, 0.5)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{path}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    replace_with_retry(temporary, path)


def main() -> None:
    started = time.perf_counter()
    manifest_path = ROOT / "transfer_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    entries = manifest["files"]
    for index, entry in enumerate(entries, start=1):
        path = ROOT / str(entry["path"])
        if not path.is_file():
            errors.append(f"missing: {entry['path']}")
            continue
        if path.stat().st_size != int(entry["bytes"]):
            errors.append(f"size mismatch: {entry['path']}")
            continue
        if sha256_file(path) != entry["sha256"]:
            errors.append(f"hash mismatch: {entry['path']}")
        if index % 500 == 0 or index == len(entries):
            print(
                f"\rVerifying files: {index}/{len(entries)}",
                end="",
                flush=True,
            )
    print(flush=True)
    result = {
        "status": "passed" if not errors else "failed",
        "verified_at": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "computer_name": platform.node(),
        "root": str(ROOT),
        "file_count": len(entries),
        "total_bytes": sum(int(entry["bytes"]) for entry in entries),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "errors": errors,
    }
    atomic_json(ROOT / "diagnostics" / "integrity_report.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
