"""Confirm that ArcPy operations cannot overwrite an existing output."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from server import run_worker


def main() -> None:
    run_dir = Path(sys.argv[1]).resolve()
    source = str(run_dir / "smoke_input.gdb" / "trees")
    existing = str(run_dir / "smoke_input.gdb" / "trees_quality_1")
    try:
        run_worker(
            "copy_features",
            {"input_path": source, "output_path": existing},
        )
    except RuntimeError as error:
        if "FileExistsError" not in str(error):
            raise
        print("existing output rejection passed")
        return
    raise AssertionError("Existing output was unexpectedly overwritten")


if __name__ == "__main__":
    main()
