"""Force a short ArcPy timeout to exercise process-tree cleanup."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from server import CONFIG, run_worker


def main() -> None:
    CONFIG["worker_timeout_seconds"] = 1
    try:
        run_worker("health", {})
    except TimeoutError:
        print("worker timeout cleanup path passed")
        return
    raise AssertionError("ArcPy health unexpectedly completed within one second")


if __name__ == "__main__":
    main()
