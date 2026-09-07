"""Request a safe D1 PTv2 pause after the active epoch checkpoint."""

from __future__ import annotations

import argparse
import json
import uuid
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    suite_root = args.suite_root.resolve()
    state_path = suite_root / "suite_state.json"
    if not state_path.is_file():
        raise FileNotFoundError(state_path)
    state = json.loads(state_path.read_text(encoding="utf-8-sig"))
    if state.get("status") != "running" or not int(state.get("active_seed", 0)):
        raise RuntimeError("No active D1 PTv2 seed is available for a pause request")
    seed = int(state["active_seed"])
    path = suite_root / "runs" / f"seed_{seed}" / "pause_request.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    payload = {
        "requested_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "active_seed": seed,
        "policy": "honor only after last.pt and training history are saved",
    }
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)
    print(json.dumps({"status": "requested", "path": str(path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
