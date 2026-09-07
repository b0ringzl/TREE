"""Request a safe end-of-epoch pause for the active D1 image seed."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SUITE = (
    PROJECT_ROOT
    / "lidar data"
    / "whu"
    / "derived"
    / "tscmdl"
    / "d1_resnet50_clean_repeats"
    / "20260817_protocol_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-root", type=Path, default=DEFAULT_SUITE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    suite_root = args.suite_root.resolve()
    state_path = suite_root / "suite_state.json"
    if not state_path.is_file():
        raise FileNotFoundError("D1 suite has not started")
    state = json.loads(state_path.read_text(encoding="utf-8-sig"))
    active_seeds = state.get("current_seeds") or []
    if not active_seeds and state.get("current_seed") is not None:
        active_seeds = [state["current_seed"]]
    if state.get("status") not in {"running", "running_parallel"} or not active_seeds:
        raise RuntimeError(f"No active D1 seed can be paused: {state.get('status')}")
    paths = []
    for seed_value in active_seeds:
        seed = int(seed_value)
        run_dir = suite_root / "runs" / f"seed_{seed}"
        pause_path = run_dir / "pause_request.json"
        payload = {
            "status": "requested",
            "requested_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "seed": seed,
            "policy": "pause after current epoch checkpoint",
        }
        temporary = Path(f"{pause_path}.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(pause_path)
        paths.append(str(pause_path))
    print(json.dumps({"status": "requested", "paths": paths}, ensure_ascii=False))


if __name__ == "__main__":
    main()
