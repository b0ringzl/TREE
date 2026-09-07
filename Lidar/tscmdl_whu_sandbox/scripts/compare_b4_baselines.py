"""Compare aligned B2, B3, and B4 predictions."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


SANDBOX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SANDBOX_ROOT / "src"))

from tscmdl_whu.status import (  # noqa: E402
    compare_prediction_models,
    compare_prediction_records,
    load_prediction_records,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--b2", type=Path, required=True)
    parser.add_argument("--b3", type=Path, required=True)
    parser.add_argument("--b4", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{path}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    paths = {"B2": args.b2, "B3": args.b3, "B4": args.b4}
    records = {
        name: load_prediction_records(path) for name, path in paths.items()
    }
    result = {
        "status": "passed",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "models": {
            name: {
                "predictions": str(path.resolve()),
                "sha256": sha256_file(path),
            }
            for name, path in paths.items()
        },
        "pairwise": {
            "B2_vs_B3": compare_prediction_records(
                records["B2"], records["B3"], 3
            ),
            "B2_vs_B4": compare_prediction_records(
                records["B2"], records["B4"], 3
            ),
            "B3_vs_B4": compare_prediction_records(
                records["B3"], records["B4"], 3
            ),
        },
        "three_model_overlap": compare_prediction_models(records, 3),
    }
    atomic_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
