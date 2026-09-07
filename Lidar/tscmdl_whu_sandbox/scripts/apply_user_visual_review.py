"""Export non-destructive action lists from the user's visual review."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SANDBOX_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SANDBOX_ROOT / "src"))

from tscmdl_whu.user_review import (  # noqa: E402
    discover_manifest,
    export_action_lists,
    load_manifest,
    load_or_create_review_state,
    inspect_user_review_state,
    refresh_validation_review_status,
    REVIEW_SCOPE_QUALITY,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT / "lidar data" / "whu" / "derived" / "tscmdl" / "c1_full_shared_dataset",
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    manifest_path = (
        args.manifest.resolve()
        if args.manifest is not None
        else discover_manifest(dataset_root)
    )
    review_path = (
        args.review.resolve()
        if args.review is not None
        else dataset_root / "user_visual_review.json"
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else dataset_root / "user_review_outputs"
    )
    _, records = load_manifest(manifest_path)
    state = load_or_create_review_state(review_path, dataset_root, manifest_path)
    summary = export_action_lists(output_dir, records, state)
    review_status = inspect_user_review_state(
        review_path, manifest_path, records, REVIEW_SCOPE_QUALITY
    )
    refresh_validation_review_status(
        dataset_root / "validation.json", review_path, review_status
    )
    summary["formal_quality_review"] = review_status
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
