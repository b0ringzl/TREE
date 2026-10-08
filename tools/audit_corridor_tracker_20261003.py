"""Replay corridor association on fixed detections; no training/model inference."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from spacescan_trees_demo_v1 import vmms_corridor


def summary(tracks):
    return {
        "stable_tracks_before_cap": len(tracks),
        "stable_observations": sum(len(track["observations"]) for track in tracks),
        "max_track_span_frames": max((track["observations"][-1]["sample_index"] -
                                      track["observations"][0]["sample_index"]
                                      for track in tracks), default=0),
        "fragments_reconnected": sum(track.get("merged_fragment_count", 0) for track in tracks),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("job_id")
    args = parser.parse_args()
    if len(args.job_id) != 32 or any(char not in "0123456789abcdef" for char in args.job_id):
        parser.error("Expected a 32-character corridor job ID")
    directory = PROJECT / "spacescan_trees_demo_v1/runtime/corridor_jobs" / args.job_id
    diagnostics = json.loads((directory / "tracking_diagnostics.json").read_text(encoding="utf-8"))
    frames = diagnostics["frames"]
    for frame in frames:
        for candidate in frame["candidates"]:
            candidate["embedding"] = np.asarray(candidate["embedding"], dtype=np.float32)
    backup = PROJECT / "spacescan_trees_demo_v1/backups/corridor_track_dedup_20261001_v1/vmms_corridor.py"
    name = "spacescan_trees_demo_v1.corridor_original_association_audit"
    spec = importlib.util.spec_from_file_location(name, backup)
    baseline = importlib.util.module_from_spec(spec)
    sys.modules[name] = baseline
    spec.loader.exec_module(baseline)
    # Remove only the output cap for counting; leave original association intact.
    baseline.MAX_TRACKS = 100000
    original_tracks = baseline._track_candidates(frames)
    current_tracks = vmms_corridor._track_candidates(frames)
    reconnect = vmms_corridor._reconnect_track_fragments
    try:
        vmms_corridor._reconnect_track_fragments = lambda tracks: tracks
        online_tracks = vmms_corridor._track_candidates(frames)
    finally:
        vmms_corridor._reconnect_track_fragments = reconnect
    report = {
        "job_id": args.job_id,
        "tracking_revision": diagnostics["tracking_revision"],
        "frames": len(frames),
        "fixed_detections": sum(len(frame["candidates"]) for frame in frames),
        "comparison_scope": "Tracking-only replay on the same new filtered detections. Original output cap disabled for counting. No identity ground truth; fewer tracks does not measure tracking accuracy.",
        "original_association": summary(original_tracks),
        "current_online_before_reconnect": summary(online_tracks),
        "current_with_reconnect": summary(current_tracks),
        "merged_tracks": [{
            "display_rank": index + 1,
            "source_ids": track["source_ids"],
            "sample_indices": [row["sample_index"] for row in track["observations"]],
        } for index, track in enumerate(current_tracks) if track.get("merged_fragment_count", 0)],
        "training_performed": False,
        "model_weights_changed": False,
    }
    (directory / "tracking_comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "merged_tracks"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
