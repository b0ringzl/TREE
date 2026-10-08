"""Fixed-detection regression, source crop recovery and identity diagnostics.

No training and no new detector predictions. Outputs always use a fresh directory.
Reviewed cases are regression evidence, not unbiased MOT/recognition accuracy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from spacescan_trees_demo_v1 import corridor_identity, panorama, tree_detection, vmms_corridor


def signature(row):
    return (row["sample_index"], tuple(round(value, 6) for value in row["box"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", default="5e5b5e4622d341e3afa091eead3006e0")
    parser.add_argument("--output", required=True)
    parser.add_argument("--reuse-motion", type=Path)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise RuntimeError("Use a new versioned output directory; never overwrite an attempt")
    output.mkdir(parents=True)
    (output / "views").mkdir()
    directory = PROJECT / "spacescan_trees_demo_v1/runtime/corridor_jobs" / args.job
    snapshot = directory / "tracking_diagnostics.json"
    diagnostics = json.loads(snapshot.read_text(encoding="utf-8"))
    result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    frames = diagnostics["frames"]
    reused = None
    if args.reuse_motion:
        reused = json.loads(args.reuse_motion.read_text(encoding="utf-8"))["frames"]
        if [(f["sample_index"], f["frame_id"]) for f in frames] != \
                [(f["sample_index"], f["frame_id"]) for f in reused]:
            raise RuntimeError("Motion cache does not match the fixed source frames")
    # These folder frames have natural-number names (5.jpg ... 217.jpg).
    folder = Path(result["route"]["folder"])
    filtered = []
    previous = None
    for index, frame in enumerate(frames):
        source_path = folder / f'{frame["frame_id"]}.jpg'
        with Image.open(source_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        if reused is not None:
            frame["motion"] = reused[index]["motion"]
        else:
            current = corridor_identity.motion_image(image)
            frame["motion"] = corridor_identity.motion_snapshot(previous, current)
            previous = current
        views = {}
        retained = []
        for row in frame["candidates"]:
            spec = row["view_spec"]
            key = json.dumps(spec, sort_keys=True)
            if key not in views:
                views[key] = panorama.render(image, spec, max_edge=1024)
            view = views[key]
            row["crop"] = view.crop(tree_detection.crop_rect(row["view_box"], view.size))
            row["embedding"] = np.asarray(row["embedding"], dtype=np.float32)
            row["crop_evidence"] = corridor_identity.crop_evidence(row["crop"])
            if row["crop_evidence"]["usable"]:
                retained.append(row)
            else:
                filtered.append({"sample_index": index, "box": row["box"], "evidence": row["crop_evidence"]})
        frame["candidates"] = retained
        if (index + 1) % 20 == 0:
            print(f'Recovered crops and flow {index + 1}/{len(frames)}', flush=True)
    audit = {}
    tracks = vmms_corridor._track_candidates(frames, audit)
    public = []
    for index, track in enumerate(tracks):
        selected = vmms_corridor._select_views(track)
        for j, row in enumerate(selected):
            vmms_corridor._jpeg(row["crop"], output / "views" / f'track_{index+1:03d}_{j+1}.jpg', edge=960)
        public.append({"rank": index + 1, "source_ids": track["source_ids"],
                       "observations": [signature(row) for row in track["observations"]],
                       "selected_images": [row["sample_index"] + 1 for row in selected],
                       "minimum_view_similarity": min((float(a["embedding"] @ b["embedding"])
                            for i, a in enumerate(selected) for b in selected[i + 1:]), default=None)})
    old_track = diagnostics["stable_tracks"][0]
    originals = {row["sample_index"]: signature(row) for row in old_track["observations"]}
    hardware = originals[75]
    travel = originals[82]
    broadleaf = originals[101]
    same_track_pairs = []
    for rank, track in enumerate(tracks, 1):
        signatures = set(signature(row) for row in track["observations"])
        for names, pair in (("hardware/travel", (hardware, travel)),
                            ("travel/broadleaf", (travel, broadleaf)),
                            ("hardware/broadleaf", (hardware, broadleaf))):
            if set(pair) <= signatures:
                same_track_pairs.append({"rank": rank, "pair": names})
    regression_cases = []
    for original_rank, incompatible in ((1, ((0, 1), (1, 2), (0, 2))),
                                        (2, ((0, 1), (1, 2))), (4, ((0, 2), (1, 2)))):
        original_observations = diagnostics["stable_tracks"][original_rank - 1]["observations"]
        view_signatures = [signature(next(row for row in original_observations
                                         if row["sample_index"] == int(view["chainage_m"])))
                           for view in result["tracks"][original_rank - 1]["views"]]
        violations = []
        for rank, track in enumerate(tracks, 1):
            signatures = set(signature(row) for row in track["observations"])
            for a, b in incompatible:
                if {view_signatures[a], view_signatures[b]} <= signatures:
                    violations.append({"new_rank": rank, "original_views": [a + 1, b + 1]})
        regression_cases.append({"original_tree_id": original_rank,
                                 "incompatible_pairs_still_linked": violations})
    report = {"source_job": args.job, "source_diagnostics_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
              "tracking_revision": vmms_corridor.TRACKING_REVISION,
              "scope": "Fixed original detections/features; new crop evidence, local flow and association. Reviewed regression, not unbiased tracking accuracy.",
              "frames": len(frames), "original_detections": sum(len(f["candidates"]) for f in frames) + len(filtered),
              "crop_quality_rejections": filtered,
              "stable_tracks_before_cap": len(tracks),
              "association_gate_counts": audit,
              "original_tree1_wrong_pairs_still_linked": same_track_pairs,
              "original_tree1_hardware_rejected": any(signature(row) == hardware for row in filtered),
              "reviewed_mixed_target_cases": regression_cases,
              "tracks": public, "training_performed": False, "model_weights_changed": False}
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "tracking_diagnostics.json").write_text(json.dumps({"frames": vmms_corridor._tracking_snapshot(frames)}, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key not in {"tracks", "crop_quality_rejections"}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
