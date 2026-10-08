"""Offline BoT-SORT + generic CLIP experiment; NOT the production tracker.

Run one real Ultralytics BoT-SORT instance for each fixed pinhole view. Reuse
the exact same detections and CLIP embeddings as the baseline. Native IDs are
proposals, never authority: spherical geometry + anchored gallery checks gate
every association, including low-confidence and reactivation stages.

No checkpoint download, training, species-score modification or production
switch. An image provider enables genuine sparse-flow camera compensation.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from . import corridor_identity, vmms_corridor

REVISION = "corridor-botsort-clip-spherical-experiment-v1-20261004"
CONFIG = dict(tracker_type="botsort", track_high_thresh=.35,
              track_low_thresh=.10, new_track_thresh=.35, track_buffer=8,
              match_thresh=.70, fuse_score=False, gmc_method="sparseOptFlow",
              proximity_thresh=.20, appearance_thresh=.90,
              with_reid=True, model="auto")


def view_key(candidate):
    # View coordinate systems must not share Kalman states.
    return json.dumps(candidate["view_spec"], sort_keys=True)


def track_candidates(frames, audit=None, image_provider=None):
    from ultralytics.engine.results import Boxes
    from ultralytics.trackers.bot_sort import BOTSORT

    audit = audit if audit is not None else {}
    keys = {view_key(row): row["view_spec"] for frame in frames for row in frame["candidates"]}
    trackers = {}
    # Construct all streams before update: upstream tracker construction resets
    # the shared BaseTrack counter. Keys still namespace IDs by view.
    for key in sorted(keys):
        tracker = BOTSORT(SimpleNamespace(**CONFIG))
        # Use the cached, normalized CLIP embeddings, not native YOLO feature
        # hooks or downloaded person/classification checkpoints.
        tracker.encoder = lambda features, boxes: [np.asarray(f, dtype=np.float32) for f in features]
        trackers[key] = tracker
    audit["botsort_streams"] = len(trackers)
    audit["gmc_enabled"] = image_provider is not None
    history = {frame["sample_index"]: frame for frame in frames}
    tracks = []
    bindings = {}
    for frame in frames:
        observations = []
        hints = {}
        groups = {key: [] for key in keys}
        for row in frame["candidates"]:
            key = view_key(row)
            index = len(observations)
            observations.append({**row, "sample_index": frame["sample_index"],
                                 "chainage_m": frame["chainage_m"], "frame_id": frame["frame_id"],
                                 "bearing": vmms_corridor._normalise_angle(frame["heading_deg"] + row["relative_yaw"]),
                                 "camera_latitude": frame["latitude"], "camera_longitude": frame["longitude"]})
            groups[key].append((index, row))
        for key, tracker in trackers.items():
            group = groups[key]
            image = image_provider(frame, keys[key]) if image_provider is not None else None
            height, width = image.shape[:2] if image is not None else (1024, 1024)
            data = np.asarray([[x*scale for x, scale in zip(row["view_box"], [width,height,width,height])]
                               + [row["score"], 0] for _, row in group], dtype=np.float32).reshape(-1, 6)
            dimension = len(group[0][1]["embedding"]) if group else 512
            features = np.asarray([row["embedding"] for _, row in group], dtype=np.float32).reshape(-1, dimension)
            tracker.update(Boxes(data, (height,width)), img=image, feats=features)
            # Preserve first observations of unconfirmed tracks, too. Do not
            # attach Kalman-predicted boxes to frames without a real detection.
            for item in tracker.tracked_stracks:
                if item.frame_id != tracker.frame_id:
                    continue
                local = int(item.idx)
                if not 0 <= local < len(group):
                    raise RuntimeError("BoT-SORT detection index mismatch")
                hints[group[local][0]] = (key, int(item.track_id))
        pairs = []
        for oi, observation in enumerate(observations):
            for ti, track in enumerate(tracks):
                gap = frame["sample_index"] - track["last_sample"]
                if not 0 < gap <= vmms_corridor.ONLINE_TRACK_MAX_GAP:
                    continue
                score, reason = corridor_identity.identity_match(track, observation, frame, history)
                if score is None:
                    audit[reason] = audit.get(reason, 0) + 1
                    if bindings.get(hints.get(oi)) == ti:
                        audit["native_identity_proposals_vetoed"] = audit.get("native_identity_proposals_vetoed", 0) + 1
                    continue
                native = bindings.get(hints.get(oi)) == ti
                # Native Kalman/ReID can rank compatible proposals; it cannot
                # override hard geometry/appearance, or resolve near ties alone.
                pairs.append((score + (.02 if native else 0), oi, ti))
        ambiguous_observations, ambiguous_tracks = set(), set()
        for index, rejected in ((1, ambiguous_observations), (2, ambiguous_tracks)):
            grouped = {}
            for pair in pairs:
                grouped.setdefault(pair[index], []).append(pair[0])
            for key, scores in grouped.items():
                scores.sort(reverse=True)
                if len(scores)>1 and scores[0]-scores[1]<.035:
                    rejected.add(key)
        audit["ambiguous_matches_skipped"] = audit.get("ambiguous_matches_skipped",0)+len(ambiguous_observations)
        pairs = [pair for pair in pairs if pair[1] not in ambiguous_observations and pair[2] not in ambiguous_tracks]
        used = set()
        for oi, ti in vmms_corridor._optimal_pairs(pairs, len(observations), len(tracks)):
            vmms_corridor._append_observation(tracks[ti], observations[oi])
            used.add(oi)
            if oi in hints:
                bindings[hints[oi]] = ti
        for oi, observation in enumerate(observations):
            if oi in used:
                continue
            ti = len(tracks)
            identity = ti + 1
            tracks.append({"id":identity,"source_ids":[identity],"observations":[observation],
                           "last_sample":frame["sample_index"],"bearing":observation["bearing"],
                           "embedding":observation["embedding"].copy(),"merged_fragment_count":0})
            if oi in hints:
                bindings[hints[oi]] = ti
    tracks = vmms_corridor._reconnect_track_fragments(tracks)
    stable = [track for track in tracks if len(track["observations"])>=2 and
              max(row["quality"] for row in track["observations"])>=.45]
    stable.sort(key=lambda track:(-len(track["observations"]),-max(row["quality"] for row in track["observations"])))
    return stable
