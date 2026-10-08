"""Conservative instance association for spherical street-view sequences.

Inference only. CLIP features are a generic appearance cue, not a trained tree
ReID encoder. Geometry must pass before appearance can link two observations.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from . import panorama

APPEARANCE_ANCHOR_MIN = .82
APPEARANCE_RECENT_MIN = .82
VIEW_PAIR_MIN = .82


def crop_evidence(image: Image.Image) -> dict:
    sample = image.copy().convert("RGB")
    sample.thumbnail((320, 320))
    pixels = np.asarray(sample, dtype=np.float32)
    red, green, blue = pixels.transpose(2, 0, 1)
    height, width = green.shape
    vegetation = ((green > 40) & (green > red * 1.03) & (green > blue * 1.02) &
                  (2 * green - red - blue > 10))
    centre = vegetation[int(height * .2):max(1, int(height * .8)),
                        int(width * .2):max(1, int(width * .8))]
    dark = float((pixels.max(axis=2) < 42).mean())
    return {"green_fraction": float(vegetation.mean()),
            "central_green_fraction": float(centre.mean()),
            "dark_fraction": dark,
            "usable": bool(vegetation.mean() >= .035 and centre.mean() >= .025 and dark < .65)}


def motion_image(image: Image.Image) -> np.ndarray:
    import cv2
    return cv2.cvtColor(np.asarray(image.resize((1280, 640))), cv2.COLOR_RGB2GRAY)


def motion_snapshot(previous: np.ndarray | None, current: np.ndarray) -> dict:
    """Forward/backward checked sparse LK flow; horizontal seam is periodic.

Local point motion inside the last target box is used, rather than assuming all
objects obey one homography in a translating, parallax-heavy 360-degree scene.
"""
    import cv2
    if previous is None:
        return {"method": "periodic sparse LK, forward/backward check", "points": []}
    height, width = previous.shape
    mask = np.zeros_like(previous)
    mask[int(height * .14):int(height * .75)] = 255
    mask[(previous < 30) | (previous > 245)] = 0
    points = cv2.goodFeaturesToTrack(previous, 1200, .015, 6, mask=mask)
    if points is None:
        return {"method": "periodic sparse LK, forward/backward check", "points": []}
    # The middle copy preserves seam-crossing point trajectories.
    points[:, :, 0] += width
    before = np.tile(previous, (1, 3))
    after = np.tile(current, (1, 3))
    settings = dict(winSize=(31, 31), maxLevel=4,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01))
    forward, status, _ = cv2.calcOpticalFlowPyrLK(before, after, points, None, **settings)
    if forward is None:
        return {"method": "periodic sparse LK, forward/backward check", "points": []}
    backward, back_status, _ = cv2.calcOpticalFlowPyrLK(after, before, forward, None, **settings)
    if backward is None:
        return {"method": "periodic sparse LK, forward/backward check", "points": []}
    source, target, back = points[:, 0], forward[:, 0], backward[:, 0]
    valid = ((status[:, 0] == 1) & (back_status[:, 0] == 1) &
             (np.linalg.norm(source - back, axis=1) <= 1.2) &
             np.isfinite(target).all(axis=1) & (target[:, 1] >= 0) & (target[:, 1] < height))
    source, target = source[valid], target[valid]
    rows = np.column_stack(((source[:, 0] - width) / width, source[:, 1] / height,
                            (target[:, 0] - width) / width, target[:, 1] / height))
    return {"method": "periodic sparse LK, forward/backward check",
            "points": np.round(rows, 6).tolist()}


def _circular_delta(left: float, right: float) -> float:
    return (left - right + .5) % 1 - .5


def _local_shift(box: list, motion: dict) -> tuple[float, float] | None:
    points = np.asarray(motion.get("points", []), dtype=np.float32)
    if len(points) < 4:
        return None
    centre = (box[0] + box[2]) / 2
    width = box[2] - box[0]
    relative_x = (points[:, 0] - centre + .5) % 1 - .5
    points = points[(np.abs(relative_x) <= width / 2) &
                    (points[:, 1] >= box[1]) & (points[:, 1] <= box[3])]
    if len(points) < 4:
        return None
    shifts = points[:, 2:] - points[:, :2]
    shifts[:, 0] = (shifts[:, 0] + .5) % 1 - .5
    median = np.median(shifts, axis=0)
    residual = np.linalg.norm(shifts - median, axis=1)
    inliers = shifts[residual <= .018]
    if len(inliers) < 4 or len(inliers) / len(points) < .65:
        return None
    median = np.median(inliers, axis=0)
    if abs(median[0]) > .10 or abs(median[1]) > .10:
        return None
    return float(median[0]), float(median[1])


def predicted_box(track: dict, frame: dict, history: dict) -> tuple[list, bool]:
    last = track["observations"][-1]
    box = list(last["box"])
    flow_supported = True
    for sample in range(last["sample_index"] + 1, frame["sample_index"] + 1):
        next_frame = history[sample]
        shift = _local_shift(box, next_frame.get("motion", {}))
        if shift is None:
            flow_supported = False
            prior = history.get(sample - 1, {})
            # Real heading is used only when provided; folder mode remains 0.
            heading_delta = (next_frame.get("heading_deg", 0) - prior.get("heading_deg", 0) + 180) % 360 - 180
            shift = (-heading_delta / 360, 0)
        box = [box[0] + shift[0], box[1] + shift[1],
               box[2] + shift[0], box[3] + shift[1]]
    return box, flow_supported


def appearance_score(track: dict, observation: dict) -> tuple[float | None, str]:
    vector = observation["embedding"]
    rows = track["observations"]
    # An immutable anchor prevents semantic drift through successive wrong links.
    anchor = float(vector @ rows[0]["embedding"])
    recent = float(vector @ rows[-1]["embedding"])
    gallery = np.asarray([float(vector @ row["embedding"]) for row in rows])
    if anchor < APPEARANCE_ANCHOR_MIN or recent < APPEARANCE_RECENT_MIN:
        return None, "appearance_anchor_or_recent"
    if float(gallery.min()) < VIEW_PAIR_MIN:
        return None, "appearance_gallery"
    return float(.4 * anchor + .35 * recent + .25 * np.median(gallery)), "accepted"


def identity_match(track: dict, observation: dict, frame: dict,
                   history: dict) -> tuple[float | None, str]:
    prediction, flow = predicted_box(track, frame, history)
    current = observation["box"]
    pw, ph = prediction[2] - prediction[0], prediction[3] - prediction[1]
    cw, ch = current[2] - current[0], current[3] - current[1]
    if min(pw, ph, cw, ch) <= 0:
        return None, "invalid_geometry"
    if not .45 <= cw / pw <= 2.2 or not .45 <= ch / ph <= 2.2:
        return None, "scale_jump"
    dx = abs(_circular_delta((current[0] + current[2]) / 2,
                             (prediction[0] + prediction[2]) / 2))
    dy = abs((current[1] + current[3] - prediction[1] - prediction[3]) / 2)
    # Appearance NEVER rescues a spatially impossible association.
    x_limit = min(.035 if flow else .025, max(.012, pw * .6))
    y_limit = min(.075, max(.025, ph * .35))
    overlap = panorama.circular_iou(prediction, current)
    if dx > x_limit or dy > y_limit or overlap < .08:
        return None, "motion_or_overlap"
    appearance, reason = appearance_score(track, observation)
    if appearance is None:
        return None, reason
    return float(appearance * .55 + overlap * .4 + (.05 if flow else 0)), "accepted"


def views_compatible(left: dict, right: dict) -> bool:
    return float(left["embedding"] @ right["embedding"]) >= VIEW_PAIR_MIN
