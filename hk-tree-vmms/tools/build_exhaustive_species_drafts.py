#!/usr/bin/env python3
"""Build reviewable, high-recall VMMS tree-instance and species draft labels.

The script never modifies human annotations. It combines a local tree segmentation
model, projected government tree-inventory points, and a crop classifier. Inventory
species are used only when a projected point matches a detected crown; classifier-only
suggestions remain ``Unknown / 待定`` until human review.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TREE_ROOT = PROJECT_ROOT.parent
COORDINATES = PROJECT_ROOT / "outputs" / "coordinates" / "frame_coordinates.csv"
INVENTORY = PROJECT_ROOT / "outputs" / "coordinates" / "nearby_trees.csv"
DEFAULT_TREE_MODEL = (
    PROJECT_ROOT
    / "derived"
    / "training_runs"
    / "combined_tree_segmentation"
    / "yolo11m_seg_1024_20260901_104757"
    / "weights"
    / "best.pt"
)
DEFAULT_CLASSIFIER = (
    TREE_ROOT
    / "full_image_species_baseline"
    / "experiments"
    / "full_web_yolo11s_cls_20260906_seed1"
    / "finetune"
    / "weights"
    / "selected_by_val_group_macro_f1.pt"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "derived" / "exhaustive_species_drafts_20260906"
UNKNOWN = "Unknown / 待定"
PANORAMA_X_SIGN = 1.0
# Estimated from prior human species labels against same-species government
# inventory points. The Ladybug panorama centre is approximately rear-facing
# relative to the INS heading in these exported images.
PANORAMA_YAW_OFFSET_BY_STREAM = {
    "hewentian_pano": -177.0,
    "stubbs_pano_1": -178.0,
    "stubbs_pano_0": -180.0,
    "jianshazui_pano_1": -180.0,
    "jianshazui_pano_2": -180.0,
}

ONLINE_SOURCES = {
    "csdi_roadside": {
        "title": "Tree (HyD) / 路政署树木",
        "url": "https://portal.csdi.gov.hk/csdi-webpage/dataset/hyd_rcd_1632210213867_60179",
        "role": "government inventory location and species attribute",
    },
    "major_parks": {
        "title": "Trees (Major Parks) / 大型公园树木",
        "url": "https://portal.csdi.gov.hk/csdi-webpage/dataset/lcsd_rcd_1730101435734_98947",
        "role": "government inventory location and species attribute",
    },
    "taxonomy_reference": {
        "title": "Hong Kong Herbarium Plant Database / 香港植物资料库",
        "url": "https://www.herbarium.gov.hk/sc/hk-plant-database/index.html",
        "role": "scientific-name, synonym and visual-reference check",
    },
}

ROUTE_CONFIG = {
    "hewentian": {
        "preview_root": PROJECT_ROOT / "derived" / "hewentian_frame_labeler" / "preview_cache",
        "default_stream": "hewentian_pano",
    },
    "jianshazui": {
        "preview_root": PROJECT_ROOT / "derived" / "jianshazui_frame_labeler" / "preview_cache",
        "default_stream": None,
    },
    "stubbs_road": {
        "preview_root": PROJECT_ROOT / "derived" / "stubbs_road_frame_labeler" / "preview_cache",
        "default_stream": None,
    },
}


@dataclass(frozen=True)
class Frame:
    route: str
    stream_id: str
    frame_id: str
    frame_key: str
    image: Path
    source_image: str
    easting: float
    northing: float
    latitude: float
    longitude: float
    heading_deg: float
    route_distance_m: float


@dataclass(frozen=True)
class InventoryTree:
    source_dataset: str
    tree_id: str
    species: str
    location_name: str
    easting: float
    northing: float
    latitude: float
    longitude: float
    dbh: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tree-model", type=Path, default=DEFAULT_TREE_MODEL)
    parser.add_argument("--classifier", type=Path, default=DEFAULT_CLASSIFIER)
    parser.add_argument("--routes", nargs="+", choices=sorted(ROUTE_CONFIG), default=sorted(ROUTE_CONFIG))
    parser.add_argument("--tree-confidence", type=float, default=0.10)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--inventory-range-m", type=float, default=45.0)
    parser.add_argument("--max-det", type=int, default=50)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def load_coordinate_rows() -> dict[tuple[str, str], dict[str, str]]:
    return {(row["stream_id"], row["frame_id"]): row for row in read_csv(COORDINATES)}


def route_distance(rows: list[dict[str, str]]) -> dict[tuple[str, str], float]:
    output: dict[tuple[str, str], float] = {}
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["stream_id"]].append(row)
    for stream_id, items in grouped.items():
        items.sort(key=lambda item: int(item["seq_id"]))
        total = 0.0
        previous: tuple[float, float] | None = None
        for row in items:
            point = (float(row["hk80_easting"]), float(row["hk80_northing"]))
            if previous is not None:
                total += math.dist(previous, point)
            output[(stream_id, row["frame_id"])] = total
            previous = point
    return output


def collect_frames(routes: Iterable[str]) -> list[Frame]:
    rows = read_csv(COORDINATES)
    coordinates = {(row["stream_id"], row["frame_id"]): row for row in rows}
    distances = route_distance(rows)
    frames: list[Frame] = []
    for route in routes:
        config = ROUTE_CONFIG[route]
        root: Path = config["preview_root"]
        for image in sorted(root.rglob("*.jpg")):
            stream_id = config["default_stream"] or image.parent.name
            frame_id = image.stem
            row = coordinates.get((stream_id, frame_id))
            if row is None:
                raise KeyError(f"Missing coordinate row for {stream_id}/{frame_id}: {image}")
            frame_key = frame_id if config["default_stream"] else f"{stream_id}__{frame_id}"
            frames.append(
                Frame(
                    route=route,
                    stream_id=stream_id,
                    frame_id=frame_id,
                    frame_key=frame_key,
                    image=image.resolve(),
                    source_image=row["source_image_relpath"],
                    easting=float(row["hk80_easting"]),
                    northing=float(row["hk80_northing"]),
                    latitude=float(row["wgs84_latitude"]),
                    longitude=float(row["wgs84_longitude"]),
                    heading_deg=float(row["heading_deg"]) % 360.0,
                    route_distance_m=distances[(stream_id, frame_id)],
                )
            )
    # Some preview caches can contain duplicate hardlinks for the same VMMS frame.
    # Annotation identity is route/stream/frame, so emit each identity once.
    unique = {(frame.route, frame.stream_id, frame.frame_id): frame for frame in frames}
    return sorted(unique.values(), key=lambda frame: (frame.route, frame.stream_id, frame.route_distance_m, frame.frame_id))


def load_inventory() -> list[InventoryTree]:
    trees: list[InventoryTree] = []
    for row in read_csv(INVENTORY):
        species = row["species_name"].strip() or UNKNOWN
        trees.append(
            InventoryTree(
                source_dataset=row["source_dataset"],
                tree_id=row["source_tree_id"],
                species=species,
                location_name=row["location_name"],
                easting=float(row["hk80_easting"]),
                northing=float(row["hk80_northing"]),
                latitude=float(row["wgs84_latitude"]),
                longitude=float(row["wgs84_longitude"]),
                dbh=row["dbh_source_value"],
            )
        )
    return trees


def signed_angle(value: float) -> float:
    return (value + 180.0) % 360.0 - 180.0


def circular_fraction_distance(a: float, b: float) -> float:
    return abs(signed_angle((a - b) * 360.0))


def project_inventory(frame: Frame, trees: list[InventoryTree], max_range_m: float) -> list[dict[str, Any]]:
    projected: list[dict[str, Any]] = []
    yaw_offset = PANORAMA_YAW_OFFSET_BY_STREAM.get(frame.stream_id, -180.0)
    for tree in trees:
        dx, dy = tree.easting - frame.easting, tree.northing - frame.northing
        distance = math.hypot(dx, dy)
        if distance > max_range_m or distance < 1.5:
            continue
        bearing = (math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0
        relative = signed_angle(bearing - frame.heading_deg)
        projected.append(
            {
                "tree": tree,
                "distance_m": distance,
                "bearing_deg": bearing,
                "relative_bearing_deg": relative,
                "x": (0.5 + (PANORAMA_X_SIGN * relative + yaw_offset) / 360.0) % 1.0,
                "panorama_yaw_offset_deg": yaw_offset,
            }
        )
    return projected


def polygon_bbox(points: list[list[float]]) -> dict[str, float]:
    xs, ys = [point[0] for point in points], [point[1] for point in points]
    left, right, top, bottom = min(xs), max(xs), min(ys), max(ys)
    return {
        "x_center": (left + right) / 2.0,
        "y_center": (top + bottom) / 2.0,
        "width": right - left,
        "height": bottom - top,
        "left": left,
        "top": top,
        "right": right,
        "bottom": bottom,
    }


def canonical_latin(value: str) -> str:
    value = re.sub(r"^\d+_", "", value).replace("_", " ")
    value = value.replace("(syn.", " (").replace("（", "(")
    latin = value.split("(", 1)[0].strip()
    words = [word.strip(",;") for word in latin.split()]
    return " ".join(words[:2]).casefold()


def classify_crops(model: YOLO, image: Image.Image, boxes: list[dict[str, float]]) -> list[dict[str, Any]]:
    if not boxes:
        return []
    crops: list[Image.Image] = []
    width, height = image.size
    for box in boxes:
        pad_x = max(8, int(box["width"] * width * 0.08))
        pad_y = max(8, int(box["height"] * height * 0.08))
        left = max(0, int(box["left"] * width) - pad_x)
        top = max(0, int(box["top"] * height) - pad_y)
        right = min(width, int(math.ceil(box["right"] * width)) + pad_x)
        bottom = min(height, int(math.ceil(box["bottom"] * height)) + pad_y)
        crops.append(image.crop((left, top, right, bottom)).convert("RGB"))
    results = model.predict(crops, imgsz=224, device=0, verbose=False)
    output: list[dict[str, Any]] = []
    for result in results:
        probabilities = result.probs
        if probabilities is None:
            output.append({"top1": UNKNOWN, "confidence": 0.0, "top5": []})
            continue
        indices = probabilities.top5
        top5 = [
            {"species": result.names[int(index)], "confidence": float(probabilities.data[int(index)])}
            for index in indices
        ]
        output.append({"top1": top5[0]["species"], "confidence": top5[0]["confidence"], "top5": top5})
    return output


def bbox_overlap(first: dict[str, float], second: dict[str, float]) -> tuple[float, float]:
    intersection_width = max(0.0, min(first["right"], second["right"]) - max(first["left"], second["left"]))
    intersection_height = max(0.0, min(first["bottom"], second["bottom"]) - max(first["top"], second["top"]))
    intersection = intersection_width * intersection_height
    first_area = max(1e-9, first["width"] * first["height"])
    second_area = max(1e-9, second["width"] * second["height"])
    union = first_area + second_area - intersection
    return intersection / union, intersection / min(first_area, second_area)


def infer_tree_instances(
    model: YOLO,
    image: Image.Image,
    image_size: int,
    confidence: float,
    max_det: int,
) -> list[dict[str, Any]]:
    """Run full-panorama plus overlapping square tiles, then suppress duplicates."""
    width, height = image.size
    windows: list[tuple[int, int, Image.Image]] = [(0, width, image)]
    if width >= int(height * 1.5):
        tile_width = min(width, height)
        last = width - tile_width
        starts = sorted({0, last, int(round(last / 2.0))})
        windows.extend((start, start + tile_width, image.crop((start, 0, start + tile_width, height))) for start in starts)

    raw: list[dict[str, Any]] = []
    results = model.predict(
        [window[2] for window in windows],
        imgsz=image_size,
        conf=confidence,
        iou=0.60,
        max_det=max_det,
        device=0,
        retina_masks=True,
        verbose=False,
    )
    for (left_px, right_px, _), result in zip(windows, results):
        if result.masks is None or result.boxes is None:
            continue
        window_width = right_px - left_px
        tiled = window_width != width
        for points_array, score in zip(result.masks.xyn, result.boxes.conf.tolist()):
            points = [
                [round((left_px + float(x) * window_width) / width, 6), round(float(y), 6)]
                for x, y in points_array.tolist()
            ]
            if len(points) < 3:
                continue
            box = polygon_bbox(points)
            area_proxy = box["width"] * box["height"]
            if area_proxy < 0.00035 or area_proxy > 0.80:
                continue
            # Ladybug panoramas contain a dark vehicle/invalid band at the bottom.
            # A genuine roadside tree crown extends above this band even when its
            # trunk reaches the lower edge.
            if box["top"] >= 0.60 or box["y_center"] >= 0.82:
                continue
            # A tile prediction cut exactly at its inner border is retained, but marked
            # lower priority because it may be a partial crown proposal.
            inner_left = left_px > 0 and box["left"] <= left_px / width + 0.003
            inner_right = right_px < width and box["right"] >= right_px / width - 0.003
            raw.append(
                {
                    "points": points,
                    "bbox": box,
                    "tree_confidence": float(score),
                    "inference_view": "tile" if tiled else "full_panorama",
                    "tile_boundary_truncated": bool(tiled and (inner_left or inner_right)),
                }
            )

    kept: list[dict[str, Any]] = []
    for proposal in sorted(raw, key=lambda item: (item["tree_confidence"], item["bbox"]["width"] * item["bbox"]["height"]), reverse=True):
        duplicate = False
        for existing in kept:
            iou, smaller_coverage = bbox_overlap(proposal["bbox"], existing["bbox"])
            if iou >= 0.58 or smaller_coverage >= 0.88:
                duplicate = True
                break
        if not duplicate:
            kept.append(proposal)
    return sorted(kept, key=lambda item: item["bbox"]["x_center"])


def match_inventory(
    predictions: list[dict[str, Any]], projected: list[dict[str, Any]]
) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
    candidates: list[tuple[float, float, int, int, float]] = []
    for pred_index, prediction in enumerate(predictions):
        box = prediction["bbox"]
        threshold_deg = max(6.0, min(30.0, box["width"] * 360.0 * 0.58 + 4.0))
        for tree_index, item in enumerate(projected):
            angle = circular_fraction_distance(box["x_center"], item["x"])
            if angle > threshold_deg:
                continue
            cost = angle / threshold_deg + 0.12 * item["distance_m"] / 45.0
            candidates.append((cost, angle, pred_index, tree_index, threshold_deg))
    assignments: dict[int, dict[str, Any]] = {}
    used_trees: set[int] = set()
    for cost, angle, pred_index, tree_index, threshold in sorted(candidates):
        if pred_index in assignments or tree_index in used_trees:
            continue
        item = dict(projected[tree_index])
        item.update({"angular_error_deg": angle, "angular_threshold_deg": threshold, "cost": cost})
        assignments[pred_index] = item
        used_trees.add(tree_index)
    unmatched = [item for index, item in enumerate(projected) if index not in used_trees]
    return assignments, unmatched


def species_tier(match: dict[str, Any] | None, classifier: dict[str, Any]) -> tuple[str, bool, bool]:
    if match is None:
        return "unknown", False, True
    tree: InventoryTree = match["tree"]
    agreement = canonical_latin(tree.species) == canonical_latin(classifier["top1"])
    distance, angle = match["distance_m"], match["angular_error_deg"]
    if distance <= 20.0 and angle <= 3.0:
        return "high", agreement, False
    if distance <= 35.0 and angle <= min(8.0, match["angular_threshold_deg"] * 0.5):
        return "medium", agreement, False
    return "low", agreement, True


def safe_key(frame: Frame) -> str:
    return f"{frame.route}__{frame.stream_id}__{frame.frame_id}"


def link_or_copy(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return "existing"
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def draw_overlay(image: Image.Image, labels: list[dict[str, Any]], destination: Path) -> None:
    canvas = image.copy().convert("RGB")
    draw = ImageDraw.Draw(canvas, "RGBA")
    width, height = canvas.size
    palette = {
        "high": (0, 210, 90, 80),
        "medium": (255, 190, 0, 80),
        "low": (255, 90, 0, 80),
        "unknown": (220, 40, 80, 80),
    }
    for label in labels:
        points = [(int(x * width), int(y * height)) for x, y in label["points"]]
        color = palette[label["species_confidence_tier"]]
        draw.polygon(points, fill=color, outline=(*color[:3], 255), width=3)
        box = label["bbox"]
        anchor = (int(box["left"] * width), max(0, int(box["top"] * height) - 18))
        short_name = label["species"].split(" ")[:2]
        text = f"{' '.join(short_name)} [{label['species_confidence_tier']}]"
        draw.rectangle((anchor[0], anchor[1], min(width, anchor[0] + 260), anchor[1] + 18), fill=(0, 0, 0, 180))
        draw.text((anchor[0] + 2, anchor[1] + 2), text, fill=(255, 255, 255, 255), font=ImageFont.load_default())
    destination.parent.mkdir(parents=True, exist_ok=True)
    if canvas.width > 1400:
        canvas.thumbnail((1400, 700), Image.Resampling.LANCZOS)
    canvas.save(destination, quality=88, optimize=True)


def write_contact_sheet(paths: list[Path], destination: Path, title: str, limit: int = 24) -> None:
    selected = paths[:limit]
    if not selected:
        return
    thumbs: list[Image.Image] = []
    for path in selected:
        with Image.open(path) as image:
            thumb = image.convert("RGB")
            thumb.thumbnail((480, 240), Image.Resampling.LANCZOS)
            thumbs.append(thumb.copy())
    columns = 3
    rows = math.ceil(len(thumbs) / columns)
    sheet = Image.new("RGB", (columns * 480, rows * 270 + 34), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((10, 10), title, fill="black", font=ImageFont.load_default())
    for index, thumb in enumerate(thumbs):
        x, y = (index % columns) * 480, 34 + (index // columns) * 270
        sheet.paste(thumb, (x, y))
        draw.text((x + 4, y + 242), selected[index].stem, fill="black", font=ImageFont.load_default())
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination, quality=90)


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists() and args.overwrite:
        if PROJECT_ROOT.resolve() not in output.parents:
            raise ValueError(f"Refusing to replace output outside project: {output}")
        shutil.rmtree(output)
    summary_path = output / "summary.json"
    if summary_path.is_file() and not args.overwrite:
        print(summary_path.read_text(encoding="utf-8"))
        return
    for path in (COORDINATES, INVENTORY, args.tree_model, args.classifier):
        if not path.is_file():
            raise FileNotFoundError(path)

    frames = collect_frames(args.routes)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        # Round-robin by route so pilots include every requested domain.
        grouped: dict[str, list[Frame]] = defaultdict(list)
        for frame in frames:
            grouped[frame.route].append(frame)
        sampled: list[Frame] = []
        route_names = sorted(grouped)
        index = 0
        while len(sampled) < min(args.limit, len(frames)):
            route = route_names[index % len(route_names)]
            route_frames = grouped[route]
            offset = index // len(route_names)
            if offset < len(route_frames):
                sampled.append(route_frames[offset])
            index += 1
        frames = sampled

    inventories = load_inventory()
    tree_model = YOLO(str(args.tree_model.resolve()))
    classifier = YOLO(str(args.classifier.resolve()))
    (output / "records").mkdir(parents=True, exist_ok=True)
    (output / "labels").mkdir(parents=True, exist_ok=True)
    (output / "images").mkdir(parents=True, exist_ok=True)
    (output / "overlays").mkdir(parents=True, exist_ok=True)

    review_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    overlay_paths: dict[str, list[Path]] = defaultdict(list)
    tier_counts: Counter[str] = Counter()
    species_counts: Counter[str] = Counter()
    link_counts: Counter[str] = Counter()
    total_instances = 0
    empty_frames = 0

    for index, frame in enumerate(frames, start=1):
        with Image.open(frame.image) as opened:
            image = opened.convert("RGB")
        predictions = infer_tree_instances(
            tree_model,
            image,
            args.image_size,
            args.tree_confidence,
            args.max_det,
        )

        classifications = classify_crops(classifier, image, [item["bbox"] for item in predictions])
        projected = project_inventory(frame, inventories, args.inventory_range_m)
        matches, unmatched_inventory = match_inventory(predictions, projected)
        labels: list[dict[str, Any]] = []
        for pred_index, (prediction, classification) in enumerate(zip(predictions, classifications)):
            match = matches.get(pred_index)
            tier, agreement, needs_review = species_tier(match, classification)
            needs_review = needs_review or prediction["tile_boundary_truncated"]
            if match is None:
                species = UNKNOWN
                inventory_payload = None
                method = "classifier_suggestion_only"
            else:
                tree: InventoryTree = match["tree"]
                species = tree.species
                method = "government_inventory_projection"
                inventory_payload = {
                    "source_dataset": tree.source_dataset,
                    "source_tree_id": tree.tree_id,
                    "species": tree.species,
                    "location_name": tree.location_name,
                    "dbh_source_value": tree.dbh,
                    "latitude": tree.latitude,
                    "longitude": tree.longitude,
                    "distance_m": round(match["distance_m"], 3),
                    "projected_x": round(match["x"], 6),
                    "angular_error_deg": round(match["angular_error_deg"], 3),
                    "angular_threshold_deg": round(match["angular_threshold_deg"], 3),
                    "panorama_yaw_offset_deg": match["panorama_yaw_offset_deg"],
                    "online_source": ONLINE_SOURCES.get(tree.source_dataset),
                }
            label = {
                "label_id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{safe_key(frame)}:{pred_index}")),
                "species": species,
                "points": prediction["points"],
                "bbox": prediction["bbox"],
                "tree_confidence": round(prediction["tree_confidence"], 6),
                "inference_view": prediction["inference_view"],
                "tile_boundary_truncated": prediction["tile_boundary_truncated"],
                "species_method": method,
                "species_confidence_tier": tier,
                "requires_human_review": needs_review,
                "inventory_classifier_agreement": agreement,
                "inventory_match": inventory_payload,
                "classifier_suggestion": classification,
            }
            labels.append(label)
            tier_counts[tier] += 1
            species_counts[species] += 1
        total_instances += len(labels)
        empty_frames += int(not labels)

        key = safe_key(frame)
        record = {
            "schema_version": "vmms_exhaustive_species_draft_v1",
            "status": "auto_draft_requires_human_acceptance",
            "frame_key": frame.frame_key,
            "dataset_route": frame.route,
            "stream_id": frame.stream_id,
            "frame_id": frame.frame_id,
            "preview_image": str(frame.image),
            "source_image_relpath": frame.source_image,
            "image_size": {"width": image.width, "height": image.height},
            "camera": {
                "easting": frame.easting,
                "northing": frame.northing,
                "latitude": frame.latitude,
                "longitude": frame.longitude,
                "heading_deg": frame.heading_deg,
                "route_distance_m": frame.route_distance_m,
            },
            "labels": labels,
            "unmatched_inventory_candidates": [
                {
                    "source_dataset": item["tree"].source_dataset,
                    "source_tree_id": item["tree"].tree_id,
                    "species": item["tree"].species,
                    "distance_m": round(item["distance_m"], 3),
                    "projected_x": round(item["x"], 6),
                }
                for item in unmatched_inventory
            ],
            "provenance": {
                "tree_model": str(args.tree_model.resolve()),
                "classifier": str(args.classifier.resolve()),
                "inventory": str(INVENTORY.resolve()),
                "online_sources": ONLINE_SOURCES,
            },
        }
        (output / "records" / f"{key}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

        class_names = sorted(set(species_counts) | {UNKNOWN})
        # Temporary species strings are retained in records. YOLO IDs are finalized below.
        label_lines = [json.dumps({"species": item["species"], "points": item["points"]}, ensure_ascii=False) for item in labels]
        (output / "labels" / f"{key}.draft.jsonl").write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")
        image_destination = output / "images" / f"{key}.jpg"
        link_counts[link_or_copy(frame.image, image_destination)] += 1
        overlay = output / "overlays" / frame.route / f"{key}.jpg"
        draw_overlay(image, labels, overlay)
        overlay_paths[frame.route].append(overlay)

        frame_rows.append(
            {
                "frame_key": frame.frame_key,
                "route": frame.route,
                "stream_id": frame.stream_id,
                "frame_id": frame.frame_id,
                "image": str(image_destination),
                "record": str(output / "records" / f"{key}.json"),
                "overlay": str(overlay),
                "instances": len(labels),
                "inventory_candidates": len(projected),
                "review_required_instances": sum(item["requires_human_review"] for item in labels),
                "route_block_id": f"{frame.stream_id}_{int(frame.route_distance_m // 100):04d}",
            }
        )
        for label in labels:
            match = label["inventory_match"] or {}
            review_rows.append(
                {
                    "frame_key": frame.frame_key,
                    "route": frame.route,
                    "stream_id": frame.stream_id,
                    "frame_id": frame.frame_id,
                    "label_id": label["label_id"],
                    "species": label["species"],
                    "species_confidence_tier": label["species_confidence_tier"],
                    "requires_human_review": label["requires_human_review"],
                    "tree_confidence": label["tree_confidence"],
                    "species_method": label["species_method"],
                    "inventory_tree_id": match.get("source_tree_id", ""),
                    "inventory_distance_m": match.get("distance_m", ""),
                    "angular_error_deg": match.get("angular_error_deg", ""),
                    "classifier_top1": label["classifier_suggestion"]["top1"],
                    "classifier_confidence": round(label["classifier_suggestion"]["confidence"], 6),
                    "inventory_classifier_agreement": label["inventory_classifier_agreement"],
                    "overlay": str(overlay),
                }
            )
        if index % 25 == 0 or index == len(frames):
            print(f"processed {index}/{len(frames)} frames, {total_instances} draft instances", flush=True)

    classes = sorted(species_counts, key=lambda name: (name == UNKNOWN, name.casefold()))
    if UNKNOWN not in classes:
        classes.append(UNKNOWN)
    class_to_id = {name: index for index, name in enumerate(classes)}
    (output / "classes.json").write_text(json.dumps(classes, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for record_path in sorted((output / "records").glob("*.json")):
        record = json.loads(record_path.read_text(encoding="utf-8"))
        lines: list[str] = []
        for label in record["labels"]:
            label["class_id"] = class_to_id[label["species"]]
            flat = " ".join(f"{coordinate:.6f}" for point in label["points"] for coordinate in point)
            lines.append(f"{label['class_id']} {flat}")
        record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        yolo_path = output / "labels" / f"{record_path.stem}.txt"
        yolo_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    (output / "data.yaml").write_text(
        "path: \"" + output.as_posix() + "\"\ntrain: images\nval: images\nnames:\n"
        + "".join(f"  {index}: {json.dumps(name, ensure_ascii=False)}\n" for index, name in enumerate(classes)),
        encoding="utf-8",
    )

    def write_table(path: Path, rows: list[dict[str, Any]]) -> None:
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    write_table(output / "frame_manifest.csv", frame_rows)
    write_table(output / "review_queue.csv", sorted(review_rows, key=lambda row: (not row["requires_human_review"], row["species_confidence_tier"], row["route"], row["frame_key"])))
    for route, paths in overlay_paths.items():
        priority = sorted(
            paths,
            key=lambda path: -sum(
                item["requires_human_review"]
                for item in json.loads((output / "records" / f"{path.stem}.json").read_text(encoding="utf-8"))["labels"]
            ),
        )
        write_contact_sheet(priority, output / "contact_sheets" / f"{route}_priority.jpg", f"{route} priority review")

    summary = {
        "status": "auto_drafts_complete_requires_human_acceptance",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "routes": args.routes,
        "frames": len(frames),
        "frames_without_predictions": empty_frames,
        "draft_instances": total_instances,
        "species_classes_including_unknown": len(classes),
        "species_confidence_tiers": dict(tier_counts),
        "species_counts": dict(species_counts.most_common()),
        "image_materialization": dict(link_counts),
        "parameters": {
            "tree_confidence": args.tree_confidence,
            "image_size": args.image_size,
            "inventory_range_m": args.inventory_range_m,
            "max_det": args.max_det,
            "panorama_x_sign": PANORAMA_X_SIGN,
            "panorama_yaw_offset_by_stream": PANORAMA_YAW_OFFSET_BY_STREAM,
        },
        "provenance": {
            "tree_model": str(args.tree_model.resolve()),
            "tree_model_sha256": sha256_file(args.tree_model.resolve()),
            "classifier": str(args.classifier.resolve()),
            "classifier_sha256": sha256_file(args.classifier.resolve()),
            "coordinate_csv": str(COORDINATES.resolve()),
            "coordinate_csv_sha256": sha256_file(COORDINATES),
            "inventory_csv": str(INVENTORY.resolve()),
            "inventory_csv_sha256": sha256_file(INVENTORY),
            "online_sources": ONLINE_SOURCES,
        },
        "safety": {
            "human_annotations_modified": False,
            "classifier_only_species_auto_accepted": False,
            "all_outputs_are_review_drafts": True,
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
