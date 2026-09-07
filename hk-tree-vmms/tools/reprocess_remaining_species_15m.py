#!/usr/bin/env python3
"""Prelabel the remaining review sequence at 15 m without touching submitted work."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from PIL import Image
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_exhaustive_species_drafts import (  # noqa: E402
    Frame,
    canonical_latin,
    infer_tree_instances,
    load_inventory,
    match_inventory,
    project_inventory,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
VMMS_ROOT = WORKSPACE_ROOT.parent / "vmms"
COORDINATES = PROJECT_ROOT / "outputs" / "coordinates" / "frame_coordinates.csv"
LABELER = PROJECT_ROOT / "derived" / "exhaustive_species_frame_labeler"
TREE_MODEL = (
    PROJECT_ROOT
    / "derived/training_runs/combined_tree_segmentation/"
    / "yolo11m_seg_1024_20260901_104757/weights/best.pt"
)
FILTER_MODEL = (
    WORKSPACE_ROOT
    / "runs/classify/binary_filter_iteration2/yolo11s_binary_tree_filter_v2/weights/best.pt"
)
ROUTER = (
    WORKSPACE_ROOT
    / "runs/classify/review_replay_species/species_fusion_adapter_v2/router_rules.json"
)
BROAD_MODEL = (
    WORKSPACE_ROOT
    / "full_image_species_baseline/experiments/full_web_yolo11s_cls_20260906_seed1/"
    / "finetune/weights/selected_by_val_group_macro_f1.pt"
)
BROAD_SUMMARY = (
    WORKSPACE_ROOT
    / "full_image_species_baseline/experiments/full_web_yolo11s_cls_20260906_seed1/"
    / "datasets/crop/dataset_summary.json"
)
MEMORY_RULES = (
    WORKSPACE_ROOT
    / "runs/classify/review_replay_species/human_memory_v1/rules.json"
)
STREAMS = [
    "hewentian_pano",
    "jianshazui_pano_1",
    "jianshazui_pano_2",
    "stubbs_pano_0",
    "stubbs_pano_1",
]
BANYAN = "榕树"
DEFAULT_PREPROCESS = {
    "recipe_version": "photo_v1",
    "exposure_ev": 0.0,
    "contrast": 1.0,
    "reason": "normal",
    "auto_suggested": False,
    "human_accepted": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spacing-m", type=float, default=15.0)
    parser.add_argument("--hwt-reviewed-through-m", type=float, default=1559.18)
    parser.add_argument("--filter-threshold", type=float, default=0.41)
    parser.add_argument(
        "--protect-drafts-after-utc",
        default="2026-09-06T13:04:08+00:00",
        help="Never replace browser-autosaved drafts newer than this timestamp",
    )
    parser.add_argument("--tree-confidence", type=float, default=0.10)
    parser.add_argument("--device", default="0")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def normalize_species(value: object) -> str:
    name = str(value or "").strip()
    folded = name.casefold()
    if folded.startswith(("ficus microcarpa", "ficus benjamina")):
        return BANYAN
    if any(token in name for token in ("細葉榕", "细叶榕", "垂葉榕", "垂叶榕")):
        return BANYAN
    return BANYAN if name == "榕樹" else name


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_frames(spacing_m: float, hwt_cutoff: float) -> list[Frame]:
    with COORDINATES.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["stream_id"] in STREAMS:
            grouped[row["stream_id"]].append(row)
    frames: list[Frame] = []
    for stream_id in STREAMS:
        route = (
            "hewentian"
            if stream_id == "hewentian_pano"
            else "jianshazui"
            if stream_id.startswith("jianshazui")
            else "stubbs_road"
        )
        items = sorted(grouped[stream_id], key=lambda row: int(row["seq_id"]))
        distances: list[float] = []
        total = 0.0
        previous: tuple[float, float] | None = None
        for row in items:
            point = (float(row["hk80_easting"]), float(row["hk80_northing"]))
            if previous is not None:
                total += math.dist(previous, point)
            previous = point
            distances.append(total)
        minimum = hwt_cutoff if stream_id == "hewentian_pano" else 0.0
        eligible = [index for index, distance in enumerate(distances) if distance >= minimum]
        if not eligible:
            continue
        selected = [eligible[0]]
        last_distance = distances[eligible[0]]
        for index in eligible[1:]:
            if distances[index] - last_distance >= spacing_m:
                selected.append(index)
                last_distance = distances[index]
        if eligible[-1] not in selected:
            selected.append(eligible[-1])
        for index in selected:
            row = items[index]
            frame_id = row["frame_id"]
            source = (VMMS_ROOT / row["source_image_relpath"]).resolve()
            frames.append(
                Frame(
                    route=route,
                    stream_id=stream_id,
                    frame_id=frame_id,
                    frame_key=f"{stream_id}__{frame_id}",
                    image=source,
                    source_image=row["source_image_relpath"],
                    easting=float(row["hk80_easting"]),
                    northing=float(row["hk80_northing"]),
                    latitude=float(row["wgs84_latitude"]),
                    longitude=float(row["wgs84_longitude"]),
                    heading_deg=float(row["heading_deg"]) % 360.0,
                    route_distance_m=distances[index],
                )
            )
    return frames


def preview(frame: Frame) -> Image.Image:
    target = LABELER / "preview_cache" / frame.stream_id / f"{frame.frame_id}.jpg"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        with Image.open(target) as image:
            return image.convert("RGB")
    with Image.open(frame.image) as opened:
        image = opened.convert("RGB")
        image.thumbnail((2048, 1024), Image.Resampling.LANCZOS)
    temporary = target.with_suffix(".tmp")
    image.save(temporary, format="JPEG", quality=90, optimize=True)
    temporary.replace(target)
    return image


def crops(image: Image.Image, predictions: list[dict[str, Any]]) -> list[Image.Image]:
    width, height = image.size
    output = []
    for prediction in predictions:
        box = prediction["bbox"]
        pad_x = max(8, int(box["width"] * width * 0.10))
        pad_y = max(8, int(box["height"] * height * 0.10))
        output.append(
            image.crop(
                (
                    max(0, int(box["left"] * width) - pad_x),
                    max(0, int(box["top"] * height) - pad_y),
                    min(width, int(math.ceil(box["right"] * width)) + pad_x),
                    min(height, int(math.ceil(box["bottom"] * height)) + pad_y),
                )
            ).convert("RGB")
        )
    return output


def map_probabilities(result: Any, mapping: dict[str, str], species: list[str]) -> np.ndarray:
    output = np.zeros(len(species), dtype=np.float32)
    target = {name: index for index, name in enumerate(species)}
    for index, probability in enumerate(result.probs.data.cpu().numpy()):
        name = normalize_species(mapping.get(result.names[index], result.names[index]))
        if name in target:
            output[target[name]] += float(probability)
    return output


def top_options(result: Any, mapping: dict[str, str], limit: int = 3) -> list[dict[str, Any]]:
    return [
        {
            "species": normalize_species(mapping.get(result.names[int(index)], result.names[int(index)])),
            "confidence": round(float(result.probs.data[int(index)]), 6),
        }
        for index in result.probs.top5[:limit]
    ]


def main() -> None:
    cfg = parse_args()
    runtime = LABELER / "runtime"
    reviews_path = runtime / "frame_review_state.json"
    drafts_path = runtime / "frame_drafts.json"
    session_path = runtime / "session.json"
    classes_path = LABELER / "annotations" / "classes.json"
    reviews = read_json(reviews_path)
    drafts = read_json(drafts_path)
    classes = [normalize_species(name) for name in json.loads(classes_path.read_text(encoding="utf-8"))]
    latin_to_ui = {canonical_latin(name): name for name in classes if canonical_latin(name)}

    rules = read_json(ROUTER)
    memory_rules = read_json(MEMORY_RULES)
    species = rules["species"]
    old_mapping = rules["old_mapping"]
    new_mapping = rules["new_mapping"]
    old_model = YOLO(rules["old_model"])
    new_model = YOLO(rules["new_model"])
    iteration3_model = YOLO(rules["iteration3_model"])
    iteration3_mapping = rules["iteration3_mapping"]
    tree_model = YOLO(str(TREE_MODEL.resolve()))
    filter_model = YOLO(str(FILTER_MODEL.resolve()))
    broad_model = YOLO(str(BROAD_MODEL.resolve()))
    memory_model = YOLO(memory_rules["embedding_model"])
    with np.load(memory_rules["memory"]) as memory_file:
        memory_embeddings = memory_file["embeddings"]
    memory_manifest = json.loads(
        Path(memory_rules["manifest"]).read_text(encoding="utf-8")
    )
    memory_rule_by_species = {
        rule["species"]: rule for rule in memory_rules.get("rules") or []
    }
    broad_summary = read_json(BROAD_SUMMARY)
    broad_mapping = {
        broad_summary["class_folders"][str(index)]: normalize_species(name)
        for index, name in enumerate(broad_summary["class_names"])
    }
    not_tree_index = next(
        index for index, name in filter_model.names.items() if name == "1_not_tree"
    )
    inventories = load_inventory()
    protected_keys = {
        frame_key
        for frame_key, value in drafts.items()
        if str(value.get("updated_at_utc") or "") > cfg.protect_drafts_after_utc
    }
    frames = [
        frame
        for frame in load_frames(cfg.spacing_m, cfg.hwt_reviewed_through_m)
        if frame.frame_key not in reviews and frame.frame_key not in protected_keys
    ]
    if cfg.limit:
        frames = frames[: cfg.limit]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = PROJECT_ROOT / "backups" / f"species_15m_reprocess_{timestamp}"
    backup.mkdir(parents=True, exist_ok=False)
    for path in (drafts_path, session_path):
        if path.is_file():
            shutil.copy2(path, backup / path.name)

    stats = Counter()
    route_stats: dict[str, Counter] = defaultdict(Counter)
    for position, frame in enumerate(frames, start=1):
        image = preview(frame)
        predictions = infer_tree_instances(
            tree_model, image, image_size=1024, confidence=cfg.tree_confidence, max_det=50
        )
        crop_images = crops(image, predictions)
        labels = []
        if crop_images:
            filter_results = filter_model.predict(crop_images, imgsz=224, batch=64, device=cfg.device, verbose=False)
            old_results = old_model.predict(crop_images, imgsz=320, batch=64, device=cfg.device, verbose=False)
            new_results = new_model.predict(crop_images, imgsz=320, batch=64, device=cfg.device, verbose=False)
            iteration3_results = iteration3_model.predict(
                crop_images, imgsz=320, batch=64, device=cfg.device, verbose=False
            )
            broad_results = broad_model.predict(crop_images, imgsz=224, batch=64, device=cfg.device, verbose=False)
            memory_vectors = memory_model.embed(
                crop_images, imgsz=320, batch=64, device=cfg.device, verbose=False
            )
            memory_vectors = np.stack(
                [tensor.cpu().numpy() for tensor in memory_vectors]
            ).astype(np.float32)
            memory_vectors /= np.maximum(
                np.linalg.norm(memory_vectors, axis=1, keepdims=True), 1e-12
            )
        else:
            filter_results = old_results = new_results = iteration3_results = broad_results = []
            memory_vectors = np.empty((0, memory_embeddings.shape[1]), dtype=np.float32)
        projected = project_inventory(frame, inventories, 45.0)
        matches, _ = match_inventory(predictions, projected)

        for index, (prediction, gate, old, new, iteration3, broad) in enumerate(
            zip(
                predictions,
                filter_results,
                old_results,
                new_results,
                iteration3_results,
                broad_results,
            )
        ):
            not_tree_probability = float(gate.probs.data[not_tree_index])
            if not_tree_probability >= cfg.filter_threshold:
                stats["filtered_false_positive"] += 1
                route_stats[frame.stream_id]["filtered_false_positive"] += 1
                continue
            old_probs = map_probabilities(old, old_mapping, species)
            new_probs = map_probabilities(new, new_mapping, species)
            iteration3_probs = map_probabilities(
                iteration3, iteration3_mapping, species
            )
            old_index = int(old_probs.argmax())
            new_index = int(new_probs.argmax())
            chosen_species = species[old_index]
            chosen_confidence = float(old_probs[old_index])
            source = "original_vmms_species_model"
            specialist = rules.get("specialist_override") or {}
            if (
                species[new_index] == specialist.get("species")
                and float(new_probs[new_index]) >= float(specialist.get("minimum_confidence", 1.1))
            ):
                chosen_species = species[new_index]
                chosen_confidence = float(new_probs[new_index])
                source = "human_review_specialist_override"
            iteration3_index = int(iteration3_probs.argmax())
            for override in rules.get("specialist_overrides") or []:
                if (
                    override.get("model_key") == "iteration3"
                    and species[iteration3_index] == override.get("species")
                    and float(iteration3_probs[iteration3_index])
                    >= float(override.get("minimum_confidence", 1.1))
                ):
                    chosen_species = species[iteration3_index]
                    chosen_confidence = float(iteration3_probs[iteration3_index])
                    source = "full_feedback_specialist_override"
                    break

            broad_options = top_options(broad, broad_mapping)
            broad_confidence = broad_options[0]["confidence"]
            broad_species = latin_to_ui.get(canonical_latin(broad_options[0]["species"]), broad_options[0]["species"])
            broad_margin = broad_options[0]["confidence"] - broad_options[1]["confidence"]
            match = matches.get(index)
            inventory_species = None
            agrees_inventory = False
            if match is not None:
                inventory_species = normalize_species(match["tree"].species)
                inventory_species = latin_to_ui.get(canonical_latin(inventory_species), inventory_species)
                agrees_inventory = canonical_latin(inventory_species) in {
                    canonical_latin(chosen_species),
                    canonical_latin(broad_species),
                }
            if agrees_inventory:
                chosen_species = inventory_species
                chosen_confidence = max(chosen_confidence, broad_confidence)
                source = "model_inventory_agreement"
            elif broad_confidence >= 0.92 and broad_margin >= 0.35 and chosen_confidence < 0.75:
                chosen_species = broad_species
                chosen_confidence = broad_confidence
                source = "broad_species_high_confidence"

            memory_choice = None
            eligible_memory = [
                other
                for other, item in enumerate(memory_manifest)
                if item.get("stream_id") == frame.stream_id
                and item.get("route_distance_m") is not None
                and abs(float(item["route_distance_m"]) - frame.route_distance_m)
                <= float(memory_rules["maximum_route_distance_m"])
            ]
            memory_by_species: dict[str, list[int]] = defaultdict(list)
            for other in eligible_memory:
                memory_by_species[normalize_species(memory_manifest[other]["species"])].append(other)
            memory_scores = []
            for memory_species, indexes in memory_by_species.items():
                if len({memory_manifest[other]["frame_key"] for other in indexes}) < 2:
                    continue
                memory_scores.append(
                    (
                        float(np.max(memory_embeddings[indexes] @ memory_vectors[index])),
                        memory_species,
                    )
                )
            memory_scores.sort(reverse=True)
            if memory_scores:
                memory_similarity, memory_species = memory_scores[0]
                memory_margin = memory_similarity - (
                    memory_scores[1][0] if len(memory_scores) > 1 else -1.0
                )
                memory_choice = (memory_species, memory_similarity, memory_margin)
                memory_rule = memory_rule_by_species.get(memory_species)
                if (
                    memory_rule
                    and memory_similarity >= float(memory_rule["minimum_similarity"])
                    and memory_margin >= float(memory_rule["minimum_margin"])
                ):
                    chosen_species = memory_species
                    chosen_confidence = memory_similarity
                    source = "route_local_human_species_memory"

            chosen_species = normalize_species(chosen_species)
            old_options = top_options(old, old_mapping)
            new_options = top_options(new, new_mapping)
            iteration3_options = top_options(iteration3, iteration3_mapping)
            note = (
                f"15m再处理；来源={source}；检测={prediction['tree_confidence']:.3f}；"
                f"误检概率={not_tree_probability:.3f}；旧模型={old_options[0]['species']} {old_options[0]['confidence']:.3f}；"
                f"纠错模型={new_options[0]['species']} {new_options[0]['confidence']:.3f}；"
                f"完整反馈模型={iteration3_options[0]['species']} {iteration3_options[0]['confidence']:.3f}；"
                f"广谱模型={broad_species} {broad_confidence:.3f}"
            )
            if inventory_species:
                note += f"；清单候选={inventory_species}"
            if memory_choice:
                note += (
                    f"；人工记忆={memory_choice[0]} "
                    f"相似度{memory_choice[1]:.3f}/间隔{memory_choice[2]:.3f}"
                )
            labels.append(
                {
                    "label_id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"15m:{frame.frame_key}:{index}")),
                    "species": chosen_species,
                    "points": prediction["points"],
                    "visibility": "clear" if chosen_confidence >= 0.75 else "uncertain",
                    "note": note,
                    "confidence": round(max(0.0, min(1.0, chosen_confidence)), 6),
                }
            )
            stats["retained_tree"] += 1
            route_stats[frame.stream_id]["retained_tree"] += 1
            stats[f"source/{source}"] += 1

        drafts[frame.frame_key] = {
            "frame_id": frame.frame_key,
            "frame_status": "annotated",
            "labels": labels,
            "draft_points": [],
            "preprocess": DEFAULT_PREPROCESS,
            "note": "按15米间距由三阶段YOLO结构重新预标注，等待人工验收。",
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        stats["frames"] += 1
        route_stats[frame.stream_id]["frames"] += 1
        stats["empty_frames"] += int(not labels)
        route_stats[frame.stream_id]["empty_frames"] += int(not labels)
        if position % 20 == 0 or position == len(frames):
            print(f"processed {position}/{len(frames)} frames; retained={stats['retained_tree']}; filtered={stats['filtered_false_positive']}", flush=True)

    # A browser may autosave while inference is running.  Re-read the file and
    # preserve every user-touched draft so batch processing can never clobber it.
    live_drafts = read_json(drafts_path)
    for frame_key, value in live_drafts.items():
        if str(value.get("updated_at_utc") or "") > cfg.protect_drafts_after_utc:
            drafts[frame_key] = value
    atomic_json(drafts_path, drafts)
    session = read_json(session_path)
    session.update(
        {
            "spacing_m": cfg.spacing_m,
            "reviewed_through_m": {"hewentian_pano": cfg.hwt_reviewed_through_m},
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    atomic_json(session_path, session)
    report = {
        "status": "ready_for_human_review",
        "spacing_m": cfg.spacing_m,
        "hwt_reviewed_through_m": cfg.hwt_reviewed_through_m,
        "filter_threshold": cfg.filter_threshold,
        "protected_autosaved_drafts": len(protected_keys),
        "frames": len(frames),
        "stats": dict(stats),
        "by_stream": {name: dict(values) for name, values in route_stats.items()},
        "models": {
            "tree_detector": str(TREE_MODEL.resolve()),
            "false_positive_filter": str(FILTER_MODEL.resolve()),
            "species_router": str(ROUTER.resolve()),
            "broad_species_fallback": str(BROAD_MODEL.resolve()),
            "human_species_memory": str(MEMORY_RULES.resolve()),
        },
        "backup": str(backup.resolve()),
        "review_url": "http://127.0.0.1:5181/frame.html",
    }
    report_path = PROJECT_ROOT / "reports" / f"{timestamp}_15米剩余树种预标注.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
