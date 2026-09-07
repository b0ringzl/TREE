#!/usr/bin/env python3
"""Build a conservative, human-reviewable VMMS species annotation package.

Existing human annotations are preserved as anchors. Automatic species names are
only retained when the independently audited confidence tier is ``high``;
medium/low inventory projections remain as evidence but the YOLO class is set to
Unknown. The source draft and all human annotation records are read-only.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_exhaustive_species_drafts import iou, load_human_records  # noqa: E402

UNKNOWN = "Unknown / 待定"


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["frame_key"])
        writer.writeheader()
        writer.writerows(rows)


def hardlink_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def bbox(points: list[list[float]]) -> dict[str, float]:
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    left, right, top, bottom = min(xs), max(xs), min(ys), max(ys)
    return {
        "x_center": (left + right) / 2,
        "y_center": (top + bottom) / 2,
        "width": right - left,
        "height": bottom - top,
        "left": left,
        "top": top,
        "right": right,
        "bottom": bottom,
    }


def conservative_auto(label: dict[str, Any], extra: bool = False) -> dict[str, Any]:
    result = json.loads(json.dumps(label, ensure_ascii=False))
    original_species = result.get("species", UNKNOWN)
    tier = result.get("species_confidence_tier", "unknown")
    if tier != "high":
        result["species"] = UNKNOWN
        if original_species != UNKNOWN:
            result["candidate_species"] = original_species
        result["species_method"] = "candidate_only_" + result.get("species_method", "auto")
        result["species_confidence_tier"] = "unknown"
    if extra:
        result["proposal_role"] = "model_extra_on_human_reviewed_frame"
        result["species"] = UNKNOWN
        if original_species != UNKNOWN:
            result["candidate_species"] = original_species
        result["species_confidence_tier"] = "unknown"
    result["requires_human_review"] = True
    return result


def merge_labels(draft: dict[str, Any], human: dict[str, Any] | None) -> tuple[list[dict[str, Any]], str]:
    automatic = draft.get("labels") or []
    if human is None:
        return [conservative_auto(label) for label in automatic], "auto_review_required"
    status = human.get("frame_status", "")
    human_labels = human.get("labels") or []
    if status == "unusable":
        return [], "excluded_human_unusable"
    if status == "no_tree":
        return [], "human_verified_no_tree"
    if not human_labels:
        return [conservative_auto(label) for label in automatic], "auto_review_required"

    merged: list[dict[str, Any]] = []
    used: set[int] = set()
    for number, source in enumerate(human_labels, 1):
        candidates = sorted(
            ((iou(source["points"], auto["points"]), index) for index, auto in enumerate(automatic) if index not in used),
            reverse=True,
        )
        best_iou, best_index = candidates[0] if candidates else (0.0, -1)
        if best_iou >= 0.20:
            used.add(best_index)
        label = json.loads(json.dumps(source, ensure_ascii=False))
        label["bbox"] = bbox(label["points"])
        label["species_method"] = "existing_human_annotation"
        label["species_confidence_tier"] = "human_verified"
        label["requires_human_review"] = False
        label["matched_auto_iou"] = round(best_iou, 6)
        label["source_human_label_number"] = number
        merged.append(label)
    for index, label in enumerate(automatic):
        if index not in used:
            merged.append(conservative_auto(label, extra=True))
    return merged, "human_anchors_with_extra_proposals"


def draw_overlay(image_path: Path, labels: list[dict[str, Any]], target: Path, status: str) -> None:
    with Image.open(image_path) as raw:
        image = raw.convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.load_default()
    width, height = image.size
    colors = {
        "human_verified": (20, 210, 90, 215),
        "high": (255, 186, 35, 220),
        "unknown": (240, 70, 70, 210),
    }
    for idx, label in enumerate(labels, 1):
        tier = label.get("species_confidence_tier", "unknown")
        color = colors.get(tier, colors["unknown"])
        points = [(round(x * width), round(y * height)) for x, y in label["points"]]
        if len(points) >= 3:
            draw.polygon(points, fill=(*color[:3], 35), outline=color, width=max(2, width // 900))
        display_species = label.get("species", UNKNOWN)
        domain_hint = label.get("vmms_domain_classifier_suggestion") or {}
        if display_species == UNKNOWN and domain_hint.get("top1_species"):
            name = "?" + domain_hint["top1_species"].split(" ")[0]
        elif display_species == UNKNOWN and label.get("candidate_species"):
            name = "?" + label["candidate_species"].split(" ")[0]
        else:
            name = display_species.split(" ")[0]
        left = min(p[0] for p in points)
        top = min(p[1] for p in points)
        text = f"{idx} {name} [{tier}]"
        box = draw.textbbox((left, top), text, font=font)
        draw.rectangle(box, fill=(0, 0, 0, 190))
        draw.text((left, top), text, fill=(255, 255, 255, 255), font=font)
    draw.rectangle((0, 0, min(width, 560), 24), fill=(0, 0, 0, 190))
    draw.text((6, 6), status, fill=(255, 255, 255, 255), font=font)
    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target, quality=86, optimize=True)


def write_html(output: Path, frames: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    cards = []
    for row in frames:
        rel = Path(row["overlay"]).relative_to(output).as_posix()
        text = " ".join(str(row[k]) for k in ("route", "stream_id", "frame_id", "review_status"))
        cards.append(
            f'<article class="card" data-route="{html.escape(row["route"])}" '
            f'data-status="{html.escape(row["review_status"])}" data-text="{html.escape(text.lower())}">'
            f'<a href="{rel}"><img loading="lazy" src="{rel}" alt="{html.escape(row["frame_key"])}"></a>'
            f'<div><b>{html.escape(row["frame_key"])}</b><br>{html.escape(row["review_status"])}<br>'
            f'实例 {row["instance_count"]} · 人工 {row["human_verified_count"]} · 高置信 {row["high_count"]} · 待定 {row["unknown_count"]}</div></article>'
        )
    doc = f"""<!doctype html><html lang="zh-Hans"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>VMMS 树种预标注验收</title>
<style>body{{font:14px system-ui;margin:20px;background:#f5f6f8;color:#18202a}}header{{position:sticky;top:0;background:#f5f6f8;padding:4px 0 12px;z-index:2}}input,select{{padding:8px;margin-right:8px}}.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(310px,1fr));gap:14px}}.card{{background:white;border-radius:9px;overflow:hidden;box-shadow:0 1px 5px #0002}}.card img{{width:100%;aspect-ratio:2/1;object-fit:cover;display:block}}.card div{{padding:9px;line-height:1.5}}small{{color:#566}}</style></head><body>
<header><h2>香港 VMMS 穷尽式多目标树种预标注验收</h2>
<p>共 {summary['frames']} 帧、{summary['instances']} 个实例。绿色=既有人工，橙色=经审计高置信清单预标注，红色=待定/额外候选。</p>
<input id="q" placeholder="搜索路线/帧号"><select id="route"><option value="">全部路线</option><option>hewentian</option><option>jianshazui</option><option>stubbs_road</option></select>
<select id="status"><option value="">全部状态</option><option>auto_review_required</option><option>human_anchors_with_extra_proposals</option><option>human_verified_no_tree</option><option>excluded_human_unusable</option></select>
<small id="shown"></small></header><main class="grid">{''.join(cards)}</main>
<script>const cards=[...document.querySelectorAll('.card')],q=document.querySelector('#q'),r=document.querySelector('#route'),s=document.querySelector('#status'),n=document.querySelector('#shown');function f(){{let k=q.value.toLowerCase(),c=0;cards.forEach(x=>{{let yes=(!k||x.dataset.text.includes(k))&&(!r.value||x.dataset.route===r.value)&&(!s.value||x.dataset.status===s.value);x.style.display=yes?'':'none';c+=yes}});n.textContent='显示 '+c+' / '+cards.length}}[q,r,s].forEach(x=>x.oninput=f);f()</script></body></html>"""
    (output / "review.html").write_text(doc, encoding="utf-8")


def main() -> None:
    cfg = args()
    draft_dir, output = cfg.draft_dir.resolve(), cfg.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing review package: {output}")
    output.mkdir(parents=True)
    human_records = load_human_records()
    records = sorted((draft_dir / "records").glob("*.json"))
    built: list[tuple[Path, dict[str, Any], list[dict[str, Any]], str]] = []
    species = {UNKNOWN}
    for record_path in records:
        draft = json.loads(record_path.read_text(encoding="utf-8"))
        identity = (draft["dataset_route"], draft["stream_id"], str(draft["frame_id"]))
        human_pair = human_records.get(identity)
        labels, status = merge_labels(draft, human_pair[0] if human_pair else None)
        for label in labels:
            species.add(label.get("species", UNKNOWN))
        built.append((record_path, draft, labels, status))
    classes = [UNKNOWN] + sorted(species - {UNKNOWN})
    class_ids = {name: idx for idx, name in enumerate(classes)}

    frame_rows: list[dict[str, Any]] = []
    instance_rows: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    for record_path, draft, labels, status in built:
        key = record_path.stem
        source_image = draft_dir / "images" / f"{key}.jpg"
        image_target = output / "images" / source_image.name
        hardlink_or_copy(source_image, image_target)
        for label in labels:
            label["class_id"] = class_ids[label.get("species", UNKNOWN)]
        record = json.loads(json.dumps(draft, ensure_ascii=False))
        record["schema_version"] = "vmms_exhaustive_species_review_v1"
        record["status"] = status
        record["labels"] = labels
        record["review_policy"] = {
            "human_annotations": "preserved",
            "auto_high": "retained as reviewable species prelabel; audited agreement 11/13",
            "auto_medium_low": "formal species downgraded to Unknown; candidate retained in metadata",
            "training_ready": False,
        }
        out_record = output / "records" / record_path.name
        out_record.parent.mkdir(parents=True, exist_ok=True)
        out_record.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        yolo_lines = []
        jsonl_lines = []
        for label in labels:
            coords = " ".join(f"{float(v):.6f}" for point in label["points"] for v in point)
            yolo_lines.append(f"{label['class_id']} {coords}")
            jsonl_lines.append(json.dumps(label, ensure_ascii=False))
            inv = label.get("inventory_match") or {}
            instance_rows.append({
                "frame_key": key, "route": draft["dataset_route"], "stream_id": draft["stream_id"],
                "frame_id": draft["frame_id"], "label_id": label.get("label_id", ""),
                "species": label.get("species", UNKNOWN), "candidate_species": label.get("candidate_species", ""),
                "tier": label.get("species_confidence_tier", "unknown"), "method": label.get("species_method", ""),
                "requires_human_review": label.get("requires_human_review", True),
                "inventory_tree_id": inv.get("source_tree_id", ""), "inventory_distance_m": inv.get("distance_m", ""),
                "overlay": str(output / "overlays" / draft["dataset_route"] / f"{key}.jpg"),
            })
        label_dir = output / "labels"
        label_dir.mkdir(parents=True, exist_ok=True)
        (label_dir / f"{key}.txt").write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8")
        (label_dir / f"{key}.review.jsonl").write_text("\n".join(jsonl_lines) + ("\n" if jsonl_lines else ""), encoding="utf-8")
        overlay = output / "overlays" / draft["dataset_route"] / f"{key}.jpg"
        draw_overlay(image_target, labels, overlay, status)
        tiers = Counter(label.get("species_confidence_tier", "unknown") for label in labels)
        row = {
            "frame_key": key, "route": draft["dataset_route"], "stream_id": draft["stream_id"],
            "frame_id": draft["frame_id"], "review_status": status, "instance_count": len(labels),
            "human_verified_count": tiers["human_verified"], "high_count": tiers["high"],
            "unknown_count": tiers["unknown"], "requires_review_count": sum(bool(x.get("requires_human_review", True)) for x in labels),
            "image": str(image_target), "overlay": str(overlay), "record": str(out_record),
        }
        frame_rows.append(row)
        counters[status] += 1
        counters["instances"] += len(labels)
        counters["human_verified_instances"] += tiers["human_verified"]
        counters["high_species_prelabels"] += tiers["high"]
        counters["unknown_instances"] += tiers["unknown"]

    (output / "classes.json").write_text(json.dumps(classes, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "DRAFT_NOT_FOR_TRAINING.txt").write_text(
        "This package contains review drafts. Accept/correct annotations and create a leakage-safe split before training.\n",
        encoding="utf-8",
    )
    write_csv(output / "frame_review_queue.csv", frame_rows)
    write_csv(output / "instance_review_queue.csv", instance_rows)
    summary = {
        "status": "ready_for_human_review_not_training",
        "frames": len(frame_rows), "instances": counters["instances"],
        "classes_including_unknown": len(classes), **dict(counters),
        "source_draft": str(draft_dir),
        "audit_basis": {"human_frames": 642, "human_instances": 905, "proposal_recall_iou20": 0.9734806629834254,
                        "high_species_agreement": "11/13", "medium_species_agreement": "8/26", "low_species_agreement": "9/74"},
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_html(output, frame_rows, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
