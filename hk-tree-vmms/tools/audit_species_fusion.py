#!/usr/bin/env python3
"""Audit government-inventory and VMMS classifier fusion on held-out blocks."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_exhaustive_species_drafts import canonical, iou, load_human_records  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft-dir", type=Path, required=True)
    parser.add_argument("--classifier-audit", type=Path, required=True)
    parser.add_argument("--dataset-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    cfg = parser.parse_args()
    folder_to_species = {v: k for k, v in json.loads(cfg.dataset_summary.read_text(encoding="utf-8"))["class_mapping"].items()}
    humans = load_human_records()
    rows = []
    with cfg.classifier_audit.open(encoding="utf-8-sig", newline="") as stream:
        predictions = list(csv.DictReader(stream))
    for prediction in predictions:
        stem = Path(prediction["path"]).stem
        match = re.match(r"(.+)__(\d+)$", stem)
        if not match:
            continue
        key, human_index = match.group(1), int(match.group(2))
        parts = key.split("__")
        route, stream, frame_id = parts[0], parts[1], parts[2]
        human = humans[(route, stream, frame_id)][0]
        human_label = human["labels"][human_index]
        draft = json.loads((cfg.draft_dir / "records" / f"{key}.json").read_text(encoding="utf-8"))
        pairs = sorted(((iou(human_label["points"], label["points"]), label) for label in draft.get("labels") or []), reverse=True, key=lambda x: x[0])
        overlap, draft_label = pairs[0] if pairs else (0.0, {})
        inventory = draft_label.get("inventory_match") or {}
        inventory_species = inventory.get("species", "")
        classifier_species = folder_to_species[prediction["prediction"]]
        classifier_confidence = float(prediction["confidence"])
        inv_cls_agree = bool(inventory_species and canonical(inventory_species) == canonical(classifier_species))
        rows.append({
            "frame_key": key, "human_species": human_label["species"], "mask_iou": overlap,
            "inventory_species": inventory_species, "inventory_tier": draft_label.get("species_confidence_tier", "unknown"),
            "classifier_species": classifier_species, "classifier_confidence": classifier_confidence,
            "classifier_correct": canonical(classifier_species) == canonical(human_label["species"]),
            "inventory_correct": bool(inventory_species and canonical(inventory_species) == canonical(human_label["species"])),
            "inventory_classifier_agreement": inv_cls_agree,
        })
    rules = []
    for threshold in (.90, .95, .98, .99):
        for tiers in (("high",), ("high", "medium"), ("high", "medium", "low")):
            selected = [r for r in rows if r["inventory_classifier_agreement"] and r["classifier_confidence"] >= threshold and r["inventory_tier"] in tiers]
            correct = sum(r["inventory_correct"] for r in selected)
            rules.append({"classifier_threshold": threshold, "inventory_tiers": list(tiers), "selected": len(selected), "correct": correct, "precision": correct / len(selected) if selected else None})
    report = {
        "held_out_instances": len(rows),
        "classifier_accuracy": sum(r["classifier_correct"] for r in rows) / len(rows),
        "fusion_rules": rules,
        "note": "Fusion rows use only route-block-held-out classifier validation crops.",
    }
    cfg.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with cfg.output.with_suffix(".csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
