#!/usr/bin/env python3
"""Audit safe bulk-curation rules for the VMMS species review package.

This script is deliberately read-only.  It estimates how many review instances
can be filled from repeated government-tree IDs, agreement between independent
signals, and the route-block-held-out domain classifier.  It also reports the
effect of removing low-confidence detector proposals from unreviewed frames.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


UNKNOWN = "Unknown / 待定"


def canonical(value: str | None) -> str:
    return " ".join((value or "").replace("（", "(").replace("）", ")").split()).casefold()


def box_iou(a: dict[str, Any], b: dict[str, Any]) -> float:
    left = max(float(a["left"]), float(b["left"]))
    right = min(float(a["right"]), float(b["right"]))
    top = max(float(a["top"]), float(b["top"]))
    bottom = min(float(a["bottom"]), float(b["bottom"]))
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, float(a["right"]) - float(a["left"])) * max(0.0, float(a["bottom"]) - float(a["top"]))
    area_b = max(0.0, float(b["right"]) - float(b["left"])) * max(0.0, float(b["bottom"]) - float(b["top"]))
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def bbox(label: dict[str, Any]) -> dict[str, float]:
    if label.get("bbox"):
        return label["bbox"]
    xs = [float(point[0]) for point in label["points"]]
    ys = [float(point[1]) for point in label["points"]]
    return {"left": min(xs), "right": max(xs), "top": min(ys), "bottom": max(ys)}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def top_vote(votes: Counter[str]) -> tuple[str, int, float]:
    if not votes:
        return "", 0, 0.0
    species, count = votes.most_common(1)[0]
    return species, count, count / sum(votes.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft-dir", type=Path, required=True)
    parser.add_argument("--review-dir", type=Path, required=True)
    parser.add_argument("--review-state", type=Path, required=True)
    parser.add_argument("--classifier-audit", type=Path, required=True)
    parser.add_argument("--classifier-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    cfg = parser.parse_args()

    draft_dir = cfg.draft_dir.resolve()
    review_dir = cfg.review_dir.resolve()
    state = load_json(cfg.review_state.resolve()) if cfg.review_state.exists() else {}
    reviewed_keys = set(state)

    summary = load_json(cfg.classifier_summary.resolve())
    species_to_folder = summary["class_mapping"]
    folder_to_species = {folder: species for species, folder in species_to_folder.items()}

    # Precision by *predicted* class is what matters for selective pseudo-labels.
    validation_rows: list[dict[str, str]] = []
    with cfg.classifier_audit.resolve().open(encoding="utf-8-sig", newline="") as stream:
        validation_rows = list(csv.DictReader(stream))
    predicted_precision: dict[str, dict[str, Any]] = {}
    for threshold in (0.95, 0.98, 0.99, 0.995):
        grouped: dict[str, Counter[str]] = defaultdict(Counter)
        for row in validation_rows:
            if float(row["confidence"]) < threshold:
                continue
            grouped[row["prediction"]]["selected"] += 1
            grouped[row["prediction"]]["correct"] += int(row["correct"].lower() == "true")
        for folder, counts in grouped.items():
            species = folder_to_species[folder]
            predicted_precision.setdefault(species, {})[str(threshold)] = {
                **counts,
                "precision": counts["correct"] / counts["selected"] if counts["selected"] else 0.0,
            }

    draft_by_key = {path.stem: load_json(path) for path in sorted((draft_dir / "records").glob("*.json"))}
    review_by_key = {path.stem: load_json(path) for path in sorted((review_dir / "records").glob("*.json"))}

    inventory_votes: dict[str, Counter[str]] = defaultdict(Counter)
    human_inventory_pairs: list[tuple[str, str]] = []
    matched_human = 0
    for key, review in review_by_key.items():
        draft_labels = draft_by_key[key].get("labels") or []
        used: set[int] = set()
        for human in [x for x in review.get("labels") or [] if x.get("species_confidence_tier") == "human_verified"]:
            choices = sorted(
                ((box_iou(bbox(human), bbox(auto)), index) for index, auto in enumerate(draft_labels) if index not in used),
                reverse=True,
            )
            score, index = choices[0] if choices else (0.0, -1)
            if score < 0.10:
                continue
            used.add(index)
            matched_human += 1
            inventory = draft_labels[index].get("inventory_match") or {}
            tree_id = str(inventory.get("source_tree_id") or "")
            if tree_id:
                inventory_votes[tree_id][human["species"]] += 1
                human_inventory_pairs.append((tree_id, human["species"]))

    # Leave-one-out audit of repeated inventory IDs. A prediction is made only
    # when every other human observation of that ID agrees.
    loo = Counter()
    for tree_id, truth in human_inventory_pairs:
        votes = inventory_votes[tree_id].copy()
        votes[truth] -= 1
        if votes[truth] <= 0:
            votes.pop(truth, None)
        if not votes:
            continue
        species, count, purity = top_vote(votes)
        if purity == 1.0:
            loo["selected"] += 1
            loo["correct"] += int(canonical(species) == canonical(truth))

    counts = Counter()
    rule_counts = Counter()
    score_bins = Counter()
    per_rule_species: dict[str, Counter[str]] = defaultdict(Counter)
    conflicts: list[dict[str, Any]] = []
    for key, review in review_by_key.items():
        state_key = f"{review['stream_id']}__{review['frame_id']}"
        frame_reviewed = state_key in reviewed_keys
        for label in review.get("labels") or []:
            tier = label.get("species_confidence_tier", "unknown")
            if tier == "human_verified":
                counts["human_verified"] += 1
                continue
            counts["nonhuman"] += 1
            counts["nonhuman_in_reviewed_frames" if frame_reviewed else "nonhuman_in_unreviewed_frames"] += 1
            if label.get("species") != UNKNOWN:
                counts["existing_formal_auto"] += 1
                continue
            counts["unknown"] += 1
            counts[f"unknown_status::{review.get('status', '')}"] += 1
            if label.get("proposal_role"):
                counts[f"unknown_role::{label['proposal_role']}"] += 1
            if frame_reviewed:
                counts["unknown_in_reviewed_frames"] += 1
                continue
            counts["unknown_in_unreviewed_frames"] += 1

            tree_score = float(label.get("tree_confidence") or 0.0)
            for threshold in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
                if tree_score >= threshold:
                    score_bins[f"tree_conf_ge_{threshold}"] += 1
                    score_bins[f"{review.get('status', '')}_tree_conf_ge_{threshold}"] += 1

            inventory = label.get("inventory_match") or {}
            inv_species = str(inventory.get("species") or "")
            tree_id = str(inventory.get("source_tree_id") or "")
            domain = label.get("vmms_domain_classifier_suggestion") or {}
            domain_species = str(domain.get("top1_species") or "")
            domain_conf = float(domain.get("confidence") or 0.0)

            decisions: list[tuple[str, str]] = []
            if tree_id and tree_id in inventory_votes:
                species, vote_count, purity = top_vote(inventory_votes[tree_id])
                if purity == 1.0:
                    decisions.append(("same_inventory_id_human", species))
                    rule_counts["same_inventory_id_human"] += 1
                    per_rule_species["same_inventory_id_human"][species] += 1
                    if vote_count >= 2:
                        rule_counts["same_inventory_id_human_2plus"] += 1
                        if tree_score >= 0.50:
                            rule_counts["same_inventory_id_human_2plus_tree_ge_050"] += 1
            if inv_species and domain_species and canonical(inv_species) == canonical(domain_species) and domain_conf >= 0.90:
                decisions.append(("inventory_domain_agreement", inv_species))
                rule_counts["inventory_domain_agreement"] += 1
                per_rule_species["inventory_domain_agreement"][inv_species] += 1
                if tree_score >= 0.50:
                    rule_counts["inventory_domain_agreement_tree_ge_050"] += 1
            if domain_species and domain_conf >= 0.99:
                decisions.append(("domain_ge_099", domain_species))
                rule_counts["domain_ge_099"] += 1
                per_rule_species["domain_ge_099"][domain_species] += 1
                metrics = predicted_precision.get(domain_species, {}).get("0.99", {})
                if metrics.get("selected", 0) >= 3 and metrics.get("precision", 0.0) >= 0.90:
                    rule_counts["domain_ge_099_vetted_class"] += 1
                    per_rule_species["domain_ge_099_vetted_class"][domain_species] += 1
                    if tree_score >= 0.50:
                        rule_counts["domain_ge_099_vetted_class_tree_ge_050"] += 1
            distinct = {canonical(species) for _, species in decisions}
            if len(distinct) > 1:
                conflicts.append({"frame_key": key, "label_id": label.get("label_id"), "decisions": decisions})

    report = {
        "reviewed_frame_records_preserved": len(reviewed_keys),
        "matched_human_to_draft": matched_human,
        "inventory_ids_with_human_votes": len(inventory_votes),
        "human_inventory_observations": len(human_inventory_pairs),
        "same_inventory_id_leave_one_out": {
            **loo,
            "accuracy": loo["correct"] / loo["selected"] if loo["selected"] else 0.0,
        },
        "counts": dict(counts),
        "unreviewed_unknown_detector_score_survivors": dict(score_bins),
        "candidate_rule_counts_overlapping": dict(rule_counts),
        "candidate_rule_species": {rule: dict(values) for rule, values in per_rule_species.items()},
        "predicted_class_precision": predicted_precision,
        "rule_conflicts": conflicts,
        "notes": [
            "Counts overlap: one instance can satisfy multiple rules.",
            "No annotation or review-state file was changed.",
        ],
    }
    cfg.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    cfg.output.resolve().write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
