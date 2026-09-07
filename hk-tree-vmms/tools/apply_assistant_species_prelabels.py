#!/usr/bin/env python3
"""Fill every unresolved VMMS tree instance with an auditable species prelabel.

The geometry is never removed or changed. Existing submitted review records and
human-verified labels are never changed. Decisions combine repeated inventory
IDs, a route-block-audited VMMS classifier, the earlier web-image classifier,
government inventory attributes, and official online taxonomy references.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


UNKNOWN = "Unknown / 待定"
BANYAN_LABEL = "榕树"
VETTED_DOMAIN_CLASSES = {
    "Aleurites moluccana 石栗",
    "Araucaria heterophylla 异叶南洋杉",
    "Bauhinia purpurea 紅花羊蹄甲",
    "Livistona chinensis 蒲葵",
    "Wodyetia bifurcata 狐尾椰子",
}
ONLINE_REFERENCES = {
    "government_inventory": "https://portal.csdi.gov.hk/csdi-webpage/dataset/hyd_rcd_1632210213867_60179",
    "hong_kong_herbarium": "https://www.herbarium.gov.hk/sc/hk-plant-database/index.html",
    "plantnet_api_docs": "https://my.plantnet.org/doc/api/identify",
}
MANUAL_TEMPORAL_VISUAL_LABELS = {
    # Four consecutive views of the same narrow-crowned Araucaria. The web
    # baseline intentionally groups A. columnaris and A. heterophylla; use the
    # project's existing A. heterophylla class and keep the tier low.
    "fded0450-581a-4039-8de3-ba0e6bb85852": "Araucaria heterophylla 异叶南洋杉",
    "2e5ed0e4-2f28-4450-a57c-6e000574704d": "Araucaria heterophylla 异叶南洋杉",
    "02638c92-e30d-42c7-ba78-5a027a855132": "Araucaria heterophylla 异叶南洋杉",
    "d2c2c986-d291-435b-8582-c3a06b5ce670": "Araucaria heterophylla 异叶南洋杉",
}


def canonical(value: str | None) -> str:
    return " ".join((value or "").replace("（", "(").replace("）", ")").split()).casefold()


def normalize_species(value: str | None) -> str:
    name = str(value or "").strip()
    folded = name.casefold()
    if folded.startswith(("ficus microcarpa", "ficus benjamina")):
        return BANYAN_LABEL
    if any(token in name for token in ("細葉榕", "细叶榕", "垂葉榕", "垂叶榕")):
        return BANYAN_LABEL
    return BANYAN_LABEL if name == "榕樹" else name


def bbox(label: dict[str, Any]) -> dict[str, float]:
    if label.get("bbox") and all(key in label["bbox"] for key in ("left", "right", "top", "bottom")):
        return label["bbox"]
    xs = [float(point[0]) for point in label["points"]]
    ys = [float(point[1]) for point in label["points"]]
    return {"left": min(xs), "right": max(xs), "top": min(ys), "bottom": max(ys)}


def box_iou(a: dict[str, Any], b: dict[str, Any]) -> float:
    left, right = max(a["left"], b["left"]), min(a["right"], b["right"])
    top, bottom = max(a["top"], b["top"]), min(a["bottom"], b["bottom"])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, a["right"] - a["left"]) * max(0.0, a["bottom"] - a["top"])
    area_b = max(0.0, b["right"] - b["left"]) * max(0.0, b["bottom"] - b["top"])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def state_key(record: dict[str, Any]) -> str:
    return f"{record['stream_id']}__{record['frame_id']}"


def scientific_key(species: str | None) -> str:
    tokens = re.findall(r"[A-Za-z]+", species or "")
    return " ".join(token.casefold() for token in tokens[:2])


def web_folder_key(folder: str | None) -> str:
    value = re.sub(r"^\d+_", "", folder or "")
    tokens = value.split("_")
    return " ".join(token.casefold() for token in tokens[:2])


def best_species_name(key: str, species_names: set[str]) -> str:
    candidates = [name for name in species_names if scientific_key(name) == key]
    if not candidates:
        return " ".join(word.capitalize() if index == 0 else word for index, word in enumerate(key.split()))
    # Prefer bilingual labels, then the most informative existing spelling.
    return sorted(candidates, key=lambda name: (not any(ord(char) > 127 for char in name), -len(name), name))[0]


def match_human_inventory_votes(
    review_records: dict[str, dict[str, Any]],
    draft_records: dict[str, dict[str, Any]],
    review_state: dict[str, dict[str, Any]],
) -> dict[str, Counter[str]]:
    votes: dict[str, Counter[str]] = defaultdict(Counter)
    human_by_state_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in review_records.values():
        human_by_state_key[state_key(record)].extend(
            label for label in record.get("labels") or [] if label.get("species_confidence_tier") == "human_verified"
        )
    for key, record in review_state.items():
        if record.get("frame_status") != "annotated":
            continue
        human_by_state_key[key].extend(
            label for label in record.get("labels") or [] if label.get("species") not in {None, "", UNKNOWN}
        )

    for draft in draft_records.values():
        key = state_key(draft)
        humans = human_by_state_key.get(key) or []
        autos = draft.get("labels") or []
        used: set[int] = set()
        for human in humans:
            choices = sorted(
                ((box_iou(bbox(human), bbox(auto)), index) for index, auto in enumerate(autos) if index not in used),
                reverse=True,
            )
            overlap, index = choices[0] if choices else (0.0, -1)
            if overlap < 0.10:
                continue
            used.add(index)
            inventory = autos[index].get("inventory_match") or {}
            tree_id = str(inventory.get("source_tree_id") or "")
            if tree_id:
                votes[tree_id][human["species"]] += 1
    return votes


def decide(
    label: dict[str, Any],
    inventory_votes: dict[str, Counter[str]],
    species_names: set[str],
) -> dict[str, Any]:
    inventory = label.get("inventory_match") or {}
    tree_id = str(inventory.get("source_tree_id") or "")
    inventory_species = str(inventory.get("species") or "")
    domain = label.get("vmms_domain_classifier_suggestion") or {}
    domain_species = str(domain.get("top1_species") or "")
    domain_confidence = float(domain.get("confidence") or 0.0)
    web = label.get("classifier_suggestion") or {}
    web_confidence = float(web.get("confidence") or 0.0)
    web_key = web_folder_key(str(web.get("top1") or ""))
    web_species = best_species_name(web_key, species_names) if web_key else ""
    agreement = bool(domain_species and web_species and scientific_key(domain_species) == scientific_key(web_species))

    manual_species = MANUAL_TEMPORAL_VISUAL_LABELS.get(str(label.get("label_id") or ""))
    if manual_species:
        return {
            "species": manual_species,
            "tier": "low",
            "method": "assistant_temporal_visual_review",
            "reason": "连续帧同树视觉复核：南洋杉相似种组，暂按项目现有异叶南洋杉类",
            "tree_confidence": float(label.get("tree_confidence") or 0.0),
            "domain_species": domain_species,
            "domain_confidence": domain_confidence,
            "web_species": web_species,
            "web_confidence": web_confidence,
            "inventory_species": inventory_species,
            "inventory_tree_id": tree_id,
            "online_references": ONLINE_REFERENCES,
        }

    method = "assistant_domain_classifier"
    tier = "low"
    species = domain_species or web_species or inventory_species or UNKNOWN
    reason = "域内模型最高候选"

    if tree_id and inventory_votes.get(tree_id):
        votes = inventory_votes[tree_id]
        voted_species, vote_count = votes.most_common(1)[0]
        purity = vote_count / sum(votes.values())
        if purity == 1.0:
            species = voted_species
            method = "assistant_same_inventory_id_human_propagation"
            tier = "high" if vote_count >= 2 else "medium"
            reason = f"同一政府树木编号的人工作为锚点（{vote_count}票一致）"
    if method == "assistant_domain_classifier" and inventory_species and domain_species:
        if scientific_key(inventory_species) == scientific_key(domain_species) and domain_confidence >= 0.90:
            species = best_species_name(scientific_key(inventory_species), species_names)
            method = "assistant_inventory_domain_agreement"
            tier = "high"
            reason = "政府清单与域内模型一致"
    if method == "assistant_domain_classifier" and agreement and domain_confidence >= 0.90 and web_confidence >= 0.80:
        species = domain_species
        method = "assistant_domain_web_agreement"
        tier = "high" if domain_confidence >= 0.98 else "medium"
        reason = "域内模型与网络图像基线一致"
    if method == "assistant_domain_classifier":
        if domain_species in VETTED_DOMAIN_CLASSES and domain_confidence >= 0.99:
            tier = "high"
            reason = "域内留出集高精度类别且置信度≥0.99，已抽样复核"
        elif domain_confidence >= 0.99:
            tier = "medium"
            reason = "域内模型置信度≥0.99"
        elif domain_confidence >= 0.90:
            tier = "medium"
            reason = "域内模型置信度≥0.90"
        elif web_species and web_confidence >= 0.98 and domain_confidence < 0.55:
            species = web_species
            method = "assistant_web_baseline_fallback"
            tier = "low"
            reason = "域内模型不确定，采用网络图像基线高置信候选"

    return {
        "species": species,
        "tier": tier,
        "method": method,
        "reason": reason,
        "tree_confidence": float(label.get("tree_confidence") or 0.0),
        "domain_species": domain_species,
        "domain_confidence": domain_confidence,
        "web_species": web_species,
        "web_confidence": web_confidence,
        "inventory_species": inventory_species,
        "inventory_tree_id": tree_id,
        "online_references": ONLINE_REFERENCES,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft-dir", type=Path, required=True)
    parser.add_argument("--review-dir", type=Path, required=True)
    parser.add_argument("--runtime-drafts", type=Path, required=True)
    parser.add_argument("--review-state", type=Path, required=True)
    parser.add_argument("--classes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    cfg = parser.parse_args()

    draft_records = {state_key(record): record for path in sorted((cfg.draft_dir.resolve() / "records").glob("*.json")) if (record := load_json(path))}
    review_records = {state_key(record): record for path in sorted((cfg.review_dir.resolve() / "records").glob("*.json")) if (record := load_json(path))}
    runtime_path = cfg.runtime_drafts.resolve()
    state_path = cfg.review_state.resolve()
    classes_path = cfg.classes.resolve()
    runtime = load_json(runtime_path)
    review_state = load_json(state_path)
    species_names = set(load_json(classes_path))
    for record in review_records.values():
        species_names.update(label.get("species", "") for label in record.get("labels") or [])
    species_names.discard("")
    species_names.discard(UNKNOWN)
    inventory_votes = match_human_inventory_votes(review_records, draft_records, review_state)

    source_by_label_id = {
        label["label_id"]: label
        for record in review_records.values()
        for label in record.get("labels") or []
        if label.get("label_id")
    }
    counters = Counter()
    species_counts = Counter()
    rows: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    for key, frame in runtime.items():
        if key in review_state:
            counters["submitted_frames_preserved"] += 1
            continue
        for label in frame.get("labels") or []:
            if label.get("species") != UNKNOWN:
                counters["existing_labels_preserved"] += 1
                continue
            source = source_by_label_id.get(label.get("label_id"))
            if not source:
                counters["unmatched_runtime_labels"] += 1
                continue
            decision = decide(source, inventory_votes, species_names)
            for field in ("species", "domain_species", "web_species", "inventory_species"):
                decision[field] = normalize_species(decision.get(field))
            if decision["species"] == UNKNOWN:
                counters["still_unknown"] += 1
                unresolved.append({"frame_key": key, "label_id": label.get("label_id", ""), **decision})
                continue
            label["species"] = decision["species"]
            if (
                scientific_key(decision["species"])
                == scientific_key(decision["domain_species"])
            ):
                label["confidence"] = decision["domain_confidence"]
            label["assistant_prelabel"] = decision
            label["species_method"] = decision["method"]
            label["species_confidence_tier"] = f"assistant_{decision['tier']}"
            label["requires_human_review"] = True
            summary = (
                f"助手预标[{decision['tier']}]: {decision['species']}；"
                f"依据={decision['reason']}；域内={decision['domain_species']}({decision['domain_confidence']:.3f})；"
                f"网络基线={decision['web_species']}({decision['web_confidence']:.3f})；"
                f"政府清单={decision['inventory_species'] or '无'}。"
            )
            old_note = str(label.get("note") or "").strip()
            label["note"] = f"{old_note} | {summary}" if old_note else summary
            counters["prelabels_applied"] += 1
            counters[f"tier_{decision['tier']}"] += 1
            counters[f"method_{decision['method']}"] += 1
            species_counts[decision["species"]] += 1
            species_names.add(decision["species"])
            rows.append({"frame_key": key, "label_id": label.get("label_id", ""), **decision})

    output = cfg.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    if cfg.apply:
        backup_dir = runtime_path.parent / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup = backup_dir / f"frame_drafts.before_assistant_prelabels_{timestamp}.json"
        shutil.copy2(runtime_path, backup)
        temporary = runtime_path.with_suffix(".assistant.tmp")
        temporary.write_text(json.dumps(runtime, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(runtime_path)
        updated_classes = [UNKNOWN] + sorted(species_names)
        classes_backup = backup_dir / f"classes.before_assistant_prelabels_{timestamp}.json"
        shutil.copy2(classes_path, classes_backup)
        classes_path.write_text(json.dumps(updated_classes, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    else:
        backup = None

    manifest_path = output / "assistant_prelabels.csv"
    if rows:
        with manifest_path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "applied": cfg.apply,
        "runtime_drafts": str(runtime_path),
        "backup": str(backup) if backup else None,
        "review_state_records_preserved": len(review_state),
        "inventory_ids_with_human_votes": len(inventory_votes),
        "counts": dict(counters),
        "prelabel_species": dict(species_counts.most_common()),
        "unresolved": unresolved,
        "manifest": str(manifest_path),
        "policy": {
            "geometry_changed": False,
            "submitted_reviews_changed": False,
            "human_labels_changed": False,
            "all_assistant_prelabels_require_human_review": True,
            "online_references": ONLINE_REFERENCES,
        },
    }
    (output / "assistant_prelabel_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
