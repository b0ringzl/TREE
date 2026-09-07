#!/usr/bin/env python3
"""Measure editable tree-proposal precision/recall against prior human frames."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_exhaustive_species_drafts import iou, load_human_records  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    cfg = parser.parse_args()
    humans = load_human_records()
    rows = []
    for threshold in (.10, .12, .15, .18, .20, .25, .30, .40, .50, .60, .70, .80):
        tp = fp = fn = proposals = human_total = 0
        for (route, stream, frame_id), (human, _) in humans.items():
            path = cfg.draft_dir / "records" / f"{route}__{stream}__{frame_id}.json"
            if not path.is_file():
                continue
            draft = json.loads(path.read_text(encoding="utf-8"))
            predicted = [x for x in draft.get("labels") or [] if float(x.get("tree_confidence", 0)) >= threshold]
            truth = human.get("labels") or []
            proposals += len(predicted); human_total += len(truth)
            pairs = sorted(
                ((iou(h["points"], p["points"]), hi, pi) for hi, h in enumerate(truth) for pi, p in enumerate(predicted)),
                reverse=True,
            )
            used_h, used_p = set(), set()
            for overlap, hi, pi in pairs:
                if overlap < .20:
                    break
                if hi not in used_h and pi not in used_p:
                    used_h.add(hi); used_p.add(pi)
            tp += len(used_h); fp += len(predicted) - len(used_p); fn += len(truth) - len(used_h)
        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        f2 = 5 * precision * recall / (4 * precision + recall) if precision + recall else 0
        rows.append({"threshold": threshold, "human_instances": human_total, "proposals": proposals,
                     "tp_iou20": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f2": f2})
    cfg.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
