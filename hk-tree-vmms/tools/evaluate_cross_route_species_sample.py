#!/usr/bin/env python3
"""Evaluate pre-review species predictions on the saved TST/Stubbs random sample."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from evaluate_human_review_accuracy import (
    DEFAULT_LABELER,
    DEFAULT_SOURCE,
    load_predictions,
    match_labels,
    normalize_species,
    source_records,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLE = PROJECT_ROOT / "reports" / "20260906_尖沙咀及司徒拔道随机分类验收样本.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "20260906_尖沙咀及司徒拔道随机分类精度.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--labeler", type=Path, default=DEFAULT_LABELER)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    cfg = parser.parse_args()
    labeler, source = cfg.labeler.resolve(), cfg.source.resolve()
    sources = source_records(source)
    predictions = load_predictions(source, sources)
    sample_rows = list(csv.DictReader(cfg.sample.resolve().open(encoding="utf-8-sig", newline="")))
    totals = Counter()
    by_route: dict[str, Counter] = defaultdict(Counter)
    details = []
    for sample in sample_rows:
        frame_key = sample["frame_key"]
        final_path = labeler / "annotations" / "records" / sample["stream_id"] / f"{sample['frame_id']}.json"
        if not final_path.is_file():
            totals["unreviewed_frames"] += 1
            by_route[sample["route"]]["unreviewed_frames"] += 1
            continue
        totals["reviewed_frames"] += 1
        by_route[sample["route"]]["reviewed_frames"] += 1
        source_record = sources[frame_key]
        final = json.loads(final_path.read_text(encoding="utf-8"))
        matches, used_source, _ = match_labels(
            source_record.get("labels") or [], final.get("labels") or [], 0.20
        )
        match_by_source = {source_index: final_index for source_index, final_index, _, _ in matches}
        for source_index, label in enumerate(source_record.get("labels") or []):
            prediction = predictions.get(str(label.get("label_id") or ""))
            if prediction is None:
                continue
            totals["predictions_before"] += 1
            by_route[sample["route"]]["predictions_before"] += 1
            if source_index not in used_source:
                outcome, final_species, correct = "deleted", "已删除实例", False
            else:
                final_label = (final.get("labels") or [])[match_by_source[source_index]]
                final_species = normalize_species(final_label.get("species"))
                correct = normalize_species(prediction["species"]) == final_species
                outcome = "unchanged_correct" if correct else "species_corrected"
            totals[outcome] += 1
            by_route[sample["route"]][outcome] += 1
            details.append({
                "route": sample["route"], "ui_frame_number": sample["ui_frame_number"],
                "frame_key": frame_key, "label_id": label.get("label_id", ""),
                "predicted_species": prediction["species"], "final_species": final_species,
                "outcome": outcome, "correct": correct,
            })

    def metrics(values: Counter) -> dict:
        denominator = values["predictions_before"]
        return {
            **dict(values),
            "adjusted_accuracy": values["unchanged_correct"] / denominator if denominator else None,
        }

    report = {
        "status": "complete" if totals["unreviewed_frames"] == 0 else "in_progress",
        "sample_frames": len(sample_rows), "overall": metrics(totals),
        "by_route": {route: metrics(values) for route, values in sorted(by_route.items())},
        "scoring": "deleted predictions are errors; corrected species are errors; unchanged species are correct; Ficus labels are merged to 榕树",
    }
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    cfg.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with cfg.output.with_suffix(".csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fields = list(details[0]) if details else ["route", "frame_key", "outcome"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(details)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
