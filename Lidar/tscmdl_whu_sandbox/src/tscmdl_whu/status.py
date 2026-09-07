"""Project-status helpers for comparing classification experiments."""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class PredictionRecord:
    sample_key: str
    split: str
    true_class: int
    predicted_class: int

    @property
    def correct(self) -> bool:
        return self.true_class == self.predicted_class


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_prediction_records(path: str | Path) -> dict[str, PredictionRecord]:
    records: dict[str, PredictionRecord] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            record = PredictionRecord(
                sample_key=str(row["sample_key"]),
                split=str(row["split"]),
                true_class=int(row["true_class"]),
                predicted_class=int(row["predicted_class"]),
            )
            if record.sample_key in records:
                raise ValueError(f"Duplicate sample key: {record.sample_key}")
            records[record.sample_key] = record
    if not records:
        raise ValueError(f"No prediction rows found in {path}")
    return records


def _comparison_summary(
    keys: list[str],
    left: Mapping[str, PredictionRecord],
    right: Mapping[str, PredictionRecord],
) -> dict[str, int | float]:
    both_correct = 0
    left_only_correct = 0
    right_only_correct = 0
    neither_correct = 0
    same_prediction = 0
    for key in keys:
        left_record = left[key]
        right_record = right[key]
        if left_record.correct and right_record.correct:
            both_correct += 1
        elif left_record.correct:
            left_only_correct += 1
        elif right_record.correct:
            right_only_correct += 1
        else:
            neither_correct += 1
        same_prediction += int(
            left_record.predicted_class == right_record.predicted_class
        )

    sample_count = len(keys)
    oracle_correct = both_correct + left_only_correct + right_only_correct
    return {
        "sample_count": sample_count,
        "both_correct": both_correct,
        "left_only_correct": left_only_correct,
        "right_only_correct": right_only_correct,
        "neither_correct": neither_correct,
        "same_prediction": same_prediction,
        "different_prediction": sample_count - same_prediction,
        "left_accuracy": (both_correct + left_only_correct) / sample_count,
        "right_accuracy": (both_correct + right_only_correct) / sample_count,
        "either_model_oracle_accuracy": oracle_correct / sample_count,
        "right_net_corrections": right_only_correct - left_only_correct,
    }


def compare_prediction_records(
    left: Mapping[str, PredictionRecord],
    right: Mapping[str, PredictionRecord],
    num_classes: int,
) -> dict[str, object]:
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    if set(left) != set(right):
        missing_left = sorted(set(right) - set(left))
        missing_right = sorted(set(left) - set(right))
        raise ValueError(
            "Prediction sample sets differ: "
            f"missing_left={missing_left[:3]} missing_right={missing_right[:3]}"
        )

    keys = sorted(left)
    for key in keys:
        left_record = left[key]
        right_record = right[key]
        if (
            left_record.split != right_record.split
            or left_record.true_class != right_record.true_class
        ):
            raise ValueError(f"Prediction metadata mismatch: {key}")
        for class_index in (
            left_record.true_class,
            left_record.predicted_class,
            right_record.predicted_class,
        ):
            if not 0 <= class_index < num_classes:
                raise ValueError(f"Class index out of range: {key}")

    splits = sorted({left[key].split for key in keys})
    by_split = {
        split: _comparison_summary(
            [key for key in keys if left[key].split == split], left, right
        )
        for split in splits
    }
    by_split_and_class = {
        split: {
            str(class_index): _comparison_summary(
                [
                    key
                    for key in keys
                    if left[key].split == split
                    and left[key].true_class == class_index
                ],
                left,
                right,
            )
            for class_index in range(num_classes)
        }
        for split in splits
    }
    return {
        "overall": _comparison_summary(keys, left, right),
        "by_split": by_split,
        "by_split_and_class": by_split_and_class,
    }


def compare_prediction_models(
    models: Mapping[str, Mapping[str, PredictionRecord]],
    num_classes: int,
) -> dict[str, object]:
    """Summarize correctness overlap for three or more aligned models."""
    if len(models) < 2:
        raise ValueError("At least two models are required")
    names = list(models)
    reference = models[names[0]]
    keys = sorted(reference)
    if not keys:
        raise ValueError("Prediction records are empty")
    for name, records in models.items():
        if set(records) != set(keys):
            raise ValueError(f"Prediction sample set differs for {name}")
        for key in keys:
            record = records[key]
            reference_record = reference[key]
            if (
                record.split != reference_record.split
                or record.true_class != reference_record.true_class
            ):
                raise ValueError(f"Prediction metadata mismatch: {name} {key}")
            if not 0 <= record.predicted_class < num_classes:
                raise ValueError(f"Class index out of range: {name} {key}")

    by_split = {}
    for split in sorted({reference[key].split for key in keys}):
        split_keys = [key for key in keys if reference[key].split == split]
        patterns: dict[str, int] = {}
        correct_by_model = {name: 0 for name in names}
        only_correct = {name: 0 for name in names}
        all_correct = 0
        none_correct = 0
        for key in split_keys:
            flags = [models[name][key].correct for name in names]
            pattern = "".join("1" if flag else "0" for flag in flags)
            patterns[pattern] = patterns.get(pattern, 0) + 1
            for name, flag in zip(names, flags):
                correct_by_model[name] += int(flag)
            all_correct += int(all(flags))
            none_correct += int(not any(flags))
            if sum(flags) == 1:
                only_correct[names[flags.index(True)]] += 1
        sample_count = len(split_keys)
        by_split[split] = {
            "sample_count": sample_count,
            "model_order": names,
            "pattern_counts": dict(sorted(patterns.items())),
            "correct_by_model": correct_by_model,
            "only_correct": only_correct,
            "all_correct": all_correct,
            "none_correct": none_correct,
            "any_model_oracle_accuracy": (sample_count - none_correct)
            / sample_count,
        }
    return {"model_order": names, "by_split": by_split}
