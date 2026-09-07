"""Deterministic grouped split planning for WHU-STree classification data."""

from __future__ import annotations

import hashlib
import itertools
from collections import Counter, defaultdict
from typing import Iterable, Mapping, Sequence


def stable_sample_seed(sample_key: str, base_seed: int) -> int:
    payload = f"{base_seed}|{sample_key}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def choose_validation_roads(
    instances: Sequence[Mapping[str, object]],
    labels: Sequence[int],
    minimum_per_class: Mapping[int, int],
    *,
    excluded_roads: Iterable[str] = (),
    max_roads: int = 4,
) -> tuple[str, ...]:
    """Choose a small road-disjoint validation group meeting every class minimum."""
    if max_roads <= 0:
        raise ValueError("max_roads must be positive")
    label_set = set(labels)
    excluded = set(excluded_roads)
    by_road: dict[str, Counter[int]] = defaultdict(Counter)
    for instance in instances:
        if bool(instance["is_official_test"]):
            continue
        road_id = str(instance["road_id"])
        label = int(instance["benchmark_label_id"])
        if road_id not in excluded and label in label_set:
            by_road[road_id][label] += 1

    roads = sorted(by_road)
    best: tuple[tuple[float, int, int, tuple[str, ...]], tuple[str, ...]] | None = None
    for group_size in range(1, min(max_roads, len(roads)) + 1):
        for combination in itertools.combinations(roads, group_size):
            counts = Counter()
            for road_id in combination:
                counts.update(by_road[road_id])
            if any(counts[label] < minimum_per_class[label] for label in labels):
                continue
            normalized_surplus = sum(
                (counts[label] - minimum_per_class[label]) / minimum_per_class[label]
                for label in labels
            )
            total_surplus = sum(
                counts[label] - minimum_per_class[label] for label in labels
            )
            score = (normalized_surplus, total_surplus, group_size, combination)
            if best is None or score < best[0]:
                best = (score, combination)
    if best is None:
        raise ValueError("No validation-road combination satisfies all class minimums")
    return best[1]


def _stable_rank(sample_key: str, split: str, seed: int) -> bytes:
    return hashlib.sha256(f"{seed}|{split}|{sample_key}".encode("utf-8")).digest()


def select_balanced_records(
    instances: Sequence[Mapping[str, object]],
    labels: Sequence[int],
    class_names: Mapping[int, str],
    validation_roads: Iterable[str],
    quotas: Mapping[str, Mapping[int, int]],
    *,
    seed: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Assign grouped pools, then deterministically sample exact per-class quotas."""
    validation = set(validation_roads)
    labels_tuple = tuple(labels)
    class_index = {label: index for index, label in enumerate(labels_tuple)}
    pools: dict[tuple[str, int], list[Mapping[str, object]]] = defaultdict(list)

    for instance in instances:
        label = int(instance["benchmark_label_id"])
        if label not in class_index:
            continue
        if bool(instance["is_official_test"]):
            split = "test"
        elif str(instance["road_id"]) in validation:
            split = "val"
        else:
            split = "train"
        pools[(split, label)].append(instance)

    selected: list[dict[str, object]] = []
    pool_histogram: dict[str, dict[str, int]] = {}
    selected_histogram: dict[str, dict[str, int]] = {}
    for split in ("train", "val", "test"):
        pool_histogram[split] = {}
        selected_histogram[split] = {}
        for label in labels_tuple:
            pool = pools[(split, label)]
            requested = int(quotas[split][label])
            pool_histogram[split][str(label)] = len(pool)
            if len(pool) < requested:
                raise ValueError(
                    f"Insufficient {split} samples for label {label}: {len(pool)} < {requested}"
                )
            ranked = sorted(
                pool,
                key=lambda item: _stable_rank(str(item["sample_key"]), split, seed),
            )
            for item in ranked[:requested]:
                record = dict(item)
                record.update(
                    {
                        "split": split,
                        "class_index": class_index[label],
                        "scientific_name": class_names[label],
                        "sample_seed": stable_sample_seed(str(item["sample_key"]), seed),
                    }
                )
                selected.append(record)
            selected_histogram[split][str(label)] = requested

    split_order = {"train": 0, "val": 1, "test": 2}
    selected.sort(
        key=lambda item: (
            split_order[str(item["split"])],
            int(item["class_index"]),
            str(item["sample_key"]),
        )
    )
    keys = [str(item["sample_key"]) for item in selected]
    if len(keys) != len(set(keys)):
        raise ValueError("A sample was selected into more than one split")

    summary = {
        "validation_roads": sorted(validation),
        "pool_histogram": pool_histogram,
        "selected_histogram": selected_histogram,
        "selected_count": len(selected),
    }
    return selected, summary
