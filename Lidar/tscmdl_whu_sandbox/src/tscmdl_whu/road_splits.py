"""Road-grouped split planning for WHU-STree classification experiments."""

from __future__ import annotations

import hashlib
import itertools
import math
import random
from collections import Counter
from typing import Iterable, Mapping, Sequence


RoadClassCounts = dict[str, Counter[int]]


def grouped_counts_can_meet_minimum(
    group_counts: Sequence[int],
    *,
    n_splits: int,
    minimum: int,
) -> bool:
    """Return whether whole groups can give every split at least a minimum."""
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if minimum < 0:
        raise ValueError("minimum must be non-negative")
    if minimum == 0:
        return True
    positive = [int(count) for count in group_counts if int(count) > 0]
    if len(positive) < n_splits or sum(positive) < n_splits * minimum:
        return False

    states = {tuple(0 for _ in range(n_splits))}
    for count in sorted(positive, reverse=True):
        next_states = set(states)
        for state in states:
            for split in range(n_splits):
                updated = list(state)
                updated[split] = min(minimum, updated[split] + count)
                next_states.add(tuple(sorted(updated)))
        states = next_states
        if tuple(minimum for _ in range(n_splits)) in states:
            return True
    return False


def maximum_grouped_minimum(
    group_counts: Sequence[int],
    *,
    n_splits: int,
    upper_bound: int | None = None,
) -> int:
    """Find the largest common minimum, optionally capped at an upper bound."""
    counts = [int(count) for count in group_counts if int(count) > 0]
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if upper_bound is not None and upper_bound < 0:
        raise ValueError("upper_bound must be non-negative")
    if len(counts) < n_splits:
        return 0
    low = 0
    high = sum(counts) // n_splits
    if upper_bound is not None:
        high = min(high, upper_bound)
    while low < high:
        middle = (low + high + 1) // 2
        if grouped_counts_can_meet_minimum(
            counts, n_splits=n_splits, minimum=middle
        ):
            low = middle
        else:
            high = middle - 1
    return low


def build_road_class_counts(
    instances: Sequence[Mapping[str, object]],
    labels: Sequence[int],
) -> RoadClassCounts:
    """Aggregate sample counts by road while retaining zero-count classes."""
    label_set = set(labels)
    counts: RoadClassCounts = {}
    for instance in instances:
        label = int(instance["benchmark_label_id"])
        if label not in label_set:
            continue
        road_id = str(instance["road_id"])
        counts.setdefault(road_id, Counter())[label] += 1
    return {road: counts[road] for road in sorted(counts)}


def _fold_counts(
    road_counts: RoadClassCounts,
    assignments: Mapping[str, int],
    n_splits: int,
) -> tuple[list[Counter[int]], list[int]]:
    counts = [Counter() for _ in range(n_splits)]
    road_sizes = [0] * n_splits
    for road, fold in assignments.items():
        counts[fold].update(road_counts[road])
        road_sizes[fold] += 1
    return counts, road_sizes


def _outer_score(
    fold_counts: Sequence[Counter[int]],
    road_sizes: Sequence[int],
    labels: Sequence[int],
    minimum_per_class: Mapping[int, int],
) -> float:
    n_splits = len(fold_counts)
    hard_deficit = 0.0
    volume_imbalance = 0.0
    for label in labels:
        minimum = float(minimum_per_class[label])
        total = float(sum(counts[label] for counts in fold_counts))
        for counts in fold_counts:
            deficit = max(0.0, minimum - counts[label]) / minimum
            hard_deficit += deficit * deficit
            share = counts[label] / total
            volume_imbalance += (share - 1.0 / n_splits) ** 2
    mean_roads = sum(road_sizes) / n_splits
    road_imbalance = sum(
        ((size - mean_roads) / max(mean_roads, 1.0)) ** 2 for size in road_sizes
    )
    return hard_deficit * 1_000_000.0 + volume_imbalance + road_imbalance


def _validate_outer_feasibility(
    road_counts: RoadClassCounts,
    labels: Sequence[int],
    n_splits: int,
    minimum_per_class: Mapping[int, int],
) -> None:
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if len(road_counts) < n_splits:
        raise ValueError("There are fewer roads than folds")
    for label in labels:
        road_support = sum(counts[label] > 0 for counts in road_counts.values())
        if road_support < n_splits:
            raise ValueError(
                f"Label {label} appears on {road_support} roads, fewer than "
                f"{n_splits} folds"
            )
        total = sum(counts[label] for counts in road_counts.values())
        required = n_splits * int(minimum_per_class[label])
        if total < required:
            raise ValueError(
                f"Label {label} has {total} samples, fewer than the required "
                f"{required}"
            )


def plan_outer_road_folds(
    road_counts: RoadClassCounts,
    labels: Sequence[int],
    *,
    n_splits: int,
    minimum_per_class: Mapping[int, int],
    seed: int,
    restarts: int = 2000,
) -> dict[str, int]:
    """Plan deterministic outer folds with whole-road class constraints."""
    if restarts <= 0:
        raise ValueError("restarts must be positive")
    labels = tuple(labels)
    roads = tuple(sorted(road_counts))
    _validate_outer_feasibility(
        road_counts, labels, n_splits, minimum_per_class
    )

    base_size, remainder = divmod(len(roads), n_splits)
    capacities = [base_size + int(fold < remainder) for fold in range(n_splits)]
    support = {
        label: sum(road_counts[road][label] > 0 for road in roads)
        for label in labels
    }
    best: tuple[float, tuple[int, ...], dict[str, int]] | None = None

    for restart in range(restarts):
        rng = random.Random(seed + restart * 1_000_003)
        jitter = {road: rng.random() for road in roads}
        order = sorted(
            roads,
            key=lambda road: (
                -sum(
                    min(
                        road_counts[road][label],
                        int(minimum_per_class[label]),
                    )
                    / max(support[label], 1)
                    for label in labels
                ),
                jitter[road],
                road,
            ),
        )
        assignments: dict[str, int] = {}
        fold_counts = [Counter() for _ in range(n_splits)]
        fold_sizes = [0] * n_splits
        for road in order:
            candidates = []
            for fold in range(n_splits):
                if fold_sizes[fold] >= capacities[fold]:
                    continue
                gain = 0.0
                for label in labels:
                    minimum = int(minimum_per_class[label])
                    before = min(fold_counts[fold][label], minimum)
                    after = min(
                        fold_counts[fold][label] + road_counts[road][label],
                        minimum,
                    )
                    gain += (after - before) / minimum
                candidates.append(
                    (
                        -gain,
                        fold_sizes[fold] / capacities[fold],
                        rng.random(),
                        fold,
                    )
                )
            chosen = min(candidates)[-1]
            assignments[road] = chosen
            fold_counts[chosen].update(road_counts[road])
            fold_sizes[chosen] += 1

        score = _outer_score(
            fold_counts, fold_sizes, labels, minimum_per_class
        )
        signature = tuple(assignments[road] for road in roads)
        candidate = (score, signature, assignments)
        if best is None or candidate[:2] < best[:2]:
            best = candidate

    if best is None:
        raise RuntimeError("Unable to create an outer-fold candidate")
    assignments = dict(best[2])

    for _ in range(100):
        current_counts, current_sizes = _fold_counts(
            road_counts, assignments, n_splits
        )
        current_score = _outer_score(
            current_counts, current_sizes, labels, minimum_per_class
        )
        best_swap: tuple[float, str, str] | None = None
        for left, right in itertools.combinations(roads, 2):
            if assignments[left] == assignments[right]:
                continue
            assignments[left], assignments[right] = (
                assignments[right],
                assignments[left],
            )
            swapped_counts, swapped_sizes = _fold_counts(
                road_counts, assignments, n_splits
            )
            swapped_score = _outer_score(
                swapped_counts, swapped_sizes, labels, minimum_per_class
            )
            assignments[left], assignments[right] = (
                assignments[right],
                assignments[left],
            )
            candidate = (swapped_score, left, right)
            if swapped_score + 1e-12 < current_score and (
                best_swap is None or candidate < best_swap
            ):
                best_swap = candidate
        if best_swap is None:
            break
        _, left, right = best_swap
        assignments[left], assignments[right] = (
            assignments[right],
            assignments[left],
        )

    final_counts, _ = _fold_counts(road_counts, assignments, n_splits)
    for fold, counts in enumerate(final_counts):
        missing = {
            label: int(minimum_per_class[label]) - counts[label]
            for label in labels
            if counts[label] < int(minimum_per_class[label])
        }
        if missing:
            raise ValueError(
                f"No feasible outer assignment was found; fold {fold} "
                f"deficits: {missing}"
            )
    return {road: assignments[road] for road in roads}


def choose_inner_validation_roads(
    road_counts: RoadClassCounts,
    labels: Sequence[int],
    test_roads: Iterable[str],
    *,
    validation_minimum: Mapping[int, int],
    training_minimum: Mapping[int, int],
    target_road_count: int,
    seed: int,
    max_candidates: int = 200_000,
) -> tuple[str, ...]:
    """Choose validation roads while retaining class support in training."""
    if target_road_count <= 0:
        raise ValueError("target_road_count must be positive")
    test_set = set(test_roads)
    candidates = tuple(sorted(set(road_counts) - test_set))
    if target_road_count >= len(candidates):
        raise ValueError("Validation roads would leave no training roads")

    total = Counter()
    test_counts = Counter()
    for road, counts in road_counts.items():
        total.update(counts)
        if road in test_set:
            test_counts.update(counts)

    rng = random.Random(seed)
    for road_count in range(
        target_road_count, min(target_road_count + 3, len(candidates) - 1) + 1
    ):
        combination_count = math.comb(len(candidates), road_count)
        if combination_count <= max_candidates:
            combinations = itertools.combinations(candidates, road_count)
        else:
            seen: set[tuple[str, ...]] = set()
            generated = []
            while len(generated) < max_candidates:
                combination = tuple(sorted(rng.sample(candidates, road_count)))
                if combination not in seen:
                    seen.add(combination)
                    generated.append(combination)
            combinations = iter(generated)

        best: tuple[float, tuple[str, ...]] | None = None
        for combination in combinations:
            validation_counts = Counter()
            for road in combination:
                validation_counts.update(road_counts[road])
            training_counts = total - test_counts - validation_counts
            if any(
                validation_counts[label] < int(validation_minimum[label])
                or training_counts[label] < int(training_minimum[label])
                for label in labels
            ):
                continue
            excess = sum(
                (
                    (
                        validation_counts[label]
                        - int(validation_minimum[label])
                    )
                    / max(total[label], 1)
                )
                ** 2
                for label in labels
            )
            sample_fraction = sum(validation_counts.values()) / max(
                sum(total.values()), 1
            )
            score = excess + sample_fraction * 0.01
            candidate = (score, combination)
            if best is None or candidate < best:
                best = candidate
        if best is not None:
            return best[1]
    raise ValueError("No road-disjoint validation subset satisfies the constraints")


def select_balanced_evaluation_keys(
    instances: Sequence[Mapping[str, object]],
    labels: Sequence[int],
    *,
    per_class: int,
    seed: int,
    namespace: str,
) -> set[str]:
    """Select an exact, deterministic per-class evaluation subset."""
    if per_class <= 0:
        raise ValueError("per_class must be positive")
    selected: set[str] = set()
    for label in labels:
        pool = [
            instance
            for instance in instances
            if int(instance["benchmark_label_id"]) == label
        ]
        if len(pool) < per_class:
            raise ValueError(
                f"Label {label} has {len(pool)} evaluation samples; "
                f"{per_class} required"
            )
        ranked = sorted(
            pool,
            key=lambda instance: hashlib.sha256(
                (
                    f"{seed}|{namespace}|{instance['sample_key']}"
                ).encode("utf-8")
            ).digest(),
        )
        selected.update(str(instance["sample_key"]) for instance in ranked[:per_class])
    return selected
