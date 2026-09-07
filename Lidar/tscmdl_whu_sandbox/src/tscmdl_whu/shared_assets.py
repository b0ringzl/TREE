"""Shared-asset planning helpers for C1 benchmark and road-domain tracks."""

from __future__ import annotations

import hashlib
import itertools
from collections import defaultdict
from pathlib import PurePosixPath
from typing import Mapping, Sequence


def _is_official_test(record: Mapping[str, object]) -> bool:
    value = record["is_official_test"]
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _trajectory_key(record: Mapping[str, object]) -> tuple[str, str]:
    return str(record["road_id"]), str(record["trajectory_id"])


def validate_shared_asset_paths(record: Mapping[str, object]) -> None:
    """Require the canonical C1b path for both physical assets."""
    sample_key = str(record["sample_key"])
    expected = {
        "point_path": PurePosixPath("assets", "points", f"{sample_key}.npz"),
        "image_path": PurePosixPath("assets", "images", f"{sample_key}.jpg"),
    }
    for field, expected_path in expected.items():
        actual = PurePosixPath(str(record[field]))
        if actual.is_absolute() or ".." in actual.parts or actual != expected_path:
            raise ValueError(
                f"{field} for {sample_key} is not the canonical shared path: {actual}"
            )


def choose_minimum_trajectory_cover(
    records: Sequence[Mapping[str, object]],
    labels: Sequence[int],
    *,
    max_trajectories: int,
    allow_official_test: bool = False,
) -> tuple[tuple[str, str], ...]:
    """Choose the smallest deterministic trajectory set covering every label."""
    if max_trajectories <= 0:
        raise ValueError("max_trajectories must be positive")
    target = set(int(label) for label in labels)
    if not target:
        raise ValueError("At least one label is required")

    by_trajectory: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(
        list
    )
    for record in records:
        label = int(record["benchmark_label_id"])
        if label not in target:
            continue
        if not allow_official_test and _is_official_test(record):
            continue
        validate_shared_asset_paths(record)
        by_trajectory[_trajectory_key(record)].append(record)

    coverage = {
        key: {int(record["benchmark_label_id"]) for record in rows}
        for key, rows in by_trajectory.items()
    }
    trajectories = sorted(coverage)
    for count in range(1, min(max_trajectories, len(trajectories)) + 1):
        candidates: list[
            tuple[tuple[int, tuple[tuple[str, str], ...]], tuple[tuple[str, str], ...]]
        ] = []
        for combination in itertools.combinations(trajectories, count):
            covered = set().union(*(coverage[key] for key in combination))
            if covered != target:
                continue
            source_sample_count = sum(len(by_trajectory[key]) for key in combination)
            candidates.append(((source_sample_count, combination), combination))
        if candidates:
            return min(candidates, key=lambda item: item[0])[1]
    raise ValueError(
        f"No set of at most {max_trajectories} eligible trajectories covers all labels"
    )


def _stable_rank(sample_key: str, label: int, seed: int) -> bytes:
    return hashlib.sha256(
        f"{seed}|c1c-smoke|{label}|{sample_key}".encode("utf-8")
    ).digest()


def select_shared_smoke_records(
    records: Sequence[Mapping[str, object]],
    labels: Sequence[int],
    *,
    samples_per_class: int,
    max_trajectories: int,
    seed: int,
    allow_official_test: bool = False,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Select a small deterministic class-complete shared-asset smoke set."""
    if samples_per_class <= 0:
        raise ValueError("samples_per_class must be positive")
    labels_tuple = tuple(int(label) for label in labels)
    selected_trajectories = choose_minimum_trajectory_cover(
        records,
        labels_tuple,
        max_trajectories=max_trajectories,
        allow_official_test=allow_official_test,
    )
    trajectory_set = set(selected_trajectories)

    by_label: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    for record in records:
        label = int(record["benchmark_label_id"])
        if label not in labels_tuple or _trajectory_key(record) not in trajectory_set:
            continue
        if not allow_official_test and _is_official_test(record):
            continue
        by_label[label].append(record)

    selected: list[dict[str, object]] = []
    for class_index, label in enumerate(labels_tuple):
        candidates = sorted(
            by_label[label],
            key=lambda record: _stable_rank(str(record["sample_key"]), label, seed),
        )
        if len(candidates) < samples_per_class:
            raise ValueError(
                f"Label {label} has only {len(candidates)} candidates in the selected "
                f"trajectories; {samples_per_class} required"
            )
        for record in candidates[:samples_per_class]:
            output = dict(record)
            output["class_index"] = class_index
            selected.append(output)

    selected.sort(
        key=lambda record: (
            str(record["road_id"]),
            str(record["trajectory_id"]),
            int(record["benchmark_label_id"]),
            str(record["sample_key"]),
        )
    )
    keys = [str(record["sample_key"]) for record in selected]
    if len(keys) != len(set(keys)):
        raise ValueError("Smoke selection contains duplicate sample keys")
    return selected, {
        "sample_count": len(selected),
        "class_count": len(labels_tuple),
        "samples_per_class": samples_per_class,
        "trajectory_count": len(selected_trajectories),
        "trajectories": [
            {"road_id": road_id, "trajectory_id": trajectory_id}
            for road_id, trajectory_id in selected_trajectories
        ],
        "official_test_allowed": allow_official_test,
    }
