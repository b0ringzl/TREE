"""Memory-mapped PTv2 neighbour-cache access for full WHU-STree datasets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class MemmapReferenceCache:
    """Expose a split memmap through the tensor-like index_select contract."""

    def __init__(
        self,
        path: str | Path,
        sample_keys: Sequence[str],
        *,
        point_count: int,
        neighbours: int,
    ) -> None:
        self.path = Path(path)
        self.sample_keys = tuple(str(value) for value in sample_keys)
        self.point_count = int(point_count)
        self.neighbours = int(neighbours)
        self.array = np.load(self.path, mmap_mode="r", allow_pickle=False)
        expected = (len(self.sample_keys), self.point_count, self.neighbours)
        if self.array.shape != expected or self.array.dtype != np.int32:
            raise ValueError(
                f"Unexpected kNN memmap contract for {self.path}: "
                f"{self.array.shape} {self.array.dtype} != {expected} int32"
            )

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(value) for value in self.array.shape)

    def index_select(self, dimension: int, indices: torch.Tensor) -> torch.Tensor:
        if dimension != 0:
            raise ValueError("MemmapReferenceCache only supports dimension 0")
        if indices.ndim != 1:
            raise ValueError("indices must be one-dimensional")
        numpy_indices = indices.detach().cpu().numpy().astype(np.int64, copy=False)
        if len(numpy_indices) and (
            numpy_indices.min() < 0 or numpy_indices.max() >= len(self.sample_keys)
        ):
            raise IndexError("kNN cache index outside the split range")
        values = np.array(
            self.array[numpy_indices],
            dtype=np.int32,
            copy=True,
            order="C",
        )
        return torch.from_numpy(values)

    def close(self) -> None:
        mmap = getattr(self.array, "_mmap", None)
        if mmap is not None:
            mmap.close()

    def __enter__(self) -> "MemmapReferenceCache":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def load_memmap_knn_cache(
    cache_root: str | Path,
    dataset_root: str | Path,
    datasets: dict[str, object],
) -> tuple[dict[str, MemmapReferenceCache], dict[str, object]]:
    root = Path(cache_root)
    index_path = root / "index.json"
    if not index_path.is_file():
        raise FileNotFoundError(index_path)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index.get("schema_version") != 2:
        raise ValueError(f"Unsupported memmap kNN schema: {index.get('schema_version')}")
    if index.get("cache_type") != "ptv2_memmap_knn":
        raise ValueError(f"Unexpected cache type: {index.get('cache_type')}")
    if int(index.get("point_count", 0)) != 8192:
        raise ValueError("PTv2 memmap cache must use 8192 points")
    if int(index.get("neighbours", 0)) != 8:
        raise ValueError("PTv2 memmap cache must use 8 neighbours")

    dataset = Path(dataset_root)
    if index["manifest_sha256"] != sha256_file(dataset / "manifest.json"):
        raise ValueError("kNN cache manifest hash does not match the dataset")
    if index["classes_sha256"] != sha256_file(dataset / "classes.json"):
        raise ValueError("kNN cache classes hash does not match the dataset")

    references: dict[str, MemmapReferenceCache] = {}
    file_entries = {}
    for split, split_dataset in datasets.items():
        entry = index["splits"].get(split)
        if entry is None:
            raise ValueError(f"kNN cache does not contain split {split}")
        sample_keys = tuple(str(value) for value in entry["sample_keys"])
        if sample_keys != tuple(split_dataset.sample_keys):
            raise ValueError(f"kNN cache sample order mismatch for split {split}")
        path = root / str(entry["path"])
        references[split] = MemmapReferenceCache(
            path,
            sample_keys,
            point_count=int(index["point_count"]),
            neighbours=int(index["neighbours"]),
        )
        file_entries[split] = {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": entry["sha256"],
            "shape": list(references[split].shape),
        }

    index_sha256 = sha256_file(index_path)
    metadata = {
        key: value for key, value in index.items() if key != "splits"
    }
    metadata.update(
        {
            "path": str(root.resolve()),
            "bytes": index_path.stat().st_size,
            "sha256": index_sha256,
            "index_sha256": index_sha256,
            "files": file_entries,
            "tensor_shapes": {
                split: list(reference.shape)
                for split, reference in references.items()
            },
        }
    )
    return references, metadata
