"""Cache the full-resolution 8-neighbour graph used by the B5 PTv2 stem."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch


SANDBOX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SANDBOX_ROOT / "src"))

from tscmdl_whu import PointManifestDataset, knn_query_packed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--neighbours", type=int, default=8)
    parser.add_argument("--query-chunk-size", type=int, default=2048)
    parser.add_argument("--full-matrix-limit", type=int, default=8192)
    parser.add_argument("--progress-every", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{path}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.neighbours <= 0:
        raise ValueError("neighbours must be positive")
    if args.query_chunk_size <= 0 or args.full_matrix_limit <= 0:
        raise ValueError("query limits must be positive")
    if args.progress_every <= 0:
        raise ValueError("progress-every must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to build the B5 kNN cache")
    if args.output.exists() and not args.force:
        raise FileExistsError(f"Cache already exists: {args.output}")

    dataset_root = args.dataset_root.resolve()
    manifest_path = dataset_root / "manifest.json"
    classes_path = dataset_root / "classes.json"
    if not manifest_path.is_file() or not classes_path.is_file():
        raise FileNotFoundError("B1 manifest.json and classes.json are required")

    started = time.perf_counter()
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    split_payload: dict[str, dict[str, object]] = {}
    split_sizes: dict[str, int] = {}
    total_samples = 0

    for split in ("train", "val", "test"):
        dataset = PointManifestDataset(dataset_root, split)
        split_sizes[split] = len(dataset)
        references: list[torch.Tensor] = []
        split_started = time.perf_counter()
        for index in range(len(dataset)):
            points, _, _ = dataset[index]
            points = points.to(device, non_blocking=False)
            offset = torch.tensor([points.shape[0]], dtype=torch.long, device=device)
            reference = knn_query_packed(
                points,
                offset,
                args.neighbours,
                query_chunk_size=args.query_chunk_size,
                full_matrix_limit=args.full_matrix_limit,
            )
            if tuple(reference.shape) != (8192, args.neighbours):
                raise ValueError(
                    f"Unexpected neighbour graph for {split}[{index}]: "
                    f"{tuple(reference.shape)}"
                )
            references.append(reference.to(device="cpu", dtype=torch.int32))
            total_samples += 1
            if (
                (index + 1) % args.progress_every == 0
                or index + 1 == len(dataset)
            ):
                elapsed = time.perf_counter() - split_started
                rate = (index + 1) / elapsed
                print(
                    f"\r{split:5s} [{index + 1:3d}/{len(dataset):3d}] "
                    f"{rate:5.2f} trees/s",
                    end="",
                    flush=True,
                )
        print(flush=True)
        split_payload[split] = {
            "sample_keys": list(dataset.sample_keys),
            "reference_index": torch.stack(references, dim=0),
        }
        del dataset

    payload = {
        "schema_version": 1,
        "cache_type": "ptv2_full_resolution_knn",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "dataset_root": str(dataset_root),
        "manifest_sha256": sha256_file(manifest_path),
        "classes_sha256": sha256_file(classes_path),
        "point_count": 8192,
        "neighbours": args.neighbours,
        "distance": "exact Euclidean kNN within each tree instance",
        "split_sizes": split_sizes,
        "splits": split_payload,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{args.output}.tmp")
    torch.save(payload, temporary)
    temporary.replace(args.output)
    cache_sha256 = sha256_file(args.output)
    metadata = {
        key: value
        for key, value in payload.items()
        if key != "splits"
    }
    metadata.update(
        {
            "output": str(args.output.resolve()),
            "output_bytes": args.output.stat().st_size,
            "output_sha256": cache_sha256,
            "sample_count": total_samples,
            "tensor_shapes": {
                split: list(split_payload[split]["reference_index"].shape)
                for split in ("train", "val", "test")
            },
        }
    )
    metadata_path = args.output.with_suffix(".json")
    atomic_json(metadata_path, metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
