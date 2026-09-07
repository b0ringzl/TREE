"""Build a resumable split-wise memory-mapped PTv2 kNN cache."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


SANDBOX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SANDBOX_ROOT / "src"))

from tscmdl_whu import PointManifestDataset, knn_query_packed  # noqa: E402


SPLITS = ("train", "val", "test")
POINT_COUNT = 8192


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--neighbours", type=int, default=8)
    parser.add_argument("--query-chunk-size", type=int, default=2048)
    parser.add_argument("--full-matrix-limit", type=int, default=8192)
    parser.add_argument("--checkpoint-every", type=int, default=32)
    parser.add_argument("--progress-every", type=int, default=32)
    parser.add_argument("--max-new-samples", type=int)
    return parser.parse_args()


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_keys_sha256(values: tuple[str, ...]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{path}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_progress(
    path: Path,
    *,
    status: str,
    split: str,
    completed: int,
    total: int,
    started: float,
    new_samples: int,
    output_dir: Path,
) -> None:
    elapsed = time.perf_counter() - started
    remaining = total - completed
    eta = elapsed / new_samples * remaining if new_samples else None
    output_bytes = sum(
        item.stat().st_size for item in output_dir.glob("*.npy") if item.is_file()
    )
    atomic_json(
        path,
        {
            "stage": "C2a-kNN-cache",
            "status": status,
            "updated_at": now(),
            "current_split": split,
            "completed_samples": completed,
            "total_samples": total,
            "progress_fraction": completed / total,
            "elapsed_seconds_this_run": round(elapsed, 3),
            "eta_seconds_this_run": round(eta, 3) if eta is not None else None,
            "output_bytes": output_bytes,
            "free_disk_bytes": shutil.disk_usage(output_dir).free,
        },
    )


def main() -> None:
    args = parse_args()
    if args.neighbours <= 0:
        raise ValueError("neighbours must be positive")
    if min(
        args.query_chunk_size,
        args.full_matrix_limit,
        args.checkpoint_every,
        args.progress_every,
    ) <= 0:
        raise ValueError("query/checkpoint/progress values must be positive")
    if args.max_new_samples is not None and args.max_new_samples <= 0:
        raise ValueError("max-new-samples must be positive when provided")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to build the C2a kNN cache")

    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = dataset_root / "manifest.json"
    classes_path = dataset_root / "classes.json"
    if not manifest_path.is_file() or not classes_path.is_file():
        raise FileNotFoundError("PTv2 manifest.json and classes.json are required")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split_sizes = {
        split: int(manifest["summary"]["split_histogram"][split])
        for split in SPLITS
    }
    total_samples = sum(split_sizes.values())
    required_bytes = total_samples * POINT_COUNT * args.neighbours * 4
    if shutil.disk_usage(output_dir).free < required_bytes + 512 * 1024**2:
        raise OSError("Insufficient disk space for C2a kNN memmaps plus safety margin")

    checkpoint_path = output_dir / "checkpoint.json"
    progress_path = output_dir / "progress.json"
    source = {
        "dataset_root": str(dataset_root),
        "manifest_sha256": sha256_file(manifest_path),
        "classes_sha256": sha256_file(classes_path),
        "point_count": POINT_COUNT,
        "neighbours": args.neighbours,
        "split_sizes": split_sizes,
    }
    if checkpoint_path.is_file():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint["source"] != source:
            raise ValueError("Existing C2a cache checkpoint uses a different source")
    else:
        checkpoint = {
            "schema_version": 2,
            "stage": "C2a",
            "status": "planned",
            "source": source,
            "completed": {split: 0 for split in SPLITS},
            "sample_keys_sha256": {},
            "run_events": [],
        }

    completed_before = sum(int(value) for value in checkpoint["completed"].values())
    started_at = now()
    started = time.perf_counter()
    new_samples = 0
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    stop_requested = False
    current_split = ""
    split_sample_keys: dict[str, tuple[str, ...]] = {}
    print(
        f"C2a kNN cache: {completed_before}/{total_samples} samples complete, "
        f"GPU={torch.cuda.get_device_name(0)}",
        flush=True,
    )

    for split in SPLITS:
        current_split = split
        dataset = PointManifestDataset(dataset_root, split)
        split_sample_keys[split] = dataset.sample_keys
        if len(dataset) != split_sizes[split]:
            raise ValueError(f"Split size mismatch for {split}")
        keys_hash = sample_keys_sha256(dataset.sample_keys)
        known_hash = checkpoint["sample_keys_sha256"].get(split)
        if known_hash is not None and known_hash != keys_hash:
            raise ValueError(f"Sample order changed for split {split}")
        checkpoint["sample_keys_sha256"][split] = keys_hash

        cache_path = output_dir / f"{split}_reference_index.npy"
        shape = (len(dataset), POINT_COUNT, args.neighbours)
        if cache_path.is_file():
            references = np.load(cache_path, mmap_mode="r+", allow_pickle=False)
            if references.shape != shape or references.dtype != np.int32:
                raise ValueError(f"Existing memmap contract mismatch: {cache_path}")
        else:
            references = np.lib.format.open_memmap(
                cache_path,
                mode="w+",
                dtype=np.int32,
                shape=shape,
            )
        start_index = int(checkpoint["completed"][split])
        split_started = time.perf_counter()
        for index in range(start_index, len(dataset)):
            points = torch.from_numpy(dataset.points[index]).to(
                device, non_blocking=False
            )
            offset = torch.tensor(
                [points.shape[0]], dtype=torch.long, device=device
            )
            reference = knn_query_packed(
                points,
                offset,
                args.neighbours,
                query_chunk_size=args.query_chunk_size,
                full_matrix_limit=args.full_matrix_limit,
            )
            if tuple(reference.shape) != (POINT_COUNT, args.neighbours):
                raise ValueError(
                    f"Unexpected reference shape for {split}[{index}]: "
                    f"{tuple(reference.shape)}"
                )
            references[index] = (
                reference.to(device="cpu", dtype=torch.int32).numpy()
            )
            checkpoint["completed"][split] = index + 1
            new_samples += 1
            completed = completed_before + new_samples
            if (
                (index + 1) % args.checkpoint_every == 0
                or index + 1 == len(dataset)
            ):
                references.flush()
                checkpoint["status"] = "running"
                atomic_json(checkpoint_path, checkpoint)
                write_progress(
                    progress_path,
                    status="running",
                    split=split,
                    completed=completed,
                    total=total_samples,
                    started=started,
                    new_samples=new_samples,
                    output_dir=output_dir,
                )
            if (
                (index + 1) % args.progress_every == 0
                or index + 1 == len(dataset)
            ):
                elapsed = time.perf_counter() - split_started
                rate = (index + 1 - start_index) / elapsed
                total_elapsed = time.perf_counter() - started
                eta = (
                    total_elapsed / new_samples * (total_samples - completed)
                    if new_samples
                    else 0.0
                )
                print(
                    f"\r{split:5s} [{index + 1:5d}/{len(dataset):5d}] "
                    f"total={completed:5d}/{total_samples} "
                    f"rate={rate:5.1f} trees/s eta={eta/60:4.1f}m",
                    end="",
                    flush=True,
                )
            if (
                args.max_new_samples is not None
                and new_samples >= args.max_new_samples
            ):
                references.flush()
                stop_requested = True
                break
        print(flush=True)
        del references
        del dataset
        gc.collect()
        if stop_requested:
            break

    complete = all(
        int(checkpoint["completed"][split]) == split_sizes[split]
        for split in SPLITS
    )
    status = "complete" if complete else "partial"
    checkpoint["run_events"].append(
        {
            "started_at": started_at,
            "finished_at": now(),
            "status": status,
            "completed_before": completed_before,
            "completed_after": completed_before + new_samples,
            "new_sample_count": new_samples,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }
    )
    checkpoint["status"] = status

    if complete:
        split_entries = {}
        for split in SPLITS:
            path = output_dir / f"{split}_reference_index.npy"
            sample_keys = split_sample_keys[split]
            split_entries[split] = {
                "path": path.name,
                "shape": [len(sample_keys), POINT_COUNT, args.neighbours],
                "dtype": "int32",
                "sample_keys": list(sample_keys),
                "sample_keys_sha256": sample_keys_sha256(sample_keys),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        index = {
            "schema_version": 2,
            "cache_type": "ptv2_memmap_knn",
            "created_at": now(),
            "dataset_root": str(dataset_root),
            "manifest_sha256": source["manifest_sha256"],
            "classes_sha256": source["classes_sha256"],
            "point_count": POINT_COUNT,
            "neighbours": args.neighbours,
            "distance": "exact Euclidean kNN within each tree instance",
            "split_sizes": split_sizes,
            "splits": split_entries,
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "elapsed_seconds_all_runs": sum(
                float(event["elapsed_seconds"])
                for event in checkpoint["run_events"]
            ),
            "run_count": len(checkpoint["run_events"]),
        }
        atomic_json(output_dir / "index.json", index)
        checkpoint["index_sha256"] = sha256_file(output_dir / "index.json")
    atomic_json(checkpoint_path, checkpoint)
    write_progress(
        progress_path,
        status=status,
        split=current_split,
        completed=completed_before + new_samples,
        total=total_samples,
        started=started,
        new_samples=new_samples,
        output_dir=output_dir,
    )
    print(
        json.dumps(
            {
                "status": status,
                "completed_samples": completed_before + new_samples,
                "total_samples": total_samples,
                "new_sample_count": new_samples,
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
