"""Run one real 19-class PTv2 train batch and one validation forward batch."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


SANDBOX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SANDBOX_ROOT / "src"))

from tscmdl_whu import (  # noqa: E402
    PointManifestDataset,
    PointTransformerV2Classifier,
    load_memmap_knn_cache,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{path}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def augment_points(points: torch.Tensor) -> torch.Tensor:
    points = points.clone()
    batch_size = points.shape[0]
    angles = torch.rand(batch_size, device=points.device) * (2.0 * torch.pi)
    cosine = torch.cos(angles)
    sine = torch.sin(angles)
    x = points[:, :, 0].clone()
    y = points[:, :, 1].clone()
    points[:, :, 0] = cosine[:, None] * x - sine[:, None] * y
    points[:, :, 1] = sine[:, None] * x + cosine[:, None] * y
    scale = 0.9 + 0.2 * torch.rand(
        batch_size, 1, 1, device=points.device
    )
    return points * scale


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the C2a GPU smoke test")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    dataset_root = args.dataset_root.resolve()
    cache_dir = args.cache_dir.resolve()

    load_started = time.perf_counter()
    datasets = {
        split: PointManifestDataset(dataset_root, split)
        for split in ("train", "val")
    }
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            generator=generator,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        ),
    }
    references, cache_metadata = load_memmap_knn_cache(
        cache_dir, dataset_root, datasets
    )
    load_seconds = time.perf_counter() - load_started

    num_classes = len(datasets["train"].class_names)
    if num_classes != 19:
        raise ValueError(f"C2a requires 19 classes, got {num_classes}")
    model = PointTransformerV2Classifier(num_classes).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=0.05,
    )
    scaler = torch.amp.GradScaler(
        "cuda", init_scale=256.0, enabled=args.amp
    )
    observed_parameter = model.classifier[-1].weight
    parameter_before = observed_parameter.detach().clone()

    try:
        torch.cuda.reset_peak_memory_stats(device)
        train_started = time.perf_counter()
        model.train()
        points, labels, indices = next(iter(loaders["train"]))
        neighbour_index = references["train"].index_select(0, indices.long())
        points = augment_points(points.to(device, non_blocking=True))
        labels = labels.to(device, non_blocking=True)
        neighbour_index = neighbour_index.to(
            device, dtype=torch.long, non_blocking=True
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=args.amp
        ):
            logits = model(points, neighbour_index)
            loss = criterion(logits, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError("Non-finite gradient norm")
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize(device)
        parameter_delta = float(
            (observed_parameter.detach() - parameter_before).abs().max()
        )
        optimizer_step_succeeded = (
            parameter_delta > 0.0
            and (not scaler.is_enabled() or scaler.get_scale() >= scale_before)
        )
        if not optimizer_step_succeeded:
            raise FloatingPointError("The single optimizer step was skipped")
        train_seconds = time.perf_counter() - train_started
        train_peak_mib = torch.cuda.max_memory_allocated(device) / (1024**2)
        train_reserved_mib = torch.cuda.max_memory_reserved(device) / (1024**2)
        train_result = {
            "batch_shape": list(points.shape),
            "reference_shape": list(neighbour_index.shape),
            "logit_shape": list(logits.shape),
            "loss": float(loss.detach()),
            "gradient_norm_before_clipping": float(gradient_norm),
            "optimizer_step_succeeded": optimizer_step_succeeded,
            "classifier_parameter_max_delta": parameter_delta,
            "elapsed_seconds": round(train_seconds, 3),
            "peak_allocated_mib": round(train_peak_mib, 3),
            "peak_reserved_mib": round(train_reserved_mib, 3),
            "stage_point_counts": model.last_stage_point_counts,
        }

        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats(device)
        validation_started = time.perf_counter()
        model.eval()
        with torch.no_grad():
            val_points, val_labels, val_indices = next(iter(loaders["val"]))
            val_reference = references["val"].index_select(
                0, val_indices.long()
            )
            val_points = val_points.to(device, non_blocking=True)
            val_labels = val_labels.to(device, non_blocking=True)
            val_reference = val_reference.to(
                device, dtype=torch.long, non_blocking=True
            )
            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=args.amp
            ):
                val_logits = model(val_points, val_reference)
                val_loss = criterion(val_logits, val_labels)
        if not torch.isfinite(val_loss):
            raise FloatingPointError("Non-finite validation loss")
        torch.cuda.synchronize(device)
        validation_result = {
            "batch_shape": list(val_points.shape),
            "reference_shape": list(val_reference.shape),
            "logit_shape": list(val_logits.shape),
            "loss": float(val_loss),
            "elapsed_seconds": round(
                time.perf_counter() - validation_started, 3
            ),
            "peak_allocated_mib": round(
                torch.cuda.max_memory_allocated(device) / (1024**2), 3
            ),
            "peak_reserved_mib": round(
                torch.cuda.max_memory_reserved(device) / (1024**2), 3
            ),
        }
    finally:
        for reference in references.values():
            reference.close()

    result = {
        "stage": "C2a",
        "status": "passed",
        "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "scope": "one training batch and one validation forward batch",
        "formal_training_started": False,
        "checkpoint_written": False,
        "official_test_split_loaded": False,
        "dataset_root": str(dataset_root),
        "cache_dir": str(cache_dir),
        "split_sizes": {
            split: len(dataset) for split, dataset in datasets.items()
        },
        "class_count": num_classes,
        "class_names": list(datasets["train"].class_names),
        "batch_size": args.batch_size,
        "amp_enabled": args.amp,
        "seed": args.seed,
        "dataset_and_cache_load_seconds": round(load_seconds, 3),
        "model": {
            "name": "PTv2 mode-1 classification encoder reimplementation",
            "parameter_count": sum(
                parameter.numel() for parameter in model.parameters()
            ),
            "configuration": model.configuration,
        },
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
        },
        "cache": cache_metadata,
        "train_batch": train_result,
        "validation_batch": validation_result,
    }
    atomic_json(args.output.resolve(), result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
