"""Train YOLO11s-cls on the frozen NJ split with balanced sampling."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from collections import Counter
from pathlib import Path

import torch
import ultralytics
from torch.utils.data import WeightedRandomSampler
from ultralytics import YOLO
from ultralytics.data.build import InfiniteDataLoader, seed_worker
from ultralytics.models.yolo.classify.train import ClassificationTrainer


class BalancedClassificationTrainer(ClassificationTrainer):
    """Use one inverse-frequency weighted draw per training sample each epoch."""

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = -1, mode: str = "train"):
        if mode != "train":
            return super().get_dataloader(dataset_path, batch_size, rank, mode)
        if rank not in {-1, 0}:
            raise RuntimeError("Balanced sampler experiment supports one GPU only")
        dataset = self.build_dataset(dataset_path, mode)
        if not dataset.samples:
            raise FileNotFoundError(f"No training images found in {dataset_path}")
        counts = Counter(int(sample[1]) for sample in dataset.samples)
        weights = torch.tensor(
            [1.0 / counts[int(sample[1])] for sample in dataset.samples], dtype=torch.double
        )
        generator = torch.Generator()
        generator.manual_seed(int(self.args.seed))
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(dataset),
            replacement=True,
            generator=generator,
        )
        workers = min(os.cpu_count() or 1, int(self.args.workers))
        return InfiniteDataLoader(
            dataset=dataset,
            batch_size=min(batch_size, len(dataset)),
            sampler=sampler,
            shuffle=False,
            num_workers=workers,
            prefetch_factor=4 if workers > 0 else None,
            pin_memory=True,
            collate_fn=getattr(dataset, "collate_fn", None),
            worker_init_fn=seed_worker,
            generator=generator,
            drop_last=False,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model", default="yolo11s-cls.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    return parser.parse_args()


def gpu_snapshot() -> str:
    try:
        return subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,memory.used",
                "--format=csv,noheader",
            ],
            text=True,
            encoding="utf-8",
        ).strip()
    except Exception as error:
        return f"unavailable: {error}"


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    training_view = dataset_root / "training_view"
    run_dir = args.run_dir.resolve()
    if not (dataset_root / "dataset_summary.json").is_file():
        raise FileNotFoundError(dataset_root / "dataset_summary.json")
    if not (training_view / "train").is_dir() or not (training_view / "val").is_dir():
        raise FileNotFoundError(f"Training-only data view is incomplete: {training_view}")
    if (run_dir / "weights" / "best.pt").exists():
        raise FileExistsError(run_dir / "weights" / "best.pt")
    run_dir.mkdir(parents=True, exist_ok=True)
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "ultralytics": ultralytics.__version__,
        "nvidia_smi": gpu_snapshot(),
        "model": args.model,
        "task": "classify",
        "dataset_root": str(dataset_root),
        "training_view": str(training_view),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "sampler": "WeightedRandomSampler inverse class frequency, replacement=True",
        "test_policy": "test is not loaded by training; evaluate once after validation selection",
    }
    (run_dir / "environment.json").write_text(
        json.dumps(environment, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    started = time.time()
    model = YOLO(args.model)
    model.train(
        trainer=BalancedClassificationTrainer,
        data=str(training_view),
        epochs=args.epochs,
        imgsz=args.image_size,
        batch=args.batch_size,
        device=0,
        workers=8,
        project=str(run_dir.parent),
        name=run_dir.name,
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        lr0=args.learning_rate,
        weight_decay=0.0002,
        seed=args.seed,
        deterministic=True,
        patience=25,
        cos_lr=True,
        amp=True,
        plots=True,
        save=True,
        save_period=1,
        val=True,
        dropout=0.1,
        verbose=True,
    )
    payload = {
        "status": "training_complete",
        "elapsed_seconds": time.time() - started,
        "last_checkpoint": str(run_dir / "weights" / "last.pt"),
        "ultralytics_best_checkpoint": str(run_dir / "weights" / "best.pt"),
        "test_loaded_during_training": False,
    }
    (run_dir / "training_complete.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
