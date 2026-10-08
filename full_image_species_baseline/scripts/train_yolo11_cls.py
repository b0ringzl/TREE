#!/usr/bin/env python3
"""Train YOLO11s-cls with balanced sampling on a materialized dataset."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault('MPLBACKEND', 'Agg')  # Headless plotting avoids Qt/PySide import hooks.

import torch
import ultralytics
from torch.utils.data import WeightedRandomSampler
from ultralytics import YOLO
from ultralytics.data.build import InfiniteDataLoader, seed_worker
from ultralytics.models.yolo.classify.train import ClassificationTrainer


class BalancedClassificationTrainer(ClassificationTrainer):
    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = -1, mode: str = "train"):
        if mode != "train":
            return super().get_dataloader(dataset_path, batch_size, rank, mode)
        if rank not in {-1, 0}:
            raise RuntimeError("Balanced classification supports one GPU")
        dataset = self.build_dataset(dataset_path, mode)
        counts = Counter(int(sample[1]) for sample in dataset.samples)
        weights = torch.tensor([1.0 / counts[int(sample[1])] for sample in dataset.samples], dtype=torch.double)
        sampling_file = os.environ.get('TREE_SAMPLE_WEIGHTS_FILE')
        if sampling_file:
            custom = json.loads(Path(sampling_file).read_text(encoding='utf-8'))
            weights = torch.tensor([custom[Path(sample[0]).name] for sample in dataset.samples], dtype=torch.double)
            if not torch.isfinite(weights).all() or not (weights > 0).all():
                raise ValueError('Invalid explicit sampling weights')
        auxiliary_weight = float(os.environ.get('TREE_HERBARIUM_SAMPLE_WEIGHT', '1'))
        if not 0 < auxiliary_weight <= 5:
            raise ValueError('Herbarium sample multiplier must be in (0, 5]')
        for i, sample in enumerate(dataset.samples):
            if Path(sample[0]).name.startswith('herbarium_'):
                weights[i] *= auxiliary_weight
        generator = torch.Generator().manual_seed(int(self.args.seed))
        sampler = WeightedRandomSampler(weights, len(dataset), replacement=True, generator=generator)
        workers = min(os.cpu_count() or 1, int(self.args.workers))
        return InfiniteDataLoader(
            dataset=dataset,
            batch_size=min(batch_size, len(dataset)),
            sampler=sampler,
            shuffle=False,
            num_workers=workers,
            prefetch_factor=4 if workers else None,
            pin_memory=True,
            collate_fn=getattr(dataset, "collate_fn", None),
            worker_init_fn=seed_worker,
            generator=generator,
            drop_last=False,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--warmup-epochs", type=float, default=3.0)
    parser.add_argument("--warmup-bias-lr", type=float, default=0.1)
    return parser.parse_args()


def gpu_snapshot() -> str:
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used", "--format=csv,noheader"],
            text=True,
            encoding="utf-8",
        ).strip()
    except Exception as error:
        return f"unavailable: {error}"


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    run_dir = args.run_dir.resolve()
    training_view = dataset_root / "training_view"
    if not (training_view / "train").is_dir() or not (training_view / "val").is_dir():
        raise FileNotFoundError(training_view)
    if (run_dir / "weights" / "best.pt").is_file():
        raise FileExistsError(run_dir / "weights" / "best.pt")
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
    from training_safety import backup_before_training, deployed_artifacts
    model_file = Path(args.model).resolve()
    if not model_file.is_file():
        raise FileNotFoundError('Provide a local initialization checkpoint for verified backup: ' + str(model_file))
    sampling_file = os.environ.get('TREE_SAMPLE_WEIGHTS_FILE')
    backup_before_training(run_dir, [model_file, *deployed_artifacts(), *([Path(sampling_file)] if sampling_file else [])], {
        'dataset_root': str(dataset_root), 'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    })
    run_dir.mkdir(parents=True, exist_ok=True)
    environment = {
        "python": platform.python_version(), "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "ultralytics": ultralytics.__version__, "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "nvidia_smi": gpu_snapshot(),
        "model": args.model, "dataset_root": str(dataset_root), "training_view": str(training_view),
        "epochs": args.epochs, "batch_size": args.batch_size, "image_size": args.image_size, "seed": args.seed,
        "learning_rate": args.learning_rate, "sampler": "inverse-frequency WeightedRandomSampler",
        "herbarium_sample_weight": float(os.environ.get('TREE_HERBARIUM_SAMPLE_WEIGHT', '1')),
        "sampling_weights_file": sampling_file,
        "test_policy": "held_out_test is outside training_view",
    }
    (run_dir / "environment.json").write_text(json.dumps(environment, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    started = time.time()
    model = YOLO(args.model)
    model.train(
        trainer=BalancedClassificationTrainer,
        data=str(training_view), epochs=args.epochs, imgsz=args.image_size, batch=args.batch_size,
        device=0, workers=8, project=str(run_dir.parent), name=run_dir.name, exist_ok=True,
        pretrained=True, optimizer="AdamW", lr0=args.learning_rate, weight_decay=0.0002,
        seed=args.seed, deterministic=True, patience=args.patience, cos_lr=True, amp=True,
        plots=True, save=True, save_period=1, val=True, dropout=0.1, verbose=True,
        warmup_epochs=args.warmup_epochs, warmup_bias_lr=args.warmup_bias_lr,
    )
    payload = {
        "status": "training_complete", "elapsed_seconds": time.time() - started,
        "best_checkpoint": str(run_dir / "weights" / "best.pt"), "last_checkpoint": str(run_dir / "weights" / "last.pt"),
        "test_loaded_during_training": False,
    }
    (run_dir / "training_complete.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
