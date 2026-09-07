"""Train and evaluate the B4b frozen-backbone TSCMDL fusion classifier."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


SANDBOX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SANDBOX_ROOT / "src"))

from tscmdl_whu import TSCMDLFusionModel, classification_metrics  # noqa: E402


class CachedFeatureDataset(Dataset):
    """Expose one aligned split from a B4 frozen feature cache."""

    def __init__(self, payload: dict[str, object], split: str) -> None:
        self.split = split
        split_payload = payload["splits"][split]
        self.point_features = split_payload["point_features"].float().contiguous()
        self.image_features = split_payload["image_features"].float().contiguous()
        self.labels = split_payload["labels"].long().contiguous()
        self.sample_keys = tuple(str(value) for value in split_payload["sample_keys"])
        sample_count = len(self.sample_keys)
        dimensions = payload.get("feature_dimensions", {})
        point_dim = int(dimensions.get("point", self.point_features.shape[1]))
        image_dim = int(dimensions.get("image", self.image_features.shape[1]))
        if self.point_features.shape != (sample_count, point_dim):
            raise ValueError(f"Unexpected point features in {split}")
        if self.image_features.shape != (sample_count, image_dim):
            raise ValueError(f"Unexpected image features in {split}")
        if self.labels.shape != (sample_count,):
            raise ValueError(f"Unexpected labels in {split}")
        if len(set(self.sample_keys)) != sample_count:
            raise ValueError(f"Duplicate sample keys in {split}")
        if not torch.isfinite(self.point_features).all():
            raise FloatingPointError(f"Non-finite point features in {split}")
        if not torch.isfinite(self.image_features).all():
            raise FloatingPointError(f"Non-finite image features in {split}")

    def __len__(self) -> int:
        return len(self.sample_keys)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        return (
            self.point_features[index],
            self.image_features[index],
            self.labels[index],
            index,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--quality-review",
        type=Path,
        default=SANDBOX_ROOT / "reports" / "B1_visual_review.json",
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--min-learning-rate", type=float, default=0.0)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument(
        "--normalization",
        choices=("l2", "batchnorm"),
        default="l2",
    )
    parser.add_argument(
        "--classifier-hidden-dims",
        type=int,
        nargs="+",
        default=(512, 256),
    )
    parser.add_argument(
        "--modality",
        choices=("fusion", "point", "image"),
        default="fusion",
    )
    parser.add_argument("--patience", type=int, default=300)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pause-request", type=Path)
    parser.add_argument(
        "--balanced-sampler",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--fused-sgd", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def json_args(args: argparse.Namespace) -> dict[str, object]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def atomic_json(path: Path, value: object) -> None:
    temporary = Path(f"{path}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def atomic_torch_save(path: Path, value: object) -> None:
    temporary = Path(f"{path}.tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def emit(message: str, log_path: Path | None = None) -> None:
    print(message, flush=True)
    if log_path is not None:
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(message + "\n")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_cache(
    path: Path,
) -> tuple[dict[str, object], dict[str, CachedFeatureDataset], str]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError("Unsupported B4 feature cache schema")
    class_names = tuple(str(value) for value in payload["class_names"])
    if len(class_names) < 2:
        raise ValueError(f"Expected at least two classes, got {class_names}")
    datasets = {
        split: CachedFeatureDataset(payload, split)
        for split in ("train", "val", "test")
    }
    expected_sizes = {
        split: int(value)
        for split, value in payload.get("split_sizes", {}).items()
    }
    actual_sizes = {split: len(dataset) for split, dataset in datasets.items()}
    if expected_sizes and actual_sizes != expected_sizes:
        raise ValueError(f"Unexpected cached split sizes: {actual_sizes}")
    all_keys = [
        sample_key
        for dataset in datasets.values()
        for sample_key in dataset.sample_keys
    ]
    if len(all_keys) != len(set(all_keys)):
        raise ValueError("Sample keys overlap across cached splits")
    return payload, datasets, sha256_file(path)


def make_loaders(
    datasets: dict[str, CachedFeatureDataset],
    batch_size: int,
    eval_batch_size: int,
    workers: int,
    seed: int,
    *,
    balanced_sampler: bool = False,
) -> dict[str, DataLoader]:
    generator = torch.Generator().manual_seed(seed)
    sampler = None
    shuffle = True
    if balanced_sampler:
        labels = datasets["train"].labels
        counts = torch.bincount(labels, minlength=len(set(labels.tolist())))
        if torch.any(counts == 0):
            raise ValueError(f"Balanced sampler found an empty class: {counts.tolist()}")
        weights = (1.0 / counts.double())[labels]
        sampler = WeightedRandomSampler(
            weights,
            num_samples=len(datasets["train"]),
            replacement=True,
            generator=generator,
        )
        shuffle = False
    return {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=workers,
            pin_memory=True,
            generator=generator if sampler is None else None,
            persistent_workers=workers > 0,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=True,
            persistent_workers=workers > 0,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=True,
            persistent_workers=workers > 0,
        ),
    }


def make_model(
    num_classes: int,
    dropout: float,
    normalization: str,
    classifier_hidden_dims: tuple[int, ...] = (512, 256),
    modality: str = "fusion",
    point_dim: int = 1024,
    image_dim: int = 2048,
    modal_dim: int = 1024,
) -> TSCMDLFusionModel:
    return TSCMDLFusionModel(
        nn.Identity(),
        nn.Identity(),
        num_classes,
        point_dim=point_dim,
        image_dim=image_dim,
        modal_dim=modal_dim,
        dropout=dropout,
        freeze_backbones=True,
        normalization=normalization,
        classifier_hidden_dims=classifier_hidden_dims,
        modality=modality,
    )


def run_epoch(
    model: TSCMDLFusionModel,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    use_amp: bool = True,
    progress_label: str,
    progress_every: int = 20,
    collect_predictions: bool = False,
    show_progress: bool = True,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    sample_count = 0
    true_labels: list[int] = []
    predicted_labels: list[int] = []
    predictions: list[dict[str, object]] = []
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)

    for batch_index, (point_raw, image_raw, labels, indices) in enumerate(
        loader, start=1
    ):
        point_raw = point_raw.to(device, non_blocking=True)
        image_raw = image_raw.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=use_amp
            ):
                logits = model.forward_from_features(point_raw, image_raw)
                loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss")
            if training:
                if scaler is None:
                    raise ValueError("Training requires a gradient scaler")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 5.0
                )
                if not torch.isfinite(gradient_norm) and not scaler.is_enabled():
                    raise FloatingPointError("Non-finite gradient norm")
                scaler.step(optimizer)
                scaler.update()

        probabilities = torch.softmax(logits.detach().float(), dim=1)
        predicted = probabilities.argmax(dim=1)
        current_batch_size = labels.shape[0]
        total_loss += loss.detach().item() * current_batch_size
        sample_count += current_batch_size
        true_labels.extend(labels.detach().cpu().tolist())
        predicted_labels.extend(predicted.cpu().tolist())
        if collect_predictions:
            for row_index, dataset_index in enumerate(indices.tolist()):
                predictions.append(
                    {
                        "dataset_index": int(dataset_index),
                        "true_class": int(labels[row_index]),
                        "predicted_class": int(predicted[row_index]),
                        "probabilities": probabilities[row_index].cpu().tolist(),
                    }
                )

        if show_progress and (
            batch_index % progress_every == 0 or batch_index == len(loader)
        ):
            accuracy = np.mean(
                np.asarray(true_labels) == np.asarray(predicted_labels)
            )
            print(
                f"\r{progress_label} [{batch_index:3d}/{len(loader):3d}] "
                f"loss={total_loss/sample_count:.4f} acc={accuracy*100:6.2f}%",
                end="",
                flush=True,
            )
    if show_progress:
        print(flush=True)
    if sample_count == 0:
        raise ValueError("No samples were processed")

    torch.cuda.synchronize(device)
    metrics = classification_metrics(true_labels, predicted_labels, num_classes)
    metrics.update(
        {
            "loss": total_loss / sample_count,
            "sample_count": sample_count,
            "elapsed_seconds": time.perf_counter() - started,
            "peak_allocated_mib": torch.cuda.max_memory_allocated(device)
            / (1024**2),
        }
    )
    return metrics, predictions


def attach_sample_metadata(
    predictions: list[dict[str, object]],
    dataset: CachedFeatureDataset,
    split: str,
    quality_reviews: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    output = []
    for prediction in predictions:
        sample_key = dataset.sample_keys[int(prediction["dataset_index"])]
        review = quality_reviews.get(sample_key, {})
        output.append(
            {
                **prediction,
                "sample_key": sample_key,
                "split": split,
                "review_quality": str(review.get("quality", "")),
                "review_note": str(review.get("note", "")),
            }
        )
    return output


def load_quality_reviews(path: Path) -> dict[str, dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    reviews = {
        str(item["sample_key"]): item for item in payload.get("reviews", [])
    }
    if len(reviews) != len(payload.get("reviews", [])):
        raise ValueError("Duplicate sample keys in quality review")
    return reviews


def write_history(output_dir: Path, history: list[dict[str, object]]) -> None:
    atomic_json(output_dir / "training_history.json", history)
    fields = [
        "epoch",
        "learning_rate",
        "train_loss",
        "train_accuracy",
        "train_macro_f1",
        "val_loss",
        "val_accuracy",
        "val_macro_f1",
        "epoch_seconds",
        "peak_allocated_mib",
    ]
    temporary = output_dir / "training_history.csv.tmp"
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)
    temporary.replace(output_dir / "training_history.csv")


def save_predictions(
    path: Path, predictions: list[dict[str, object]], num_classes: int
) -> None:
    fields = [
        "sample_key",
        "split",
        "true_class",
        "predicted_class",
        "correct",
        "review_quality",
        "review_note",
    ] + [f"probability_{index}" for index in range(num_classes)]
    temporary = Path(f"{path}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in predictions:
            row = {
                "sample_key": item["sample_key"],
                "split": item["split"],
                "true_class": item["true_class"],
                "predicted_class": item["predicted_class"],
                "correct": int(item["true_class"] == item["predicted_class"]),
                "review_quality": item["review_quality"],
                "review_note": item["review_note"],
            }
            row.update(
                {
                    f"probability_{index}": probability
                    for index, probability in enumerate(item["probabilities"])
                }
            )
            writer.writerow(row)
    temporary.replace(path)


def plot_results(
    output_dir: Path,
    history: list[dict[str, object]],
    metrics_by_split: dict[str, dict[str, object]],
    class_names: tuple[str, ...],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [int(item["epoch"]) for item in history]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].plot(
        epochs,
        [item["train_loss"] for item in history],
        label="train",
        color="#1565C0",
    )
    axes[0].plot(
        epochs,
        [item["val_loss"] for item in history],
        label="val",
        color="#D84315",
    )
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Cross entropy")
    axes[1].plot(
        epochs,
        [100 * item["train_accuracy"] for item in history],
        label="train",
        color="#00897B",
    )
    axes[1].plot(
        epochs,
        [100 * item["val_accuracy"] for item in history],
        label="val",
        color="#6A1B9A",
    )
    axes[1].set(title="Accuracy", xlabel="Epoch", ylabel="Percent")
    axes[2].plot(
        epochs,
        [100 * item["train_macro_f1"] for item in history],
        label="train",
        color="#2E7D32",
    )
    axes[2].plot(
        epochs,
        [100 * item["val_macro_f1"] for item in history],
        label="val",
        color="#EF6C00",
    )
    axes[2].set(title="Macro F1", xlabel="Epoch", ylabel="Percent")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "training_curves.png", dpi=180)
    plt.close(figure)

    short_names = ["C. camphora", "L. indica", "M. grandiflora"]
    if len(class_names) != 3:
        short_names = list(class_names)
    for split, metrics in metrics_by_split.items():
        matrix = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
        row_sum = matrix.sum(axis=1, keepdims=True)
        normalized = np.divide(
            matrix,
            row_sum,
            out=np.zeros_like(matrix, dtype=np.float64),
            where=row_sum != 0,
        )
        figure, axis = plt.subplots(figsize=(7, 6))
        image = axis.imshow(normalized, cmap="viridis", vmin=0.0, vmax=1.0)
        for row in range(len(matrix)):
            for column in range(len(matrix)):
                color = "white" if normalized[row, column] < 0.45 else "black"
                axis.text(
                    column,
                    row,
                    f"{matrix[row, column]}\n{normalized[row, column]*100:.1f}%",
                    ha="center",
                    va="center",
                    color=color,
                )
        axis.set_xticks(
            range(len(short_names)), short_names, rotation=20, ha="right"
        )
        axis.set_yticks(range(len(short_names)), short_names)
        axis.set_xlabel("Predicted class")
        axis.set_ylabel("True class")
        axis.set_title(f"TSCMDL fusion {split} confusion matrix")
        figure.colorbar(image, ax=axis, label="Row-normalized recall")
        figure.tight_layout()
        figure.savefig(output_dir / f"confusion_matrix_{split}.png", dpi=180)
        plt.close(figure)


def checkpoint_payload(
    model: TSCMDLFusionModel,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineAnnealingLR,
    scaler: torch.amp.GradScaler,
    history: list[dict[str, object]],
    epoch: int,
    best_epoch: int,
    best_score: tuple[float, float],
    epochs_without_improvement: int,
    class_names: tuple[str, ...],
    args: argparse.Namespace,
    feature_cache_sha256: str,
) -> dict[str, object]:
    return {
        "run_type": "b4b_frozen_feature_fusion",
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "history": history,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "epochs_without_improvement": epochs_without_improvement,
        "class_names": class_names,
        "args": json_args(args),
        "feature_cache_sha256": feature_cache_sha256,
    }


def main() -> None:
    args = parse_args()
    numeric_values = (
        args.epochs,
        args.batch_size,
        args.eval_batch_size,
        args.patience,
        args.checkpoint_every,
        args.progress_every,
    )
    if any(value <= 0 for value in numeric_values):
        raise ValueError("epochs, batch sizes, patience, and intervals must be positive")
    if args.workers < 0:
        raise ValueError("workers cannot be negative")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("dropout must be in [0, 1)")
    if any(
        not math.isfinite(value) or value < 0
        for value in (
            args.learning_rate,
            args.min_learning_rate,
            args.momentum,
            args.weight_decay,
        )
    ):
        raise ValueError("Optimizer values must be finite and non-negative")
    if args.learning_rate <= 0:
        raise ValueError("learning-rate must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for B4b fusion training")

    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.pause_request is not None:
        args.pause_request = args.pause_request.resolve()
    last_path = args.output_dir / "last.pt"
    if args.resume and not last_path.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {last_path}")

    set_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    cache_payload, datasets, feature_cache_sha256 = load_cache(args.feature_cache)
    class_names = tuple(str(value) for value in cache_payload["class_names"])
    num_classes = len(class_names)
    feature_dimensions = cache_payload.get("feature_dimensions", {})
    point_dim = int(feature_dimensions.get("point", datasets["train"].point_features.shape[1]))
    image_dim = int(feature_dimensions.get("image", datasets["train"].image_features.shape[1]))
    modal_dim = int(cache_payload.get("fusion_modal_dim", 1024))
    loaders = make_loaders(
        datasets,
        args.batch_size,
        args.eval_batch_size,
        args.workers,
        args.seed,
        balanced_sampler=args.balanced_sampler,
    )
    classifier_hidden_dims = tuple(int(value) for value in args.classifier_hidden_dims)
    model = make_model(
        num_classes,
        args.dropout,
        args.normalization,
        classifier_hidden_dims,
        args.modality,
        point_dim,
        image_dim,
        modal_dim,
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        fused=args.fused_sgd,
    )
    scheduler = CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.min_learning_rate
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    criterion = nn.CrossEntropyLoss()
    quality_reviews = load_quality_reviews(args.quality_review)
    log_path = args.output_dir / "train.log"
    run_state_path = args.output_dir / "run_state.json"
    live_progress_path = args.output_dir / "live_progress.json"
    config = {
        "args": json_args(args),
        "run_type": "b4b_frozen_feature_fusion",
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "class_names": list(class_names),
        "parameter_count": parameter_count,
        "split_sizes": {
            split: len(dataset) for split, dataset in datasets.items()
        },
        "model": "TSCMDL frozen-backbone cached-feature fusion",
        "feature_cache": {
            "path": str(args.feature_cache.resolve()),
            "sha256": feature_cache_sha256,
            "source_checkpoint_paths": cache_payload[
                "source_checkpoint_paths"
            ],
            "source_checkpoint_sha256": cache_payload[
                "source_checkpoint_sha256"
            ],
            "checkpoint_load": cache_payload["checkpoint_load"],
        },
        "fusion_architecture": {
            "point_raw": point_dim,
            "image_raw": image_dim,
            "modal_projection": modal_dim,
            "concatenated": 2 * modal_dim,
            "classifier": [*classifier_hidden_dims, num_classes],
            "dropout": args.dropout,
            "normalization": args.normalization,
            "modality": args.modality,
        },
        "execution_optimizations": {
            "frozen_feature_cache": True,
            "amp": args.amp,
            "fused_sgd": args.fused_sgd,
            "checkpoint_every": args.checkpoint_every,
            "balanced_sampler": args.balanced_sampler,
        },
        "quality_review_records": len(quality_reviews),
        "paper_protocol": {
            "parser": "PyMuPDF in pdf-reading conda environment",
            "intermediate_text": str(
                SANDBOX_ROOT.parents[1]
                / "paper reading"
                / "output"
                / "TSCMDL_extracted.txt"
            ),
            "visual_checks": "PDF pages 5 and 6 rendered with PyMuPDF",
            "adaptation": (
                "WHU-STree uses 8192 points; both validation-selected pretrained "
                "backbones are frozen for this B4b experiment."
            ),
        },
    }
    atomic_json(args.output_dir / "run_config.json", config)
    atomic_json(
        run_state_path,
        {
            "status": "running",
            "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "pid": os.getpid(),
            "resume": bool(args.resume),
        },
    )
    emit(json.dumps(config, ensure_ascii=False), log_path)

    history: list[dict[str, object]] = []
    start_epoch = 0
    best_epoch = 0
    best_score = (-1.0, -1.0)
    epochs_without_improvement = 0
    resumed = False
    if args.resume:
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        if checkpoint["feature_cache_sha256"] != feature_cache_sha256:
            raise ValueError("Resume checkpoint uses another feature cache")
        model.load_state_dict(checkpoint["model_state"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        history = checkpoint["history"]
        start_epoch = int(checkpoint["epoch"])
        best_epoch = int(checkpoint["best_epoch"])
        best_score = tuple(float(value) for value in checkpoint["best_score"])
        epochs_without_improvement = int(checkpoint["epochs_without_improvement"])
        resumed = True
        emit(f"Resuming from epoch {start_epoch}", log_path)

    training_started = time.perf_counter()
    for epoch_index in range(start_epoch, args.epochs):
        epoch_number = epoch_index + 1
        learning_rate = optimizer.param_groups[0]["lr"]
        emit(f"Epoch {epoch_number}/{args.epochs} lr={learning_rate:.7f}", log_path)
        epoch_started = time.perf_counter()
        train_metrics, _ = run_epoch(
            model,
            loaders["train"],
            criterion,
            device,
            num_classes,
            optimizer=optimizer,
            scaler=scaler,
            use_amp=args.amp,
            progress_label="train",
            progress_every=args.progress_every,
        )
        val_metrics, _ = run_epoch(
            model,
            loaders["val"],
            criterion,
            device,
            num_classes,
            use_amp=args.amp,
            progress_label="val  ",
            progress_every=args.progress_every,
        )
        epoch_seconds = time.perf_counter() - epoch_started
        row = {
            "epoch": epoch_number,
            "learning_rate": learning_rate,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "epoch_seconds": epoch_seconds,
            "peak_allocated_mib": max(
                float(train_metrics["peak_allocated_mib"]),
                float(val_metrics["peak_allocated_mib"]),
            ),
        }
        history.append(row)
        current_score = (
            float(val_metrics["macro_f1"]),
            float(val_metrics["accuracy"]),
        )
        if current_score > best_score:
            best_score = current_score
            best_epoch = epoch_number
            epochs_without_improvement = 0
            atomic_torch_save(
                args.output_dir / "best.pt",
                {
                    "run_type": "b4b_frozen_feature_fusion",
                    "epoch": epoch_number,
                    "model_state": model.state_dict(),
                    "val_metrics": val_metrics,
                    "class_names": class_names,
                    "args": json_args(args),
                    "feature_cache_sha256": feature_cache_sha256,
                },
            )
        else:
            epochs_without_improvement += 1
        scheduler.step()
        write_history(args.output_dir, history)
        should_stop = epochs_without_improvement >= args.patience
        pause_requested = (
            args.pause_request is not None and args.pause_request.is_file()
        )
        should_checkpoint = (
            epoch_number % args.checkpoint_every == 0
            or epoch_number == args.epochs
            or should_stop
            or pause_requested
        )
        if should_checkpoint:
            atomic_torch_save(
                last_path,
                checkpoint_payload(
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    history,
                    epoch_number,
                    best_epoch,
                    best_score,
                    epochs_without_improvement,
                    class_names,
                    args,
                    feature_cache_sha256,
                ),
            )
        emit(
            f"epoch={epoch_number} train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']*100:.2f}% "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']*100:.2f}% "
            f"val_macro_f1={val_metrics['macro_f1']*100:.2f}% "
            f"best_epoch={best_epoch} time={epoch_seconds:.2f}s",
            log_path,
        )
        atomic_json(
            live_progress_path,
            {
                "status": "running",
                "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "epoch": epoch_number,
                "total_epochs": args.epochs,
                "best_epoch": best_epoch,
                "val_macro_f1": val_metrics["macro_f1"],
                "val_accuracy": val_metrics["accuracy"],
                "epochs_without_improvement": epochs_without_improvement,
            },
        )
        if pause_requested:
            atomic_json(
                run_state_path,
                {
                    "status": "paused",
                    "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                    "pid": os.getpid(),
                    "epoch": epoch_number,
                },
            )
            emit(f"Pause requested after epoch {epoch_number}; checkpoint saved", log_path)
            return
        if should_stop:
            emit(
                f"Early stopping after {args.patience} epochs without improvement",
                log_path,
            )
            break

    final_epoch = len(history)
    if not last_path.is_file() or int(
        torch.load(last_path, map_location="cpu", weights_only=False)["epoch"]
    ) != final_epoch:
        atomic_torch_save(
            last_path,
            checkpoint_payload(
                model,
                optimizer,
                scheduler,
                scaler,
                history,
                final_epoch,
                best_epoch,
                best_score,
                epochs_without_improvement,
                class_names,
                args,
                feature_cache_sha256,
            ),
        )

    best_checkpoint = torch.load(
        args.output_dir / "best.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(best_checkpoint["model_state"], strict=True)
    final_metrics: dict[str, dict[str, object]] = {}
    all_predictions = []
    for split in ("val", "test"):
        metrics, predictions = run_epoch(
            model,
            loaders[split],
            criterion,
            device,
            num_classes,
            use_amp=args.amp,
            progress_label=f"final {split}",
            progress_every=args.progress_every,
            collect_predictions=True,
        )
        for item, class_name in zip(metrics["per_class"], class_names):
            item["scientific_name"] = class_name
        final_metrics[split] = metrics
        all_predictions.extend(
            attach_sample_metadata(
                predictions, datasets[split], split, quality_reviews
            )
        )

    training_session_seconds = time.perf_counter() - training_started
    training_epoch_seconds = sum(float(item["epoch_seconds"]) for item in history)
    result = {
        "status": "complete",
        "run_type": "b4b_frozen_feature_fusion",
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "training_elapsed_seconds": training_epoch_seconds,
        "training_session_elapsed_seconds": training_session_seconds,
        "resumed": resumed,
        "best_validation_score": {
            "macro_f1": best_score[0],
            "accuracy": best_score[1],
        },
        "metrics": final_metrics,
        "class_names": list(class_names),
        "feature_cache_sha256": feature_cache_sha256,
    }
    atomic_json(args.output_dir / "final_metrics.json", result)
    save_predictions(args.output_dir / "predictions.csv", all_predictions, num_classes)
    plot_results(args.output_dir, history, final_metrics, class_names)
    atomic_json(
        run_state_path,
        {
            "status": "complete",
            "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "pid": os.getpid(),
            "best_epoch": best_epoch,
            "epochs_completed": len(history),
        },
    )
    emit(json.dumps(result, ensure_ascii=False, indent=2), log_path)


if __name__ == "__main__":
    main()
