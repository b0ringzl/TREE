"""Train and evaluate the B3 three-class ResNet50 image baseline."""

from __future__ import annotations

import argparse
import csv
import json
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
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.transforms import v2


SANDBOX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SANDBOX_ROOT / "src"))

from tscmdl_whu import ImageManifestDataset, classification_metrics  # noqa: E402


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--quality-review",
        type=Path,
        default=SANDBOX_ROOT / "reports" / "B1_visual_review.json",
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--min-learning-rate", type=float, default=0.0)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--patience", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=5)
    parser.add_argument("--smoke-steps", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pretrained-weights", type=Path)
    parser.add_argument("--pause-request", type=Path)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument(
        "--balanced-sampler",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--channels-last",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--fused-sgd",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--imagenet-pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def json_args(args: argparse.Namespace) -> dict[str, object]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{path}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def atomic_torch_save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{path}.tmp")
    torch.save(value, temporary)
    temporary.replace(path)


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


def make_transforms() -> tuple[v2.Compose, v2.Compose]:
    normalize = v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    train_transform = v2.Compose(
        [
            v2.RandomCrop((224, 224)),
            v2.RandomHorizontalFlip(),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            normalize,
        ]
    )
    eval_transform = v2.Compose(
        [
            v2.CenterCrop((224, 224)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            normalize,
        ]
    )
    return train_transform, eval_transform


def make_loaders(
    dataset_root: Path,
    batch_size: int,
    eval_batch_size: int,
    workers: int,
    seed: int,
    *,
    splits: tuple[str, ...] = ("train", "val"),
    balanced_sampler: bool = False,
) -> tuple[dict[str, ImageManifestDataset], dict[str, DataLoader]]:
    train_transform, eval_transform = make_transforms()
    datasets = {
        split: ImageManifestDataset(
            dataset_root,
            split,
            train_transform if split == "train" else eval_transform,
        )
        for split in splits
    }
    generator = torch.Generator().manual_seed(seed)
    loaders: dict[str, DataLoader] = {}
    for split, dataset in datasets.items():
        sampler = None
        shuffle = split == "train"
        if split == "train" and balanced_sampler:
            counts = np.bincount(dataset.labels, minlength=len(dataset.class_names))
            if np.any(counts == 0):
                raise ValueError(f"Balanced sampler found an empty class: {counts}")
            sample_weights = torch.as_tensor(
                1.0 / counts[dataset.labels], dtype=torch.double
            )
            sampler = WeightedRandomSampler(
                sample_weights,
                num_samples=len(dataset),
                replacement=True,
                generator=generator,
            )
            shuffle = False
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size if split == "train" else eval_batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=workers,
            pin_memory=True,
            generator=generator if sampler is None else None,
            persistent_workers=workers > 0,
        )
    return datasets, loaders


def make_model(
    num_classes: int,
    imagenet_pretrained: bool,
    pretrained_weights: Path | None = None,
) -> nn.Module:
    if pretrained_weights is not None:
        if not pretrained_weights.is_file():
            raise FileNotFoundError(pretrained_weights)
        model = resnet50(weights=None)
        state = torch.load(pretrained_weights, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
    else:
        weights = ResNet50_Weights.IMAGENET1K_V2 if imagenet_pretrained else None
        model = resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    use_amp: bool = True,
    progress_label: str,
    progress_every: int = 5,
    max_batches: int = 0,
    collect_predictions: bool = False,
    channels_last: bool = True,
    live_progress_path: Path | None = None,
    epoch_number: int = 0,
    total_epochs: int = 0,
    learning_rate: float = 0.0,
    phase: str = "train",
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

    for batch_index, (images, labels, indices) in enumerate(loader, start=1):
        if max_batches and batch_index > max_batches:
            break
        images = images.to(
            device,
            non_blocking=True,
            memory_format=(
                torch.channels_last if channels_last else torch.contiguous_format
            ),
        )
        labels = labels.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss")
            if training:
                if scaler is None:
                    raise ValueError("Training requires a gradient scaler")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                if not torch.isfinite(gradient_norm) and not scaler.is_enabled():
                    raise FloatingPointError("Non-finite gradient norm")
                scaler.step(optimizer)
                scaler.update()

        probabilities = torch.softmax(logits.detach().float(), dim=1)
        predicted = probabilities.argmax(dim=1)
        current_batch_size = labels.shape[0]
        total_loss += float(loss.detach()) * current_batch_size
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

        if batch_index % progress_every == 0 or batch_index == len(loader):
            accuracy = np.mean(np.asarray(true_labels) == np.asarray(predicted_labels))
            shown_total = min(len(loader), max_batches) if max_batches else len(loader)
            elapsed = time.perf_counter() - started
            rate = batch_index / elapsed if elapsed > 0 else 0.0
            phase_eta = (shown_total - batch_index) / rate if rate > 0 else None
            print(
                f"\r{progress_label} [{batch_index:3d}/{shown_total:3d}] "
                f"loss={total_loss/sample_count:.4f} acc={accuracy*100:6.2f}%",
                end="",
                flush=True,
            )
            if live_progress_path is not None:
                live_metrics = classification_metrics(
                    true_labels, predicted_labels, num_classes
                )
                atomic_json(
                    live_progress_path,
                    {
                        "status": "running",
                        "updated_at": datetime.now().astimezone().isoformat(
                            timespec="seconds"
                        ),
                        "phase": phase,
                        "epoch": epoch_number,
                        "total_epochs": total_epochs,
                        "batch": batch_index,
                        "total_batches": shown_total,
                        "phase_eta_seconds": phase_eta,
                        "learning_rate": learning_rate,
                        "loss": total_loss / sample_count,
                        "accuracy": live_metrics["accuracy"],
                        "macro_f1": live_metrics["macro_f1"],
                        "peak_allocated_mib": torch.cuda.max_memory_allocated(device)
                        / (1024**2),
                    },
                )
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
            "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / (1024**2),
        }
    )
    return metrics, predictions


def attach_sample_metadata(
    predictions: list[dict[str, object]],
    dataset: ImageManifestDataset,
    split: str,
    quality_reviews: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    output = []
    for prediction in predictions:
        index = int(prediction["dataset_index"])
        sample_key = dataset.sample_keys[index]
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
        str(item["sample_key"]): item
        for item in payload.get("reviews", [])
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
    axes[0].plot(epochs, [item["train_loss"] for item in history], label="train", color="#1565C0")
    axes[0].plot(epochs, [item["val_loss"] for item in history], label="val", color="#D84315")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Cross entropy")
    axes[1].plot(epochs, [100 * item["train_accuracy"] for item in history], label="train", color="#00897B")
    axes[1].plot(epochs, [100 * item["val_accuracy"] for item in history], label="val", color="#6A1B9A")
    axes[1].set(title="Accuracy", xlabel="Epoch", ylabel="Percent")
    axes[2].plot(epochs, [100 * item["train_macro_f1"] for item in history], label="train", color="#2E7D32")
    axes[2].plot(epochs, [100 * item["val_macro_f1"] for item in history], label="val", color="#EF6C00")
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
        axis.set_xticks(range(len(short_names)), short_names, rotation=20, ha="right")
        axis.set_yticks(range(len(short_names)), short_names)
        axis.set_xlabel("Predicted class")
        axis.set_ylabel("True class")
        axis.set_title(f"ResNet50 {split} confusion matrix")
        figure.colorbar(image, ax=axis, label="Row-normalized recall")
        figure.tight_layout()
        figure.savefig(output_dir / f"confusion_matrix_{split}.png", dpi=180)
        plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.eval_batch_size <= 0:
        raise ValueError("epochs and batch sizes must be positive")
    if args.workers < 0 or args.progress_every <= 0 or args.smoke_steps < 0:
        raise ValueError("workers/smoke-steps cannot be negative and progress-every must be positive")
    if args.patience <= 0:
        raise ValueError("patience must be positive")
    if not 0.0 <= args.label_smoothing < 1.0:
        raise ValueError("label-smoothing must be in [0, 1)")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for B3 ResNet50 training")

    args.dataset_root = args.dataset_root.resolve()
    args.output_dir = args.output_dir.resolve()
    args.quality_review = args.quality_review.resolve()
    if args.pretrained_weights is not None:
        args.pretrained_weights = args.pretrained_weights.resolve()
    if args.pause_request is not None:
        args.pause_request = args.pause_request.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    last_path = args.output_dir / "last.pt"
    if last_path.exists() and not args.resume and not args.smoke_steps:
        raise FileExistsError(f"Checkpoint already exists; use --resume: {last_path}")
    log_path = args.output_dir / "train.log"
    live_progress_path = args.output_dir / "live_progress.json"
    run_state_path = args.output_dir / "run_state.json"
    set_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda")

    datasets, loaders = make_loaders(
        args.dataset_root,
        args.batch_size,
        args.eval_batch_size,
        args.workers,
        args.seed,
        splits=("train", "val"),
        balanced_sampler=args.balanced_sampler,
    )
    class_names = datasets["train"].class_names
    num_classes = len(class_names)
    model = make_model(
        num_classes,
        args.imagenet_pretrained,
        args.pretrained_weights,
    ).to(device)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
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
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    quality_reviews = load_quality_reviews(args.quality_review)
    manifest = json.loads(
        (args.dataset_root / "manifest.json").read_text(encoding="utf-8")
    )
    configured_split_sizes = manifest.get("summary", {}).get("split_sizes", {})
    if not configured_split_sizes:
        configured_split_sizes = {
            split: sum(
                1 for record in manifest["records"] if record["split"] == split
            )
            for split in ("train", "val", "test")
        }
    config = {
        "args": json_args(args),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "torchvision": __import__("torchvision").__version__,
        "class_names": list(class_names),
        "parameter_count": parameter_count,
        "split_sizes": configured_split_sizes,
        "augmentation": "random 224x224 crop, random horizontal flip, ImageNet normalization",
        "evaluation_transform": "center 224x224 crop, ImageNet normalization",
        "model": f"torchvision ResNet50 with {num_classes}-class linear head",
        "pretrained_weights": (
            str(args.pretrained_weights)
            if args.pretrained_weights is not None
            else (
                "ResNet50_Weights.IMAGENET1K_V2"
                if args.imagenet_pretrained
                else "none"
            )
        ),
        "execution_optimizations": {
            "amp": args.amp,
            "channels_last": args.channels_last,
            "fused_sgd": args.fused_sgd,
        },
        "quality_review_records": len(quality_reviews),
        "test_policy": "test split loaded only after validation-selected training completes",
        "test_loaded_during_training": False,
        "paper_protocol": {
            "parser": "PyMuPDF in pdf-reading conda environment",
            "intermediate_text": str(
                SANDBOX_ROOT.parents[1] / "paper reading" / "output" / "TSCMDL_extracted.txt"
            ),
            "table_visual_check": "PDF page 6 rendered with PyMuPDF",
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
            "smoke_steps": args.smoke_steps,
        },
    )
    emit(json.dumps(config, ensure_ascii=False), log_path)

    if args.smoke_steps:
        smoke_started = time.perf_counter()
        train_metrics, _ = run_epoch(
            model,
            loaders["train"],
            criterion,
            device,
            num_classes,
            optimizer=optimizer,
            scaler=scaler,
            use_amp=args.amp,
            progress_label="smoke train",
            progress_every=1,
            max_batches=args.smoke_steps,
            channels_last=args.channels_last,
            live_progress_path=live_progress_path,
            phase="smoke_train",
        )
        val_metrics, _ = run_epoch(
            model,
            loaders["val"],
            criterion,
            device,
            num_classes,
            use_amp=args.amp,
            progress_label="smoke val  ",
            progress_every=1,
            max_batches=1,
            channels_last=args.channels_last,
            live_progress_path=live_progress_path,
            phase="smoke_val",
        )
        result = {
            "status": "passed",
            "train": train_metrics,
            "val": val_metrics,
            "elapsed_seconds": time.perf_counter() - smoke_started,
        }
        atomic_json(args.output_dir / "smoke_result.json", result)
        atomic_json(
            run_state_path,
            {
                "status": "smoke_complete",
                "updated_at": datetime.now().astimezone().isoformat(
                    timespec="seconds"
                ),
                "pid": os.getpid(),
            },
        )
        emit(json.dumps(result, ensure_ascii=False, indent=2), log_path)
        return

    history: list[dict[str, object]] = []
    start_epoch = 0
    best_epoch = 0
    best_score = (-1.0, -1.0)
    epochs_without_improvement = 0
    training_started = time.perf_counter()
    if args.resume:
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        history = checkpoint["history"]
        start_epoch = int(checkpoint["epoch"])
        best_epoch = int(checkpoint["best_epoch"])
        best_score = tuple(float(value) for value in checkpoint["best_score"])
        epochs_without_improvement = int(checkpoint["epochs_without_improvement"])
        emit(f"Resuming from epoch {start_epoch}", log_path)

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
            channels_last=args.channels_last,
            live_progress_path=live_progress_path,
            epoch_number=epoch_number,
            total_epochs=args.epochs,
            learning_rate=learning_rate,
            phase="train",
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
            channels_last=args.channels_last,
            live_progress_path=live_progress_path,
            epoch_number=epoch_number,
            total_epochs=args.epochs,
            learning_rate=learning_rate,
            phase="val",
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
                    "epoch": epoch_number,
                    "model_state": model.state_dict(),
                    "val_metrics": val_metrics,
                    "class_names": class_names,
                    "args": json_args(args),
                },
            )
        else:
            epochs_without_improvement += 1
        scheduler.step()
        atomic_torch_save(
            last_path,
            {
                "epoch": epoch_number,
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
            },
        )
        write_history(args.output_dir, history)
        emit(
            f"epoch={epoch_number} train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']*100:.2f}% "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']*100:.2f}% "
            f"val_macro_f1={val_metrics['macro_f1']*100:.2f}% "
            f"best_epoch={best_epoch} time={epoch_seconds:.1f}s",
            log_path,
        )
        if args.pause_request is not None and args.pause_request.is_file():
            try:
                pause = json.loads(args.pause_request.read_text(encoding="utf-8-sig"))
            except (OSError, json.JSONDecodeError):
                pause = {}
            if pause.get("status") == "requested":
                pause.update(
                    {
                        "status": "complete",
                        "completed_epoch": epoch_number,
                        "completed_at": datetime.now().astimezone().isoformat(
                            timespec="seconds"
                        ),
                    }
                )
                atomic_json(args.pause_request, pause)
                atomic_json(
                    run_state_path,
                    {
                        "status": "paused",
                        "updated_at": datetime.now().astimezone().isoformat(
                            timespec="seconds"
                        ),
                        "pid": os.getpid(),
                        "completed_epoch": epoch_number,
                    },
                )
                emit(f"Safe pause completed after epoch {epoch_number}", log_path)
                return
        if epochs_without_improvement >= args.patience:
            emit(
                f"Early stopping after {args.patience} epochs without improvement",
                log_path,
            )
            break

    best_checkpoint = torch.load(
        args.output_dir / "best.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(best_checkpoint["model_state"])
    test_datasets, test_loaders = make_loaders(
        args.dataset_root,
        args.batch_size,
        args.eval_batch_size,
        args.workers,
        args.seed,
        splits=("test",),
    )
    datasets.update(test_datasets)
    loaders.update(test_loaders)
    emit(
        "Best validation-selected checkpoint locked; official test split loaded",
        log_path,
    )
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
            channels_last=args.channels_last,
            live_progress_path=live_progress_path,
            epoch_number=len(history),
            total_epochs=len(history),
            learning_rate=float(optimizer.param_groups[0]["lr"]),
            phase=f"final_{split}",
        )
        for item, class_name in zip(metrics["per_class"], class_names):
            item["scientific_name"] = class_name
        final_metrics[split] = metrics
        all_predictions.extend(
            attach_sample_metadata(predictions, datasets[split], split, quality_reviews)
        )

    training_session_seconds = time.perf_counter() - training_started
    training_epoch_seconds = sum(float(item["epoch_seconds"]) for item in history)
    result = {
        "status": "complete",
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "training_elapsed_seconds": training_epoch_seconds,
        "training_session_elapsed_seconds": training_session_seconds,
        "resumed": bool(args.resume),
        "best_validation_score": {
            "macro_f1": best_score[0],
            "accuracy": best_score[1],
        },
        "metrics": final_metrics,
        "class_names": list(class_names),
        "parameter_count": parameter_count,
        "test_evaluated_after_training": True,
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
