"""Train and evaluate a three-class 8192-point PointMLP baseline."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader


SANDBOX_ROOT = Path(__file__).resolve().parents[1]
POINTMLP_ROOT = SANDBOX_ROOT / "vendor" / "pointMLP-pytorch"
sys.path.insert(0, str(SANDBOX_ROOT / "src"))
sys.path.insert(0, str(POINTMLP_ROOT / "pointnet2_ops_lib"))
sys.path.insert(0, str(POINTMLP_ROOT / "classification_ModelNet40"))

from models.pointmlp import Model  # noqa: E402
from tscmdl_whu import PointManifestDataset, classification_metrics  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.0125)
    parser.add_argument("--min-learning-rate", type=float, default=0.000625)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--smoke-steps", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
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


def make_model(num_classes: int) -> Model:
    return Model(
        points=8192,
        class_num=num_classes,
        embed_dim=64,
        groups=1,
        res_expansion=1.0,
        activation="relu",
        bias=False,
        use_xyz=False,
        normalize="anchor",
        dim_expansion=[2, 2, 2, 2],
        pre_blocks=[2, 2, 2, 2],
        pos_blocks=[2, 2, 2, 2],
        k_neighbors=[24, 24, 24, 24],
        reducers=[2, 2, 2, 2],
    )


def augment_points(points: torch.Tensor) -> torch.Tensor:
    """Apply vertical-axis rotation, isotropic scale, jitter, and point shuffle."""
    points = points.clone()
    batch_size = points.shape[0]
    angles = torch.rand(batch_size, device=points.device) * (2.0 * torch.pi)
    cosine = torch.cos(angles)
    sine = torch.sin(angles)
    x = points[:, :, 0].clone()
    y = points[:, :, 1].clone()
    points[:, :, 0] = cosine[:, None] * x - sine[:, None] * y
    points[:, :, 1] = sine[:, None] * x + cosine[:, None] * y
    scale = 0.9 + 0.2 * torch.rand(batch_size, 1, 1, device=points.device)
    points = points * scale
    jitter = torch.randn_like(points).mul_(0.005).clamp_(-0.02, 0.02)
    points = points + jitter
    permutation = torch.randperm(points.shape[1], device=points.device)
    return points[:, permutation, :]


def make_loaders(
    dataset_root: Path,
    batch_size: int,
    eval_batch_size: int,
    workers: int,
    seed: int,
) -> tuple[dict[str, PointManifestDataset], dict[str, DataLoader]]:
    datasets = {
        split: PointManifestDataset(dataset_root, split)
        for split in ("train", "val", "test")
    }
    generator = torch.Generator().manual_seed(seed)
    drop_last = len(datasets["train"]) % batch_size == 1
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            pin_memory=True,
            drop_last=drop_last,
            generator=generator,
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
    return datasets, loaders


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
    progress_every: int = 1,
    max_batches: int = 0,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    sample_count = 0
    true_labels: list[int] = []
    predicted_labels: list[int] = []
    predictions: list[dict[str, object]] = []
    skipped_optimizer_steps = 0
    successful_optimizer_steps = 0
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)

    for batch_index, (points, labels, indices) in enumerate(loader, start=1):
        if max_batches and batch_index > max_batches:
            break
        points = points.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if training:
            points = augment_points(points)
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(points.permute(0, 2, 1).contiguous())
                loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss")
            if training:
                if scaler is None:
                    raise ValueError("Training requires a gradient scaler")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if not torch.isfinite(gradient_norm) and not scaler.is_enabled():
                    raise FloatingPointError("Non-finite gradient norm")
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.is_enabled() and scaler.get_scale() < previous_scale:
                    skipped_optimizer_steps += 1
                else:
                    successful_optimizer_steps += 1

        probabilities = torch.softmax(logits.detach().float(), dim=1)
        predicted = probabilities.argmax(dim=1)
        batch_size = labels.shape[0]
        total_loss += float(loss.detach()) * batch_size
        sample_count += batch_size
        true_labels.extend(labels.detach().cpu().tolist())
        predicted_labels.extend(predicted.cpu().tolist())
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
            print(
                f"\r{progress_label} [{batch_index:3d}/{shown_total:3d}] "
                f"loss={total_loss/sample_count:.4f} acc={accuracy*100:6.2f}%",
                end="",
                flush=True,
            )
    print(flush=True)
    if sample_count == 0:
        raise ValueError("No samples were processed")
    if training and successful_optimizer_steps == 0:
        raise FloatingPointError(
            f"All {skipped_optimizer_steps} optimizer steps were skipped by AMP"
        )
    torch.cuda.synchronize(device)
    metrics = classification_metrics(true_labels, predicted_labels, num_classes)
    metrics.update(
        {
            "loss": total_loss / sample_count,
            "sample_count": sample_count,
            "elapsed_seconds": time.perf_counter() - started,
            "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / (1024**2),
            "successful_optimizer_steps": successful_optimizer_steps,
            "skipped_optimizer_steps": skipped_optimizer_steps,
        }
    )
    return metrics, predictions


def attach_sample_keys(
    predictions: list[dict[str, object]], dataset: PointManifestDataset, split: str
) -> list[dict[str, object]]:
    output = []
    for prediction in predictions:
        index = int(prediction["dataset_index"])
        output.append(
            {
                **prediction,
                "sample_key": dataset.sample_keys[index],
                "split": split,
            }
        )
    return output


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
        axis.set_title(f"PointMLP {split} confusion matrix")
        figure.colorbar(image, ax=axis, label="Row-normalized recall")
        figure.tight_layout()
        figure.savefig(output_dir / f"confusion_matrix_{split}.png", dpi=180)
        plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size < 2 or args.eval_batch_size <= 0:
        raise ValueError("epochs/eval-batch-size must be positive and batch-size must be at least two")
    if args.workers < 0 or args.progress_every <= 0 or args.smoke_steps < 0:
        raise ValueError("workers/smoke-steps cannot be negative and progress-every must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for B2 PointMLP training")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    last_path = args.output_dir / "last.pt"
    if last_path.exists() and not args.resume and not args.smoke_steps:
        raise FileExistsError(f"Checkpoint already exists; use --resume: {last_path}")
    log_path = args.output_dir / "train.log"
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
    )
    class_names = datasets["train"].class_names
    num_classes = len(class_names)
    model = make_model(num_classes).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.min_learning_rate
    )
    scaler = torch.amp.GradScaler("cuda", init_scale=8.0, enabled=args.amp)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    config = {
        "args": json_args(args),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "class_names": list(class_names),
        "parameter_count": parameter_count,
        "split_sizes": {split: len(dataset) for split, dataset in datasets.items()},
        "augmentation": "random Z rotation, isotropic 0.9-1.1 scale, clipped Gaussian jitter, point shuffle",
        "model": "full PointMLP with 8192 input points and 3 output classes",
    }
    atomic_json(args.output_dir / "run_config.json", config)
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
        )
        result = {
            "status": "passed",
            "train": train_metrics,
            "val": val_metrics,
            "elapsed_seconds": time.perf_counter() - smoke_started,
        }
        atomic_json(args.output_dir / "smoke_result.json", result)
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
        emit(
            f"Epoch {epoch_number}/{args.epochs} lr={learning_rate:.7f}",
            log_path,
        )
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
        improved = current_score > best_score
        if improved:
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
        )
        for item, class_name in zip(metrics["per_class"], class_names):
            item["scientific_name"] = class_name
        final_metrics[split] = metrics
        all_predictions.extend(attach_sample_keys(predictions, datasets[split], split))

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
    }
    atomic_json(args.output_dir / "final_metrics.json", result)
    save_predictions(args.output_dir / "predictions.csv", all_predictions, num_classes)
    plot_results(args.output_dir, history, final_metrics, class_names)
    emit(json.dumps(result, ensure_ascii=False, indent=2), log_path)


if __name__ == "__main__":
    main()
