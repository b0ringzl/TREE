#!/usr/bin/env python3
"""Fine-tune a copied four-class ResNet50 backbone on ten tree species."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.models import resnet50
from torchvision.transforms import v2


PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--warmup-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--backbone-lr", type=float, default=3e-4)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--progress-every", type=int, default=5)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


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
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_rows(manifest: Path, split: str) -> list[dict[str, str]]:
    with manifest.open("r", encoding="utf-8", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["split"] == split]
    if not rows:
        raise ValueError(f"No rows for split={split}")
    return rows


class CropDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], transform: Any) -> None:
        self.rows = rows
        self.transform = transform
        self.labels = np.asarray([int(row["class_index"]) for row in rows])

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        path = PROJECT_ROOT / row["image_path"]
        with Image.open(path) as source:
            source = source.convert("RGB")
            width, height = source.size
            box = (
                max(0, math.floor(float(row["x0"]) * width)),
                max(0, math.floor(float(row["y0"]) * height)),
                min(width, math.ceil(float(row["x1"]) * width)),
                min(height, math.ceil(float(row["y1"]) * height)),
            )
            if box[2] <= box[0] or box[3] <= box[1]:
                raise ValueError(f"Invalid crop for {row['sample_id']}: {box}")
            image = source.crop(box)
            tensor = self.transform(image)
        return tensor, int(row["class_index"]), index


def transforms(image_size: int) -> tuple[v2.Compose, v2.Compose]:
    normalize = v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    train_transform = v2.Compose(
        [
            v2.RandomResizedCrop(
                (image_size, image_size), scale=(0.72, 1.0), antialias=True
            ),
            v2.RandomHorizontalFlip(),
            v2.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            normalize,
        ]
    )
    eval_transform = v2.Compose(
        [
            v2.Resize((image_size, image_size), antialias=True),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            normalize,
        ]
    )
    return train_transform, eval_transform


def class_names(rows: list[dict[str, str]]) -> tuple[str, ...]:
    mapping: dict[int, str] = {}
    for row in rows:
        index = int(row["class_index"])
        name = row["species"]
        if index in mapping and mapping[index] != name:
            raise ValueError(f"Class index {index} maps to multiple names")
        mapping[index] = name
    expected = list(range(len(mapping)))
    if sorted(mapping) != expected:
        raise ValueError(f"Non-contiguous class indices: {sorted(mapping)}")
    return tuple(mapping[index] for index in expected)


def build_model(
    checkpoint_path: Path, number_of_classes: int
) -> tuple[nn.Module, dict[str, object]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state" not in checkpoint:
        raise ValueError("Source checkpoint has no model_state")
    state = dict(checkpoint["model_state"])
    source_fc_weight = state.pop("fc.weight", None)
    source_fc_bias = state.pop("fc.bias", None)
    if source_fc_weight is None or source_fc_bias is None:
        raise ValueError("Source checkpoint has no ResNet fc parameters")
    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, number_of_classes)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if set(missing) != {"fc.weight", "fc.bias"} or unexpected:
        raise ValueError(f"Unexpected transfer mismatch: missing={missing}, unexpected={unexpected}")
    nn.init.normal_(model.fc.weight, mean=0.0, std=0.01)
    nn.init.zeros_(model.fc.bias)
    transfer = {
        "source_checkpoint": str(checkpoint_path),
        "source_sha256": sha256_file(checkpoint_path),
        "source_epoch": checkpoint.get("epoch"),
        "source_class_names": list(checkpoint.get("class_names", [])),
        "source_fc_shape": list(source_fc_weight.shape),
        "target_fc_shape": list(model.fc.weight.shape),
        "loaded_backbone_keys": len(state),
        "discarded_parameters": ["fc.weight", "fc.bias"],
    }
    return model, transfer


def classification_metrics(
    truth: list[int], predicted: list[int], number_of_classes: int
) -> dict[str, object]:
    confusion = np.zeros((number_of_classes, number_of_classes), dtype=np.int64)
    for actual, guess in zip(truth, predicted):
        confusion[actual, guess] += 1
    per_class: list[dict[str, object]] = []
    for index in range(number_of_classes):
        true_positive = int(confusion[index, index])
        false_positive = int(confusion[:, index].sum() - true_positive)
        false_negative = int(confusion[index, :].sum() - true_positive)
        support = int(confusion[index, :].sum())
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append(
            {
                "class_index": index,
                "support": support,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    count = int(confusion.sum())
    return {
        "accuracy": float(np.trace(confusion) / count) if count else 0.0,
        "balanced_accuracy": float(np.mean([row["recall"] for row in per_class])),
        "macro_precision": float(np.mean([row["precision"] for row in per_class])),
        "macro_recall": float(np.mean([row["recall"] for row in per_class])),
        "macro_f1": float(np.mean([row["f1"] for row in per_class])),
        "confusion_matrix": confusion.tolist(),
        "per_class": per_class,
        "sample_count": count,
    }


def evaluate_group_level(
    rows: list[dict[str, str]], probabilities: np.ndarray, number_of_classes: int
) -> tuple[dict[str, object], list[dict[str, object]]]:
    grouped: defaultdict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[row["group_id"]].append(index)
    truth: list[int] = []
    predicted: list[int] = []
    group_predictions: list[dict[str, object]] = []
    for group_id in sorted(grouped):
        indices = grouped[group_id]
        labels = {int(rows[index]["class_index"]) for index in indices}
        if len(labels) != 1:
            raise ValueError(f"Group {group_id} has multiple labels")
        label = labels.pop()
        average = probabilities[indices].mean(axis=0)
        guess = int(average.argmax())
        truth.append(label)
        predicted.append(guess)
        group_predictions.append(
            {
                "group_id": group_id,
                "true_class": label,
                "predicted_class": guess,
                "crop_count": len(indices),
                "probabilities": average.tolist(),
            }
        )
    return classification_metrics(truth, predicted, number_of_classes), group_predictions


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    number_of_classes: int,
    *,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    rows: list[dict[str, str]],
    epoch: int,
    epochs: int,
    phase: str,
    live_path: Path,
    progress_every: int,
    max_batches: int = 0,
) -> tuple[dict[str, object], np.ndarray, list[dict[str, object]]]:
    training = optimizer is not None
    model.train(training)
    truth: list[int] = []
    predicted: list[int] = []
    probability_rows: list[np.ndarray] = []
    ordered_indices: list[int] = []
    loss_sum = 0.0
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    shown_batches = min(len(loader), max_batches) if max_batches else len(loader)
    for batch_index, (images, labels, indices) in enumerate(loader, start=1):
        if max_batches and batch_index > max_batches:
            break
        images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
        labels = labels.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logits = model(images)
                loss = criterion(logits, labels)
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
        probs = torch.softmax(logits.detach().float(), dim=1).cpu().numpy()
        guesses = probs.argmax(axis=1)
        label_list = labels.detach().cpu().tolist()
        truth.extend(label_list)
        predicted.extend(guesses.tolist())
        probability_rows.extend(probs)
        ordered_indices.extend(indices.tolist())
        loss_sum += float(loss.detach()) * len(label_list)
        if batch_index % progress_every == 0 or batch_index == shown_batches:
            elapsed = time.perf_counter() - started
            atomic_json(
                live_path,
                {
                    "status": "running",
                    "updated_at": timestamp(),
                    "phase": phase,
                    "epoch": epoch,
                    "epochs": epochs,
                    "batch": batch_index,
                    "batches": shown_batches,
                    "running_loss": loss_sum / len(truth),
                    "running_accuracy": float(np.mean(np.asarray(truth) == np.asarray(predicted))),
                    "elapsed_seconds": elapsed,
                    "peak_allocated_mib": (
                        torch.cuda.max_memory_allocated(device) / 1024**2
                        if device.type == "cuda"
                        else 0.0
                    ),
                },
            )
    elapsed = time.perf_counter() - started
    metrics = classification_metrics(truth, predicted, number_of_classes)
    metrics["loss"] = loss_sum / len(truth)
    metrics["elapsed_seconds"] = elapsed
    metrics["peak_allocated_mib"] = (
        torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else 0.0
    )
    probabilities = np.zeros((len(rows), number_of_classes), dtype=np.float32)
    used_rows: list[dict[str, object]] = []
    for dataset_index, probability in zip(ordered_indices, probability_rows):
        probabilities[dataset_index] = probability
        used_rows.append({"dataset_index": dataset_index})
    if len(ordered_indices) != len(rows):
        if not max_batches:
            raise ValueError(f"{phase} did not cover the full dataset")
        probabilities = probabilities[ordered_indices]
    return metrics, probabilities, used_rows


def enrich_metrics(metrics: dict[str, object], names: tuple[str, ...]) -> dict[str, object]:
    enriched = json.loads(json.dumps(metrics))
    for row in enriched["per_class"]:
        row["scientific_name"] = names[int(row["class_index"])]
    return enriched


def save_predictions(
    path: Path,
    rows: list[dict[str, str]],
    probabilities: np.ndarray,
    names: tuple[str, ...],
) -> None:
    fields = [
        "sample_id",
        "group_id",
        "source",
        "image_path",
        "true_class",
        "true_species",
        "predicted_class",
        "predicted_species",
        *[f"probability_{index}" for index in range(len(names))],
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row, probability in zip(rows, probabilities):
            true_class = int(row["class_index"])
            guess = int(probability.argmax())
            output: dict[str, object] = {
                "sample_id": row["sample_id"],
                "group_id": row["group_id"],
                "source": row["source"],
                "image_path": row["image_path"],
                "true_class": true_class,
                "true_species": names[true_class],
                "predicted_class": guess,
                "predicted_species": names[guess],
            }
            output.update(
                {f"probability_{index}": float(value) for index, value in enumerate(probability)}
            )
            writer.writerow(output)


def plot_confusion(path: Path, matrix: list[list[int]], names: tuple[str, ...], title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    array = np.asarray(matrix)
    figure, axis = plt.subplots(figsize=(11, 9))
    image = axis.imshow(array, cmap="Blues")
    figure.colorbar(image, ax=axis)
    short = [name.split()[0] for name in names]
    axis.set_xticks(range(len(names)), short, rotation=45, ha="right")
    axis.set_yticks(range(len(names)), short)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Ground truth")
    axis.set_title(title)
    for row in range(len(names)):
        for column in range(len(names)):
            axis.text(column, row, str(array[row, column]), ha="center", va="center", fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_history(path: Path, history: list[dict[str, object]]) -> None:
    if not history:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This formal baseline requires CUDA")
    if args.warmup_epochs < 0 or args.warmup_epochs >= args.epochs:
        raise ValueError("warmup epochs must be in [0, epochs)")
    manifest = args.manifest.resolve()
    source_checkpoint = args.source_checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / "matplotlib_cache"))
    seed_everything(args.seed)

    train_rows = read_rows(manifest, "train")
    val_rows = read_rows(manifest, "val")
    names = class_names(train_rows + val_rows)
    if len(names) != 10:
        raise ValueError(f"Expected ten classes, found {len(names)}")
    train_transform, eval_transform = transforms(args.image_size)
    train_dataset = CropDataset(train_rows, train_transform)
    val_dataset = CropDataset(val_rows, eval_transform)
    counts = np.bincount(train_dataset.labels, minlength=len(names))
    if np.any(counts == 0):
        raise ValueError(f"Training has an empty class: {counts.tolist()}")
    weights = torch.as_tensor(1.0 / counts[train_dataset.labels], dtype=torch.double)
    sampler = WeightedRandomSampler(
        weights,
        num_samples=len(train_dataset),
        replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )

    model, transfer = build_model(source_checkpoint, len(names))
    device = torch.device("cuda:0")
    model = model.to(device, memory_format=torch.channels_last)
    backbone_parameters = [parameter for name, parameter in model.named_parameters() if not name.startswith("fc.")]
    head_parameters = list(model.fc.parameters())
    optimizer = AdamW(
        [
            {"params": backbone_parameters, "lr": args.backbone_lr},
            {"params": head_parameters, "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    for parameter in backbone_parameters:
        parameter.requires_grad = args.warmup_epochs == 0

    config = {
        "schema_version": 1,
        "stage": "independent-hk-web-resnet50-ten-class-transfer",
        "started_at": timestamp(),
        "branch_expected": "codex/hk-image-resnet50-10class",
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "class_names": list(names),
        "split_sizes": {"train": len(train_rows), "val": len(val_rows)},
        "selection_metric": "HK tree-group macro-F1, then group accuracy",
        "test_policy": "test rows and images are loaded only after best validation checkpoint is selected",
        "transfer": transfer,
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    atomic_json(output_dir / "run_config.json", config)
    live_path = output_dir / "live_progress.json"
    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"
    history: list[dict[str, object]] = []
    start_epoch = 1
    best_score = (-1.0, -1.0)
    if args.resume and last_path.is_file():
        resume = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(resume["model_state"])
        optimizer.load_state_dict(resume["optimizer_state"])
        scheduler.load_state_dict(resume["scheduler_state"])
        scaler.load_state_dict(resume["scaler_state"])
        history = list(resume["history"])
        start_epoch = int(resume["epoch"]) + 1
        best_score = tuple(resume["best_score"])
        if start_epoch > args.warmup_epochs:
            for parameter in backbone_parameters:
                parameter.requires_grad = True

    print(json.dumps(config, ensure_ascii=False, indent=2), flush=True)
    training_started = time.perf_counter()
    for epoch in range(start_epoch, args.epochs + 1):
        if epoch == args.warmup_epochs + 1:
            for parameter in backbone_parameters:
                parameter.requires_grad = True
        train_metrics, _, _ = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            len(names),
            optimizer=optimizer,
            scaler=scaler,
            rows=train_rows,
            epoch=epoch,
            epochs=args.epochs,
            phase="train",
            live_path=live_path,
            progress_every=args.progress_every,
            max_batches=args.max_train_batches,
        )
        val_metrics, val_probabilities, _ = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            len(names),
            optimizer=None,
            scaler=scaler,
            rows=val_rows,
            epoch=epoch,
            epochs=args.epochs,
            phase="val",
            live_path=live_path,
            progress_every=args.progress_every,
        )
        val_group_metrics, _ = evaluate_group_level(val_rows, val_probabilities, len(names))
        score = (float(val_group_metrics["macro_f1"]), float(val_group_metrics["accuracy"]))
        history_row = {
            "epoch": epoch,
            "backbone_frozen": epoch <= args.warmup_epochs,
            "backbone_lr": optimizer.param_groups[0]["lr"],
            "head_lr": optimizer.param_groups[1]["lr"],
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_crop_accuracy": val_metrics["accuracy"],
            "val_crop_macro_f1": val_metrics["macro_f1"],
            "val_group_accuracy": val_group_metrics["accuracy"],
            "val_group_macro_f1": val_group_metrics["macro_f1"],
            "peak_allocated_mib": max(
                float(train_metrics["peak_allocated_mib"]),
                float(val_metrics["peak_allocated_mib"]),
            ),
        }
        history.append(history_row)
        if score > best_score:
            best_score = score
            atomic_torch_save(
                best_path,
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "class_names": names,
                    "val_crop_metrics": val_metrics,
                    "val_group_metrics": val_group_metrics,
                    "transfer": transfer,
                },
            )
        scheduler.step()
        atomic_torch_save(
            last_path,
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict(),
                "history": history,
                "best_score": best_score,
                "class_names": names,
            },
        )
        write_history(output_dir / "training_history.csv", history)
        atomic_json(output_dir / "training_history.json", history)
        print(
            f"epoch={epoch:03d}/{args.epochs} train_loss={train_metrics['loss']:.4f} "
            f"val_crop_f1={val_metrics['macro_f1']:.4f} "
            f"val_group_f1={val_group_metrics['macro_f1']:.4f} "
            f"best={best_score[0]:.4f} vram={history_row['peak_allocated_mib']:.0f}MiB",
            flush=True,
        )

    best = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model_state"])
    val_metrics, val_probabilities, _ = run_epoch(
        model,
        val_loader,
        criterion,
        device,
        len(names),
        optimizer=None,
        scaler=scaler,
        rows=val_rows,
        epoch=args.epochs,
        epochs=args.epochs,
        phase="final_val",
        live_path=live_path,
        progress_every=args.progress_every,
    )
    val_group_metrics, val_group_predictions = evaluate_group_level(
        val_rows, val_probabilities, len(names)
    )

    # Test rows and images are intentionally not touched until model selection is complete.
    test_rows = read_rows(manifest, "test")
    test_dataset = CropDataset(test_rows, eval_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    test_metrics, test_probabilities, _ = run_epoch(
        model,
        test_loader,
        criterion,
        device,
        len(names),
        optimizer=None,
        scaler=scaler,
        rows=test_rows,
        epoch=args.epochs,
        epochs=args.epochs,
        phase="test",
        live_path=live_path,
        progress_every=args.progress_every,
    )
    test_group_metrics, test_group_predictions = evaluate_group_level(
        test_rows, test_probabilities, len(names)
    )
    val_metrics = enrich_metrics(val_metrics, names)
    val_group_metrics = enrich_metrics(val_group_metrics, names)
    test_metrics = enrich_metrics(test_metrics, names)
    test_group_metrics = enrich_metrics(test_group_metrics, names)
    save_predictions(output_dir / "predictions_val.csv", val_rows, val_probabilities, names)
    save_predictions(output_dir / "predictions_test.csv", test_rows, test_probabilities, names)
    atomic_json(output_dir / "group_predictions_val.json", val_group_predictions)
    atomic_json(output_dir / "group_predictions_test.json", test_group_predictions)
    plot_confusion(
        output_dir / "confusion_matrix_test_group.png",
        test_group_metrics["confusion_matrix"],
        names,
        "HK Street View test: tree-group confusion matrix",
    )
    plot_confusion(
        output_dir / "confusion_matrix_test_crop.png",
        test_metrics["confusion_matrix"],
        names,
        "HK Street View test: crop confusion matrix",
    )
    final = {
        "status": "complete",
        "completed_at": timestamp(),
        "best_epoch": int(best["epoch"]),
        "epochs_completed": args.epochs,
        "training_elapsed_seconds": time.perf_counter() - training_started,
        "class_names": list(names),
        "split_sizes": {
            "train": len(train_rows),
            "val": len(val_rows),
            "test": len(test_rows),
        },
        "metrics": {
            "val_crop": val_metrics,
            "val_tree_group": val_group_metrics,
            "test_crop": test_metrics,
            "test_tree_group": test_group_metrics,
        },
        "selection_metric": "val_tree_group.macro_f1",
        "test_evaluated_after_training": True,
        "transfer": transfer,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "limitations": [
            "This is a ten-class closed-set baseline.",
            "Bauhinia purpurea has one HK validation tree and one HK test tree.",
            "Validation and test cover the current reviewed HK Street View subset, not all deployment conditions.",
        ],
    }
    atomic_json(output_dir / "final_metrics.json", final)
    atomic_json(
        live_path,
        {
            "status": "complete",
            "updated_at": timestamp(),
            "best_epoch": final["best_epoch"],
            "test_group_accuracy": test_group_metrics["accuracy"],
            "test_group_macro_f1": test_group_metrics["macro_f1"],
            "test_crop_accuracy": test_metrics["accuracy"],
            "test_crop_macro_f1": test_metrics["macro_f1"],
        },
    )
    print(json.dumps(final, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
