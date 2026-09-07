#!/usr/bin/env python3
"""Train a dynamic-class ResNet50 from a leakage-controlled image manifest."""

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
    parser.add_argument("--stage-name", required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=192)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--backbone-lr", type=float, default=2e-4)
    parser.add_argument("--head-lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def atomic_json(path: Path, value: object) -> None:
    temporary = Path(f"{path}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
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


def class_names(rows: list[dict[str, str]]) -> tuple[str, ...]:
    mapping: dict[int, str] = {}
    for row in rows:
        index = int(row["class_index"])
        name = row["species"]
        if index in mapping and mapping[index] != name:
            raise ValueError(f"Class index {index} maps to multiple names")
        mapping[index] = name
    if sorted(mapping) != list(range(len(mapping))):
        raise ValueError("Class indices are not contiguous")
    return tuple(mapping[index] for index in range(len(mapping)))


class ManifestDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], transform) -> None:
        self.rows = rows
        self.transform = transform
        self.labels = np.asarray([int(row["class_index"]) for row in rows])

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        with Image.open(PROJECT_ROOT / row["image_path"]) as source:
            source = source.convert("RGB")
            if row["sample_type"] == "crop":
                width, height = source.size
                box = (
                    max(0, math.floor(float(row["x0"]) * width)),
                    max(0, math.floor(float(row["y0"]) * height)),
                    min(width, math.ceil(float(row["x1"]) * width)),
                    min(height, math.ceil(float(row["y1"]) * height)),
                )
                if box[2] <= box[0] or box[3] <= box[1]:
                    raise ValueError(f"Invalid crop: {row['sample_id']}")
                source = source.crop(box)
            tensor = self.transform(source)
        return tensor, int(row["class_index"]), index


def make_transforms(image_size: int):
    normalize = v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    train = v2.Compose(
        [
            v2.RandomResizedCrop((image_size, image_size), scale=(0.65, 1.0), antialias=True),
            v2.RandomHorizontalFlip(),
            v2.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.12, hue=0.02),
            v2.RandomRotation(8),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            normalize,
        ]
    )
    evaluate = v2.Compose(
        [
            v2.Resize((image_size, image_size), antialias=True),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            normalize,
        ]
    )
    return train, evaluate


def build_model(checkpoint_path: Path, number_of_classes: int):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = dict(checkpoint["model_state"])
    source_fc_weight = state.pop("fc.weight")
    source_fc_bias = state.pop("fc.bias")
    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, number_of_classes)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if set(missing) != {"fc.weight", "fc.bias"} or unexpected:
        raise ValueError(f"Transfer mismatch: missing={missing}, unexpected={unexpected}")
    nn.init.normal_(model.fc.weight, mean=0.0, std=0.01)
    nn.init.zeros_(model.fc.bias)
    return model, {
        "source_checkpoint": str(checkpoint_path),
        "source_sha256": sha256_file(checkpoint_path),
        "source_epoch": checkpoint.get("epoch"),
        "source_class_names": list(checkpoint.get("class_names", [])),
        "source_fc_shape": list(source_fc_weight.shape),
        "source_fc_bias_shape": list(source_fc_bias.shape),
        "target_fc_shape": list(model.fc.weight.shape),
        "loaded_backbone_keys": len(state),
    }


def metrics(truth: list[int], predicted: list[int], probabilities: np.ndarray, classes: int) -> dict[str, object]:
    confusion = np.zeros((classes, classes), dtype=np.int64)
    for actual, guess in zip(truth, predicted):
        confusion[actual, guess] += 1
    per_class = []
    for index in range(classes):
        tp = int(confusion[index, index])
        fp = int(confusion[:, index].sum() - tp)
        support = int(confusion[index, :].sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append({"class_index": index, "support": support, "precision": precision, "recall": recall, "f1": f1})
    count = len(truth)
    top_k = min(5, classes)
    top5 = np.argpartition(probabilities, -top_k, axis=1)[:, -top_k:]
    top5_accuracy = float(np.mean([actual in guesses for actual, guesses in zip(truth, top5)]))
    return {
        "accuracy": float(np.trace(confusion) / count),
        "top5_accuracy": top5_accuracy,
        "balanced_accuracy": float(np.mean([row["recall"] for row in per_class])),
        "macro_f1": float(np.mean([row["f1"] for row in per_class])),
        "sample_count": count,
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }


def run_epoch(model, loader, criterion, device, classes, optimizer, scaler, rows, live_path, phase, epoch, epochs, progress_every):
    training = optimizer is not None
    model.train(training)
    truth: list[int] = []
    predictions: list[int] = []
    probability_rows: list[np.ndarray] = []
    ordered_indices: list[int] = []
    loss_sum = 0.0
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for batch_index, (images, labels, indices) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
        labels = labels.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels)
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
        probs = torch.softmax(logits.detach().float(), dim=1).cpu().numpy()
        actual = labels.detach().cpu().tolist()
        truth.extend(actual)
        predictions.extend(probs.argmax(axis=1).tolist())
        probability_rows.extend(probs)
        ordered_indices.extend(indices.tolist())
        loss_sum += float(loss.detach()) * len(actual)
        if batch_index % progress_every == 0 or batch_index == len(loader):
            atomic_json(
                live_path,
                {
                    "status": "running",
                    "updated_at": timestamp(),
                    "phase": phase,
                    "epoch": epoch,
                    "epochs": epochs,
                    "batch": batch_index,
                    "batches": len(loader),
                    "running_loss": loss_sum / len(truth),
                    "running_accuracy": float(np.mean(np.asarray(truth) == np.asarray(predictions))),
                    "elapsed_seconds": time.perf_counter() - started,
                    "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
                },
            )
    probabilities = np.zeros((len(rows), classes), dtype=np.float32)
    for index, probability in zip(ordered_indices, probability_rows):
        probabilities[index] = probability
    result = metrics(truth, predictions, np.asarray(probability_rows), classes)
    result["loss"] = loss_sum / len(truth)
    result["elapsed_seconds"] = time.perf_counter() - started
    result["peak_allocated_mib"] = torch.cuda.max_memory_allocated(device) / 1024**2
    return result, probabilities


def group_metrics(rows, probabilities, classes):
    grouped: defaultdict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[row["group_id"]].append(index)
    truth, predicted, averaged = [], [], []
    for group in sorted(grouped):
        indices = grouped[group]
        labels = {int(rows[index]["class_index"]) for index in indices}
        if len(labels) != 1:
            raise ValueError(f"Multiple labels in group {group}")
        average = probabilities[indices].mean(axis=0)
        truth.append(labels.pop())
        predicted.append(int(average.argmax()))
        averaged.append(average)
    return metrics(truth, predicted, np.asarray(averaged), classes)


def enrich(result: dict[str, object], names: tuple[str, ...]) -> dict[str, object]:
    output = json.loads(json.dumps(result))
    for row in output["per_class"]:
        row["scientific_name"] = names[int(row["class_index"])]
    return output


def save_predictions(path: Path, rows, probabilities, names):
    fields = ["sample_id", "group_id", "image_path", "true_class", "true_species", "predicted_class", "predicted_species", "confidence"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row, probability in zip(rows, probabilities):
            actual = int(row["class_index"])
            guess = int(probability.argmax())
            writer.writerow(
                {
                    "sample_id": row["sample_id"], "group_id": row["group_id"], "image_path": row["image_path"],
                    "true_class": actual, "true_species": names[actual], "predicted_class": guess,
                    "predicted_species": names[guess], "confidence": float(probability[guess]),
                }
            )


def write_history(path: Path, history: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not 0 <= args.warmup_epochs < args.epochs:
        raise ValueError("warmup epochs must be in [0, epochs)")
    manifest = args.manifest.resolve()
    checkpoint_path = args.source_checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / "matplotlib_cache"))
    seed_everything(args.seed)

    train_rows = read_rows(manifest, "train")
    val_rows = read_rows(manifest, "val")
    names = class_names(train_rows + val_rows)
    train_transform, eval_transform = make_transforms(args.image_size)
    train_dataset = ManifestDataset(train_rows, train_transform)
    val_dataset = ManifestDataset(val_rows, eval_transform)
    counts = np.bincount(train_dataset.labels, minlength=len(names))
    weights = torch.as_tensor(1.0 / counts[train_dataset.labels], dtype=torch.double)
    sampler = WeightedRandomSampler(weights, len(train_dataset), replacement=True, generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
    val_loader = DataLoader(val_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)

    model, transfer = build_model(checkpoint_path, len(names))
    device = torch.device("cuda:0")
    model = model.to(device, memory_format=torch.channels_last)
    backbone = [parameter for name, parameter in model.named_parameters() if not name.startswith("fc.")]
    head = list(model.fc.parameters())
    optimizer = AdamW([{"params": backbone, "lr": args.backbone_lr}, {"params": head, "lr": args.head_lr}], weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    for parameter in backbone:
        parameter.requires_grad = args.warmup_epochs == 0

    live_path = output_dir / "live_progress.json"
    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"
    history: list[dict[str, object]] = []
    best_score = (-1.0, -1.0)
    start_epoch = 1
    stale_epochs = 0
    if args.resume and last_path.is_file():
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        history = checkpoint["history"]
        best_score = tuple(checkpoint["best_score"])
        start_epoch = int(checkpoint["epoch"]) + 1
        stale_epochs = int(checkpoint.get("stale_epochs", 0))
    config = {
        "schema_version": 1, "stage": args.stage_name, "started_at": timestamp(), "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest), "class_names": list(names),
        "split_sizes": {"train": len(train_rows), "val": len(val_rows)},
        "selection_metric": "validation observation-group macro-F1 then accuracy", "test_policy": "loaded after checkpoint selection",
        "transfer": transfer, "device": torch.cuda.get_device_name(device), "torch": torch.__version__,
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    atomic_json(output_dir / "run_config.json", config)
    print(json.dumps(config, ensure_ascii=False, indent=2), flush=True)
    started = time.perf_counter()
    completed_epochs = start_epoch - 1
    if start_epoch > args.warmup_epochs:
        for parameter in backbone:
            parameter.requires_grad = True
    for epoch in range(start_epoch, args.epochs + 1):
        if epoch == args.warmup_epochs + 1:
            for parameter in backbone:
                parameter.requires_grad = True
        train_result, _ = run_epoch(model, train_loader, criterion, device, len(names), optimizer, scaler, train_rows, live_path, "train", epoch, args.epochs, args.progress_every)
        val_result, val_probabilities = run_epoch(model, val_loader, criterion, device, len(names), None, scaler, val_rows, live_path, "val", epoch, args.epochs, args.progress_every)
        val_group = group_metrics(val_rows, val_probabilities, len(names))
        score = (float(val_group["macro_f1"]), float(val_group["accuracy"]))
        improved = score > best_score
        if improved:
            best_score = score
            stale_epochs = 0
            atomic_torch_save(best_path, {"epoch": epoch, "model_state": model.state_dict(), "class_names": names, "val_metrics": val_result, "val_group_metrics": val_group, "transfer": transfer})
        else:
            stale_epochs += 1
        scheduler.step()
        row = {
            "epoch": epoch, "backbone_frozen": epoch <= args.warmup_epochs, "backbone_lr": optimizer.param_groups[0]["lr"], "head_lr": optimizer.param_groups[1]["lr"],
            "train_loss": train_result["loss"], "train_accuracy": train_result["accuracy"], "train_macro_f1": train_result["macro_f1"],
            "val_loss": val_result["loss"], "val_accuracy": val_result["accuracy"], "val_macro_f1": val_result["macro_f1"],
            "val_group_accuracy": val_group["accuracy"], "val_group_macro_f1": val_group["macro_f1"], "peak_allocated_mib": max(train_result["peak_allocated_mib"], val_result["peak_allocated_mib"]),
        }
        history.append(row)
        completed_epochs = epoch
        atomic_torch_save(last_path, {"epoch": epoch, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(), "scaler_state": scaler.state_dict(), "history": history, "best_score": best_score, "stale_epochs": stale_epochs, "class_names": names})
        write_history(output_dir / "training_history.csv", history)
        print(f"epoch={epoch:03d}/{args.epochs} train_loss={train_result['loss']:.4f} val_group_f1={val_group['macro_f1']:.4f} best={best_score[0]:.4f} stale={stale_epochs}", flush=True)
        if stale_epochs >= args.patience:
            print(f"early_stop epoch={epoch} patience={args.patience}", flush=True)
            break

    best = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model_state"])
    val_result, val_probabilities = run_epoch(model, val_loader, criterion, device, len(names), None, scaler, val_rows, live_path, "final_val", completed_epochs, completed_epochs, args.progress_every)
    val_group = group_metrics(val_rows, val_probabilities, len(names))
    test_rows = read_rows(manifest, "test")
    test_loader = DataLoader(ManifestDataset(test_rows, eval_transform), batch_size=args.eval_batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
    test_result, test_probabilities = run_epoch(model, test_loader, criterion, device, len(names), None, scaler, test_rows, live_path, "test", completed_epochs, completed_epochs, args.progress_every)
    test_group = group_metrics(test_rows, test_probabilities, len(names))
    save_predictions(output_dir / "predictions_val.csv", val_rows, val_probabilities, names)
    save_predictions(output_dir / "predictions_test.csv", test_rows, test_probabilities, names)
    final = {
        "status": "complete", "completed_at": timestamp(), "best_epoch": int(best["epoch"]), "epochs_completed": completed_epochs,
        "training_elapsed_seconds": time.perf_counter() - started, "class_names": list(names),
        "split_sizes": {"train": len(train_rows), "val": len(val_rows), "test": len(test_rows)},
        "metrics": {"val_sample": enrich(val_result, names), "val_group": enrich(val_group, names), "test_sample": enrich(test_result, names), "test_group": enrich(test_group, names)},
        "selection_metric": "val_group.macro_f1", "test_evaluated_after_selection": True, "transfer": transfer,
    }
    atomic_json(output_dir / "final_metrics.json", final)
    atomic_json(live_path, {"status": "complete", "updated_at": timestamp(), "best_epoch": final["best_epoch"], "test_accuracy": test_result["accuracy"], "test_macro_f1": test_result["macro_f1"], "test_top5_accuracy": test_result["top5_accuracy"], "test_group_accuracy": test_group["accuracy"], "test_group_macro_f1": test_group["macro_f1"]})
    print(json.dumps(final, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
