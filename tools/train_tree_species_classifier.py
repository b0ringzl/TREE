#!/usr/bin/env python3
"""Train a small PyTorch tree-species classifier from local image folders.

This script is intentionally self-contained so it can run in the existing
Miniconda yolo environment without sklearn or pandas.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_CACHE = PROJECT_ROOT / "models" / "_cache"
os.environ.setdefault("TORCH_HOME", str(PROJECT_CACHE / "torch"))
os.environ.setdefault("HF_HOME", str(PROJECT_CACHE / "huggingface"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_CACHE / "xdg"))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_CACHE / "matplotlib"))
os.environ.setdefault("TMP", str(PROJECT_CACHE / "tmp"))
os.environ.setdefault("TEMP", str(PROJECT_CACHE / "tmp"))
for cache_dir in {
    Path(os.environ["TORCH_HOME"]),
    Path(os.environ["HF_HOME"]),
    Path(os.environ["XDG_CACHE_HOME"]),
    Path(os.environ["MPLCONFIGDIR"]),
    Path(os.environ["TMP"]),
}:
    cache_dir.mkdir(parents=True, exist_ok=True)

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True)
class Sample:
    path: str
    species: str
    source: str
    original_species: str


def stable_float(text: str) -> float:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    return int(digest, 16) / float(16**12)


def iter_species_images(root: Path, source: str) -> list[Sample]:
    samples: list[Sample] = []
    if not root.exists():
        return samples
    for species_dir in sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name.casefold()):
        image_dir = species_dir / "images"
        if not image_dir.exists():
            continue
        for image in sorted(image_dir.rglob("*")):
            if image.is_file() and image.suffix.lower() in IMAGE_EXTS:
                samples.append(
                    Sample(
                        path=str(image.resolve()),
                        species=species_dir.name,
                        source=source,
                        original_species=species_dir.name,
                    )
                )
    return samples


def read_label_map(path: Path | None) -> dict[str, str]:
    if not path or not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return {
            row["original_species"].strip(): row["label_species"].strip()
            for row in reader
            if row.get("original_species") and row.get("label_species")
        }


def apply_label_map(samples: list[Sample], label_map: dict[str, str]) -> list[Sample]:
    if not label_map:
        return samples
    return [
        Sample(
            path=sample.path,
            species=label_map.get(sample.species, sample.species),
            source=sample.source,
            original_species=sample.original_species,
        )
        for sample in samples
    ]


def label_variant_map(samples: list[Sample]) -> dict[str, list[str]]:
    variants: dict[str, set[str]] = {}
    for sample in samples:
        if sample.original_species != sample.species:
            variants.setdefault(sample.species, set()).add(sample.original_species)
    return {label: sorted(names, key=str.casefold) for label, names in sorted(variants.items())}


def cap_per_species(samples: list[Sample], max_per_species: int, seed: int) -> list[Sample]:
    if max_per_species <= 0:
        return samples
    grouped: dict[str, list[Sample]] = {}
    for sample in samples:
        grouped.setdefault(sample.species, []).append(sample)
    rng = random.Random(seed)
    capped: list[Sample] = []
    for species, items in grouped.items():
        items = items[:]
        rng.shuffle(items)
        capped.extend(items[:max_per_species])
    return sorted(capped, key=lambda item: (item.species.casefold(), item.path))


def split_samples(samples: list[Sample], val_fraction: float) -> tuple[list[Sample], list[Sample]]:
    train: list[Sample] = []
    val: list[Sample] = []
    for sample in samples:
        bucket = stable_float(f"{sample.species}|{sample.path}")
        if bucket < val_fraction:
            val.append(sample)
        else:
            train.append(sample)
    train_species = {item.species for item in train}
    val_species = {item.species for item in val}
    missing_val = sorted(train_species - val_species)
    if missing_val:
        by_species: dict[str, list[Sample]] = {}
        for item in train:
            by_species.setdefault(item.species, []).append(item)
        move_to_val = {by_species[name][-1].path for name in missing_val if len(by_species.get(name, [])) > 1}
        train = [item for item in train if item.path not in move_to_val]
        val.extend(item for name in missing_val for item in by_species.get(name, [])[-1:] if item.path in move_to_val)
    return train, val


class TreeSpeciesDataset(Dataset):
    def __init__(self, samples: list[Sample], class_to_idx: dict[str, int], transform) -> None:
        self.samples = samples
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        with Image.open(sample.path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        return tensor, self.class_to_idx[sample.species]


def make_model(num_classes: int, pretrained: bool) -> nn.Module:
    weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, num_classes: int) -> dict:
    model.eval()
    correct = 0
    total = 0
    per_class_total = [0] * num_classes
    per_class_correct = [0] * num_classes
    confusion = [[0] * num_classes for _ in range(num_classes)]
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            preds = logits.argmax(dim=1)
            matches = preds.eq(labels)
            correct += int(matches.sum().item())
            total += int(labels.numel())
            for label, pred, match in zip(labels.cpu().tolist(), preds.cpu().tolist(), matches.cpu().tolist()):
                per_class_total[label] += 1
                per_class_correct[label] += int(match)
                confusion[label][pred] += 1
    per_class_acc = [
        (per_class_correct[index] / per_class_total[index]) if per_class_total[index] else None
        for index in range(num_classes)
    ]
    return {
        "accuracy": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
        "per_class_total": per_class_total,
        "per_class_correct": per_class_correct,
        "per_class_accuracy": per_class_acc,
        "confusion": confusion,
    }


def write_manifest(path: Path, samples: list[Sample], class_to_idx: dict[str, int]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "species", "original_species", "class_index", "source"])
        writer.writeheader()
        for sample in samples:
            writer.writerow({
                "path": sample.path,
                "species": sample.species,
                "original_species": sample.original_species,
                "class_index": class_to_idx[sample.species],
                "source": sample.source,
            })


def metrics_for_checkpoint(metrics: dict) -> dict:
    return {key: value for key, value in metrics.items() if key != "confusion"}


def write_per_class_metrics(path: Path, species: list[str], metrics: dict) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["class_index", "species", "val_total", "val_correct", "val_accuracy"],
        )
        writer.writeheader()
        for index, name in enumerate(species):
            accuracy = metrics["per_class_accuracy"][index]
            writer.writerow({
                "class_index": index,
                "species": name,
                "val_total": metrics["per_class_total"][index],
                "val_correct": metrics["per_class_correct"][index],
                "val_accuracy": "" if accuracy is None else round(accuracy, 6),
            })


def write_confusion_pairs(path: Path, species: list[str], metrics: dict, limit: int = 100) -> None:
    pairs = []
    for actual_index, row in enumerate(metrics["confusion"]):
        for predicted_index, count in enumerate(row):
            if actual_index != predicted_index and count:
                pairs.append({
                    "actual_species": species[actual_index],
                    "predicted_species": species[predicted_index],
                    "count": count,
                })
    pairs.sort(key=lambda item: item["count"], reverse=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["actual_species", "predicted_species", "count"])
        writer.writeheader()
        writer.writerows(pairs[:limit])


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a local tree species image classifier.")
    parser.add_argument("--data-root", type=Path, default=Path("external_datasets/inat_max150"))
    parser.add_argument("--supplement-root", type=Path, default=Path("external_datasets/inat_parent_name_supplement"))
    parser.add_argument("--output", type=Path, default=Path("models/tree_species_baseline"))
    parser.add_argument("--max-per-species", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrained", action="store_true", help="Use torchvision ImageNet weights if available.")
    parser.add_argument("--label-map", type=Path, default=Path("tools/tree_species_label_merge_map.csv"), help="CSV with original_species,label_species columns.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)

    samples = iter_species_images(args.data_root, "inat_max150")
    supplement_samples = iter_species_images(args.supplement_root, "inat_parent_name_supplement")
    by_species: dict[str, list[Sample]] = {}
    for sample in samples:
        by_species.setdefault(sample.species, []).append(sample)
    for sample in supplement_samples:
        if not by_species.get(sample.species):
            samples.append(sample)

    label_map = read_label_map(args.label_map)
    samples = apply_label_map(samples, label_map)
    variants_by_label = label_variant_map(samples)
    samples = cap_per_species(samples, args.max_per_species, args.seed)
    species = sorted({item.species for item in samples}, key=str.casefold)
    class_to_idx = {name: index for index, name in enumerate(species)}
    train_samples, val_samples = split_samples(samples, args.val_fraction)

    train_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(
        TreeSpeciesDataset(train_samples, class_to_idx, train_transform),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        TreeSpeciesDataset(val_samples, class_to_idx, val_transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    model = make_model(len(species), args.pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    history: list[dict] = []
    best_accuracy = -1.0
    best_epoch = 0
    best_metrics: dict | None = None
    best_model_path = args.output / "tree_species_mobilenetv3_best.pt"
    started = time.time()

    print(f"device={device}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}")
    print(f"classes={len(species)} train={len(train_samples)} val={len(val_samples)}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for step, (images, labels) in enumerate(train_loader, 1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            batch_size = int(labels.numel())
            running_loss += float(loss.item()) * batch_size
            seen += batch_size
            if step == 1 or step % 20 == 0:
                print(f"epoch={epoch} step={step}/{len(train_loader)} loss={loss.item():.4f}", flush=True)
        metrics = evaluate(model, val_loader, device, len(species))
        epoch_row = {
            "epoch": epoch,
            "train_loss": running_loss / max(seen, 1),
            "val_accuracy": metrics["accuracy"],
            "val_correct": metrics["correct"],
            "val_total": metrics["total"],
        }
        history.append(epoch_row)
        print(json.dumps(epoch_row, ensure_ascii=False), flush=True)
        if metrics["accuracy"] > best_accuracy:
            best_accuracy = metrics["accuracy"]
            best_epoch = epoch
            best_metrics = metrics
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "class_to_idx": class_to_idx,
                    "args": vars(args),
                    "history": history,
                    "best_epoch": best_epoch,
                    "metrics": metrics_for_checkpoint(metrics),
                    "label_map": label_map,
                    "label_variant_map": variants_by_label,
                },
                best_model_path,
            )
            print(f"saved_best={best_model_path} val_accuracy={best_accuracy:.6f}", flush=True)

    torch.save(
        {
            "model_state": model.state_dict(),
            "class_to_idx": class_to_idx,
            "args": vars(args),
            "history": history,
            "label_map": label_map,
            "label_variant_map": variants_by_label,
        },
        args.output / "tree_species_mobilenetv3_final.pt",
    )
    write_manifest(args.output / "train_manifest.csv", train_samples, class_to_idx)
    write_manifest(args.output / "val_manifest.csv", val_samples, class_to_idx)
    (args.output / "class_to_idx.json").write_text(json.dumps(class_to_idx, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "label_merge_map.json").write_text(json.dumps({
        "label_map": label_map,
        "label_variant_map": variants_by_label,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    if best_metrics is not None:
        write_per_class_metrics(args.output / "per_class_metrics.csv", species, best_metrics)
        write_confusion_pairs(args.output / "confusion_top_pairs.csv", species, best_metrics)
    report = {
        "status": "completed",
        "duration_seconds": round(time.time() - started, 2),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "classes": len(species),
        "label_merges": len(label_map),
        "labels_with_variants": len(variants_by_label),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "history": history,
        "best_epoch": best_epoch,
        "best_val_accuracy": best_accuracy,
        "best_model_path": str(best_model_path.resolve()),
        "final_model_path": str((args.output / "tree_species_mobilenetv3_final.pt").resolve()),
        "per_class_metrics": str((args.output / "per_class_metrics.csv").resolve()),
        "confusion_top_pairs": str((args.output / "confusion_top_pairs.csv").resolve()),
    }
    (args.output / "run_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
