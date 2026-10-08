#!/usr/bin/env python3
"""Run inference with a trained tree species classifier checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import os
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
from torchvision import models, transforms


def make_model(num_classes: int) -> nn.Module:
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def load_images(path: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    if path.is_file():
        return [path]
    return sorted(item for item in path.rglob("*") if item.is_file() and item.suffix.lower() in exts)


def main() -> int:
    parser = argparse.ArgumentParser(description="Predict tree species for one image or a directory.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("models/tree_species_predictions.csv"))
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    class_to_idx = checkpoint["class_to_idx"]
    label_variant_map = checkpoint.get("label_variant_map", {})
    idx_to_class = {index: name for name, index in class_to_idx.items()}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(len(class_to_idx))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    rows = []
    images = load_images(args.input)
    with torch.inference_mode():
        for image_path in images:
            with Image.open(image_path) as image:
                tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
            probs = torch.softmax(model(tensor), dim=1)[0]
            values, indices = torch.topk(probs, k=min(args.topk, probs.numel()))
            prediction = {
                "path": str(image_path.resolve()),
                "top1": idx_to_class[int(indices[0])],
                "top1_prob": float(values[0]),
                "top1_variant_references": json.dumps(
                    label_variant_map.get(idx_to_class[int(indices[0])], []),
                    ensure_ascii=False,
                ),
                "topk": json.dumps(
                    [
                        {
                            "species": idx_to_class[int(index)],
                            "prob": float(value),
                            "variant_references": label_variant_map.get(idx_to_class[int(index)], []),
                        }
                        for value, index in zip(values.cpu(), indices.cpu())
                    ],
                    ensure_ascii=False,
                ),
            }
            rows.append(prediction)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "top1", "top1_prob", "top1_variant_references", "topk"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"device": str(device), "images": len(rows), "output": str(args.output.resolve())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
