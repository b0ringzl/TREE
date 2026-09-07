"""Cache aligned B2/B3 features for frozen-backbone B4b training."""

from __future__ import annotations

import argparse
import hashlib
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
from torchvision.models import resnet50
from torchvision.transforms import v2


SANDBOX_ROOT = Path(__file__).resolve().parents[1]
POINTMLP_ROOT = SANDBOX_ROOT / "vendor" / "pointMLP-pytorch"
sys.path.insert(0, str(SANDBOX_ROOT / "src"))
sys.path.insert(0, str(POINTMLP_ROOT / "pointnet2_ops_lib"))
sys.path.insert(0, str(POINTMLP_ROOT / "classification_ModelNet40"))

from models.pointmlp import Model  # noqa: E402
from tscmdl_whu import (  # noqa: E402
    MultimodalManifestDataset,
    PointMLPFeatureEncoder,
    ResNet50FeatureEncoder,
    TSCMDLFusionModel,
)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--b2-checkpoint", type=Path, required=True)
    parser.add_argument("--b3-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--channels-last", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_pointmlp(num_classes: int) -> Model:
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


def make_transform() -> v2.Compose:
    return v2.Compose(
        [
            v2.CenterCrop((224, 224)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_model(
    class_names: tuple[str, ...],
    b2_checkpoint_path: Path,
    b3_checkpoint_path: Path,
) -> tuple[TSCMDLFusionModel, dict[str, object]]:
    b2_checkpoint = torch.load(
        b2_checkpoint_path, map_location="cpu", weights_only=False
    )
    b3_checkpoint = torch.load(
        b3_checkpoint_path, map_location="cpu", weights_only=False
    )
    if tuple(b2_checkpoint["class_names"]) != class_names:
        raise ValueError("B2 checkpoint class names do not match the dataset")
    if tuple(b3_checkpoint["class_names"]) != class_names:
        raise ValueError("B3 checkpoint class names do not match the dataset")

    point_model = make_pointmlp(len(class_names))
    point_load = point_model.load_state_dict(b2_checkpoint["model_state"], strict=True)
    image_model = resnet50(weights=None)
    image_model.fc = nn.Linear(image_model.fc.in_features, len(class_names))
    image_load = image_model.load_state_dict(b3_checkpoint["model_state"], strict=True)
    point_encoder = PointMLPFeatureEncoder(point_model)
    image_encoder = ResNet50FeatureEncoder(image_model)
    model = TSCMDLFusionModel(
        point_encoder,
        image_encoder,
        len(class_names),
        point_dim=point_encoder.output_dim,
        image_dim=image_encoder.output_dim,
        modal_dim=1024,
        dropout=0.5,
        freeze_backbones=True,
    )
    return model, {
        "b2_epoch": int(b2_checkpoint["epoch"]),
        "b3_epoch": int(b3_checkpoint["epoch"]),
        "b2_strict_load": not point_load.missing_keys
        and not point_load.unexpected_keys,
        "b3_strict_load": not image_load.missing_keys
        and not image_load.unexpected_keys,
        "point_feature_dim": point_encoder.output_dim,
        "image_feature_dim": image_encoder.output_dim,
    }


def extract_split(
    model: TSCMDLFusionModel,
    dataset: MultimodalManifestDataset,
    batch_size: int,
    workers: int,
    device: torch.device,
    use_amp: bool,
    channels_last: bool,
) -> dict[str, object]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )
    point_parts = []
    image_parts = []
    label_parts = []
    seen_indices = []
    model.eval()
    started = time.perf_counter()
    for batch_index, (points, images, labels, indices) in enumerate(loader, start=1):
        points = points.to(device, non_blocking=True).permute(0, 2, 1).contiguous()
        images = images.to(
            device,
            non_blocking=True,
            memory_format=(
                torch.channels_last if channels_last else torch.contiguous_format
            ),
        )
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=use_amp
        ):
            point_raw, image_raw = model.encode_modalities(points, images)
        point_parts.append(point_raw.float().cpu())
        image_parts.append(image_raw.float().cpu())
        label_parts.append(labels.long().cpu())
        seen_indices.extend(int(value) for value in indices.tolist())
        print(
            f"\r{dataset.split:5s} [{batch_index:3d}/{len(loader):3d}] "
            f"samples={len(seen_indices):3d}/{len(dataset):3d}",
            end="",
            flush=True,
        )
    print(flush=True)

    expected_indices = list(range(len(dataset)))
    if seen_indices != expected_indices:
        raise ValueError(f"Non-sequential cached indices in split {dataset.split}")
    point_features = torch.cat(point_parts)
    image_features = torch.cat(image_parts)
    labels = torch.cat(label_parts)
    if point_features.shape != (len(dataset), 1024):
        raise ValueError(f"Unexpected point cache shape: {point_features.shape}")
    if image_features.shape != (len(dataset), 2048):
        raise ValueError(f"Unexpected image cache shape: {image_features.shape}")
    if labels.shape != (len(dataset),):
        raise ValueError(f"Unexpected label cache shape: {labels.shape}")
    if not torch.isfinite(point_features).all() or not torch.isfinite(
        image_features
    ).all():
        raise FloatingPointError(f"Non-finite cached features in {dataset.split}")
    if labels.tolist() != dataset.labels.tolist():
        raise ValueError(f"Cached labels changed order in {dataset.split}")
    return {
        "point_features": point_features.contiguous(),
        "image_features": image_features.contiguous(),
        "labels": labels.contiguous(),
        "sample_keys": list(dataset.sample_keys),
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.workers < 0:
        raise ValueError("batch-size must be positive and workers cannot be negative")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for B4 feature extraction")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda")
    transform = make_transform()
    datasets = {
        split: MultimodalManifestDataset(args.dataset_root, split, transform)
        for split in ("train", "val", "test")
    }
    class_names = datasets["train"].class_names
    for split, dataset in datasets.items():
        if dataset.class_names != class_names:
            raise ValueError(f"Class names differ in split {split}")

    model, checkpoint_load = build_model(
        class_names, args.b2_checkpoint, args.b3_checkpoint
    )
    model.to(device)
    if args.channels_last:
        model.image_encoder.to(memory_format=torch.channels_last)
    if any(parameter.requires_grad for parameter in model.point_encoder.parameters()):
        raise ValueError("Point backbone is not frozen")
    if any(parameter.requires_grad for parameter in model.image_encoder.parameters()):
        raise ValueError("Image backbone is not frozen")

    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    split_payloads = {
        split: extract_split(
            model,
            dataset,
            args.batch_size,
            args.workers,
            device,
            args.amp,
            args.channels_last,
        )
        for split, dataset in datasets.items()
    }
    cache_path = args.output_dir / "features.pt"
    source_hashes = {
        "b2_best": sha256_file(args.b2_checkpoint),
        "b3_best": sha256_file(args.b3_checkpoint),
    }
    cache = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "dataset_root": str(args.dataset_root.resolve()),
        "class_names": list(class_names),
        "source_checkpoint_paths": {
            "b2_best": str(args.b2_checkpoint.resolve()),
            "b3_best": str(args.b3_checkpoint.resolve()),
        },
        "source_checkpoint_sha256": source_hashes,
        "checkpoint_load": checkpoint_load,
        "preprocessing": {
            "point_input": "8192 normalized XYZ points",
            "image_transform": "center 224x224 crop and ImageNet normalization",
            "amp": args.amp,
        },
        "splits": split_payloads,
    }
    atomic_torch_save(cache_path, cache)
    manifest = {
        "status": "complete",
        "cache_path": str(cache_path.resolve()),
        "cache_bytes": cache_path.stat().st_size,
        "cache_sha256": sha256_file(cache_path),
        "class_names": list(class_names),
        "split_sizes": {
            split: len(dataset) for split, dataset in datasets.items()
        },
        "feature_shapes": {
            split: {
                "point": list(payload["point_features"].shape),
                "image": list(payload["image_features"].shape),
            }
            for split, payload in split_payloads.items()
        },
        "checkpoint_load": checkpoint_load,
        "source_checkpoint_sha256": source_hashes,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / (1024**2),
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / (1024**2),
        "elapsed_seconds": time.perf_counter() - started,
        "temporary_files": [],
    }
    atomic_json(args.output_dir / "cache_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
