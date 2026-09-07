"""Cache aligned PTv2 and ResNet50 features for the D1 fusion baseline."""

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
sys.path.insert(0, str(SANDBOX_ROOT / "src"))

from tscmdl_whu import (  # noqa: E402
    MultimodalManifestDataset,
    PointTransformerV2Classifier,
    load_memmap_knn_cache,
)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--knn-cache", type=Path, required=True)
    parser.add_argument("--point-checkpoint", type=Path, required=True)
    parser.add_argument("--image-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=56)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--channels-last", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def image_transform() -> v2.Compose:
    return v2.Compose(
        [
            v2.CenterCrop((224, 224)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_models(
    class_names: tuple[str, ...],
    point_checkpoint_path: Path,
    image_checkpoint_path: Path,
) -> tuple[PointTransformerV2Classifier, nn.Module, dict[str, object]]:
    point_checkpoint = torch.load(
        point_checkpoint_path, map_location="cpu", weights_only=False
    )
    image_checkpoint = torch.load(
        image_checkpoint_path, map_location="cpu", weights_only=False
    )
    if tuple(point_checkpoint["class_names"]) != class_names:
        raise ValueError("Point checkpoint class order does not match the dataset")
    if tuple(image_checkpoint["class_names"]) != class_names:
        raise ValueError("Image checkpoint class order does not match the dataset")

    point_model = PointTransformerV2Classifier(len(class_names))
    point_load = point_model.load_state_dict(
        point_checkpoint["model_state"], strict=True
    )
    image_model = resnet50(weights=None)
    image_model.fc = nn.Linear(image_model.fc.in_features, len(class_names))
    image_load = image_model.load_state_dict(
        image_checkpoint["model_state"], strict=True
    )
    image_dim = int(image_model.fc.in_features)
    image_model.fc = nn.Identity()
    point_model.eval()
    image_model.eval()
    return point_model, image_model, {
        "point_epoch": int(point_checkpoint["epoch"]),
        "image_epoch": int(image_checkpoint["epoch"]),
        "point_strict_load": not point_load.missing_keys
        and not point_load.unexpected_keys,
        "image_strict_load": not image_load.missing_keys
        and not image_load.unexpected_keys,
        "point_feature_dim": int(point_model.configuration["encoder_channels"][-1]),
        "image_feature_dim": image_dim,
    }


def ptv2_forward_features(
    model: PointTransformerV2Classifier,
    points: torch.Tensor,
    initial_reference_index: torch.Tensor,
) -> torch.Tensor:
    """Expose the global-average PTv2 feature without changing accepted code."""
    if points.ndim != 3 or points.shape[2] != 3:
        raise ValueError(f"Expected points [B,N,3], got {tuple(points.shape)}")
    batch_size, point_count, _ = points.shape
    expected = (batch_size, point_count, model.patch_neighbours)
    if tuple(initial_reference_index.shape) != expected:
        raise ValueError(
            f"Reference shape mismatch: {tuple(initial_reference_index.shape)} != {expected}"
        )
    coord = points.reshape(batch_size * point_count, 3).contiguous()
    feat = model.patch_projection(coord)
    offset = torch.arange(
        1, batch_size + 1, dtype=torch.long, device=points.device
    ) * point_count
    base = (
        torch.arange(batch_size, device=points.device)
        .mul(point_count)
        .view(batch_size, 1, 1)
    )
    packed_reference = (
        initial_reference_index.long() + base
    ).reshape(batch_size * point_count, model.patch_neighbours)
    feat = model.patch_blocks(coord, feat, offset, packed_reference)
    for stage in model.encoder_stages:
        coord, feat, offset = stage(coord, feat, offset)
    ends = [int(value) for value in offset.detach().cpu().tolist()]
    starts = [0, *ends[:-1]]
    return torch.stack(
        [feat[start:end].mean(dim=0) for start, end in zip(starts, ends)], dim=0
    )


def extract_split(
    split: str,
    dataset: MultimodalManifestDataset,
    reference_cache: object,
    point_model: PointTransformerV2Classifier,
    image_model: nn.Module,
    *,
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
    point_parts: list[torch.Tensor] = []
    image_parts: list[torch.Tensor] = []
    label_parts: list[torch.Tensor] = []
    seen_indices: list[int] = []
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    for batch_index, (points, images, labels, indices) in enumerate(loader, start=1):
        reference = reference_cache.index_select(0, indices)
        points = points.to(device, non_blocking=True)
        reference = reference.to(device, non_blocking=True)
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
            point_features = ptv2_forward_features(point_model, points, reference)
            image_features = image_model(images)
        point_parts.append(point_features.float().cpu())
        image_parts.append(image_features.float().cpu())
        label_parts.append(labels.long().cpu())
        seen_indices.extend(int(value) for value in indices.tolist())
        print(
            f"\r{split:5s} [{batch_index:3d}/{len(loader):3d}] "
            f"samples={len(seen_indices):4d}/{len(dataset):4d}",
            end="",
            flush=True,
        )
    print(flush=True)
    if seen_indices != list(range(len(dataset))):
        raise ValueError(f"Non-sequential feature extraction in {split}")
    point_features = torch.cat(point_parts).contiguous()
    image_features = torch.cat(image_parts).contiguous()
    labels = torch.cat(label_parts).contiguous()
    expected_point_dim = int(point_model.configuration["encoder_channels"][-1])
    if point_features.shape != (len(dataset), expected_point_dim):
        raise ValueError(f"Unexpected point features in {split}: {point_features.shape}")
    if image_features.shape != (len(dataset), 2048):
        raise ValueError(f"Unexpected image features in {split}: {image_features.shape}")
    if labels.tolist() != dataset.labels.tolist():
        raise ValueError(f"Labels changed order in {split}")
    if not torch.isfinite(point_features).all():
        raise FloatingPointError(f"Non-finite point features in {split}")
    if not torch.isfinite(image_features).all():
        raise FloatingPointError(f"Non-finite image features in {split}")
    return {
        "point_features": point_features,
        "image_features": image_features,
        "labels": labels,
        "sample_keys": list(dataset.sample_keys),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
    }


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.workers < 0:
        raise ValueError("batch-size must be positive and workers cannot be negative")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for D1 fusion feature extraction")
    args.dataset_root = args.dataset_root.resolve()
    args.knn_cache = args.knn_cache.resolve()
    args.point_checkpoint = args.point_checkpoint.resolve()
    args.image_checkpoint = args.image_checkpoint.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda")
    transform = image_transform()
    datasets = {
        split: MultimodalManifestDataset(args.dataset_root, split, transform)
        for split in ("train", "val", "test")
    }
    class_names = datasets["train"].class_names
    for split, dataset in datasets.items():
        if dataset.class_names != class_names:
            raise ValueError(f"Class order differs in {split}")
    references, cache_metadata = load_memmap_knn_cache(
        args.knn_cache, args.dataset_root, datasets
    )
    point_model, image_model, checkpoint_load = build_models(
        class_names, args.point_checkpoint, args.image_checkpoint
    )
    point_model.to(device)
    image_model.to(device)
    if args.channels_last:
        image_model.to(memory_format=torch.channels_last)

    started = time.perf_counter()
    split_payloads = {
        split: extract_split(
            split,
            dataset,
            references[split],
            point_model,
            image_model,
            batch_size=args.batch_size,
            workers=args.workers,
            device=device,
            use_amp=args.amp,
            channels_last=args.channels_last,
        )
        for split, dataset in datasets.items()
    }
    output_path = args.output_dir / "features.pt"
    payload = {
        "schema_version": 1,
        "cache_type": "d1_ptv2_resnet50_frozen_features",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "seed": args.seed,
        "dataset_root": str(args.dataset_root),
        "class_names": list(class_names),
        "feature_dimensions": {
            "point": checkpoint_load["point_feature_dim"],
            "image": checkpoint_load["image_feature_dim"],
        },
        "fusion_modal_dim": 1024,
        "split_sizes": {
            split: len(dataset) for split, dataset in datasets.items()
        },
        "source_checkpoint_paths": {
            "point_best": str(args.point_checkpoint),
            "image_best": str(args.image_checkpoint),
        },
        "source_checkpoint_sha256": {
            "point_best": sha256_file(args.point_checkpoint),
            "image_best": sha256_file(args.image_checkpoint),
        },
        "checkpoint_load": checkpoint_load,
        "knn_cache": {
            "path": cache_metadata["path"],
            "index_sha256": cache_metadata["index_sha256"],
        },
        "preprocessing": {
            "point": "accepted normalized 8192 XYZ with accepted exact k=8 cache",
            "image": "deterministic center crop 224 and ImageNet normalization",
            "amp": args.amp,
            "channels_last": args.channels_last,
        },
        "splits": split_payloads,
    }
    atomic_torch_save(output_path, payload)
    manifest = {
        "status": "complete",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "seed": args.seed,
        "cache_path": str(output_path),
        "cache_bytes": output_path.stat().st_size,
        "cache_sha256": sha256_file(output_path),
        "elapsed_seconds": time.perf_counter() - started,
        "feature_dimensions": payload["feature_dimensions"],
        "split_sizes": payload["split_sizes"],
        "source_checkpoint_sha256": payload["source_checkpoint_sha256"],
        "knn_index_sha256": cache_metadata["index_sha256"],
    }
    atomic_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
