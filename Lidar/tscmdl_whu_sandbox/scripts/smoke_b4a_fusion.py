"""Validate frozen-backbone TSCMDL fusion without starting formal training."""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
import time
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
    classification_metrics,
)
from tscmdl_whu.status import sha256_file  # noqa: E402


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--b2-checkpoint", type=Path, required=True)
    parser.add_argument("--b3-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--workers", type=int, default=0)
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


def emit(message: str, log_path: Path) -> None:
    print(message, flush=True)
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(message + "\n")


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


def make_image_transform() -> v2.Compose:
    return v2.Compose(
        [
            v2.CenterCrop((224, 224)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def make_datasets(
    root: Path,
) -> dict[str, MultimodalManifestDataset]:
    transform = make_image_transform()
    return {
        split: MultimodalManifestDataset(root, split, transform)
        for split in ("train", "val", "test")
    }


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
    details = {
        "b2_epoch": int(b2_checkpoint["epoch"]),
        "b3_epoch": int(b3_checkpoint["epoch"]),
        "b2_strict_load": not point_load.missing_keys
        and not point_load.unexpected_keys,
        "b3_strict_load": not image_load.missing_keys
        and not image_load.unexpected_keys,
        "point_feature_dim": point_encoder.output_dim,
        "image_feature_dim": image_encoder.output_dim,
        "modal_feature_dim": model.modal_dim,
        "fused_feature_dim": 2 * model.modal_dim,
    }
    return model, details


def parameter_counts(model: TSCMDLFusionModel) -> dict[str, int]:
    point = sum(parameter.numel() for parameter in model.point_encoder.parameters())
    image = sum(parameter.numel() for parameter in model.image_encoder.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    return {
        "point_backbone": point,
        "image_backbone": image,
        "fusion_trainable": trainable,
        "total": total,
        "frozen": total - trainable,
    }


def batch_metrics(
    labels: torch.Tensor, logits: torch.Tensor, num_classes: int
) -> dict[str, object]:
    predicted = logits.detach().float().argmax(dim=1)
    return classification_metrics(
        labels.detach().cpu().tolist(),
        predicted.cpu().tolist(),
        num_classes,
    )


def evaluate_batch(
    model: TSCMDLFusionModel,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    use_amp: bool,
) -> dict[str, object]:
    points, images, labels, _ = batch
    points = points.to(device, non_blocking=True).permute(0, 2, 1).contiguous()
    images = images.to(device, non_blocking=True).contiguous(
        memory_format=torch.channels_last
    )
    labels = labels.to(device, non_blocking=True)
    model.eval()
    with torch.no_grad(), torch.autocast(
        device_type="cuda", dtype=torch.float16, enabled=use_amp
    ):
        logits = model(points, images)
        loss = criterion(logits, labels)
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite evaluation loss")
    metrics = batch_metrics(labels, logits, num_classes)
    metrics["loss"] = loss.detach().item()
    metrics["sample_count"] = int(labels.shape[0])
    return metrics


def save_checkpoint(
    path: Path,
    model: TSCMDLFusionModel,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    history: list[dict[str, object]],
    step: int,
    class_names: tuple[str, ...],
    args: argparse.Namespace,
) -> None:
    atomic_torch_save(
        path,
        {
            "run_type": "b4a_frozen_fusion_smoke",
            "epoch": step,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "history": history,
            "class_names": class_names,
            "args": json_args(args),
        },
    )


def main() -> None:
    args = parse_args()
    if args.steps < 2:
        raise ValueError("B4a restore validation requires at least two steps")
    if args.batch_size < 2:
        raise ValueError("Batch size must be at least two for BatchNorm")
    if args.workers < 0 or args.learning_rate <= 0:
        raise ValueError("workers cannot be negative and learning-rate must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for B4a fusion validation")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "train.log"
    set_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda")

    datasets = make_datasets(args.dataset_root)
    class_names = datasets["train"].class_names
    for split, dataset in datasets.items():
        expected = {"train": 420, "val": 90, "test": 90}[split]
        if len(dataset) != expected:
            raise ValueError(f"Unexpected {split} size: {len(dataset)} != {expected}")
        if dataset.class_names != class_names:
            raise ValueError(f"Class-name mismatch in split {split}")

    train_loader = DataLoader(
        datasets["train"],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        datasets["val"],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    test_loader = DataLoader(
        datasets["test"],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )

    model, load_details = build_model(
        class_names, args.b2_checkpoint, args.b3_checkpoint
    )
    model.image_encoder.to(memory_format=torch.channels_last)
    model.to(device)
    counts = parameter_counts(model)
    if any(parameter.requires_grad for parameter in model.point_encoder.parameters()):
        raise ValueError("Point backbone is not frozen")
    if any(parameter.requires_grad for parameter in model.image_encoder.parameters()):
        raise ValueError("Image backbone is not frozen")

    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.SGD(
        trainable_parameters,
        lr=args.learning_rate,
        momentum=0.9,
        weight_decay=2e-4,
    )
    scaler = torch.amp.GradScaler("cuda", init_scale=8.0, enabled=args.amp)
    criterion = nn.CrossEntropyLoss()
    config = {
        "args": {**json_args(args), "epochs": args.steps},
        "run_type": "b4a_frozen_fusion_smoke",
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "model": "TSCMDL 1024+1024 feature concatenation with frozen backbones",
        "class_names": list(class_names),
        "split_sizes": {split: len(dataset) for split, dataset in datasets.items()},
        "checkpoint_load": load_details,
        "parameter_counts": counts,
        "image_transform": "center 224x224 crop and ImageNet normalization",
        "point_input": "8192 XYZ points from the B1 WHU-STree protocol",
    }
    atomic_json(args.output_dir / "run_config.json", config)
    emit(json.dumps(config, ensure_ascii=False), log_path)

    history: list[dict[str, object]] = []
    train_iterator = iter(train_loader)
    fixed_val_batch = next(iter(val_loader))
    restore_verified = False
    backbone_gradient_free = True
    fusion_gradient_finite = True
    successful_steps = 0
    skipped_steps = 0
    feature_shapes: dict[str, list[int]] | None = None
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)

    for step in range(1, args.steps + 1):
        step_started = time.perf_counter()
        points, images, labels, _ = next(train_iterator)
        points = points.to(device, non_blocking=True).permute(0, 2, 1).contiguous()
        images = images.to(device, non_blocking=True).contiguous(
            memory_format=torch.channels_last
        )
        labels = labels.to(device, non_blocking=True)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=args.amp
        ):
            features = model.forward_features(points, images)
            logits = model.classifier(features["fused"])
            loss = criterion(logits, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss")
        if feature_shapes is None:
            feature_shapes = {
                name: list(tensor.shape) for name, tensor in features.items()
            }
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        backbone_gradients = [
            parameter.grad
            for encoder in (model.point_encoder, model.image_encoder)
            for parameter in encoder.parameters()
            if parameter.grad is not None
        ]
        backbone_gradient_free &= not backbone_gradients
        fusion_gradients = [
            parameter.grad
            for parameter in trainable_parameters
            if parameter.grad is not None
        ]
        if not fusion_gradients:
            raise FloatingPointError("Fusion head received no gradients")
        gradient_norm = torch.linalg.vector_norm(
            torch.stack(
                [
                    torch.linalg.vector_norm(gradient.detach().float())
                    for gradient in fusion_gradients
                ]
            )
        )
        fusion_gradient_finite &= bool(torch.isfinite(gradient_norm))
        previous_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.is_enabled() and scaler.get_scale() < previous_scale:
            skipped_steps += 1
        else:
            successful_steps += 1

        train_metrics = batch_metrics(labels, logits, len(class_names))
        val_metrics = evaluate_batch(
            model,
            fixed_val_batch,
            criterion,
            device,
            len(class_names),
            args.amp,
        )
        row = {
            "epoch": step,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": loss.detach().item(),
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "epoch_seconds": time.perf_counter() - step_started,
            "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / (1024**2),
        }
        history.append(row)
        atomic_json(args.output_dir / "training_history.json", history)
        save_checkpoint(
            args.output_dir / "last.pt",
            model,
            optimizer,
            scaler,
            history,
            step,
            class_names,
            args,
        )
        emit(
            f"step={step}/{args.steps} loss={loss.detach().item():.4f} "
            f"train_acc={100*float(train_metrics['accuracy']):.2f}% "
            f"val_acc={100*float(val_metrics['accuracy']):.2f}% "
            f"grad_norm={float(gradient_norm):.4f}",
            log_path,
        )

        if step == 1:
            probe_name = next(
                name
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            )
            expected = model.state_dict()[probe_name].detach().cpu().clone()
            with torch.no_grad():
                dict(model.named_parameters())[probe_name].add_(1.0)
            checkpoint = torch.load(
                args.output_dir / "last.pt",
                map_location="cpu",
                weights_only=False,
            )
            model.load_state_dict(checkpoint["model_state"], strict=True)
            optimizer = torch.optim.SGD(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                lr=args.learning_rate,
                momentum=0.9,
                weight_decay=2e-4,
            )
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scaler.load_state_dict(checkpoint["scaler_state"])
            history = checkpoint["history"]
            restored = model.state_dict()[probe_name].detach().cpu()
            restore_verified = torch.equal(restored, expected)
            del checkpoint, restored, expected
            gc.collect()
            if not restore_verified:
                raise ValueError("Checkpoint restore did not recover the fusion state")

    final_val = evaluate_batch(
        model,
        fixed_val_batch,
        criterion,
        device,
        len(class_names),
        args.amp,
    )
    final_test = evaluate_batch(
        model,
        next(iter(test_loader)),
        criterion,
        device,
        len(class_names),
        args.amp,
    )
    final = {
        "status": "complete",
        "run_type": "b4a_frozen_fusion_smoke",
        "best_epoch": max(
            history,
            key=lambda item: (
                float(item["val_macro_f1"]),
                float(item["val_accuracy"]),
            ),
        )["epoch"],
        "epochs_completed": len(history),
        "metrics": {"val": final_val, "test": final_test},
        "class_names": list(class_names),
        "note": "Metrics cover one fixed batch per split and are not performance results.",
    }
    atomic_json(args.output_dir / "final_metrics.json", final)
    result = {
        "status": "passed",
        "run_type": "b4a_frozen_fusion_smoke",
        "dataset_alignment": {
            split: {
                "sample_count": len(dataset),
                "first_sample_key": dataset.sample_keys[0],
                "last_sample_key": dataset.sample_keys[-1],
            }
            for split, dataset in datasets.items()
        },
        "checkpoint_load": load_details,
        "source_checkpoint_sha256": {
            "b2_best": sha256_file(args.b2_checkpoint),
            "b3_best": sha256_file(args.b3_checkpoint),
        },
        "parameter_counts": counts,
        "feature_shapes": feature_shapes,
        "backbones_frozen": True,
        "backbone_gradient_free": backbone_gradient_free,
        "fusion_gradient_finite": fusion_gradient_finite,
        "checkpoint_restore_verified": restore_verified,
        "successful_optimizer_steps": successful_steps,
        "skipped_optimizer_steps": skipped_steps,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / (1024**2),
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / (1024**2),
        "elapsed_seconds": time.perf_counter() - started,
        "metrics_are_smoke_only": True,
    }
    atomic_json(args.output_dir / "smoke_result.json", result)
    emit(json.dumps(result, ensure_ascii=False, indent=2), log_path)


if __name__ == "__main__":
    main()
