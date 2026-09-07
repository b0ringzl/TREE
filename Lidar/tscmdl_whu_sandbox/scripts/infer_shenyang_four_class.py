"""Run the frozen D1 four-class ensemble on WHU-STree Shenyang instances.

The public Shenyang package provides per-point tree-instance annotations but no
species labels.  This script therefore produces external-domain predictions,
confidence, modal agreement and input diagnostics; it deliberately does not
report classification accuracy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import Counter, OrderedDict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn
from torchvision.models import resnet50
from torchvision.transforms import v2


SANDBOX_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SANDBOX_ROOT / "src"))

from tscmdl_whu import (  # noqa: E402
    PointTransformerV2Classifier,
    TSCMDLFusionModel,
    knn_query_packed,
    nearest_poses,
    normalize_unit_sphere,
    parse_ply_header,
    project_equirectangular,
    projection_crop_bounds,
    read_trajectory_csv,
    stable_sample_seed,
    uniform_sample_indices,
)

from cache_d1_fusion_features import ptv2_forward_features  # noqa: E402
from prepare_a4_pairing_preview import draw_overlay, wrapped_crop  # noqa: E402
from prepare_d1_rework_candidates import image_metrics, quality_risk  # noqa: E402


DATA_ROOT = PROJECT_ROOT / "lidar data" / "whu"
DERIVED_ROOT = DATA_ROOT / "derived" / "tscmdl"
POINT_SUITE = DERIVED_ROOT / "d1_ptv2_clean_repeats" / "20260818_protocol_v1"
IMAGE_SUITE = DERIVED_ROOT / "d1_resnet50_clean_repeats" / "20260817_protocol_v1"
FUSION_SUITE = DERIVED_ROOT / "d1_fusion_clean_repeats" / "20260818_protocol_v1"
SEEDS = (20260728, 20260729, 20260730)
BASE_SAMPLE_SEED = 20260823
TARGET_POINTS = 8192
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root", type=Path, default=DATA_ROOT / "whu-stree-sy"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DERIVED_ROOT
        / "e2_shenyang_four_class_inference"
        / "20260823_v1",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--no-overlays", action="store_true")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def atomic_json(path: Path, value: object) -> None:
    temporary = Path(f"{path}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def stable_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


class PanoramaCache:
    def __init__(self, capacity: int = 4) -> None:
        self.capacity = capacity
        self.values: OrderedDict[Path, Image.Image] = OrderedDict()

    def get(self, path: Path) -> Image.Image:
        image = self.values.pop(path, None)
        if image is None:
            with Image.open(path) as source:
                image = source.convert("RGB")
        self.values[path] = image
        while len(self.values) > self.capacity:
            _, evicted = self.values.popitem(last=False)
            evicted.close()
        return image


def discover_trajectories(dataset_root: Path) -> list[dict[str, Path | str]]:
    trajectories = []
    for label_path in sorted((dataset_root / "reference_data").glob("*.npy")):
        road_id, trajectory_id = label_path.stem.split("_", 1)
        ply_path = dataset_root / road_id / "PCD" / f"{trajectory_id}.ply"
        trajectory_csv = (
            dataset_root / road_id / "hdi" / trajectory_id / "traj.csv"
        )
        image_dir = dataset_root / road_id / "image" / trajectory_id
        for required in (ply_path, trajectory_csv, image_dir):
            if not required.exists():
                raise FileNotFoundError(required)
        trajectories.append(
            {
                "road_id": road_id,
                "trajectory_id": trajectory_id,
                "label_path": label_path,
                "ply_path": ply_path,
                "trajectory_csv": trajectory_csv,
                "image_dir": image_dir,
            }
        )
    if not trajectories:
        raise FileNotFoundError(f"No Shenyang reference_data found in {dataset_root}")
    return trajectories


def memmap_xyz(ply_path: Path) -> tuple[np.ndarray, object]:
    header = parse_ply_header(ply_path)
    names = set(header.dtype.names or ())
    if not {"x", "y", "z"}.issubset(names):
        raise ValueError(f"PLY has no XYZ coordinates: {ply_path}")
    records = np.memmap(
        ply_path,
        dtype=header.dtype,
        mode="r",
        offset=header.data_offset,
        shape=(header.vertex_count,),
    )
    return records, header


def transform_image(image: Image.Image) -> torch.Tensor:
    transform = v2.Compose(
        [
            v2.CenterCrop((224, 224)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return transform(image)


def quality_tier(reasons: list[str]) -> str:
    return "risk" if reasons else "normal"


def prepare_instances(args: argparse.Namespace) -> tuple[list[dict], Path]:
    output_dir = args.output_dir.resolve()
    image_root = output_dir / "assets" / "images"
    overlay_root = output_dir / "assets" / "overlays"
    point_root = output_dir / "assets" / "points"
    for path in (image_root, point_root):
        path.mkdir(parents=True, exist_ok=True)
    if not args.no_overlays:
        overlay_root.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        records = payload["records"]
        expected = args.limit if args.limit is not None else 2664
        if len(records) != expected:
            raise ValueError(
                f"Existing manifest has {len(records)} records, expected {expected}: {manifest_path}"
            )
        print(f"Using existing prepared manifest: {manifest_path}", flush=True)
        return records, manifest_path

    progress_path = output_dir / "progress.json"
    records: list[dict] = []
    panorama_cache = PanoramaCache()
    started = time.perf_counter()
    stop = False
    for trajectory in discover_trajectories(args.dataset_root.resolve()):
        road_id = str(trajectory["road_id"])
        trajectory_id = str(trajectory["trajectory_id"])
        ply_path = Path(trajectory["ply_path"])
        label_path = Path(trajectory["label_path"])
        trajectory_csv = Path(trajectory["trajectory_csv"])
        image_dir = Path(trajectory["image_dir"])
        ply, header = memmap_xyz(ply_path)
        labels = np.load(label_path, mmap_mode="r", allow_pickle=False)
        if labels.ndim != 1 or len(labels) != header.vertex_count:
            raise ValueError(
                f"Reference length mismatch: {label_path} {labels.shape} != {header.vertex_count}"
            )

        foreground = np.flatnonzero(labels >= 0)
        order = np.argsort(labels[foreground], kind="stable")
        ordered_indices = foreground[order]
        ordered_labels = np.asarray(labels[ordered_indices])
        tree_ids, starts, counts = np.unique(
            ordered_labels, return_index=True, return_counts=True
        )
        poses = read_trajectory_csv(trajectory_csv)
        print(
            f"Preparing road={road_id} trajectory={trajectory_id}: {len(tree_ids)} trees",
            flush=True,
        )
        for tree_id_value, start, count in zip(tree_ids, starts, counts):
            if args.limit is not None and len(records) >= args.limit:
                stop = True
                break
            tree_id = int(tree_id_value)
            sample_key = f"SY_{road_id}_{trajectory_id}_{tree_id}"
            source_indices = ordered_indices[int(start) : int(start + count)]
            sample_seed = stable_sample_seed(sample_key, BASE_SAMPLE_SEED)
            local_indices = uniform_sample_indices(
                int(count), TARGET_POINTS, sample_seed
            )
            chosen = source_indices[local_indices]
            raw_points = np.empty((TARGET_POINTS, 3), dtype=np.float64)
            for axis, field in enumerate(("x", "y", "z")):
                raw_points[:, axis] = ply[field][chosen]
            normalized, transform = normalize_unit_sphere(raw_points)

            point_rel = Path("assets") / "points" / f"{sample_key}.npz"
            np.savez_compressed(
                output_dir / point_rel,
                points_xyz=normalized,
                centroid_xyz=np.asarray(transform.centroid, dtype=np.float64),
                scale=np.float64(transform.scale),
                source_point_count=np.int64(count),
                sampled_unique_point_count=np.int64(len(np.unique(local_indices))),
                sample_seed=np.uint32(sample_seed),
            )

            center = raw_points.mean(axis=0)
            pose, center_distance = nearest_poses(center, poses, count=1)[0]
            image_path = image_dir / pose.image_name
            panorama = panorama_cache.get(image_path)
            u, v, distances = project_equirectangular(
                raw_points, pose, panorama.width, panorama.height, apply_tilt=True
            )
            bounds = projection_crop_bounds(
                u, v, panorama.width, panorama.height, margin=1.25, minimum_size=256
            )
            source_crop = wrapped_crop(
                panorama,
                bounds.left_unwrapped_px,
                bounds.top_px,
                bounds.width_px,
                bounds.height_px,
            )
            raw_crop = source_crop.resize((768, 512), Image.Resampling.LANCZOS)
            image_rel = Path("assets") / "images" / f"{sample_key}.jpg"
            raw_crop.save(output_dir / image_rel, quality=args.jpeg_quality)

            overlay_rel = None
            visible_count = 0
            if not args.no_overlays:
                overlay, visible_count = draw_overlay(
                    source_crop,
                    u,
                    v,
                    bounds.left_unwrapped_px,
                    bounds.top_px,
                    panorama.width,
                    bounds.width_px,
                    bounds.height_px,
                    f"{sample_key} | {center_distance:.1f} m",
                )
                overlay_rel = Path("assets") / "overlays" / f"{sample_key}.jpg"
                overlay.save(output_dir / overlay_rel, quality=args.jpeg_quality)
            metrics = image_metrics(raw_crop)
            risk_score, risk_reasons = quality_risk(metrics)
            record = {
                "sample_key": sample_key,
                "dataset": "whu_stree_shenyang",
                "road_id": road_id,
                "trajectory_id": trajectory_id,
                "tree_id": tree_id,
                "species_ground_truth_available": False,
                "source_point_count": int(count),
                "sampled_unique_point_count": int(len(np.unique(local_indices))),
                "point_path": point_rel.as_posix(),
                "image_path": image_rel.as_posix(),
                "overlay_path": overlay_rel.as_posix() if overlay_rel else None,
                "source_panorama": pose.image_name,
                "camera_distance_m": round(float(center_distance), 4),
                "point_distance_range_m": [
                    round(float(distances.min()), 4),
                    round(float(distances.max()), 4),
                ],
                "visible_projected_point_count": int(visible_count),
                "visible_projected_point_ratio": round(
                    float(visible_count / TARGET_POINTS), 6
                ),
                "projection_status": "coarse_public_pose_projection_without_exact_camera_extrinsics",
                "automatic_quality_tier": quality_tier(risk_reasons),
                "automatic_risk_score": risk_score,
                "automatic_risk_reasons": risk_reasons,
                "image_metrics": metrics,
            }
            records.append(record)
            if len(records) == 1 or len(records) % 25 == 0:
                elapsed = time.perf_counter() - started
                atomic_json(
                    progress_path,
                    {
                        "stage": "preprocessing",
                        "completed": len(records),
                        "elapsed_seconds": round(elapsed, 2),
                        "rate_trees_per_second": round(len(records) / elapsed, 3),
                        "updated_at": timestamp(),
                    },
                )
                print(
                    f"  prepared {len(records)} trees ({len(records) / elapsed:.2f}/s)",
                    flush=True,
                )
        del ordered_labels, ordered_indices, order, foreground, labels, ply
        if stop:
            break

    manifest = {
        "schema_version": 1,
        "stage": "external_domain_inference_prepared",
        "dataset": "WHU-STree Shenyang",
        "dataset_root": str(args.dataset_root.resolve()),
        "species_ground_truth_available": False,
        "accuracy_computable": False,
        "instance_annotation_role": "tree-instance segmentation ground truth",
        "classification_scope": "forced closed-set four-class prediction; not an accuracy claim",
        "projection_status": "coarse public-pose projection; exact camera lever arm/extrinsics unavailable",
        "target_point_count": TARGET_POINTS,
        "record_count": len(records),
        "created_at": timestamp(),
        "records": records,
    }
    atomic_json(manifest_path, manifest)
    atomic_json(
        progress_path,
        {
            "stage": "preprocessing_complete",
            "completed": len(records),
            "elapsed_seconds": round(time.perf_counter() - started, 2),
            "updated_at": timestamp(),
        },
    )
    return records, manifest_path


def build_knn_cache(
    output_dir: Path, records: list[dict], device: torch.device
) -> Path:
    path = output_dir / "knn_reference_index.npy"
    expected_shape = (len(records), TARGET_POINTS, 8)
    if path.exists():
        existing = np.load(path, mmap_mode="r", allow_pickle=False)
        if tuple(existing.shape) != expected_shape or existing.dtype != np.int32:
            raise ValueError(f"Invalid existing kNN cache: {path}")
        print(f"Using existing kNN cache: {path}", flush=True)
        return path
    temporary = Path(f"{path}.tmp.npy")
    cache = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.int32, shape=expected_shape
    )
    progress_path = output_dir / "progress.json"
    started = time.perf_counter()
    for index, record in enumerate(records):
        with np.load(output_dir / record["point_path"], allow_pickle=False) as archive:
            points_np = np.asarray(archive["points_xyz"], dtype=np.float32)
        points = torch.from_numpy(points_np).to(device)
        offset = torch.tensor([TARGET_POINTS], dtype=torch.long, device=device)
        reference = knn_query_packed(points, offset, 8)
        cache[index] = reference.cpu().numpy().astype(np.int32, copy=False)
        if index == 0 or (index + 1) % 25 == 0:
            cache.flush()
            elapsed = time.perf_counter() - started
            atomic_json(
                progress_path,
                {
                    "stage": "knn_cache",
                    "completed": index + 1,
                    "total": len(records),
                    "elapsed_seconds": round(elapsed, 2),
                    "updated_at": timestamp(),
                },
            )
            print(f"  kNN {index + 1}/{len(records)}", flush=True)
    cache.flush()
    del cache
    temporary.replace(path)
    return path


def load_seed_models(seed: int, class_names: tuple[str, ...], device: torch.device):
    point_path = POINT_SUITE / "runs" / f"seed_{seed}" / "best.pt"
    image_path = IMAGE_SUITE / "runs" / f"seed_{seed}" / "best.pt"
    fusion_path = FUSION_SUITE / "runs" / f"seed_{seed}" / "best.pt"
    point_checkpoint = torch.load(point_path, map_location="cpu", weights_only=False)
    image_checkpoint = torch.load(image_path, map_location="cpu", weights_only=False)
    fusion_checkpoint = torch.load(fusion_path, map_location="cpu", weights_only=False)
    for name, checkpoint in (
        ("point", point_checkpoint),
        ("image", image_checkpoint),
        ("fusion", fusion_checkpoint),
    ):
        if tuple(checkpoint["class_names"]) != class_names:
            raise ValueError(f"{name} checkpoint class order mismatch for seed {seed}")

    point_config = dict(point_checkpoint["model_config"])
    point_config.pop("num_classes", None)
    point_model = PointTransformerV2Classifier(
        len(class_names), **point_config
    )
    point_model.load_state_dict(point_checkpoint["model_state"], strict=True)

    image_model = resnet50(weights=None)
    image_model.fc = nn.Linear(image_model.fc.in_features, len(class_names))
    image_model.load_state_dict(image_checkpoint["model_state"], strict=True)
    image_classifier = image_model.fc
    image_model.fc = nn.Identity()

    fusion_args = fusion_checkpoint["args"]
    fusion_model = TSCMDLFusionModel(
        nn.Identity(),
        nn.Identity(),
        len(class_names),
        point_dim=384,
        image_dim=2048,
        modal_dim=1024,
        dropout=float(fusion_args["dropout"]),
        freeze_backbones=True,
        normalization=str(fusion_args["normalization"]),
        classifier_hidden_dims=tuple(fusion_args["classifier_hidden_dims"]),
        modality="fusion",
    )
    fusion_model.load_state_dict(fusion_checkpoint["model_state"], strict=True)

    point_model.to(device).eval()
    image_model.to(device, memory_format=torch.channels_last).eval()
    image_classifier.to(device).eval()
    fusion_model.to(device).eval()
    return point_model, image_model, image_classifier, fusion_model, {
        "point": stable_hash(point_path),
        "image": stable_hash(image_path),
        "fusion": stable_hash(fusion_path),
    }


def infer(
    args: argparse.Namespace,
    records: list[dict],
    manifest_path: Path,
    knn_path: Path,
    device: torch.device,
) -> Path:
    output_dir = args.output_dir.resolve()
    predictions_path = output_dir / "predictions.csv"
    if predictions_path.exists():
        print(f"Using existing predictions: {predictions_path}", flush=True)
        return predictions_path
    class_names = (
        "Cinnamomum camphora",
        "Lagerstroemia indica",
        "Magnolia grandiflora",
        "Other",
    )
    probability_sums = {
        name: np.zeros((len(records), len(class_names)), dtype=np.float64)
        for name in ("image", "point", "fusion")
    }
    knn = np.load(knn_path, mmap_mode="r", allow_pickle=False)
    image_transform = v2.Compose(
        [
            v2.CenterCrop((224, 224)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    progress_path = output_dir / "progress.json"
    checkpoint_hashes: dict[str, dict[str, str]] = {}
    started = time.perf_counter()
    for seed_index, seed in enumerate(SEEDS, start=1):
        print(f"Loading seed {seed} models", flush=True)
        point_model, image_model, image_classifier, fusion_model, hashes = (
            load_seed_models(seed, class_names, device)
        )
        checkpoint_hashes[str(seed)] = hashes
        for batch_start in range(0, len(records), args.batch_size):
            batch_records = records[batch_start : batch_start + args.batch_size]
            points_list = []
            images_list = []
            for record in batch_records:
                with np.load(
                    output_dir / record["point_path"], allow_pickle=False
                ) as archive:
                    points_list.append(
                        np.asarray(archive["points_xyz"], dtype=np.float32)
                    )
                with Image.open(output_dir / record["image_path"]) as image:
                    images_list.append(image_transform(image.convert("RGB")))
            batch_stop = batch_start + len(batch_records)
            points = torch.from_numpy(np.stack(points_list)).to(
                device, non_blocking=True
            )
            references = torch.from_numpy(
                np.asarray(knn[batch_start:batch_stop], dtype=np.int64)
            ).to(device, non_blocking=True)
            images = torch.stack(images_list).to(
                device,
                non_blocking=True,
                memory_format=torch.channels_last,
            )
            with torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=args.amp
            ):
                point_features = ptv2_forward_features(
                    point_model, points, references
                )
                point_logits = point_model.classifier(point_features)
                image_features = image_model(images)
                image_logits = image_classifier(image_features)
                fusion_logits = fusion_model.forward_from_features(
                    point_features, image_features
                )
            for name, logits in (
                ("image", image_logits),
                ("point", point_logits),
                ("fusion", fusion_logits),
            ):
                probability_sums[name][batch_start:batch_stop] += (
                    torch.softmax(logits.float(), dim=1).cpu().numpy()
                )
            completed = (seed_index - 1) * len(records) + batch_stop
            total = len(SEEDS) * len(records)
            if batch_start == 0 or batch_stop == len(records) or completed % 100 < args.batch_size:
                atomic_json(
                    progress_path,
                    {
                        "stage": "ensemble_inference",
                        "seed": seed,
                        "completed_model_samples": completed,
                        "total_model_samples": total,
                        "elapsed_seconds": round(time.perf_counter() - started, 2),
                        "updated_at": timestamp(),
                    },
                )
                print(
                    f"  seed {seed}: {batch_stop}/{len(records)} trees",
                    flush=True,
                )
        del point_model, image_model, image_classifier, fusion_model
        torch.cuda.empty_cache()

    probabilities = {name: values / len(SEEDS) for name, values in probability_sums.items()}
    fieldnames = [
        "sample_key",
        "dataset",
        "road_id",
        "trajectory_id",
        "tree_id",
        "species_ground_truth_available",
        "relationship",
    ]
    for modality in ("image", "point", "fusion"):
        fieldnames.extend(
            [
                f"{modality}_predicted_class",
                f"{modality}_class_name",
                f"{modality}_confidence",
                *[
                    f"{modality}_probability_{index}"
                    for index in range(len(class_names))
                ],
            ]
        )
    with predictions_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for index, record in enumerate(records):
            predicted = {
                modality: int(np.argmax(probabilities[modality][index]))
                for modality in probabilities
            }
            relationship = (
                "all_agree"
                if len(set(predicted.values())) == 1
                else "disagreement"
            )
            row = {
                "sample_key": record["sample_key"],
                "dataset": record["dataset"],
                "road_id": record["road_id"],
                "trajectory_id": record["trajectory_id"],
                "tree_id": record["tree_id"],
                "species_ground_truth_available": False,
                "relationship": relationship,
            }
            for modality in ("image", "point", "fusion"):
                class_index = predicted[modality]
                row[f"{modality}_predicted_class"] = class_index
                row[f"{modality}_class_name"] = class_names[class_index]
                row[f"{modality}_confidence"] = f"{probabilities[modality][index, class_index]:.9f}"
                for candidate in range(len(class_names)):
                    row[f"{modality}_probability_{candidate}"] = (
                        f"{probabilities[modality][index, candidate]:.9f}"
                    )
            writer.writerow(row)

    distributions = {}
    for modality in ("image", "point", "fusion"):
        predicted = np.argmax(probabilities[modality], axis=1)
        distributions[modality] = {
            class_names[index]: int(np.count_nonzero(predicted == index))
            for index in range(len(class_names))
        }
    agreement = np.stack(
        [np.argmax(probabilities[name], axis=1) for name in ("image", "point", "fusion")],
        axis=1,
    )
    validation = {
        "stage": "external_domain_inference_complete",
        "dataset": "WHU-STree Shenyang",
        "record_count": len(records),
        "species_ground_truth_available": False,
        "accuracy_computable": False,
        "accuracy_reason": "The local Shenyang package has instance segmentation labels but no species labels.",
        "classification_scope": "forced closed-set four-class predictions",
        "seeds": list(SEEDS),
        "class_names": list(class_names),
        "model_checkpoint_sha256": checkpoint_hashes,
        "manifest_sha256": stable_hash(manifest_path),
        "knn_cache_shape": list(knn.shape),
        "prediction_distribution": distributions,
        "three_modal_agreement_count": int(
            np.count_nonzero(
                (agreement[:, 0] == agreement[:, 1])
                & (agreement[:, 1] == agreement[:, 2])
            )
        ),
        "probability_checks": {
            name: {
                "finite": bool(np.isfinite(values).all()),
                "max_sum_error": float(
                    np.max(np.abs(values.sum(axis=1) - 1.0))
                ),
            }
            for name, values in probabilities.items()
        },
        "completed_at": timestamp(),
    }
    atomic_json(output_dir / "validation.json", validation)
    atomic_json(
        progress_path,
        {
            "stage": "complete",
            "completed": len(records),
            "validation_path": str(output_dir / "validation.json"),
            "updated_at": timestamp(),
        },
    )
    print(json.dumps(validation, ensure_ascii=False, indent=2), flush=True)
    return predictions_path


def main() -> int:
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the accepted PTv2 inference protocol")
    device = torch.device("cuda")
    print(
        f"Device: {torch.cuda.get_device_name(device)}; output={args.output_dir.resolve()}",
        flush=True,
    )
    records, manifest_path = prepare_instances(args)
    knn_path = build_knn_cache(args.output_dir.resolve(), records, device)
    infer(args, records, manifest_path, knn_path, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
