"""Run validation-only B4c selection, then locked three-seed modality ablations."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader


SANDBOX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SANDBOX_ROOT / "src"))
sys.path.insert(0, str(SANDBOX_ROOT / "scripts"))

from train_b4_tscmdl import (  # noqa: E402
    CachedFeatureDataset,
    atomic_json,
    atomic_torch_save,
    make_model,
    plot_results,
    run_epoch,
    set_seed,
    sha256_file,
)


CANDIDATES = (
    {
        "name": "original",
        "hidden_dims": [512, 256],
        "dropout": 0.5,
        "learning_rate": 0.01,
        "weight_decay": 2e-4,
    },
    {
        "name": "compact",
        "hidden_dims": [256, 64],
        "dropout": 0.5,
        "learning_rate": 0.005,
        "weight_decay": 1e-3,
    },
    {
        "name": "small_strong",
        "hidden_dims": [128],
        "dropout": 0.6,
        "learning_rate": 0.003,
        "weight_decay": 2e-3,
    },
)
MODALITIES = ("fusion", "image", "point")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=(20260727, 20260728, 20260729),
    )
    parser.add_argument("--workers", type=int, default=0)
    return parser.parse_args()


def json_args(args: argparse.Namespace) -> dict[str, object]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def emit(message: str, log_path: Path) -> None:
    print(message, flush=True)
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(message + "\n")


def make_trial_loaders(
    train_dataset: CachedFeatureDataset,
    val_dataset: CachedFeatureDataset,
    *,
    batch_size: int,
    eval_batch_size: int,
    workers: int,
    seed: int,
    test_dataset: CachedFeatureDataset | None = None,
) -> dict[str, DataLoader]:
    generator = torch.Generator().manual_seed(seed)
    loaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            pin_memory=True,
            generator=generator,
            persistent_workers=workers > 0,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=True,
            persistent_workers=workers > 0,
        ),
    }
    if test_dataset is not None:
        loaders["test"] = DataLoader(
            test_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=True,
            persistent_workers=workers > 0,
        )
    return loaders


def state_to_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def train_trial(
    *,
    candidate: dict[str, object],
    modality: str,
    seed: int,
    train_dataset: CachedFeatureDataset,
    val_dataset: CachedFeatureDataset,
    test_dataset: CachedFeatureDataset | None,
    epochs: int,
    patience: int,
    batch_size: int,
    eval_batch_size: int,
    workers: int,
    device: torch.device,
) -> tuple[dict[str, object], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    set_seed(seed)
    loaders = make_trial_loaders(
        train_dataset,
        val_dataset,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        workers=workers,
        seed=seed,
        test_dataset=test_dataset,
    )
    model = make_model(
        3,
        float(candidate["dropout"]),
        "l2",
        tuple(int(value) for value in candidate["hidden_dims"]),
        modality,
    ).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(candidate["learning_rate"]),
        momentum=0.9,
        weight_decay=float(candidate["weight_decay"]),
        fused=True,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    criterion = nn.CrossEntropyLoss()
    history = []
    best_epoch = 0
    best_score = (-1.0, -1.0)
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    started = time.perf_counter()

    for epoch_index in range(epochs):
        epoch_number = epoch_index + 1
        learning_rate = optimizer.param_groups[0]["lr"]
        epoch_started = time.perf_counter()
        train_metrics, _ = run_epoch(
            model,
            loaders["train"],
            criterion,
            device,
            3,
            optimizer=optimizer,
            scaler=scaler,
            use_amp=False,
            progress_label="train",
            show_progress=False,
        )
        val_metrics, _ = run_epoch(
            model,
            loaders["val"],
            criterion,
            device,
            3,
            use_amp=False,
            progress_label="val",
            show_progress=False,
        )
        row = {
            "epoch": epoch_number,
            "learning_rate": learning_rate,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "epoch_seconds": time.perf_counter() - epoch_started,
            "peak_allocated_mib": max(
                float(train_metrics["peak_allocated_mib"]),
                float(val_metrics["peak_allocated_mib"]),
            ),
        }
        history.append(row)
        score = (
            float(val_metrics["macro_f1"]),
            float(val_metrics["accuracy"]),
        )
        if score > best_score:
            best_score = score
            best_epoch = epoch_number
            best_state = state_to_cpu(model)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        scheduler.step()
        if epochs_without_improvement >= patience:
            break

    if best_state is None:
        raise RuntimeError("Trial did not create a best state")
    last_state = state_to_cpu(model)
    model.load_state_dict(best_state, strict=True)
    val_metrics, val_predictions = run_epoch(
        model,
        loaders["val"],
        criterion,
        device,
        3,
        use_amp=False,
        progress_label="final val",
        collect_predictions=test_dataset is not None,
        show_progress=False,
    )
    metrics = {"val": val_metrics}
    predictions = {"val": val_predictions} if test_dataset is not None else {}
    if test_dataset is not None:
        test_metrics, test_predictions = run_epoch(
            model,
            loaders["test"],
            criterion,
            device,
            3,
            use_amp=False,
            progress_label="final test",
            collect_predictions=True,
            show_progress=False,
        )
        metrics["test"] = test_metrics
        predictions["test"] = test_predictions

    result = {
        "candidate": str(candidate["name"]),
        "modality": modality,
        "seed": seed,
        "parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "best_validation_score": {
            "macro_f1": best_score[0],
            "accuracy": best_score[1],
        },
        "metrics": metrics,
        "history": history,
        "predictions": predictions,
        "elapsed_seconds": time.perf_counter() - started,
    }
    return result, best_state, last_state


def metric_summary(
    runs: list[dict[str, object]],
    split: str,
) -> dict[str, float]:
    output = {}
    for metric in ("accuracy", "macro_f1"):
        values = [float(run["metrics"][split][metric]) for run in runs]
        output[f"{metric}_mean"] = mean(values)
        output[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
        output[f"{metric}_min"] = min(values)
        output[f"{metric}_max"] = max(values)
    return output


def attach_predictions(
    trial: dict[str, object],
    datasets: dict[str, CachedFeatureDataset],
) -> list[dict[str, object]]:
    rows = []
    for split, predictions in trial["predictions"].items():
        dataset = datasets[split]
        for prediction in predictions:
            index = int(prediction["dataset_index"])
            rows.append(
                {
                    "modality": trial["modality"],
                    "seed": trial["seed"],
                    "sample_key": dataset.sample_keys[index],
                    "split": split,
                    "true_class": prediction["true_class"],
                    "predicted_class": prediction["predicted_class"],
                    "correct": int(
                        prediction["true_class"]
                        == prediction["predicted_class"]
                    ),
                    **{
                        f"probability_{class_index}": probability
                        for class_index, probability in enumerate(
                            prediction["probabilities"]
                        )
                    },
                }
            )
    return rows


def save_predictions(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "modality",
        "seed",
        "sample_key",
        "split",
        "true_class",
        "predicted_class",
        "correct",
        "probability_0",
        "probability_1",
        "probability_2",
    ]
    temporary = Path(f"{path}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def save_selected_history(
    output_dir: Path, history: list[dict[str, object]]
) -> None:
    atomic_json(output_dir / "training_history.json", history)
    fields = list(history[0])
    temporary = output_dir / "training_history.csv.tmp"
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)
    temporary.replace(output_dir / "training_history.csv")


def plot_suite(
    output_dir: Path,
    candidate_summaries: list[dict[str, object]],
    ablation_summaries: list[dict[str, object]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [item["candidate"] for item in candidate_summaries]
    values = [100 * float(item["val_macro_f1_mean"]) for item in candidate_summaries]
    errors = [100 * float(item["val_macro_f1_std"]) for item in candidate_summaries]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.bar(labels, values, yerr=errors, capsize=6, color=("#2E7D32", "#1565C0", "#D84315"))
    axis.set(ylabel="Validation macro F1 (%)", title="B4c regularization selection (3 seeds)")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "regularization_validation.png", dpi=180)
    plt.close(figure)

    modalities = [item["modality"] for item in ablation_summaries]
    x = np.arange(len(modalities))
    width = 0.36
    val_values = [
        100 * float(item["val_macro_f1_mean"]) for item in ablation_summaries
    ]
    test_values = [
        100 * float(item["test_macro_f1_mean"]) for item in ablation_summaries
    ]
    val_errors = [
        100 * float(item["val_macro_f1_std"]) for item in ablation_summaries
    ]
    test_errors = [
        100 * float(item["test_macro_f1_std"]) for item in ablation_summaries
    ]
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(
        x - width / 2,
        val_values,
        width,
        yerr=val_errors,
        capsize=5,
        label="val",
        color="#00897B",
    )
    axis.bar(
        x + width / 2,
        test_values,
        width,
        yerr=test_errors,
        capsize=5,
        label="test",
        color="#EF6C00",
    )
    axis.set_xticks(x, modalities)
    axis.set(ylabel="Macro F1 (%)", title="Locked-config modality ablation (3 seeds)")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "modality_ablation.png", dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if (
        args.epochs <= 0
        or args.patience <= 0
        or args.batch_size <= 1
        or args.eval_batch_size <= 0
        or args.workers < 0
    ):
        raise ValueError("Invalid B4c training arguments")
    if len(args.seeds) != 3 or len(set(args.seeds)) != 3:
        raise ValueError("B4c requires exactly three unique seeds")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for B4c")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output_dir / "checkpoints"
    checkpoint_dir.mkdir()
    log_path = args.output_dir / "train.log"

    set_seed(args.seeds[0])
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    feature_cache_sha256 = sha256_file(args.feature_cache)
    cache = torch.load(args.feature_cache, map_location="cpu", weights_only=False)
    class_names = tuple(str(value) for value in cache["class_names"])
    if len(class_names) != 3:
        raise ValueError("B4c expects three classes")

    # The tuning phase instantiates only train and val datasets.
    train_dataset = CachedFeatureDataset(cache, "train")
    val_dataset = CachedFeatureDataset(cache, "val")
    config = {
        "args": json_args(args),
        "run_type": "b4c_regularization_ablation",
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "model": "B4c validation-locked regularization and modality ablation",
        "class_names": list(class_names),
        "split_sizes": {"train": len(train_dataset), "val": len(val_dataset), "test": 90},
        "feature_cache": {
            "path": str(args.feature_cache.resolve()),
            "sha256": feature_cache_sha256,
            "source_checkpoint_sha256": cache["source_checkpoint_sha256"],
        },
        "candidates": list(CANDIDATES),
        "modalities": list(MODALITIES),
        "selection_metric": "mean validation macro F1 across three seeds, then mean accuracy",
        "test_access_policy": (
            "No test dataset is instantiated during candidate selection. "
            "locked_config.json is written before test evaluation."
        ),
    }
    atomic_json(args.output_dir / "run_config.json", config)
    emit(json.dumps(config, ensure_ascii=False), log_path)
    started = time.perf_counter()

    tuning_runs = []
    total_tuning = len(CANDIDATES) * len(args.seeds)
    for candidate_index, candidate in enumerate(CANDIDATES):
        for seed_index, seed in enumerate(args.seeds):
            ordinal = candidate_index * len(args.seeds) + seed_index + 1
            emit(
                f"Tuning {ordinal}/{total_tuning}: {candidate['name']} seed={seed}",
                log_path,
            )
            trial, _, _ = train_trial(
                candidate=candidate,
                modality="fusion",
                seed=seed,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                test_dataset=None,
                epochs=args.epochs,
                patience=args.patience,
                batch_size=args.batch_size,
                eval_batch_size=args.eval_batch_size,
                workers=args.workers,
                device=device,
            )
            if set(trial["metrics"]) != {"val"} or trial["predictions"]:
                raise RuntimeError("Tuning trial accessed non-validation outputs")
            tuning_runs.append(trial)
            atomic_json(
                args.output_dir / "suite_progress.json",
                {
                    "phase": "validation_selection",
                    "completed": ordinal,
                    "total": total_tuning,
                    "latest": {
                        "candidate": candidate["name"],
                        "seed": seed,
                        "val_macro_f1": trial["metrics"]["val"]["macro_f1"],
                    },
                },
            )

    candidate_summaries = []
    for candidate in CANDIDATES:
        runs = [
            run for run in tuning_runs if run["candidate"] == candidate["name"]
        ]
        summary = {
            "candidate": candidate["name"],
            "config": candidate,
            "parameter_count": int(runs[0]["parameter_count"]),
            **{
                f"val_{key}": value
                for key, value in metric_summary(runs, "val").items()
            },
        }
        candidate_summaries.append(summary)
    selected_summary = max(
        candidate_summaries,
        key=lambda item: (
            float(item["val_macro_f1_mean"]),
            float(item["val_accuracy_mean"]),
            -int(item["parameter_count"]),
        ),
    )
    selected_candidate = next(
        candidate
        for candidate in CANDIDATES
        if candidate["name"] == selected_summary["candidate"]
    )
    tuning_payload = {
        "status": "complete",
        "test_metrics_present": False,
        "runs": tuning_runs,
        "candidate_summaries": candidate_summaries,
        "selected_candidate": selected_candidate,
    }
    atomic_json(args.output_dir / "tuning_results.json", tuning_payload)
    tuning_sha256 = sha256_file(args.output_dir / "tuning_results.json")
    locked_at = datetime.now().astimezone().isoformat(timespec="seconds")
    locked_config = {
        "status": "locked_before_test",
        "locked_at": locked_at,
        "selected_candidate": selected_candidate,
        "selection_summary": selected_summary,
        "selection_metric": config["selection_metric"],
        "tuning_results_sha256": tuning_sha256,
        "test_evaluated_before_lock": False,
    }
    atomic_json(args.output_dir / "locked_config.json", locked_config)
    emit(
        f"Locked candidate before test: {selected_candidate['name']}",
        log_path,
    )

    # Test data becomes available only after the selected configuration is locked.
    test_dataset = CachedFeatureDataset(cache, "test")
    datasets = {"val": val_dataset, "test": test_dataset}
    ablation_runs = []
    prediction_rows = []
    saved_states = {}
    total_ablation = len(MODALITIES) * len(args.seeds)
    for modality_index, modality in enumerate(MODALITIES):
        for seed_index, seed in enumerate(args.seeds):
            ordinal = modality_index * len(args.seeds) + seed_index + 1
            emit(
                f"Ablation {ordinal}/{total_ablation}: {modality} seed={seed}",
                log_path,
            )
            trial, best_state, last_state = train_trial(
                candidate=selected_candidate,
                modality=modality,
                seed=seed,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                test_dataset=test_dataset,
                epochs=args.epochs,
                patience=args.patience,
                batch_size=args.batch_size,
                eval_batch_size=args.eval_batch_size,
                workers=args.workers,
                device=device,
            )
            checkpoint_path = checkpoint_dir / f"{modality}_seed{seed}_best.pt"
            atomic_torch_save(
                checkpoint_path,
                {
                    "run_type": "b4c_regularization_ablation",
                    "candidate": selected_candidate,
                    "modality": modality,
                    "seed": seed,
                    "epoch": trial["best_epoch"],
                    "model_state": best_state,
                    "class_names": class_names,
                    "feature_cache_sha256": feature_cache_sha256,
                    "metrics": trial["metrics"],
                    "locked_config_sha256": sha256_file(
                        args.output_dir / "locked_config.json"
                    ),
                },
            )
            trial["checkpoint"] = {
                "path": str(checkpoint_path.resolve()),
                "bytes": checkpoint_path.stat().st_size,
                "sha256": sha256_file(checkpoint_path),
            }
            prediction_rows.extend(attach_predictions(trial, datasets))
            ablation_runs.append(trial)
            if modality == "fusion":
                saved_states[seed] = {
                    "best": best_state,
                    "last": last_state,
                    "trial": trial,
                }
            atomic_json(
                args.output_dir / "suite_progress.json",
                {
                    "phase": "locked_ablation",
                    "completed": ordinal,
                    "total": total_ablation,
                    "latest": {
                        "modality": modality,
                        "seed": seed,
                        "val_macro_f1": trial["metrics"]["val"]["macro_f1"],
                        "test_macro_f1": trial["metrics"]["test"]["macro_f1"],
                    },
                },
            )

    ablation_summaries = []
    for modality in MODALITIES:
        runs = [run for run in ablation_runs if run["modality"] == modality]
        ablation_summaries.append(
            {
                "modality": modality,
                **{
                    f"val_{key}": value
                    for key, value in metric_summary(runs, "val").items()
                },
                **{
                    f"test_{key}": value
                    for key, value in metric_summary(runs, "test").items()
                },
            }
        )
    ablation_payload = {
        "status": "complete",
        "locked_at": locked_at,
        "test_evaluation_started_after_lock": True,
        "selected_candidate": selected_candidate,
        "runs": ablation_runs,
        "modality_summaries": ablation_summaries,
    }
    atomic_json(args.output_dir / "ablation_results.json", ablation_payload)
    save_predictions(args.output_dir / "predictions.csv", prediction_rows)

    selected_fusion = max(
        (run for run in ablation_runs if run["modality"] == "fusion"),
        key=lambda run: (
            float(run["metrics"]["val"]["macro_f1"]),
            float(run["metrics"]["val"]["accuracy"]),
        ),
    )
    selected_seed = int(selected_fusion["seed"])
    selected_states = saved_states[selected_seed]
    selected_pointer = {
        "selection_basis": "best validation macro F1 among locked fusion seeds",
        "seed": selected_seed,
        "candidate": selected_candidate,
        "checkpoint": selected_fusion["checkpoint"],
        "metrics": selected_fusion["metrics"],
    }
    atomic_json(args.output_dir / "selected_fusion_model.json", selected_pointer)
    atomic_torch_save(
        args.output_dir / "best.pt",
        {
            "run_type": "b4c_regularization_ablation",
            "epoch": selected_fusion["best_epoch"],
            "model_state": selected_states["best"],
            "class_names": class_names,
            "candidate": selected_candidate,
            "modality": "fusion",
            "seed": selected_seed,
            "feature_cache_sha256": feature_cache_sha256,
            "metrics": selected_fusion["metrics"],
        },
    )
    atomic_torch_save(
        args.output_dir / "last.pt",
        {
            "run_type": "b4c_regularization_ablation",
            "epoch": selected_fusion["epochs_completed"],
            "model_state": selected_states["last"],
            "history": selected_fusion["history"],
            "best_epoch": selected_fusion["best_epoch"],
            "class_names": class_names,
            "candidate": selected_candidate,
            "modality": "fusion",
            "seed": selected_seed,
            "feature_cache_sha256": feature_cache_sha256,
        },
    )
    save_selected_history(args.output_dir, selected_fusion["history"])
    final_metrics = {
        "status": "complete",
        "run_type": "b4c_regularization_ablation",
        "best_epoch": selected_fusion["best_epoch"],
        "epochs_completed": selected_fusion["epochs_completed"],
        "best_validation_score": selected_fusion["best_validation_score"],
        "metrics": selected_fusion["metrics"],
        "class_names": list(class_names),
        "selected_seed": selected_seed,
        "selected_candidate": selected_candidate,
        "feature_cache_sha256": feature_cache_sha256,
        "suite_elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(args.output_dir / "final_metrics.json", final_metrics)
    plot_results(
        args.output_dir,
        selected_fusion["history"],
        selected_fusion["metrics"],
        class_names,
    )
    plot_suite(args.output_dir, candidate_summaries, ablation_summaries)

    summary_rows = []
    for item in candidate_summaries:
        summary_rows.append(
            {
                "phase": "selection",
                "name": item["candidate"],
                "val_accuracy_mean": item["val_accuracy_mean"],
                "val_accuracy_std": item["val_accuracy_std"],
                "val_macro_f1_mean": item["val_macro_f1_mean"],
                "val_macro_f1_std": item["val_macro_f1_std"],
                "test_accuracy_mean": "",
                "test_accuracy_std": "",
                "test_macro_f1_mean": "",
                "test_macro_f1_std": "",
            }
        )
    for item in ablation_summaries:
        summary_rows.append(
            {
                "phase": "ablation",
                "name": item["modality"],
                "val_accuracy_mean": item["val_accuracy_mean"],
                "val_accuracy_std": item["val_accuracy_std"],
                "val_macro_f1_mean": item["val_macro_f1_mean"],
                "val_macro_f1_std": item["val_macro_f1_std"],
                "test_accuracy_mean": item["test_accuracy_mean"],
                "test_accuracy_std": item["test_accuracy_std"],
                "test_macro_f1_mean": item["test_macro_f1_mean"],
                "test_macro_f1_std": item["test_macro_f1_std"],
            }
        )
    summary_path = args.output_dir / "summary.csv"
    with Path(f"{summary_path}.tmp").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    Path(f"{summary_path}.tmp").replace(summary_path)
    final_summary = {
        "status": "complete",
        "selected_candidate": selected_candidate,
        "selected_fusion_seed": selected_seed,
        "candidate_summaries": candidate_summaries,
        "modality_summaries": ablation_summaries,
        "artifacts": {
            "locked_config": str(
                (args.output_dir / "locked_config.json").resolve()
            ),
            "predictions": str((args.output_dir / "predictions.csv").resolve()),
            "checkpoint_count": total_ablation,
        },
        "suite_elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(args.output_dir / "final_summary.json", final_summary)
    atomic_json(
        args.output_dir / "suite_progress.json",
        {
            "phase": "complete",
            "completed": total_tuning + total_ablation,
            "total": total_tuning + total_ablation,
            "selected_candidate": selected_candidate["name"],
            "selected_fusion_seed": selected_seed,
        },
    )
    emit(json.dumps(final_summary, ensure_ascii=False, indent=2), log_path)


if __name__ == "__main__":
    main()
