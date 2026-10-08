"""Aggregate three matched YOLO11s-cls and WHU ResNet50 NJ runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


SEEDS = (20260728, 20260729, 20260730)
METRIC_KEYS = ("accuracy", "balanced_accuracy", "macro_f1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments-root", type=Path, required=True)
    parser.add_argument("--resnet-suite", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_test_predictions(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["split"] == "test"]
    return {row["sample_key"]: row for row in rows}


def metric_bundle(true: np.ndarray, predicted: np.ndarray, class_names: list[str]) -> dict[str, object]:
    labels = list(range(len(class_names)))
    precision, recall, f1, support = precision_recall_fscore_support(
        true, predicted, labels=labels, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(true, predicted)),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "sample_count": int(len(true)),
        "confusion_matrix": confusion_matrix(true, predicted, labels=labels).tolist(),
        "per_class": [
            {
                "class_index": index,
                "scientific_name": class_names[index],
                "support": int(support[index]),
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
            }
            for index in labels
        ],
    }


def exact_binomial_two_sided(successes: int, trials: int) -> float:
    if trials == 0:
        return 1.0
    lower = min(successes, trials - successes)
    probability = sum(math.comb(trials, value) for value in range(lower + 1)) / (2**trials)
    return min(1.0, 2.0 * probability)


def paired_comparison(
    yolo_correct: np.ndarray, resnet_correct: np.ndarray, rng_seed: int
) -> dict[str, object]:
    resnet_only = int(np.sum((resnet_correct == 1) & (yolo_correct == 0)))
    yolo_only = int(np.sum((resnet_correct == 0) & (yolo_correct == 1)))
    rng = np.random.default_rng(rng_seed)
    sampled = rng.integers(0, len(yolo_correct), size=(20000, len(yolo_correct)))
    differences = yolo_correct[sampled].mean(axis=1) - resnet_correct[sampled].mean(axis=1)
    return {
        "accuracy_difference_yolo_minus_resnet": float(yolo_correct.mean() - resnet_correct.mean()),
        "paired_bootstrap_95ci": [float(value) for value in np.quantile(differences, [0.025, 0.975])],
        "paired_outcomes": {
            "both_correct": int(np.sum((resnet_correct == 1) & (yolo_correct == 1))),
            "both_wrong": int(np.sum((resnet_correct == 0) & (yolo_correct == 0))),
            "resnet_only_correct": resnet_only,
            "yolo_only_correct": yolo_only,
        },
        "mcnemar_exact_two_sided_p": exact_binomial_two_sided(
            min(resnet_only, yolo_only), resnet_only + yolo_only
        ),
    }


def aggregate_metrics(rows: list[dict[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {"runs": rows}
    for key in METRIC_KEYS:
        values = [float(row[key]) for row in rows]
        result[f"{key}_mean"] = statistics.mean(values)
        result[f"{key}_sample_std"] = statistics.stdev(values)
        result[f"{key}_min"] = min(values)
        result[f"{key}_max"] = max(values)
    return result


def probability_matrix(
    rows: dict[str, dict[str, str]], sample_keys: list[str], class_count: int
) -> np.ndarray:
    return np.asarray(
        [
            [float(rows[key][f"probability_{index}"]) for index in range(class_count)]
            for key in sample_keys
        ],
        dtype=np.float64,
    )


def save_figure(payload: dict[str, object], path: Path) -> None:
    yolo = payload["yolo11s_cls_three_seed"]
    resnet = payload["whu_resnet50_three_seed"]
    yolo_ensemble = payload["probability_ensemble"]["yolo11s_cls"]["metrics"]
    resnet_ensemble = payload["probability_ensemble"]["whu_resnet50"]["metrics"]
    labels = ["Accuracy", "Balanced accuracy", "Macro-F1"]
    figure, axes = plt.subplots(1, 2, figsize=(13.2, 5.6), constrained_layout=True)
    x = np.arange(3)
    width = 0.34
    yolo_means = [float(yolo[f"{key}_mean"]) for key in METRIC_KEYS]
    resnet_means = [float(resnet[f"{key}_mean"]) for key in METRIC_KEYS]
    yolo_std = [float(yolo[f"{key}_sample_std"]) for key in METRIC_KEYS]
    resnet_std = [float(resnet[f"{key}_sample_std"]) for key in METRIC_KEYS]
    axes[0].bar(x - width / 2, yolo_means, width, yerr=yolo_std, capsize=4, label="YOLO11s-cls")
    axes[0].bar(x + width / 2, resnet_means, width, yerr=resnet_std, capsize=4, label="WHU ResNet50")
    for offset, runs, color in ((-width / 2, yolo["runs"], "#1f77b4"), (width / 2, resnet["runs"], "#ff7f0e")):
        for metric_index, key in enumerate(METRIC_KEYS):
            values = [float(run[key]) for run in runs]
            jitter = np.linspace(-0.045, 0.045, len(values))
            axes[0].scatter(metric_index + offset + jitter, values, color=color, edgecolor="white", zorder=3)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0.65, 1.0)
    axes[0].set_ylabel("Score")
    axes[0].set_title("Three matched seeds: mean ± sample SD")
    axes[0].legend(loc="lower left")

    short_names = ["C. camphora", "L. indica", "M. grandiflora", "Other"]
    class_x = np.arange(len(short_names))
    yolo_f1 = [float(row["f1"]) for row in yolo_ensemble["per_class"]]
    resnet_f1 = [float(row["f1"]) for row in resnet_ensemble["per_class"]]
    axes[1].bar(class_x - width / 2, yolo_f1, width, label="YOLO11s-cls ensemble")
    axes[1].bar(class_x + width / 2, resnet_f1, width, label="WHU ResNet50 ensemble")
    axes[1].set_xticks(class_x, short_names, rotation=18, ha="right")
    axes[1].set_ylim(0.65, 1.0)
    axes[1].set_ylabel("F1")
    axes[1].set_title("Three-seed probability ensemble")
    axes[1].legend(loc="lower left")
    for container in axes[1].containers:
        axes[1].bar_label(container, fmt="%.3f", padding=2, fontsize=8)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_report(payload: dict[str, object], path: Path) -> None:
    yolo = payload["yolo11s_cls_three_seed"]
    resnet = payload["whu_resnet50_three_seed"]
    ensemble = payload["probability_ensemble"]
    paired = ensemble["paired_comparison"]
    lines = [
        "# 南京四分类：YOLO11s-cls 与 WHU ResNet50 三种子正式汇总",
        "",
        "## 结论",
        "",
        f"在同一冻结划分（测试集 n={payload['test_samples']}）上，YOLO11s-cls 三种子平均 Accuracy 为 "
        f"{100 * yolo['accuracy_mean']:.2f}% ± {100 * yolo['accuracy_sample_std']:.2f}%，平均 Macro-F1 为 "
        f"{100 * yolo['macro_f1_mean']:.2f}% ± {100 * yolo['macro_f1_sample_std']:.2f}%。WHU ResNet50 对应结果为 "
        f"{100 * resnet['accuracy_mean']:.2f}% ± {100 * resnet['accuracy_sample_std']:.2f}% 和 "
        f"{100 * resnet['macro_f1_mean']:.2f}% ± {100 * resnet['macro_f1_sample_std']:.2f}%。",
        "",
        "三种子逐树概率平均后：YOLO11s-cls Accuracy/Macro-F1 为 "
        f"{100 * ensemble['yolo11s_cls']['metrics']['accuracy']:.2f}% / "
        f"{100 * ensemble['yolo11s_cls']['metrics']['macro_f1']:.2f}%，WHU ResNet50 为 "
        f"{100 * ensemble['whu_resnet50']['metrics']['accuracy']:.2f}% / "
        f"{100 * ensemble['whu_resnet50']['metrics']['macro_f1']:.2f}%。集成 Accuracy 差值为 "
        f"{100 * paired['accuracy_difference_yolo_minus_resnet']:.2f} 个百分点。",
        "",
        "## 单种子结果",
        "",
        "| 种子 | YOLO Accuracy | YOLO Macro-F1 | ResNet Accuracy | ResNet Macro-F1 | Accuracy 差值 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for yolo_run, resnet_run in zip(yolo["runs"], resnet["runs"]):
        lines.append(
            f"| {yolo_run['seed']} | {100 * yolo_run['accuracy']:.2f}% | "
            f"{100 * yolo_run['macro_f1']:.2f}% | {100 * resnet_run['accuracy']:.2f}% | "
            f"{100 * resnet_run['macro_f1']:.2f}% | "
            f"{100 * (yolo_run['accuracy'] - resnet_run['accuracy']):+.2f} pp |"
        )
    outcomes = paired["paired_outcomes"]
    ci = paired["paired_bootstrap_95ci"]
    lines.extend(
        [
            "",
            "## 集成后的成对检验",
            "",
            f"- YOLO 正确而 ResNet 错误：{outcomes['yolo_only_correct']} 棵；ResNet 正确而 YOLO 错误：{outcomes['resnet_only_correct']} 棵。",
            f"- Accuracy 差值的样本级 bootstrap 95% CI：[{100 * ci[0]:.2f}, {100 * ci[1]:.2f}] 个百分点。",
            f"- McNemar 精确双侧 p={paired['mcnemar_exact_two_sided_p']:.6g}。同一路段树木可能相关，因此该 p 值仅作探索性证据。",
            "",
            "## 真实性边界",
            "",
            "- 三次运行使用相同冻结 train/val/test、相同四分类标签和相同测试树；每次先按验证集 Macro-F1 选权重，再解封测试集。",
            "- 这是已裁剪单树图像的分类比较，不代表完整街景中的目标检测或自动分割精度。",
            "- 两模型使用各自既定训练配方（YOLO11 为 AdamW，ResNet50 为 SGD），因此结论是‘当前完整基线方案比较’，不是纯主干架构消融。",
            "- 仅有三个随机种子；均值和标准差用于描述随机初始化波动，不替代跨路段、跨城市外部验证。",
            "",
            f"机器验收状态：`{payload['status']}`；全部检查：`{all(payload['checks'].values())}`。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    experiments_root = args.experiments_root.resolve()
    resnet_suite = args.resnet_suite.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    yolo_runs: list[dict[str, object]] = []
    resnet_runs: list[dict[str, object]] = []
    yolo_predictions: dict[int, dict[str, dict[str, str]]] = {}
    resnet_predictions: dict[int, dict[str, dict[str, str]]] = {}
    class_names: list[str] | None = None
    checks: dict[str, bool] = {}

    for seed in SEEDS:
        yolo_dir = experiments_root / f"nj_yolo11s_cls_seed_{seed}"
        resnet_dir = resnet_suite / "runs" / f"seed_{seed}"
        yolo_metrics = read_json(yolo_dir / "final_metrics.json")
        resnet_metrics = read_json(resnet_dir / "final_metrics.json")
        yolo_environment = read_json(yolo_dir / "environment.json")
        yolo_acceptance = read_json(yolo_dir / "acceptance.json")
        names = list(yolo_metrics["class_names"])
        if class_names is None:
            class_names = names
        checks[f"seed_{seed}_class_order_matches"] = names == class_names
        checks[f"seed_{seed}_environment_seed_matches"] = int(yolo_environment["seed"]) == seed
        checks[f"seed_{seed}_yolo_acceptance_passes"] = (
            yolo_acceptance["status"] == "passed_with_limitations"
            and all(yolo_acceptance["checks"].values())
        )
        yolo_test = yolo_metrics["metrics"]["test"]
        resnet_test = resnet_metrics["metrics"]["test"]
        selection = yolo_metrics["selection"]["selected"]
        yolo_runs.append(
            {
                "seed": seed,
                "selected_epoch": int(selection["epoch"]),
                **{key: float(yolo_test[key]) for key in METRIC_KEYS},
                "checkpoint_sha256": yolo_metrics["checkpoint_sha256"],
            }
        )
        resnet_runs.append(
            {"seed": seed, **{key: float(resnet_test[key]) for key in METRIC_KEYS}}
        )
        yolo_predictions[seed] = read_test_predictions(yolo_dir / "predictions.csv")
        resnet_predictions[seed] = read_test_predictions(resnet_dir / "predictions.csv")

    assert class_names is not None
    sample_keys = sorted(yolo_predictions[SEEDS[0]])
    checks["test_count_is_151"] = len(sample_keys) == 151
    baseline_key_set = set(sample_keys)
    true_by_key = {key: int(yolo_predictions[SEEDS[0]][key]["true_class"]) for key in sample_keys}
    for seed in SEEDS:
        checks[f"seed_{seed}_paired_sample_keys_match"] = (
            set(yolo_predictions[seed]) == baseline_key_set
            and set(resnet_predictions[seed]) == baseline_key_set
        )
        checks[f"seed_{seed}_ground_truth_matches"] = all(
            int(yolo_predictions[seed][key]["true_class"]) == true_by_key[key]
            and int(resnet_predictions[seed][key]["true_class"]) == true_by_key[key]
            for key in sample_keys
        )

    true = np.asarray([true_by_key[key] for key in sample_keys], dtype=np.int64)
    class_count = len(class_names)
    yolo_probability_stack = np.stack(
        [probability_matrix(yolo_predictions[seed], sample_keys, class_count) for seed in SEEDS]
    )
    resnet_probability_stack = np.stack(
        [probability_matrix(resnet_predictions[seed], sample_keys, class_count) for seed in SEEDS]
    )
    checks["yolo_probabilities_are_finite"] = bool(np.isfinite(yolo_probability_stack).all())
    checks["resnet_probabilities_are_finite"] = bool(np.isfinite(resnet_probability_stack).all())
    checks["yolo_probability_rows_sum_to_one"] = bool(
        np.allclose(yolo_probability_stack.sum(axis=2), 1.0, atol=1e-5)
    )
    checks["resnet_probability_rows_sum_to_one"] = bool(
        np.allclose(resnet_probability_stack.sum(axis=2), 1.0, atol=1e-5)
    )

    yolo_seed_predictions = yolo_probability_stack.argmax(axis=2)
    resnet_seed_predictions = resnet_probability_stack.argmax(axis=2)
    yolo_ensemble_probability = yolo_probability_stack.mean(axis=0)
    resnet_ensemble_probability = resnet_probability_stack.mean(axis=0)
    yolo_ensemble_prediction = yolo_ensemble_probability.argmax(axis=1)
    resnet_ensemble_prediction = resnet_ensemble_probability.argmax(axis=1)
    yolo_ensemble_metrics = metric_bundle(true, yolo_ensemble_prediction, class_names)
    resnet_ensemble_metrics = metric_bundle(true, resnet_ensemble_prediction, class_names)
    paired = paired_comparison(
        (yolo_ensemble_prediction == true).astype(np.int8),
        (resnet_ensemble_prediction == true).astype(np.int8),
        20260824,
    )
    payload: dict[str, object] = {
        "status": "passed_with_limitations" if all(checks.values()) else "failed",
        "scope": "NJ frozen four-class single-tree image classification, three matched seeds",
        "seeds": list(SEEDS),
        "test_samples": len(sample_keys),
        "class_names": class_names,
        "yolo11s_cls_three_seed": aggregate_metrics(yolo_runs),
        "whu_resnet50_three_seed": aggregate_metrics(resnet_runs),
        "matched_seed_differences": [
            {
                "seed": seed,
                **{
                    f"{key}_yolo_minus_resnet": float(yolo_runs[index][key])
                    - float(resnet_runs[index][key])
                    for key in METRIC_KEYS
                },
            }
            for index, seed in enumerate(SEEDS)
        ],
        "probability_ensemble": {
            "method": "arithmetic mean of per-class probabilities across the three seeds",
            "yolo11s_cls": {
                "metrics": yolo_ensemble_metrics,
                "prediction_unanimity_rate": float(
                    np.mean(np.all(yolo_seed_predictions == yolo_seed_predictions[0], axis=0))
                ),
            },
            "whu_resnet50": {
                "metrics": resnet_ensemble_metrics,
                "prediction_unanimity_rate": float(
                    np.mean(np.all(resnet_seed_predictions == resnet_seed_predictions[0], axis=0))
                ),
            },
            "paired_comparison": paired,
        },
        "checks": checks,
        "limitations": [
            "The recipes are model-native rather than optimizer-identical (YOLO11 AdamW; ResNet50 SGD).",
            "The task is classification of pre-cropped target-tree images, not full-scene detection or segmentation.",
            "Only three seeds are available, so uncertainty of the seed-level mean remains imprecise.",
            "Trees from the same road or trajectory may be correlated; sample-level paired inference is exploratory.",
            "External validation across unseen roads and cities is still required for deployment claims.",
        ],
    }

    with (output_dir / "three_seed_runs.csv").open("w", encoding="utf-8", newline="") as stream:
        fieldnames = [
            "seed",
            "yolo_selected_epoch",
            "yolo_accuracy",
            "yolo_balanced_accuracy",
            "yolo_macro_f1",
            "resnet_accuracy",
            "resnet_balanced_accuracy",
            "resnet_macro_f1",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for yolo_run, resnet_run in zip(yolo_runs, resnet_runs):
            writer.writerow(
                {
                    "seed": yolo_run["seed"],
                    "yolo_selected_epoch": yolo_run["selected_epoch"],
                    **{f"yolo_{key}": yolo_run[key] for key in METRIC_KEYS},
                    **{f"resnet_{key}": resnet_run[key] for key in METRIC_KEYS},
                }
            )

    ensemble_fields = ["sample_key", "true_class", "yolo_predicted_class", "resnet_predicted_class"]
    ensemble_fields += [f"yolo_probability_{index}" for index in range(class_count)]
    ensemble_fields += [f"resnet_probability_{index}" for index in range(class_count)]
    with (output_dir / "ensemble_predictions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=ensemble_fields)
        writer.writeheader()
        for row_index, key in enumerate(sample_keys):
            writer.writerow(
                {
                    "sample_key": key,
                    "true_class": int(true[row_index]),
                    "yolo_predicted_class": int(yolo_ensemble_prediction[row_index]),
                    "resnet_predicted_class": int(resnet_ensemble_prediction[row_index]),
                    **{
                        f"yolo_probability_{index}": float(yolo_ensemble_probability[row_index, index])
                        for index in range(class_count)
                    },
                    **{
                        f"resnet_probability_{index}": float(resnet_ensemble_probability[row_index, index])
                        for index in range(class_count)
                    },
                }
            )

    summary_path = output_dir / "three_seed_summary.json"
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    save_figure(payload, output_dir / "three_seed_comparison.png")
    write_report(payload, output_dir / "三种子正式汇总.md")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if payload["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
