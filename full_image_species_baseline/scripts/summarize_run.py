#!/usr/bin/env python3
"""Validate both stages and write a concise Markdown experiment report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    dataset = read_json(run_dir / "dataset" / "dataset_summary.json")
    stages = {name: read_json(run_dir / name / "final_metrics.json") for name in ("pretrain", "finetune")}
    errors: list[str] = []
    warnings: list[str] = []
    for name, manifest_name in (("pretrain", "full_manifest.csv"), ("finetune", "crop_manifest.csv")):
        stage_dir = run_dir / name
        manifest = run_dir / "dataset" / manifest_name
        config = read_json(stage_dir / "run_config.json")
        if sha256_file(manifest) != config["manifest_sha256"]:
            errors.append(f"{name}: manifest hash mismatch")
        checkpoint = torch.load(stage_dir / "best.pt", map_location="cpu", weights_only=False)
        if len(checkpoint["class_names"]) != checkpoint["model_state"]["fc.weight"].shape[0]:
            errors.append(f"{name}: checkpoint head mismatch")
        with manifest.open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        group_splits: defaultdict[str, set[str]] = defaultdict(set)
        hash_splits: defaultdict[str, set[str]] = defaultdict(set)
        for row in rows:
            group_splits[row["group_id"]].add(row["split"])
            hash_splits[row["sha256"]].add(row["split"])
        if any(len(value) > 1 for value in group_splits.values()):
            errors.append(f"{name}: group leakage")
        if any(len(value) > 1 for value in hash_splits.values()):
            errors.append(f"{name}: content leakage")
    if stages["finetune"]["transfer"]["source_sha256"] != sha256_file(run_dir / "pretrain" / "best.pt"):
        errors.append("finetune did not use the pretrain best checkpoint")
    for name, stage in stages.items():
        small_support = [
            row["scientific_name"]
            for row in stage["metrics"]["test_group"]["per_class"]
            if int(row["support"]) < 3
        ]
        if small_support:
            warnings.append(
                f"{name}: {len(small_support)} classes have fewer than 3 test observations"
            )

    def line(label: str, stage: dict) -> str:
        sample, group = stage["metrics"]["test_sample"], stage["metrics"]["test_group"]
        return f"| {label} | {len(stage['class_names'])} | {stage['split_sizes']['train']} | {sample['accuracy']:.4f} | {sample['top5_accuracy']:.4f} | {sample['macro_f1']:.4f} | {group['accuracy']:.4f} | {group['macro_f1']:.4f} |"

    report = "\n".join(
        [
            "# 全量图像树种识别两阶段基线报告",
            "",
            "## 数据审计",
            "",
            f"- 扫描图像候选：{dataset['raw_candidates']:,} 张；去重后：{dataset['unique_images']:,} 张。",
            f"- 全图预训练：{dataset['full']['classes']} 类、{dataset['full']['samples']:,} 张。",
            f"- 人工标注目标裁剪：{dataset['crop']['classes']} 类、{dataset['crop']['samples']:,} 个目标实例。",
            f"- 隔离跨类别重复标注：{dataset['label_conflicts']['count']} 组，详见 `dataset/label_conflicts.csv`。",
            "- 划分以 iNaturalist observation 为组，完全相同图像内容不跨集合；裁剪继承原图划分。",
            "",
            "## 独立测试集结果",
            "",
            "| 阶段 | 类别 | 训练样本 | Sample Acc | Top-5 Acc | Sample Macro-F1 | Observation Acc | Observation Macro-F1 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
            line("全图弱标签预训练", stages["pretrain"]),
            line("人工裁剪精调", stages["finetune"]),
            "",
            "## 解释边界",
            "",
            "当前测试集来自网络图像，与香港街景部署域不同；这些指标用于建立数据与训练基线，不能直接当作街景应用精度。人工裁剪精调阶段只覆盖已完成标注的类别。",
            "少数类别的独立测试 observation 数低于 3，其逐类指标只用于定位问题，不应作稳定精度结论。",
            "",
        ]
    )
    (run_dir / "REPORT.md").write_text(report, encoding="utf-8")
    acceptance = {
        "status": "passed_with_warnings" if not errors and warnings else "passed" if not errors else "failed",
        "errors": errors,
        "warnings": warnings,
        "dataset_checks": dataset["checks"],
        "headline": {
            name: {
                "classes": len(stage["class_names"]),
                "test_accuracy": stage["metrics"]["test_sample"]["accuracy"],
                "test_top5_accuracy": stage["metrics"]["test_sample"]["top5_accuracy"],
                "test_macro_f1": stage["metrics"]["test_sample"]["macro_f1"],
                "test_group_macro_f1": stage["metrics"]["test_group"]["macro_f1"],
            }
            for name, stage in stages.items()
        },
    }
    (run_dir / "acceptance.json").write_text(json.dumps(acceptance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(acceptance, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
