"""Build the frozen D2a automatic-exposure visualization package."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DERIVED_ROOT = PROJECT_ROOT / "lidar data" / "whu" / "derived" / "tscmdl"
DEFAULT_SOURCE = DERIVED_ROOT / "d1_four_class_image_quality" / "candidate_manifest.json"
DEFAULT_DATASET = DERIVED_ROOT / "c1_full_shared_dataset"
DEFAULT_OUTPUT = (
    DERIVED_ROOT
    / "d2_exposure_stratified_evaluation"
    / "20260819_visual_confirmation_v1"
)
DEFAULT_REPORT = (
    PROJECT_ROOT
    / "Lidar"
    / "tscmdl_whu_sandbox"
    / "reports"
    / "D2a_自动曝光检测可视化确认_启动报告.md"
)


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{path}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def exposure_group(record: dict[str, object]) -> str:
    reasons = {str(value) for value in record.get("automatic_risk_reasons", [])}
    dark = "excessive_dark_pixels" in reasons
    bright = "excessive_bright_pixels" in reasons
    if dark and bright:
        return "mixed"
    if dark:
        return "dark"
    if bright:
        return "bright"
    return ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_path = args.source.resolve()
    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    report_path = args.report.resolve()

    payload = json.loads(source_path.read_text(encoding="utf-8"))
    source_records = payload.get("records", [])
    if not isinstance(source_records, list):
        raise ValueError("candidate manifest records must be a list")

    selected: list[dict[str, object]] = []
    missing_assets: list[str] = []
    for raw in source_records:
        record = dict(raw)
        if str(record.get("model_split", "")) != "test_silver":
            continue
        group = exposure_group(record)
        if not group:
            continue
        record["d2_exposure_group"] = group
        record["d2_detection_source"] = "D1 deterministic automatic quality audit"
        record["d2_visual_confirmation_required"] = True
        record["quality_preview_path"] = "__dynamic_overlay__"
        selected.append(record)
        for field in ("image_path", "point_path"):
            asset = dataset_root / str(record[field])
            if not asset.is_file():
                missing_assets.append(f"{record['sample_key']}:{field}:{asset}")

    selected.sort(
        key=lambda item: (
            str(item["d2_exposure_group"]),
            int(item["model_class_index"]),
            str(item["road_id"]),
            str(item["trajectory_id"]),
            int(item["tree_id"]),
        )
    )
    keys = [str(record["sample_key"]) for record in selected]
    if len(keys) != len(set(keys)):
        raise ValueError("D2 visualization sample keys are duplicated")
    if missing_assets:
        raise FileNotFoundError("\n".join(missing_assets[:20]))

    group_counts = Counter(str(record["d2_exposure_group"]) for record in selected)
    class_counts = Counter(str(record["model_class_name"]) for record in selected)
    group_class_counts: dict[str, Counter[str]] = defaultdict(Counter)
    road_counts: dict[str, set[str]] = defaultdict(set)
    for record in selected:
        group = str(record["d2_exposure_group"])
        group_class_counts[group][str(record["model_class_name"])] += 1
        road_counts[group].add(str(record["road_id"]))

    classes = payload.get("classes", [])
    generated_at = timestamp()
    source_sha256 = sha256_file(source_path)
    manifest = {
        "format_version": 1,
        "stage": "D2a-automatic-exposure-visual-confirmation",
        "status": "awaiting_user_confirmation",
        "generated_at": generated_at,
        "source_dataset_root": str(dataset_root),
        "source_candidate_manifest": str(source_path),
        "source_candidate_manifest_sha256": source_sha256,
        "selection_policy": {
            "split": "test_silver",
            "included_automatic_reasons": [
                "excessive_dark_pixels",
                "excessive_bright_pixels",
            ],
            "manual_case_registry_used_for_selection": False,
            "model_predictions_used_for_selection": False,
        },
        "classes": classes,
        "summary": {
            "sample_count": len(selected),
            "group_counts": dict(sorted(group_counts.items())),
            "class_counts": dict(sorted(class_counts.items())),
            "group_class_counts": {
                group: dict(sorted(counts.items()))
                for group, counts in sorted(group_class_counts.items())
            },
            "group_road_counts": {
                group: len(roads) for group, roads in sorted(road_counts.items())
            },
        },
        "records": selected,
    }
    protocol = {
        "schema_version": 1,
        "stage": "D2a-automatic-exposure-visual-confirmation",
        "status": "frozen",
        "frozen_at": generated_at,
        "scope": "visualize and confirm automatic exposure grouping only",
        "prohibitions": [
            "no model training",
            "no checkpoint selection",
            "no source asset modification",
            "no use of the seven illustrative cases for sample selection",
        ],
        "source_candidate_manifest": {
            "path": str(source_path),
            "sha256": source_sha256,
        },
        "dataset_root": str(dataset_root),
        "selection": {
            "split": "test_silver",
            "exposure_groups": {
                "dark": "automatic_risk_reasons contains excessive_dark_pixels",
                "bright": "automatic_risk_reasons contains excessive_bright_pixels",
            },
            "expected_counts": dict(sorted(group_counts.items())),
        },
        "detector_thresholds": {
            "dark": "dark_ratio > 0.35",
            "bright": "bright_ratio > 0.35",
            "pixel_metrics_source": "automatic_metrics in D1 candidate manifest",
        },
        "next_gate": "user visually confirms automatic grouping before frozen three-modality inference",
    }
    validation = {
        "stage": "D2a-visualization-package-validation",
        "status": "passed",
        "validated_at": generated_at,
        "checks": {
            "source_manifest_exists": source_path.is_file(),
            "dataset_root_exists": dataset_root.is_dir(),
            "sample_keys_unique": len(keys) == len(set(keys)),
            "all_samples_are_test_silver": all(
                str(record.get("model_split")) == "test_silver" for record in selected
            ),
            "all_samples_have_exposure_flag": all(
                str(record.get("d2_exposure_group")) in {"dark", "bright", "mixed"}
                for record in selected
            ),
            "all_assets_exist": not missing_assets,
            "manual_case_registry_used_for_selection": False,
            "model_predictions_used_for_selection": False,
        },
        "sample_count": len(selected),
        "group_counts": dict(sorted(group_counts.items())),
        "missing_assets": missing_assets,
    }

    manifest_path = output_dir / "visualization_manifest.json"
    protocol_path = output_dir / "protocol.json"
    validation_path = output_dir / "validation.json"
    write_json(manifest_path, manifest)
    write_json(protocol_path, protocol)
    write_json(validation_path, validation)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "\n".join(
            [
                "# D2a 自动曝光检测可视化确认启动报告",
                "",
                f"启动时间：{generated_at}",
                "",
                "## 一、阶段范围",
                "",
                "本小阶段仅建立自动过暗/过亮结果的可视化确认入口，不启动训练，",
                "不依据模型预测筛样，也不把用户登记的 7 棵论文案例作为实验划分。",
                "",
                "## 二、输入与选择原则",
                "",
                f"- 来源清单：`{source_path}`",
                "- 数据划分：仅 `test_silver`，未进入 D1 训练、验证或核心测试指标。",
                "- 过暗：`dark_ratio > 0.35`。",
                "- 过亮：`bright_ratio > 0.35`。",
                "",
                "## 三、样本规模",
                "",
                f"- 过暗：{group_counts.get('dark', 0)} 张。",
                f"- 过亮：{group_counts.get('bright', 0)} 张。",
                f"- 合计：{len(selected)} 张。",
                "",
                "## 四、产物",
                "",
                f"- 冻结协议：`{protocol_path}`",
                f"- 可视化清单：`{manifest_path}`",
                f"- 结构验证：`{validation_path}`",
                "- 用户确认记录由可视化服务首次启动时单独创建。",
                "",
                "## 五、验证结果",
                "",
                "- 样本键唯一。",
                "- 所有样本均属于 test_silver。",
                "- 所有图像与点云资产存在。",
                "- 选择过程未读取模型预测。",
                "",
                "## 六、风险与边界",
                "",
                "自动曝光标签只描述像素亮暗比例，不等同于目标树可辨识性或配对质量。",
                "本阶段的人工确认只验证曝光分组是否合理，不重新执行 D1 数据清洗。",
                "",
                "## 七、下一阶段门",
                "",
                "用户完成或抽查可视化确认后，另行批准冻结三模态推理清单；",
                "在此之前不启动 D2 模型推理或训练。",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "passed",
                "output_dir": str(output_dir),
                "manifest": str(manifest_path),
                "protocol": str(protocol_path),
                "validation": str(validation_path),
                "report": str(report_path),
                "sample_count": len(selected),
                "group_counts": dict(sorted(group_counts.items())),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
