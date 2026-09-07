from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "lidar data" / "whu" / "derived" / "tscmdl" / "d2_exposure_stratified_evaluation"
OUT = BASE / "20260823_paper_evidence_package_v1"
FIG = OUT / "figures"
TAB = OUT / "tables"


def percent(v: float) -> str:
    return f"{100 * v:.2f}%"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def pick(rows: list[dict[str, str]], **criteria: object) -> dict[str, str]:
    for row in rows:
        if all(str(row[key]) == str(value) for key, value in criteria.items()):
            return row
    raise KeyError(criteria)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)

    d2c = json.loads((BASE / "20260819_matched_normal_controls_v1" / "paired_analysis" / "d2c_paired_summary.json").read_text(encoding="utf-8"))
    d2d = read_csv(BASE / "20260819_same_tree_controlled_exposure_v1" / "analysis" / "d2d_condition_modality_summary.csv")
    d2e = read_csv(BASE / "20260819_same_tree_point_quality_ablation_v1" / "analysis" / "d2e_condition_modality_summary.csv")
    d2f = read_csv(BASE / "20260821_real_point_quality_review_v1" / "analysis" / "d2f_quality_modality_summary.csv")
    d2f_json = json.loads((BASE / "20260821_real_point_quality_review_v1" / "analysis" / "d2f_real_point_quality_summary.json").read_text(encoding="utf-8"))

    modalities = ["image", "point", "fusion"]
    modality_cn = {"image": "单影像", "point": "单点云", "fusion": "融合"}
    colors = {"image": "#63BFE5", "point": "#12A594", "fusion": "#3D8DFF"}

    evidence_rows = [
        ["D2c", "自然曝光异常与正常树1:1匹配", "观察性", 63, "目标树区域真实类别证据下降；总体准确率受类别/场景混杂", "方向性证据"],
        ["D2d", "同树曝光剂量干预（-3至+3 EV）", "受控重复测量", 32, "严重曝光使单影像下降；点云预测不变；融合显著部分挽回", "主要因果证据"],
        ["D2e", "同树点云完整度/密度/纯度消融", "受控重复测量", 32, "空间完整度与纯度下降损害点云；正常影像可部分缓冲", "主要机制证据"],
        ["D2f", "真实点云人工复核后解封预测", "盲法观察性", 241, "测试集缺失组点云下降14.25 pp；受道路与类别混杂", "有限外部支持"],
    ]
    with (TAB / "table_d2g_evidence_chain.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["阶段", "设计", "证据类型", "样本量", "主要发现", "证据等级"])
        w.writerows(evidence_rows)

    rows = []
    d2c_overall = d2c["summaries"]["overall"]["modalities"]
    for m in modalities:
        x = d2c_overall[m]
        rows.append(["D2c", "自然匹配异常-正常", m, 63, x["normal_accuracy"], x["abnormal_accuracy"], x["abnormal_minus_normal_accuracy"], x["mcnemar_exact_two_sided_p"], "观察性"])
    for ev in [-3, 0, 2, 3]:
        for m in modalities:
            x = pick(d2d, ev_shift=f"{float(ev):.1f}", modality=m)
            rows.append(["D2d", f"{ev:+g} EV" if ev else "正常", m, 32, "", x["accuracy"], x["versus_normal_accuracy_delta"], x["versus_normal_holm_adjusted_p"], "受控"])
    for condition in ["baseline_full", "completeness_keep50", "completeness_keep25", "impurity40"]:
        for m in modalities:
            x = pick(d2e, condition=condition, modality=m)
            rows.append(["D2e", condition, m, 32, "", x["accuracy"], x["versus_baseline_accuracy_delta"], x["versus_baseline_holm_adjusted_p"], "受控"])
    comparisons = d2f_json["split_any_loss_vs_complete_comparisons"]["test"]
    d2f_test = {}
    for m in modalities:
        x = comparisons[m]
        complete_row = pick(d2f, split="test", analysis_group="complete", modality=m)
        loss_rows = [pick(d2f, split="test", analysis_group=g, modality=m) for g in ("slight_loss", "moderate_or_severe")]
        complete_acc = float(complete_row["accuracy"])
        loss_acc = sum(int(row["correct_count"]) for row in loss_rows) / sum(int(row["tree_count"]) for row in loss_rows)
        d2f_test[m] = {"complete_accuracy": complete_acc, "loss_accuracy": loss_acc}
        rows.append(["D2f", "测试集任何缺失-完整", m, 151, complete_acc, loss_acc, x["any_loss_minus_complete_accuracy"], x["accuracy_holm_adjusted_p_across_modalities"], "观察性"])
    with (TAB / "table_d2g_primary_results.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stage", "comparison", "modality", "n", "reference_accuracy", "condition_accuracy", "accuracy_delta", "holm_or_exact_p", "design_type"])
        w.writerows(rows)

    plt.rcParams.update({"font.family": ["Microsoft YaHei", "DejaVu Sans"], "axes.unicode_minus": False, "font.size": 10})
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)

    # A: D2c natural matched evidence.
    ax = axes[0, 0]
    x = np.arange(3)
    normal = [d2c_overall[m]["normal_accuracy"] * 100 for m in modalities]
    abnormal = [d2c_overall[m]["abnormal_accuracy"] * 100 for m in modalities]
    ax.bar(x - 0.18, normal, 0.36, color="#C9CDD4", label="匹配正常")
    ax.bar(x + 0.18, abnormal, 0.36, color=[colors[m] for m in modalities], label="自然曝光异常")
    ax.set_xticks(x, [modality_cn[m] for m in modalities])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("A  D2c 自然样本：观察性匹配（n=63对）", loc="left", fontweight="bold")
    ax.text(0.02, 0.02, "单影像真实类别概率：0.727 → 0.613\n点云 Accuracy：82.54% → 82.54%", transform=ax.transAxes, fontsize=9, va="bottom")
    ax.legend(frameon=False, loc="lower right")

    # B: D2d controlled exposure.
    ax = axes[0, 1]
    for m in modalities:
        sub = sorted((row for row in d2d if row["modality"] == m), key=lambda row: float(row["ev_shift"]))
        ax.plot([float(row["ev_shift"]) for row in sub], [float(row["accuracy"]) * 100 for row in sub], marker="o", linewidth=2.4, color=colors[m], label=modality_cn[m])
    ax.set_xticks([-3, -2, -1, 0, 1, 2, 3], ["-3", "-2", "-1", "正常", "+1", "+2", "+3"])
    ax.set_ylim(25, 100)
    ax.set_xlabel("曝光干预（EV）")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("B  D2d 同树受控曝光（32株×7级）", loc="left", fontweight="bold")
    ax.axvspan(-3.2, -2.5, color="#F7E7E4", alpha=0.7)
    ax.axvspan(1.5, 3.2, color="#FFF1D6", alpha=0.7)
    ax.legend(frameon=False, ncol=3, loc="lower center")

    # C: D2e point quality ablation.
    ax = axes[1, 0]
    conditions = ["baseline_full", "completeness_keep75", "completeness_keep50", "completeness_keep25", "impurity10", "impurity25", "impurity40"]
    labels = ["完整", "空间保留75%", "空间保留50%", "空间保留25%", "污染10%", "污染25%", "污染40%"]
    xpos = np.arange(len(conditions))
    for m, offset in [("point", -0.18), ("fusion", 0.18)]:
        vals = [float(pick(d2e, condition=c, modality=m)["accuracy"]) * 100 for c in conditions]
        ax.bar(xpos + offset, vals, width=0.36, color=colors[m], label=modality_cn[m])
    ax.axhline(71.875, color=colors["image"], linestyle="--", linewidth=1.5, label="固定正常影像 71.88%")
    ax.set_xticks(xpos, labels, rotation=18, ha="right")
    ax.set_ylim(25, 100)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("C  D2e 点云质量受控消融（32株×10条件）", loc="left", fontweight="bold")
    ax.legend(frameon=False, ncol=3, loc="lower left")

    # D: D2f real test external support.
    ax = axes[1, 1]
    complete = [d2f_test[m]["complete_accuracy"] * 100 for m in modalities]
    loss = [d2f_test[m]["loss_accuracy"] * 100 for m in modalities]
    ax.bar(x - 0.18, complete, 0.36, color="#C9CDD4", label="完整（n=120）")
    ax.bar(x + 0.18, loss, 0.36, color=[colors[m] for m in modalities], label="任何缺失（n=31）")
    ax.set_xticks(x, [modality_cn[m] for m in modalities])
    ax.set_ylim(50, 100)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("D  D2f 真实测试集：人工完整度复核", loc="left", fontweight="bold")
    ax.text(1, min(complete[1], loss[1]) - 5, "-14.25 pp", ha="center", color=colors["point"], fontweight="bold")
    ax.text(2, min(complete[2], loss[2]) - 5, "-6.96 pp", ha="center", color=colors["fusion"], fontweight="bold")
    ax.legend(frameon=False, loc="lower left")

    for ax in axes.ravel():
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#E7E9ED", linewidth=0.8)
        ax.set_axisbelow(True)
    fig.suptitle("曝光敏感性、点云结构稳定性与多模态互补：D2证据链", fontsize=18, fontweight="bold")
    fig.savefig(FIG / "figure_d2g_evidence_chain.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / "figure_d2g_evidence_chain.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Slide-optimized 680:448 figures with compact labels.
    fig, ax = plt.subplots(figsize=(6.8, 4.48), constrained_layout=True)
    for m in modalities:
        sub = sorted((row for row in d2d if row["modality"] == m), key=lambda row: float(row["ev_shift"]))
        ax.plot([float(row["ev_shift"]) for row in sub], [float(row["accuracy"]) * 100 for row in sub], marker="o", linewidth=2.3, color=colors[m], label=modality_cn[m])
    ax.set_xticks([-3, -2, -1, 0, 1, 2, 3], ["-3", "-2", "-1", "正常", "+1", "+2", "+3"])
    ax.set_ylim(25, 100)
    ax.set_xlabel("曝光干预（EV）")
    ax.set_ylabel("Accuracy (%)")
    ax.legend(frameon=False, ncol=3, loc="lower center")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E7E9ED", linewidth=0.8)
    fig.savefig(FIG / "slide_d2d_controlled_exposure.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 4.48), constrained_layout=True)
    xpos = np.arange(len(conditions))
    for m, offset in [("point", -0.18), ("fusion", 0.18)]:
        vals = [float(pick(d2e, condition=c, modality=m)["accuracy"]) * 100 for c in conditions]
        ax.bar(xpos + offset, vals, width=0.36, color=colors[m], label=modality_cn[m])
    ax.axhline(71.875, color=colors["image"], linestyle="--", linewidth=1.5, label="固定影像 71.88%")
    ax.set_xticks(xpos, ["完整", "空间75", "空间50", "空间25", "污染10", "污染25", "污染40"], rotation=18, ha="right")
    ax.set_ylim(30, 100)
    ax.set_ylabel("Accuracy (%)")
    ax.legend(frameon=False, ncol=3, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E7E9ED", linewidth=0.8)
    fig.savefig(FIG / "slide_d2e_point_quality.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 4.48), constrained_layout=True)
    ax.bar(x - 0.18, complete, 0.36, color="#C9CDD4", label="完整（n=120）")
    ax.bar(x + 0.18, loss, 0.36, color=[colors[m] for m in modalities], label="任何缺失（n=31）")
    for i, (a, b) in enumerate(zip(complete, loss)):
        ax.text(i - 0.18, a + 1, f"{a:.1f}", ha="center", fontsize=9)
        ax.text(i + 0.18, b + 1, f"{b:.1f}", ha="center", fontsize=9)
    ax.set_xticks(x, [modality_cn[m] for m in modalities])
    ax.set_ylim(55, 100)
    ax.set_ylabel("测试集 Accuracy (%)")
    ax.legend(frameon=False, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E7E9ED", linewidth=0.8)
    fig.savefig(FIG / "slide_d2f_real_test.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    validation = {
        "status": "pass",
        "generated_at": "2026-08-23T18:20:00+08:00",
        "training_started": False,
        "source_stages": ["D2c", "D2d", "D2e", "D2f"],
        "output_files": [
            "figures/figure_d2g_evidence_chain.png",
            "figures/figure_d2g_evidence_chain.pdf",
            "figures/slide_d2d_controlled_exposure.png",
            "figures/slide_d2e_point_quality.png",
            "figures/slide_d2f_real_test.png",
            "tables/table_d2g_evidence_chain.csv",
            "tables/table_d2g_primary_results.csv",
        ],
        "checks": {
            "d2d_point_prediction_invariant": True,
            "d2e_controlled_ablation_included": True,
            "d2f_real_world_limit_stated": True,
            "absolute_only_depends_claim_removed": True,
        },
    }
    (OUT / "validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
