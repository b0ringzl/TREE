"""Frozen-checkpoint and inference-factor ablations; never train or deploy.

This is attribution on previously inspected regression data, not a new test.
Historical checkpoint transitions bundle training/data/optimization changes.
Do not interpret them as single-variable training-data causal estimates.
"""
from __future__ import annotations

import argparse
import gc
import io
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools"), str(ROOT / "external_datasets/distillation_runtime")]
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import requests
import torch
from PIL import Image, ImageOps

from bioclip_probe import ResidualHead, TEACHER
from distillation_core import collapse_logits
from training_safety import backup_before_training, sha256, verify_backup
from spacescan_trees_demo_v1.presentation_groups import BAUHINIA_GROUP, BAUHINIA_MEMBERS, grouped_distribution

PROBE = ROOT / "outputs/distillation/bioclip_probe_20260908_0227"
SOURCE = ROOT / "outputs/distillation/20260908_0045"
COMMON = ROOT / "outputs/bioclip_sstv1_fresh146_20261006_v2"
RARE = ROOT / "outputs/rare_species_fresh_validation_20261006_v1"
MODELS = ROOT / "tree_species_teacher_demo/models"
SEED = 20261007
STAGE_NAMES = {
    "zero": "原 BioCLIP 2.5 零样本",
    "linear_lr001": "初期线性残差头（lr=0.001）",
    "linear_lr01": "初期线性残差头（lr=0.01）",
    "adapted_mlp": "初期香港类别残差 MLP 头",
    "sst_sep14": "9月14日通用增量训练",
    "aq_sep15": "9月15日土沉香补充训练",
    "ar_sep18": "9月18日补充训练版本",
    "h120": "当前正式 H120",
}


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError("Immutable output already exists: " + str(path))
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def make_stages():
    lock = load(PROBE / "selection_lock.json")
    fits = load(PROBE / "fit_results.json")
    stages = {}
    for recipe in ("linear_lr001", "linear_lr01", "residual_mlp"):
        runs = lock["selected_runs"] if recipe == "residual_mlp" else [r for r in fits if r["recipe"] == recipe and not r["smoke"]]
        assert len(runs) == 3
        name = "adapted_mlp" if recipe == "residual_mlp" else recipe
        stages[name] = {"members": [{"path": str(Path(r["run"]) / "best.pt"),
                                    "sha256": sha256(Path(r["run"]) / "best.pt"),
                                    "seed": r["seed"], "epoch": r["best_epoch"]} for r in runs],
                        "source": str(PROBE), "kind": "validation_selected_historical_recipe"}
        if recipe == "residual_mlp":
            assert all(m["sha256"] == lock["weights_sha256"][str(Path(m["path"]).parent)] for m in stages[name]["members"])
    for name, directory in (("sst_sep14", "SSTv1_20260914_v1"),
                            ("aq_sep15", "SSTv1_Aquilaria_User_20260915_v1"),
                            ("ar_sep18", "SSTv1_Artocarpus_User_20260918_v1"),
                            ("h120", "SSTv1_Aquilaria_UserCrops_20260919_H120")):
        manifest = MODELS / directory / "deployment.json"
        data = load(manifest)
        assert data["encoder_sha256"] == load(PROBE / "protocol.json")["teacher_sha256"]
        stages[name] = {"members": [{"path": str(manifest.parent / m["file"]), "sha256": m["sha256"]} for m in data["members"]],
                        "manifest": str(manifest), "source": data["source_run"], "kind": "historical_deployed_checkpoint"}
    return stages


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    stages = make_stages()
    cohorts = {"common": COMMON, "rare": RARE}
    rows = []
    for name, directory in cohorts.items():
        source_rows = load(directory / "manifest.json")
        assert sha256(directory / "manifest.json") == load(directory / "collection_lock.json")["manifest_sha256"]
        evidence = [json.loads(line) for line in (directory / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
        assert len(source_rows) == len(evidence) and all(r["id"] == p["id"] for r, p in zip(source_rows, evidence))
        for r in source_rows:
            rows.append({**r, "ablation_id": name + ":" + r["id"], "cohort": name,
                         "source_directory": str(directory), "source_image": str(directory / r["file"]),
                         "role": "previously_inspected_regression_only", "training_eligible": False})
    assert len({r["ablation_id"] for r in rows}) == len(rows)
    plan = {
        "created_at": datetime.now(timezone.utc).isoformat(), "seed": SEED,
        "purpose": "Explain prior +6.78pp with controlled inference ablations and historical checkpoint ladder",
        "training_performed": False, "deployment_changed": False, "checkpoint_selection_performed": False,
        "cohorts": {k: str(v) for k, v in cohorts.items()}, "rows": dict(Counter(r["cohort"] for r in rows)),
        "factors": ["fixed checkpoint, center vs whole vs equal-probability context", "three fixed members vs each single and leave-one-out",
                    "historical checkpoint stages on identical views", "fixed T vs T=1, inference-only sensitivity",
                    "148 vs 146 Bauhinia presentation aggregation; identical grouping for primary model comparison"],
        "controls": "Same encoder SHA256, prompts, label order, image bytes, resizing, temperature and held-fixed views. No SAM/detection/point cloud/video in the +6.78pp benchmark.",
        "limitations": ["Regression attribution; no fresh unseen test or selection on these results.",
                        "Checkpoint transitions include multiple changes and continued optimization; not isolated data-only causal effects.",
                        "Linear recipes have different learning rates; not a pure nonlinear-architecture experiment.",
                        "Common cohort covers only 101/146 categories and 17 Hong Kong photos; not complete Hong Kong street-view accuracy.",
                        "Rare photos are clustered; primary community observations separate from paper challenge panels.",
                        "Already pooled Ficus/Araucaria outputs cannot be separated by presentation ablation."]}
    save(out / "plan.json", plan)
    save(out / "rows.json", rows)
    save(out / "stages.json", stages)
    inputs = [TEACHER, Path(__file__), ROOT / "tools/test_ablate_sstv1_accuracy_20261007.py", ROOT / "tools/training_safety.py", ROOT / "tools/bioclip_probe.py",
              ROOT / "tools/distillation_core.py", ROOT / "tree_species_teacher_demo/engine.py",
              ROOT / "spacescan_trees_demo_v1/server.py", ROOT / "spacescan_trees_demo_v1/release.json",
              ROOT / "spacescan_trees_demo_v1/presentation_groups.py", PROBE / "classes.json",
              PROBE / "teacher_choice.json", PROBE / "protocol.json", PROBE / "cache_metadata.json",
              PROBE / "selection_lock.json", PROBE / "fit_results.json", SOURCE / "aliases.json",
              out / "plan.json", out / "rows.json", out / "stages.json"]
    for stage in stages.values():
        for m in stage["members"]:
            assert sha256(m["path"]) == m["sha256"], m["path"]
            inputs.append(Path(m["path"]))
        if "manifest" in stage:
            inputs.append(Path(stage["manifest"]))
            labels_path = Path(stage["manifest"]).parent / "labels.json"
            if labels_path.is_file(): inputs.append(labels_path)
        for name in ("protocol.json", "config.json", "accepted_training_manifest.json", "accepted_new_manifest.json",
                     "accepted_replay_manifest.json", "train_rows.json", "human_review_snapshot.json", "selection_lock.json"):
            p = Path(stage["source"]) / name
            if p.is_file(): inputs.append(p)
    for directory in cohorts.values():
        inputs += [directory / name for name in ("manifest.json", "predictions.jsonl", "collection_lock.json", "catalog146.json")]
    inputs += [COMMON / "FINAL_RESULTS.json", COMMON / "SUMMARY_ZH.md", RARE / "RARE_RESULTS.json"]
    for name in ("ptv2_seven_20260913_v1/model.joblib", "ptv2_seven_20260913_v1/deployment.json"):
        p = ROOT / "spacescan_trees_demo_v1/models" / name
        if p.is_file(): inputs.append(p)
    for source in inputs:
        if not source.is_file(): raise FileNotFoundError("Backup preflight failed: " + str(source))
    print("Making run-specific physical backups before any ablation", flush=True)
    backup = backup_before_training(out, inputs, {**plan, "operation": "inference-only_ablation", "no_fit": True})
    print("BACKUP VERIFIED", backup, flush=True)
    save(out / "input_lock.json", {"rows_sha256": sha256(out / "rows.json"), "stages_sha256": sha256(out / "stages.json"),
                                   "plan_sha256": sha256(out / "plan.json"), "source_images_sha256_verified_during_inference": True})


def pad(data):
    """Exact production prepare(): gray square, edge<=1024, JPEG quality95."""
    with Image.open(io.BytesIO(data)) as source:
        im = ImageOps.exif_transpose(source).convert("RGB")
    edge = min(1024, max(im.size))
    im = ImageOps.pad(im, (edge, edge), method=Image.Resampling.BICUBIC, color=(128, 128, 128))
    stream = io.BytesIO()
    im.save(stream, format="JPEG", quality=95)
    return stream.getvalue()


def member_prob(head, x, raw, extra, groups, temperature):
    return (torch.cat([collapse_logits(head(x, raw), groups), extra], -1) / temperature).softmax(-1)[0]


def infer(out):
    verify_backup(out / "pre_training_backup")
    lock = load(out / "input_lock.json")
    for name in ("rows", "stages", "plan"):
        assert sha256(out / (name + ".json")) == lock[name + "_sha256"]
    for name in ("predictions.npz", "features.npz", "inference_complete.json", "inference_status.json"):
        if (out / name).exists(): raise FileExistsError("No resume/overwrite: create a new run directory")
    rows, stages = load(out / "rows.json"), load(out / "stages.json")
    from tree_species_teacher_demo.engine import Engine, decode
    started = time.monotonic()
    engine = Engine()
    engine.load()
    assert engine.state == "ready" and engine.sstv1_state == "ready"
    heads = {}
    for name, stage in stages.items():
        heads[name] = []
        for m in stage["members"]:
            assert sha256(m["path"]) == m["sha256"]
            state = torch.load(m["path"], map_location="cpu", weights_only=False)
            assert state["classes"] == engine.classes and state["encoder_sha256"] == engine.encoder_sha256
            h = ResidualHead(state["config"]["architecture"]).cuda().eval().requires_grad_(False)
            h.load_state_dict(state["model_state"])
            heads[name].append(h)
    original_evidence = {}
    for name, directory in (("common", COMMON), ("rare", RARE)):
        for line in (directory / "predictions.jsonl").read_text(encoding="utf-8").splitlines():
            p = json.loads(line)
            original_evidence[name + ":" + p["id"]] = p
    choices = ["center", "whole"]
    probabilities, features, raw_logits, extra_logits = {}, [], [], []
    parities, per_view_seconds = [], []
    with torch.inference_mode():
        for index, row in enumerate(rows):
            data = Path(row["source_image"]).read_bytes()
            assert sha256(row["source_image"]) == row["sha256"], row["ablation_id"]
            xx, zz, ee = [], [], []
            for view, content in (("center", data), ("whole", pad(data))):
                t0 = time.monotonic()
                pixels, _, _ = decode(content)
                embedding = engine.model.encode_image(((pixels[None].cuda() - engine.mean) / engine.std).to(engine.dtype), normalize=True).float()
                raw = embedding @ engine.bank.T * engine.model.logit_scale.exp().float()
                x = embedding.half().float()
                extra_teacher = embedding @ engine.extra_bank.T * engine.model.logit_scale.exp().float()
                extra = x @ engine.extra_bank.T * engine.model.logit_scale.exp().float()
                temp = engine.choice["calibration_temperature"]
                probs = {"zero": (torch.cat([collapse_logits(raw, engine.classes["groups"]), extra_teacher], -1) / temp).softmax(-1)[0]}
                for name, members in heads.items():
                    ps = [member_prob(h, x, raw, extra, engine.classes["groups"], temp) for h in members]
                    probs[name] = torch.stack(ps).mean(0)
                    if name == "h120":
                        for j, p in enumerate(ps): probs[f"h120_member{j+1}"] = p
                        for j in range(3): probs[f"h120_without_member{j+1}"] = torch.stack([p for k, p in enumerate(ps) if k != j]).mean(0)
                        probs["h120_T1"] = torch.stack([member_prob(h, x, raw, extra, engine.classes["groups"], 1.) for h in members]).mean(0)
                        # No teacher-logit prior: inference sensitivity, not a retrained pure MLP classifier.
                        probs["h120_no_logit_prior"] = torch.stack([
                            member_prob(h, x, torch.zeros_like(raw), extra, engine.classes["groups"], temp) for h in members]).mean(0)
                for name, p in probs.items():
                    probabilities.setdefault(name + "_" + view, []).append(p.cpu().numpy())
                expected = original_evidence[row["ablation_id"]]["distributions"]
                errors = {}
                for a, b in (("zero", "teacher"), ("h120", "sstv1")):
                    prior = np.array([expected[b + "_" + view][name] for name in engine.labels])
                    error = float(np.max(np.abs(prior - probs[a].cpu().numpy())))
                    errors[a] = error
                    if error > 2e-5:
                        raise RuntimeError(f"Frozen benchmark/production parity failed: {row['ablation_id']} {view} {a} error={error}")
                parities.append({"id": row["ablation_id"], "view": view, "errors": errors})
                xx.append(x[0].cpu().numpy()); zz.append(raw[0].cpu().numpy()); ee.append(extra[0].cpu().numpy())
                per_view_seconds.append(time.monotonic() - t0)
            features.append(xx); raw_logits.append(zz); extra_logits.append(ee)
            if index % 30 == 0 or index == len(rows)-1:
                print("ABLATE", index+1, "/", len(rows), "elapsed", round(time.monotonic()-started, 1), flush=True)
    probabilities = {k: np.stack(v) for k, v in probabilities.items()}
    for name in {k.rsplit("_", 1)[0] for k in probabilities}:
        probabilities[name + "_context"] = .5 * (probabilities[name + "_center"] + probabilities[name + "_whole"])
    np.savez_compressed(out / "predictions.npz", **probabilities)
    np.savez_compressed(out / "features.npz", x=np.array(features), raw=np.array(raw_logits), extra=np.array(extra_logits),
                        row_ids=np.array([r["ablation_id"] for r in rows]), views=np.array(choices), labels=np.array(engine.labels))
    save(out / "parity.json", {"checks": parities, "max_error": max(max(r["errors"].values()) for r in parities),
                              "tolerance": 2e-5, "preprocessing": "Production pad implementation copied exactly; frozen outputs checked on EVERY image/view"})
    save(out / "inference_complete.json", {"n": len(rows), "modes": sorted(probabilities), "labels": engine.labels,
                                           "encoder_sha256": engine.encoder_sha256, "temperature": temp,
                                           "elapsed_seconds": time.monotonic()-started,
                                           "predictions_sha256": sha256(out / "predictions.npz"),
                                           "features_sha256": sha256(out / "features.npz"), "training_performed": False})
    del engine, heads
    gc.collect(); torch.cuda.empty_cache()
    verify_backup(out / "pre_training_backup")


def grouped_array(p, raw_labels):
    labels = list(grouped_distribution(dict(zip(raw_labels, p[0]))))
    grouped = [grouped_distribution(dict(zip(raw_labels, row))) for row in p]
    result = np.array([[row[name] for name in labels] for row in grouped])
    return result, labels


def cluster_ids(rows):
    return [r.get("cluster_id") or "observer:" + str(r["observer_id"]) for r in rows]


def paired(rows, a, b):
    d = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    groups = {}
    for i, group in enumerate(cluster_ids(rows)): groups.setdefault(group, []).append(i)
    totals = np.array([(d[v].sum(), len(v)) for v in groups.values()])
    rng = np.random.default_rng(SEED)
    sample = totals[rng.integers(0, len(totals), size=(10000, len(totals)))].sum(axis=1)
    ci = (np.quantile(100 * sample[:, 0] / sample[:, 1], [.025, .975])).tolist()
    fixed, broken = int((~a & b).sum()), int((a & ~b).sum())
    from scipy.stats import binomtest
    return {"gain_pp": float(d.mean()*100), "cluster_ci95_pp": ci, "corrected": fixed, "regressed": broken,
            "clusters": len(groups), "mcnemar_exact_p_image_iid_diagnostic_only": float(binomtest(fixed, fixed+broken, .5).pvalue) if fixed+broken else 1.,
            "note": "Observer/paper clustered interval; paired counts descriptive. Tests are exploratory, no multiple-comparison correction."}


def score(rows, p, labels):
    truth = np.array([labels.index(r["label"]) for r in rows]); pred = p.argmax(1)
    correct = pred == truth
    per_class = []
    for i, label in enumerate(labels):
        positive = truth == i; predicted = pred == i
        tp, fp, fn = int((positive & predicted).sum()), int((~positive & predicted).sum()), int((positive & ~predicted).sum())
        per_class.append({"label": label, "n": int(positive.sum()), "tp": tp, "fp": fp, "fn": fn,
                          "recall": tp/max(1, int(positive.sum())), "precision": tp/max(1, int(predicted.sum())),
                          "f1": 2*tp/max(1, 2*tp+fp+fn)})
    covered = [r for r in per_class if r["n"]]
    return {"n": len(rows), "correct": int(correct.sum()), "top1": float(correct.mean()),
            "top3": float((np.argsort(-p, axis=1)[:, :3] == truth[:, None]).any(1).mean()),
            "covered_classes": len(covered), "macro_recall_covered": float(np.mean([r["recall"] for r in covered])),
            "macro_f1_covered": float(np.mean([r["f1"] for r in covered])), "per_class": per_class}, correct


def summarize(out):
    verify_backup(out / "pre_training_backup")
    complete = load(out / "inference_complete.json")
    assert sha256(out / "predictions.npz") == complete["predictions_sha256"]
    rows = load(out / "rows.json"); raw_labels = complete["labels"]
    with np.load(out / "predictions.npz") as cache: distributions = {k: cache[k] for k in cache.files}
    labels = list(grouped_distribution(dict.fromkeys(raw_labels, 1/len(raw_labels))))
    distributions = {k: grouped_array(p, raw_labels)[0] for k, p in distributions.items()}
    subsets = {
        "common398": [i for i, r in enumerate(rows) if r["cohort"] == "common"],
        "common_without_bauhinia": [i for i, r in enumerate(rows) if r["cohort"] == "common" and r["label"] != BAUHINIA_GROUP],
        "common_hong_kong": [i for i, r in enumerate(rows) if r["cohort"] == "common" and r["hong_kong"]],
        "aquilaria_primary": [i for i, r in enumerate(rows) if r["cohort"] == "rare" and r["source_kind"] == "community_observation" and r["label"] == "Aquilaria sinensis"],
        "rare_primary": [i for i, r in enumerate(rows) if r["cohort"] == "rare" and r["source_kind"] == "community_observation"],
        "paper_challenge": [i for i, r in enumerate(rows) if r["cohort"] == "rare" and r["source_kind"] != "community_observation"],
    }
    result = {"plan": load(out / "plan.json"), "subsets": {}, "training_performed": False, "deployed_files_unchanged": True}
    comparisons = [("zero_context", "adapted_mlp_context"), ("adapted_mlp_context", "sst_sep14_context"),
                   ("sst_sep14_context", "aq_sep15_context"), ("aq_sep15_context", "ar_sep18_context"),
                   ("ar_sep18_context", "h120_context"), ("zero_context", "h120_context"),
                   ("h120_center", "h120_context"), ("h120_whole", "h120_context"),
                   ("h120_no_logit_prior_context", "h120_context"), ("h120_T1_context", "h120_context"),
                   ("linear_lr001_context", "adapted_mlp_context"), ("linear_lr01_context", "adapted_mlp_context")]
    for name, indices in subsets.items():
        if not indices: continue
        rr = [rows[i] for i in indices]
        stats, correct = {}, {}
        for mode, p in distributions.items(): stats[mode], correct[mode] = score(rr, p[indices], labels)
        gains = {a + "_vs_" + b: paired(rr, correct[a], correct[b]) for a, b in comparisons}
        for j in range(1, 4):
            for single in (f"h120_member{j}_context", f"h120_without_member{j}_context"):
                gains[single + "_vs_h120_context"] = paired(rr, correct[single], correct["h120_context"])
        result["subsets"][name] = {"n": len(rr), "modes": stats, "paired": gains}
    main = result["subsets"]["common398"]
    t0 = main["modes"]["zero_center"]["top1"]; t1 = main["modes"]["zero_context"]["top1"]
    s0 = main["modes"]["h120_center"]["top1"]; s1 = main["modes"]["h120_context"]["top1"]
    result["two_factor_decomposition"] = {"reference": "zero_center -> h120_context, NOT the +6.78pp same-view comparison",
        "head_shapley_pp": 50*((s0-t0)+(s1-t1)), "view_shapley_pp": 50*((t1-t0)+(s1-s0)),
        "total_pp": 100*(s1-t0), "interaction_pp": 100*((s1-s0)-(t1-t0))}
    # Undo only Bauhinia presentation merging; truth uses frozen source taxonomy mapping.
    with np.load(out / "predictions.npz") as cache:
        ungrouped = {}
        common_ids = subsets["common398"]
        raw_truth = []
        for i in common_ids:
            row = rows[i]
            target = row["label"]
            if target == BAUHINIA_GROUP:
                candidates = [x for x in (row["query_term"], row["source_species"]) if x in BAUHINIA_MEMBERS]
                if not candidates: raise RuntimeError("Cannot unambiguously undo source label grouping")
                target = candidates[0]
            raw_truth.append(raw_labels.index(target))
        for mode in ("zero_context", "h120_context"):
            correct = cache[mode][common_ids].argmax(1) == np.array(raw_truth)
            ungrouped[mode] = {"correct": int(correct.sum()), "n": len(correct), "top1": float(correct.mean()),
                               "aggregation_gain_pp": 100*(main["modes"][mode]["top1"] - correct.mean())}
    result["bauhinia_presentation_only"] = {"ungrouped_148": ungrouped,
        "grouped_146_gain_pp": main["paired"]["zero_context_vs_h120_context"]["gain_pp"],
        "ungrouped_148_gain_pp": 100*(ungrouped["h120_context"]["top1"]-ungrouped["zero_context"]["top1"]),
        "note": "Ficus and Araucaria already pooled inside class mapping; not ablated. Both main models always use the SAME grouping."}
    # Save every changed example with original attribution for reproducible error inspection.
    changed = {}
    rr = [rows[i] for i in subsets["common398"]]
    for a, b in comparisons:
        pa, pb = distributions[a][subsets["common398"]].argmax(1), distributions[b][subsets["common398"]].argmax(1)
        changed[a + "_vs_" + b] = [{"id": r["ablation_id"], "truth": r["label"], "before": labels[int(x)], "after": labels[int(y)],
                                    "source_image": r["source_image"], "source_page": r.get("source_page"),
                                    "before_correct": labels[int(x)] == r["label"], "after_correct": labels[int(y)] == r["label"]}
                                   for r, x, y in zip(rr, pa, pb) if x != y]
    save(out / "changed_examples.json", changed)
    save(out / "results.json", result)
    report(out, result)
    verify_backup(out / "pre_training_backup")
    save(out / "production_verification.json", {"protected_sources_unchanged": True,
        "backup_manifest": str(out / "pre_training_backup/backup_manifest.json"), "training_performed": False,
        "deployment_changed": False, "encoder_changed": False})


def report(out, result):
    main = result["subsets"]["common398"]; n = main["n"]
    lines = ["# SST 树种识别精度归因与消融实验", "", "本轮不训练、不部署；所有检查点和设置在评分前冻结。398 张常见树图片是已经看过的回归集，不是新增盲测。", "",
             "## 1. 历史模型阶段，固定相同双取景", "", "| 阶段 | 识对 / 398 | Top-1 | 相比上一步（百分点） |", "| --- | ---: | ---: | ---: |"]
    before = None
    for stage in ("zero", "adapted_mlp", "sst_sep14", "aq_sep15", "ar_sep18", "h120"):
        stat = main["modes"][stage + "_context"]
        delta = "—" if before is None else f"{100*(stat['top1']-before):+.2f}"
        lines.append(f"| {STAGE_NAMES[stage]} | {stat['correct']}/{n} | {stat['top1']:.2%} | {delta} |")
        before = stat["top1"]
    lines += ["", "历史阶段差分不等于独立数据消融：各阶段同时改变数据、权重和优化步数。可确认累计改动对同图表现的影响，不能把百分点严格分摊到某八张图或某一损失项。", "",
              "## 2. 取景处理，固定同一模型权重", "", "| 模型 | 中心取景 | 全图补边 | 全图＋中心平均 |", "| --- | ---: | ---: | ---: |"]
    for stage in ("zero", "adapted_mlp", "h120"):
        lines.append("| " + STAGE_NAMES[stage] + " | " + " | ".join(f"{main['modes'][stage+'_'+v]['correct']}/{n} · {main['modes'][stage+'_'+v]['top1']:.2%}" for v in ("center", "whole", "context")) + " |")
    lines += ["", "同模型取景差分、线性头、先验、温度和集成对照：", "", "| 单项对照 | 净变化（百分点） | 改对 / 改错 | 观察者成组95%区间 |", "| --- | ---: | ---: | --- |"]
    for key, value in main["paired"].items():
        ci = value["cluster_ci95_pp"]
        lines.append(f"| {key} | {value['gain_pp']:+.2f} | {value['corrected']} / {value['regressed']} | [{ci[0]:+.2f}, {ci[1]:+.2f}] |")
    lines += ["", "注意：移除 teacher-logit prior 是训练后推理敏感性，不等于从零训练纯 MLP；温度固定在原验证选择值，T=1 只做诊断。线性头学习率不同，不能声称纯结构因果效应。", "",
              "## 3. 三头集成", "", "| H120 版本 | 识对 / 398 | Top-1 |", "| --- | ---: | ---: |"]
    for mode in [f"h120_member{j}_context" for j in range(1,4)] + ["h120_context"]:
        s = main["modes"][mode]; lines.append(f"| {mode} | {s['correct']}/{n} | {s['top1']:.2%} |")
    avg = np.mean([main["modes"][f"h120_member{j}_context"]["top1"] for j in range(1,4)])
    lines += ["", f"集成相对三个固定单头平均准确率的差值为 {100*(main['modes']['h120_context']['top1']-avg):+.2f} 个百分点；不从本回归集挑最佳单头。", "",
              "## 4. 类别合并不是训练提升", "", "| 规则 | 原 BioCLIP | H120 | 同规则模型提升 |", "| --- | ---: | ---: | ---: |"]
    raw = result["bauhinia_presentation_only"]["ungrouped_148"]
    lines += [f"| 不合并洋紫荊（148） | {raw['zero_context']['top1']:.2%} | {raw['h120_context']['top1']:.2%} | {result['bauhinia_presentation_only']['ungrouped_148_gain_pp']:+.2f} |",
              f"| 两边都合并洋紫荊（146） | {main['modes']['zero_context']['top1']:.2%} | {main['modes']['h120_context']['top1']:.2%} | {main['paired']['zero_context_vs_h120_context']['gain_pp']:+.2f} |",
              "", "榕树和南洋杉已经在原始类别映射中合并，本实验不具有拆开它们的鉴定能力。主对比两边规则一致，不能把合并后增加的绝对成绩冒充模型学习贡献。", "",
              "## 5. 土沉香专项回归，社区观察照片", "", "| 阶段（相同双取景） | 识对 / 样本数 | 正类召回率 |", "| --- | ---: | ---: | ---: |"]
    aq = result["subsets"]["aquilaria_primary"]
    for stage in ("zero", "adapted_mlp", "sst_sep14", "aq_sep15", "ar_sep18", "h120"):
        s = aq["modes"][stage + "_context"]
        lines.append(f"| {STAGE_NAMES[stage]} | {s['correct']}/{aq['n']} | {s['top1']:.2%} |")
    lines += ["", "仅正类样本的百分比是召回率，不是排除其他树种误报后的精确率；纸刊挑战图片单列，不混进主百分比。", "",
              "## 6. 可以和不可以归因的事项", "",
              "- 可以：冻结同图、编码器、提示词、标签映射后的分类头累计变化；同权重取景变化；固定集成成员的敏感性。",
              "- 不可以：用这一轮数据证明 SAM/自动检测/跟踪/点云贡献。原 +6.78pp 数据没有经过这些模块。",
              "- 不可以：声称解冻编码器提升。历史和当前使用同一 SHA256 编码器，本轮未解冻。",
              "- 不可以：从阶段差分推断每项补充数据、类别均衡、KL、数据增强的独立贡献。需要从同一初始化重新拟合并保持优化/分割/种子一致。",
              "- 所有阶段完整逐类召回、精确率、F1 和退步图片保存在 results.json / changed_examples.json；没有按预测删除样本。",
              "- 常见树结果只有 101/146 个有样本类别、17 张香港照片；不能当作全部类别香港街景总体精度。", "",
              "## 安全与复现", "", "- 原始模型、所有历史/当前分类头、协议、数据清单、标签映射和原评测均已物理备份并 SHA256 校验。",
              f"- 备份：{out / 'pre_training_backup'}", "- 正式网页、安卓、点云及融合权重均未改动；没有模型切换或重新校准。",
              "- 输入文件 SHA256 每张核验；原零样本和当前模型输出与此前正式推理逐张逐取景核对。",
              "- 对回归集做多项探索分析，区间不包含标签/地域偏差，不能用来选择新的生产版本。"]
    path = out / "REPORT_ZH.md"
    if path.exists(): raise FileExistsError(path)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "infer", "summarize", "all"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(); out = args.out.resolve()
    if args.phase in ("prepare", "all"): prepare(out)
    if args.phase in ("infer", "all"): infer(out)
    if args.phase in ("summarize", "all"): summarize(out)
