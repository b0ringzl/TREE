"""Conservative, separately gated Jacaranda embedding adapter experiment.

User images are training data. Development selects the override threshold; the
already inspected test is regression evidence only. This script never deploys.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]

import joblib
import numpy as np
import torch
from sklearn.svm import SVC

from audit_artocarpus_preprocessing_20260917 import exact_demo_prepare
from distillation_core import read_json, save_json
from train_artocarpus_user_20260918 import protected
from aq_ar_dualview_next_20260919 import load_pair
from training_safety import backup_before_training, verify_backup, sha256
from tree_species_teacher_demo.engine import Engine, SSTV1_MANIFEST, decode

OUT = ROOT / "outputs" / "SSTv1_Jacaranda_Adapter_20260922_v2"
SOURCE = ROOT / "outputs" / "SSTv1_Jacaranda_User_20260922_v3"
PARENT = ROOT / "outputs" / "SSTv1_Aquilaria_UserCrops_20260919_v2"
TARGET = 75
SEED = 20260922


def normalize(array):
    array = np.asarray(array, dtype="float32")
    return array / np.linalg.norm(array, axis=1, keepdims=True).clip(min=1e-12)


def pair_feature(x, xc):
    return normalize(normalize(x) + normalize(xc))


def source_paths():
    manifest = read_json(SSTV1_MANIFEST)
    rows = read_json(SOURCE / "accepted_data_manifest.json")
    return list(dict.fromkeys([Path(__file__), ROOT / "tools" / "training_safety.py", SSTV1_MANIFEST,
        *[SSTV1_MANIFEST.parent / m["file"] for m in manifest["members"]],
        SOURCE / "accepted_data_manifest.json", SOURCE / "split_assignment.json",
        SOURCE / "config.json", SOURCE / "protocol.json", SOURCE / "user_features.npz",
        SOURCE / "baseline_predictions.npz", SOURCE / "baseline_evaluation.json",
        PARENT / "train_rows.json", PARENT / "train_features.npz",
        PARENT / "train_center_features.npz",
        *[Path(r["original_copy"]) for r in rows],
        *[Path(r["source_image"]) for r in rows]]))


def preflight():
    if OUT.exists():
        raise RuntimeError("Versioned output already exists")
    OUT.mkdir(parents=True)
    paths = source_paths()
    rows = read_json(SOURCE / "accepted_data_manifest.json")
    if len(rows) != 6 or any(r["target"] != TARGET for r in rows):
        raise RuntimeError("The six user-confirmed targets changed")
    manifest = read_json(SSTV1_MANIFEST)
    config = dict(seed=SEED, source_run=str(SOURCE), method="RBF SVC on normalized mean of the two Demo views",
                  kernel="rbf", C=2.0, gamma=2.0, class_weight="balanced",
                  old_jacaranda_weight=1.0, user_crop_weight=4.0, user_original_weight=4.0,
                  minimum_svc_margin=0.5, development_negative_gap=0.25,
                  threshold="max(SVC margin 0.5, maximum development non-Jacaranda margin + 0.25)",
                  selection="Fixed development only; no additional false positives allowed",
                  six_user_images="training only, grouped into two sites; not independent accuracy evidence",
                  historical_test="previously inspected; regression only", deployment=False)
    save_json(OUT / "config.json", config)
    save_json(OUT / "accepted_data_manifest.json", rows)
    save_json(OUT / "split_assignment.json", dict(new_photos="train_only", old_train=str(PARENT / "train_rows.json"),
        development="fixed 2655 images", historical_test="regression_only", site_groups=sorted({r["source_group"] for r in rows})))
    save_json(OUT / "labels.json", read_json(SOURCE / "labels.json"))
    save_json(OUT / "protocol.json", dict(created_at=datetime.now(timezone.utc).isoformat(),
        deployment_manifest=manifest, sources={str(p.resolve()): sha256(p) for p in paths}))
    print("PREFLIGHT", len(paths), "sources", flush=True)


def verify_sources(protocol):
    for path, expected in protocol["sources"].items():
        if sha256(path) != expected:
            raise RuntimeError("Source changed: " + path)


def features():
    protocol = read_json(OUT / "protocol.json")
    verify_sources(protocol)
    torch.set_num_threads(4)
    engine = Engine()
    engine.load()
    if engine.sstv1_version != protocol["deployment_manifest"]["version"]:
        raise RuntimeError("Initializer changed")
    prep = exact_demo_prepare()
    rows = read_json(OUT / "accepted_data_manifest.json")
    embeddings = []
    with torch.inference_mode():
        for row in rows:
            blob = Path(row["original_copy"]).read_bytes()
            views = [prep(blob)[0], blob]
            pieces = []
            for content in views:
                pixels = decode(content)[0]
                embedding = engine.model.encode_image(((pixels[None].cuda() - engine.mean) / engine.std).to(engine.dtype), normalize=True).float()
                pieces.append(embedding.cpu().numpy()[0])
            embeddings.append(pieces)
    np.savez_compressed(OUT / "original_features.npz", x=np.asarray(embeddings)[:, 0],
                        xc=np.asarray(embeddings)[:, 1], encoder_sha256=engine.encoder_sha256)
    save_json(OUT / "features_manifest.json", dict(original_features_sha256=sha256(OUT / "original_features.npz"),
        user_crop_features_sha256=sha256(SOURCE / "user_features.npz"), encoder_sha256=engine.encoder_sha256))
    print("FEATURES", len(rows), flush=True)


def fit():
    protocol = read_json(OUT / "protocol.json")
    verify_sources(protocol)
    if (OUT / "pre_training_backup").exists():
        raise RuntimeError("Fit already attempted; use a new versioned directory")
    config = read_json(OUT / "config.json")
    feature_manifest = read_json(OUT / "features_manifest.json")
    if sha256(OUT / "original_features.npz") != feature_manifest["original_features_sha256"]:
        raise RuntimeError("Original feature cache changed")
    if sha256(SOURCE / "user_features.npz") != feature_manifest["user_crop_features_sha256"]:
        raise RuntimeError("Crop feature cache changed")
    train_rows = read_json(PARENT / "train_rows.json")
    with np.load(PARENT / "train_features.npz") as whole, np.load(PARENT / "train_center_features.npz") as center:
        old_x = pair_feature(whole["x"], center["x"])
    with np.load(SOURCE / "user_features.npz") as crop, np.load(OUT / "original_features.npz") as original:
        crop_x = pair_feature(crop["x"], crop["xc"])
        original_x = pair_feature(original["x"], original["xc"])
    y_old = np.asarray([r["target"] == TARGET for r in train_rows], dtype="int32")
    x_train = np.concatenate([old_x, crop_x, original_x])
    y_train = np.concatenate([y_old, np.ones(12, dtype="int32")])
    weights = np.concatenate([np.ones(len(old_x), dtype="float32"),
                              np.full(12, config["user_crop_weight"], dtype="float32")])
    save_json(OUT / "training_manifest.json", dict(old_train_n=len(old_x), old_jacaranda_n=int(y_old.sum()),
        user_photographs=6, user_views_per_photo=2, user_originals_and_crops_are_not_independent=True,
        train_feature_dimension=x_train.shape[1], parameter_config=config))
    with np.load(SOURCE / "baseline_predictions.npz") as base:
        base_dev = base["development"]
    dev = load_pair(PARENT, "dev")
    dev_rows = dev[0]
    baseline_dev = dict(correct=int((base_dev.argmax(1) == np.asarray([r["target"] for r in dev_rows])).sum()),
                        jacaranda_tp=int(((base_dev.argmax(1) == TARGET) & (np.asarray([r["target"] for r in dev_rows]) == TARGET)).sum()),
                        jacaranda_fp=int(((base_dev.argmax(1) == TARGET) & (np.asarray([r["target"] for r in dev_rows]) != TARGET)).sum()))
    save_json(OUT / "baseline_evaluation.json", baseline_dev)
    # Initialization is the frozen SSTv1 encoder/heads; the SVC is newly fitted.
    inputs = list(dict.fromkeys([*protected(), *source_paths(), OUT / "config.json",
        OUT / "accepted_data_manifest.json", OUT / "split_assignment.json", OUT / "labels.json",
        OUT / "protocol.json", OUT / "original_features.npz", OUT / "features_manifest.json",
        OUT / "training_manifest.json", OUT / "baseline_evaluation.json"]))
    backup = backup_before_training(OUT, inputs, metadata=dict(seed=SEED,
        frozen_initial_weights=protocol["deployment_manifest"]["version"], task="Jacaranda gated correction adapter"))
    verify_backup(backup)
    print("BACKUP_VERIFIED", backup, flush=True)
    model = SVC(C=config["C"], gamma=config["gamma"], class_weight=config["class_weight"], random_state=SEED)
    model.fit(x_train, y_train, sample_weight=weights)
    joblib.dump(model, OUT / "adapter.joblib")
    dev_x = pair_feature(dev[1].cpu().numpy(), dev[3].cpu().numpy())
    dev_score = model.decision_function(dev_x)
    old_pred = base_dev.argmax(1)
    dev_true = np.asarray([r["target"] for r in dev_rows])
    eligible_negative = (dev_true != TARGET) & (old_pred != TARGET)
    threshold = float(max(config["minimum_svc_margin"],
                          dev_score[eligible_negative].max() + config["development_negative_gap"]))
    proposed = old_pred.copy()
    proposed[(old_pred != TARGET) & (dev_score > threshold)] = TARGET
    validation = dict(baseline=baseline_dev, candidate=dict(correct=int((proposed == dev_true).sum()),
        jacaranda_tp=int(((proposed == TARGET) & (dev_true == TARGET)).sum()),
        jacaranda_fp=int(((proposed == TARGET) & (dev_true != TARGET)).sum())),
        threshold=threshold, max_non_jacaranda_dev_margin=float(dev_score[eligible_negative].max()),
        development_top_score=float(dev_score.max()))
    user_crop_scores = model.decision_function(crop_x).tolist()
    user_original_scores = model.decision_function(original_x).tolist()
    save_json(OUT / "selection_lock.json", dict(validation=validation,
        user_crop_scores=user_crop_scores, user_original_scores=user_original_scores,
        user_crops_above_threshold=sum(v > threshold for v in user_crop_scores),
        user_originals_above_threshold=sum(v > threshold for v in user_original_scores),
        selected_on="fixed development", adapter_sha256=sha256(OUT / "adapter.joblib"), deployed=False))
    verify_backup(backup)
    verify_sources(protocol)
    print("LOCKED", "threshold", threshold, "crop", sum(v > threshold for v in user_crop_scores),
          "original", sum(v > threshold for v in user_original_scores), flush=True)


def regression():
    lock = read_json(OUT / "selection_lock.json")
    if sha256(OUT / "adapter.joblib") != lock["adapter_sha256"]:
        raise RuntimeError("Adapter changed")
    if (OUT / "regression_results.json").exists():
        raise RuntimeError("Regression already exists")
    model = joblib.load(OUT / "adapter.joblib")
    test = load_pair(PARENT, "test")
    test_x = pair_feature(test[1].cpu().numpy(), test[3].cpu().numpy())
    scores = model.decision_function(test_x)
    # Baseline predictions use the pinned deployed SSTv1 heads, exact two-view features.
    from train_artocarpus_user_20260918 import heads_from_deployment
    from aq_ar_dualview_next_20260919 import tensor, paired_predict
    with np.load(PARENT / "train_features.npz") as data:
        params = dict(groups=read_json(PARENT / "classes.json")["groups"],
            temp=read_json(SSTV1_MANIFEST)["temperature"], extra=tensor(data["extra_bank"]),
            scale=float(data["scale"]))
    torch.set_num_threads(4)
    old_prob = paired_predict(heads_from_deployment(read_json(OUT / "protocol.json")), test, params)
    old_pred = old_prob.argmax(1)
    actual = np.asarray([r["target"] for r in test[0]])
    changed = (old_pred != TARGET) & (scores > lock["validation"]["threshold"])
    proposed = old_pred.copy()
    proposed[changed] = TARGET
    save_json(OUT / "regression_results.json", dict(n=len(actual),
        baseline_correct=int((old_pred == actual).sum()), candidate_correct=int((proposed == actual).sum()),
        added_true_positives=int((changed & (actual == TARGET)).sum()),
        added_false_positives=int((changed & (actual != TARGET)).sum()), changed_indices=np.where(changed)[0].tolist(),
        interpretation="Previously inspected historical test; regression check, not new unbiased accuracy evidence.",
        deployed=False))
    print("REGRESSION", int(changed.sum()), "changed", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in {"preflight", "features", "fit", "regression"}:
        raise SystemExit("Usage: train_jacaranda_adapter_20260922.py preflight|features|fit|regression")
    {"preflight": preflight, "features": features, "fit": fit, "regression": regression}[sys.argv[1]]()
