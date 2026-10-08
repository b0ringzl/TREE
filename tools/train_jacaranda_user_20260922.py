"""User-confirmed Jacaranda correction trial; validation selection, no auto-deploy.

The six provided pictures are training evidence, not six independent trees or
held-out accuracy. Each crop is a recorded approximation of the centered target.
"""

from __future__ import annotations

import argparse
import gc
import io
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

from audit_artocarpus_preprocessing_20260917 import exact_demo_prepare
from distillation_core import read_json, save_json, digest
from train_artocarpus_user_20260918 import protected, heads_from_deployment, predict, binary, cpu_state
from aq_ar_dualview_next_20260919 import tensor, load_pair, paired_predict, paired_logprob
from training_safety import backup_before_training, verify_backup, sha256
from tree_species_teacher_demo.engine import Engine, SSTV1_MANIFEST, decode

OUT = ROOT / "outputs" / "SSTv1_Jacaranda_User_20260922_v3"
PARENT = ROOT / "outputs" / "SSTv1_Aquilaria_UserCrops_20260919_v2"
AUDIT = ROOT / "outputs" / "artocarpus_preprocess_audit_20260917_v1"
SEED = 20260922
TARGET = 75  # Jacaranda mimosifolia in the fixed 148-class map.
SAMPLES = [
    ("C:/Users/25041725G/Desktop/验证/Snipaste_2026-09-10_14-38-57.png", "site_a", (.28, .14, .70, 1.0)),
    ("C:/Users/25041725G/Desktop/验证/Snipaste_2026-09-10_15-54-52.png", "site_a", (0, 0, 1, 1)),
    ("C:/Users/25041725G/Desktop/验证/微信图片_20260910144105_125_2.jpg", "site_b", (.15, 0, .60, .72)),
    ("C:/Users/25041725G/Desktop/test/yhh.png", "site_a", (0, 0, .57, 1)),
    ("C:/Users/25041725G/Desktop/test/yhh2.png", "site_a", (.38, .27, .64, 1)),
    ("C:/Users/25041725G/Desktop/test/微信图片_20260908143139_110_2.jpg", "site_b", (.14, 0, .82, 1)),
]


def checked_sources(protocol):
    for path, expected in protocol["source_sha256"].items():
        if sha256(path) != expected:
            raise RuntimeError("Source changed: " + path)


def prepare(out: Path):
    out.mkdir(parents=True, exist_ok=False)
    (out / "user_originals").mkdir()
    (out / "target_crops").mkdir()
    labels = read_json(PARENT / "labels.json")
    if len(labels) != 148 or labels[TARGET] != "Jacaranda mimosifolia":
        raise RuntimeError("148-class label map changed")
    dep = read_json(SSTV1_MANIFEST)
    if dep["version"] != "sstv1-aquilaria-user-crops-20260919-h120":
        raise RuntimeError("Deployed initializer changed")
    rows = []
    for i, (source, site, box) in enumerate(SAMPLES, 1):
        source = Path(source).resolve()
        raw = out / "user_originals" / f"{i:02d}{source.suffix.lower()}"
        shutil.copy2(source, raw)
        if sha256(source) != sha256(raw):
            raise RuntimeError("User image copy failed")
        with Image.open(raw) as handle:
            image = ImageOps.exif_transpose(handle).convert("RGB")
        width, height = image.size
        left, top, right, bottom = box
        pixel_box = (round(left * width), round(top * height), round(right * width), round(bottom * height))
        crop = image.crop(pixel_box)
        if min(crop.size) < 224:
            raise RuntimeError("Target crop too small: " + source.name)
        crop_path = out / "target_crops" / f"{i:02d}.png"
        crop.save(crop_path)
        rows.append(dict(sample_id=f"user_jacaranda_{i:02d}", source_image=str(crop_path.resolve()),
                         original_source=str(source), original_copy=str(raw.resolve()),
                         original_sha256=sha256(raw), sha256=sha256(crop_path),
                         source_group=site, group_id=f"user_jacaranda|{site}",
                         original_size=[width, height], crop_box=list(pixel_box), target=TARGET,
                         species="Jacaranda mimosifolia", phase="train", training_eligible=True,
                         label_authority="User confirmed centered target tree as 藍花楹 on 2026-09-22",
                         boundary_authority="Researcher approximate rectangle; not user-reviewed pixel mask",
                         rights="Private training/evaluation only; no redistribution assertion"))
    old_train = read_json(PARENT / "train_rows.json")
    dev = read_json(AUDIT / "rows.json")
    prior_hashes = {r.get("sha256") or r.get("source_sha256") for r in old_train + dev}
    if any(r["original_sha256"] in prior_hashes or r["sha256"] in prior_hashes for r in rows):
        raise RuntimeError("User image overlaps old training or development by SHA-256")
    source_files = [SSTV1_MANIFEST, *[SSTV1_MANIFEST.parent / m["file"] for m in dep["members"]],
                    PARENT / "train_rows.json", PARENT / "train_features.npz", PARENT / "train_center_features.npz",
                    AUDIT / "rows.json", AUDIT / "paired_features_predictions.npz", Path(__file__),
                    ROOT / "tools" / "training_safety.py"]
    config = dict(seed=SEED, target=TARGET, target_name=labels[TARGET], steps=120, eval_every=10,
                  learning_rate=1e-5, weight_decay=.01, new_loss_weight=.35,
                  hard_negative_loss_weight=.7, hard_negative_count=200, replay_kl_weight=1.5,
                  source_groups={"site_a": "First, second, fourth and fifth supplied views; same scene family",
                                 "site_b": "Third and sixth supplied views; same scene family"},
                  trainable="Three existing native-148 MLP heads only; frozen BioCLIP encoder/text bank/temperature",
                  inference="Exact Demo whole letterbox + original center view, equal probability mean",
                  selection="Validation only: Jacaranda F1 cannot drop; no Jacaranda TP drop, no additional FP, <=1 overall lost correct, macro-F1 drop <0.001; no Aquilaria/Artocarpus TP drop or FP rise. Initial checkpoint is allowed.",
                  user_images="All six and their crops are training-only; no accuracy claim on them",
                  development="Existing 2,655-image previously inspected development set; regression evidence, not fresh unbiased test",
                  deployment=False)
    save_json(out / "accepted_data_manifest.json", rows)
    save_json(out / "split_assignment.json", {"new_user_images": "train_only", "development": str(AUDIT / "rows.json"),
                                              "historical_test": "regression_only_after_selection", "source_groups": config["source_groups"]})
    save_json(out / "labels.json", labels)
    save_json(out / "config.json", config)
    save_json(out / "protocol.json", dict(created_at=datetime.now(timezone.utc).isoformat(),
                                            deployment_manifest=dep, config=config,
                                            source_sha256={str(p.resolve()): sha256(p) for p in source_files},
                                            accepted_sha256={r["source_image"]: r["sha256"] for r in rows},
                                            original_sha256={r["original_copy"]: r["original_sha256"] for r in rows}))
    print("PREPARED", len(rows), "photographs,", len({r["source_group"] for r in rows}), "scene groups", flush=True)


def features(out: Path):
    protocol = read_json(out / "protocol.json")
    checked_sources(protocol)
    rows = read_json(out / "accepted_data_manifest.json")
    if (out / "user_features.npz").exists():
        raise RuntimeError("Features already exist; use a new run to retry")
    torch.set_num_threads(4)
    engine = Engine()
    engine.load()
    if engine.sstv1_version != protocol["deployment_manifest"]["version"]:
        raise RuntimeError("Loaded classifier differs from pinned initializer")
    production = exact_demo_prepare()
    arrays = {name: [] for name in ("x", "z", "xc", "zc")}
    with torch.inference_mode():
        for row in rows:
            if sha256(row["source_image"]) != row["sha256"]:
                raise RuntimeError("User crop changed")
            blob = Path(row["source_image"]).read_bytes()
            for content, xkey, zkey in ((production(blob)[0], "x", "z"), (blob, "xc", "zc")):
                pixels = decode(content)[0]
                embedding = engine.model.encode_image(((pixels[None].cuda() - engine.mean) / engine.std).to(engine.dtype), normalize=True).float()
                arrays[xkey].append(embedding.half().cpu().numpy())
                arrays[zkey].append((embedding @ engine.bank.T * engine.model.logit_scale.exp().float()).cpu().numpy())
    np.savez_compressed(out / "user_features.npz", **{k: np.concatenate(v) for k, v in arrays.items()},
                        rows_hash=digest(rows), encoder_sha256=engine.encoder_sha256)
    save_json(out / "FEATURES_COMPLETE.json", dict(user_features_sha256=sha256(out / "user_features.npz"),
                                                    encoder_sha256=engine.encoder_sha256, rows=len(rows)))
    print("FEATURES_COMPLETE", len(rows), flush=True)


def summary(rows, probabilities):
    target = np.asarray([r["target"] for r in rows])
    predicted = probabilities.argmax(1)
    count = int((predicted == target).sum())
    # Fixed 148-class macro-F1; labels absent from this development set contribute zero.
    f1s = []
    for klass in range(148):
        tp = int(((target == klass) & (predicted == klass)).sum())
        fp = int(((target != klass) & (predicted == klass)).sum())
        fn = int(((target == klass) & (predicted != klass)).sum())
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.)
    return dict(n=len(rows), correct=count, top1=count / len(rows), macro_f1=float(np.mean(f1s)),
                jacaranda=binary(target, probabilities, TARGET),
                aquilaria=binary(target, probabilities, 11), artocarpus=binary(target, probabilities, 16))


def qualifies(candidate, baseline):
    j, b = candidate["jacaranda"], baseline["jacaranda"]
    if j["tp"] < b["tp"] or j["fp"] > b["fp"] or j["f1"] < b["f1"]:
        return False
    if candidate["correct"] < baseline["correct"] - 1 or candidate["macro_f1"] < baseline["macro_f1"] - .001:
        return False
    for key in ("aquilaria", "artocarpus"):
        if candidate[key]["tp"] < baseline[key]["tp"] or candidate[key]["fp"] > baseline[key]["fp"]:
            return False
    return True


def fit(out: Path):
    protocol = read_json(out / "protocol.json")
    checked_sources(protocol)
    if (out / "fit").exists():
        raise RuntimeError("A fit already exists; retries require a new versioned run")
    features_complete = read_json(out / "FEATURES_COMPLETE.json")
    if sha256(out / "user_features.npz") != features_complete["user_features_sha256"]:
        raise RuntimeError("User feature cache changed")
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    parent = load_pair(PARENT, "train")
    dev = load_pair(PARENT, "dev")
    user_rows = read_json(out / "accepted_data_manifest.json")
    with np.load(out / "user_features.npz") as data:
        if str(data["rows_hash"]) != digest(user_rows):
            raise RuntimeError("User feature order changed")
        new = (user_rows, *[tensor(data[k]) for k in ("x", "z", "xc", "zc")])
    with np.load(PARENT / "train_features.npz") as data:
        params = dict(groups=read_json(PARENT / "classes.json")["groups"],
                      temp=protocol["deployment_manifest"]["temperature"],
                      extra=tensor(data["extra_bank"]), scale=float(data["scale"]))
    heads = heads_from_deployment(protocol)
    initial = cpu_state(heads)
    baseline_dev = paired_predict(heads, dev, params)
    baseline_user = paired_predict(heads, new, params)
    baseline = summary(dev[0], baseline_dev)
    save_json(out / "baseline_evaluation.json", dict(development=baseline,
                                                       user_train_correct=int((baseline_user.argmax(1) == TARGET).sum()),
                                                       user_train_jacaranda_scores=[float(p[TARGET]) for p in baseline_user]))
    np.savez_compressed(out / "baseline_predictions.npz", development=baseline_dev, user_train=baseline_user)
    torch.save(dict(model_state=initial), out / "initial.pt")
    run = out / "fit"
    run.mkdir(exist_ok=False)
    save_json(run / "config.json", protocol["config"])
    replay_rows = parent[0]
    with torch.no_grad():
        similarity = torch.stack([parent[1] @ new[1].T, parent[1] @ new[3].T,
                                  parent[3] @ new[1].T, parent[3] @ new[3].T]).amax(dim=(0, 2))
        similarity[torch.tensor([r["target"] == TARGET for r in replay_rows], device="cuda")] = -1
        hard_indices = similarity.topk(protocol["config"]["hard_negative_count"]).indices
    save_json(run / "hard_negative_rows.json", [dict(train_index=int(i),
        group_id=replay_rows[int(i)]["group_id"], target=replay_rows[int(i)]["target"],
        similarity=float(similarity[int(i)])) for i in hard_indices.cpu().tolist()])
    backup_inputs = list(dict.fromkeys([*protected(), *[Path(p) for p in protocol["source_sha256"]],
        *[Path(p) for p in protocol["accepted_sha256"]], *[Path(p) for p in protocol["original_sha256"]],
        out / "initial.pt", out / "user_features.npz", out / "accepted_data_manifest.json",
        out / "split_assignment.json", out / "labels.json", out / "config.json", out / "protocol.json",
        out / "baseline_evaluation.json", out / "baseline_predictions.npz", run / "config.json",
        run / "hard_negative_rows.json"]))
    backup = backup_before_training(run, backup_inputs, metadata=dict(seed=SEED, initial_version=protocol["deployment_manifest"]["version"],
                                                              data_role="six user-confirmed training photographs; two scene groups"))
    verify_backup(backup)
    print("BACKUP_VERIFIED", backup, flush=True)
    counts = Counter((r["target"], r["group_id"]) for r in replay_rows)
    class_groups = {target: len({r["group_id"] for r in replay_rows if r["target"] == target})
                    for target in {r["target"] for r in replay_rows}}
    replay_weights = tensor([1 / (class_groups[r["target"]] * counts[(r["target"], r["group_id"])]) for r in replay_rows])
    replay_weights /= replay_weights.sum()
    new_counts = Counter(r["group_id"] for r in user_rows)
    new_weights = tensor([1 / (len(new_counts) * new_counts[r["group_id"]]) for r in user_rows])
    labels_replay = torch.tensor([r["target"] for r in replay_rows], device="cuda")
    labels_new = torch.full((len(user_rows),), TARGET, device="cuda", dtype=torch.long)
    with torch.no_grad():
        anchor = paired_logprob(heads, parent, params).exp().detach()
    optimizer = torch.optim.AdamW(heads.parameters(), lr=protocol["config"]["learning_rate"], weight_decay=.01)
    history = []
    best = dict(step=0, checkpoint=str(out / "initial.pt"), metrics=baseline, user_train_correct=int((baseline_user.argmax(1) == TARGET).sum()))
    for step in range(1, protocol["config"]["steps"] + 1):
        heads.train()
        optimizer.zero_grad(set_to_none=True)
        old_logp = paired_logprob(heads, parent, params)
        new_logp = paired_logprob(heads, new, params)
        old_loss = (F.nll_loss(old_logp, labels_replay, reduction="none") * replay_weights).sum()
        hard_loss = F.nll_loss(old_logp[hard_indices], labels_replay[hard_indices])
        new_loss = (F.nll_loss(new_logp, labels_new, reduction="none") * new_weights).sum()
        kl = F.kl_div(old_logp, anchor, reduction="batchmean")
        loss = (old_loss + protocol["config"]["hard_negative_loss_weight"] * hard_loss
                + protocol["config"]["new_loss_weight"] * new_loss + protocol["config"]["replay_kl_weight"] * kl)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(heads.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        if step % protocol["config"]["eval_every"]:
            continue
        dev_prob = paired_predict(heads, dev, params)
        user_prob = paired_predict(heads, new, params)
        metrics = summary(dev[0], dev_prob)
        user_correct = int((user_prob.argmax(1) == TARGET).sum())
        eligible = qualifies(metrics, baseline)
        checkpoint = run / f"step_{step:03d}.pt"
        torch.save(dict(model_state=cpu_state(heads), step=step, seed=SEED), checkpoint)
        item = dict(step=step, loss=float(loss.detach()), old_loss=float(old_loss.detach()),
                    hard_loss=float(hard_loss.detach()),
                    new_loss=float(new_loss.detach()), kl=float(kl.detach()), metrics=metrics,
                    user_train_correct=user_correct, eligible=eligible, checkpoint=str(checkpoint))
        history.append(item)
        if eligible and (metrics["jacaranda"]["f1"], metrics["correct"], metrics["macro_f1"]) > (
            best["metrics"]["jacaranda"]["f1"], best["metrics"]["correct"], best["metrics"]["macro_f1"]):
            best = dict(step=step, checkpoint=str(checkpoint), metrics=metrics, user_train_correct=user_correct)
        print("FIT", step, "user", user_correct, "jac_dev", metrics["jacaranda"],
              "correct", metrics["correct"], "eligible", eligible, flush=True)
    save_json(run / "history.json", history)
    best.update(sha256=sha256(best["checkpoint"]), selected_on="fixed development only",
                user_train_not_used_for_selection=True, deployed=False)
    save_json(out / "selection_lock.json", best)
    verify_backup(backup)
    checked_sources(protocol)
    print("LOCKED", best["step"], flush=True)


def evaluate(out: Path):
    protocol = read_json(out / "protocol.json")
    lock = read_json(out / "selection_lock.json")
    if sha256(lock["checkpoint"]) != lock["sha256"]:
        raise RuntimeError("Locked checkpoint changed")
    if (out / "regression_results.json").exists():
        raise RuntimeError("Regression already evaluated; no overwrite")
    torch.set_num_threads(4)
    test = load_pair(PARENT, "test")
    with np.load(PARENT / "train_features.npz") as data:
        params = dict(groups=read_json(PARENT / "classes.json")["groups"],
                      temp=protocol["deployment_manifest"]["temperature"],
                      extra=tensor(data["extra_bank"]), scale=float(data["scale"]))
    base = heads_from_deployment(protocol)
    candidate = heads_from_deployment(protocol)
    candidate.load_state_dict(torch.load(lock["checkpoint"], map_location="cpu", weights_only=False)["model_state"])
    bp = paired_predict(base, test, params)
    cp = paired_predict(candidate, test, params)
    save_json(out / "regression_results.json", dict(baseline=summary(test[0], bp), candidate=summary(test[0], cp),
        interpretation="Historical inspected test is regression-only. User photos are training replay, not heldout accuracy.",
        selected_step=lock["step"], deployed=False))
    np.savez_compressed(out / "regression_predictions.npz", baseline=bp, candidate=cp)
    checked_sources(protocol)
    print("REGRESSION_COMPLETE", lock["step"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "features", "fit", "evaluate"))
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    globals()[args.phase](args.out.resolve())
