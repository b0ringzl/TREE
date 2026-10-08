"""Isolated last-vision-block BioCLIP test for two Araucaria source labels.

Experimental only: no 153-class deployment and no street-view accuracy claim.
Every optimizer update is gated by a verified physical backup of initialization,
deployed weights, accepted manifest, split, configuration and baseline results.
"""

import argparse
import copy
import json
import math
import random
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "external_datasets/distillation_runtime")]
from distillation_core import canonical_image, read_json, save_json
from training_safety import backup_before_training, deployed_artifacts, sha256, verify_backup

PROBE = ROOT / "outputs/SSTv1_Urban_Split_Probe_20260924_v1"
ENCODER = ROOT / "external_datasets/distillation_models/bioclip25/open_clip_model.safetensors"
MODEL_DIR = ENCODER.parent
H120 = ROOT / "tree_species_teacher_demo/models/SSTv1_Aquilaria_UserCrops_20260919_H120"
LABELS = ["Araucaria columnaris", "Araucaria heterophylla"]


class Images(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return canonical_image(row["source_image"]), LABELS.index(row["source_label"])


def direct_head(probe):
    scaler = probe.named_steps["standardscaler"]
    lr = probe.named_steps["logisticregression"]
    w = (lr.coef_[0] / scaler.scale_).astype(np.float32)
    b = float(lr.intercept_[0] - np.dot(lr.coef_[0], scaler.mean_ / scaler.scale_))
    head = torch.nn.Linear(len(w), 2)
    with torch.no_grad():
        head.weight.zero_()
        head.bias.zero_()
        head.weight[1].copy_(torch.tensor(w))
        head.bias[1] = b
    return head


def score(model, head, rows, batch_size, device):
    loader = DataLoader(Images(rows), batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval(); head.eval()
    true, pred, loss = [], [], []
    with torch.no_grad():
        for pixels, y in loader:
            x = pixels.to(device)
            x = ((x - MEAN) / STD)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                emb = model.encode_image(x, normalize=True).float()
            logits = head(emb)
            true.extend(y.tolist())
            pred.extend(logits.argmax(-1).cpu().tolist())
            loss.extend(F.cross_entropy(logits, y.to(device), reduction="none").cpu().tolist())
    matrix = [[sum(t == i and p == j for t, p in zip(true, pred)) for j in range(2)] for i in range(2)]
    return {"n": len(true), "correct": sum(t == p for t, p in zip(true, pred)),
            "balanced_accuracy": sum(matrix[i][i] / max(1, sum(matrix[i])) for i in range(2)) / 2,
            "mean_ce": float(np.mean(loss)), "matrix_true_rows_pred_columns": matrix}


def run(out, steps):
    out.mkdir(parents=True, exist_ok=False)
    random.seed(20260924); np.random.seed(20260924); torch.manual_seed(20260924)
    torch.cuda.manual_seed_all(20260924)
    all_rows = read_json(PROBE / "accepted_data_manifest.json")
    by_phase = {p: [r for r in all_rows if r["pair"] == "araucaria" and r["phase"] == p]
                for p in ("train", "planning", "selection", "test")}
    for phase, rows in by_phase.items():
        for row in rows:
            if sha256(row["source_image"]) != row["source_sha256"]:
                raise ValueError(f"Source image changed: {phase} {row['sample_id']}")
    save_json(out / "accepted_data_manifest.json", by_phase)
    save_json(out / "split_assignment.json", [{"sample_id": r["sample_id"], "phase": phase,
                                                 "source_label": r["source_label"], "group_id": r["group_id"]}
                                                for phase, rows in by_phase.items() for r in rows])
    save_json(out / "label_mapping.json", LABELS)
    config = {"seed": 20260924, "model": "BioCLIP 2.5 ViT-H-14", "trainable": "last visual resblock + conditional 2-way head",
              "initial_head": str(PROBE / "araucaria_probe.joblib"), "max_steps": steps, "batch_size": 2,
              "encoder_lr": 1e-6, "head_lr": 1e-4, "weight_decay": 0.01,
              "pixel_preprocessing": "EXIF RGB, shortest-side bicubic 224, center crop 224; BioCLIP mean/std",
              "selection": "planning balanced accuracy, then lower cross-entropy; initial state eligible",
              "test": "historically inspected regression only; not a fresh blind city-street test",
              "production": "unchanged; only conditional binary specialist tested"}
    save_json(out / "config.json", config)
    save_json(out / "baseline_evaluation.json", read_json(PROBE / "results.json")["araucaria"])
    inputs = [ENCODER, PROBE / "araucaria_probe.joblib", PROBE / "accepted_data_manifest.json",
              H120 / "deployment.json", H120 / "labels.json", *H120.glob("member_*.pt"),
              ROOT / "spacescan_trees_demo_v1/release.json", ROOT / "spacescan_trees_demo_v1/models/ptv2_seven_20260913_v1/model.joblib",
              ROOT / "outputs/SSTv1_Inference_Optimization_20260920_v2/regression_results.json",
              Path(__file__), out / "accepted_data_manifest.json", out / "split_assignment.json",
              out / "label_mapping.json", out / "config.json", out / "baseline_evaluation.json",
              *deployed_artifacts()]
    protected = {str(Path(p).resolve()): sha256(p) for p in inputs}
    backup = backup_before_training(out, inputs, {"run": out.name, "seed": 20260924, "kind": "partial_BioCLIP_unfreeze"})
    verify_backup(backup)
    print("BACKUP VERIFIED", backup, flush=True)

    import open_clip
    device = "cuda"
    model, _, _ = open_clip.create_model_and_transforms("local-dir:" + str(MODEL_DIR.resolve()),
                                                        precision="fp32", device=device)
    model.eval().requires_grad_(False)
    last = model.visual.transformer.resblocks[-1]
    last.requires_grad_(True)
    head = direct_head(joblib.load(PROBE / "araucaria_probe.joblib")).to(device)
    initial_block = {k: v.detach().cpu().clone() for k, v in last.state_dict().items()}
    initial_head = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    initial_planning = score(model, head, by_phase["planning"], 2, device)
    save_json(out / "initial_live_planning.json", initial_planning)
    best = (initial_planning["balanced_accuracy"], -initial_planning["mean_ce"])
    best_step = 0
    best_block, best_head = initial_block, initial_head
    optimizer = torch.optim.AdamW([{"params": last.parameters(), "lr": config["encoder_lr"]},
                                   {"params": head.parameters(), "lr": config["head_lr"]}],
                                  weight_decay=config["weight_decay"])
    loader = DataLoader(Images(by_phase["train"]), batch_size=config["batch_size"],
                        shuffle=True, generator=torch.Generator().manual_seed(config["seed"]), num_workers=0)
    history = []
    step = 0
    while step < steps:
        for pixels, y in loader:
            model.eval(); last.train(); head.train()
            x = (pixels.to(device) - MEAN) / STD
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                emb = model.encode_image(x, normalize=True).float()
            logits = head(emb)
            loss = F.cross_entropy(logits, y)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_([*last.parameters(), *head.parameters()], 1.0, error_if_nonfinite=True)
            optimizer.step()
            step += 1
            if step % 40 == 0 or step == steps:
                val = score(model, head, by_phase["planning"], 2, device)
                item = {"step": step, "train_loss": float(loss.detach().cpu()), "planning": val}
                history.append(item)
                ranking = (val["balanced_accuracy"], -val["mean_ce"])
                if ranking > best:
                    best = ranking; best_step = step
                    best_block = {k: v.detach().cpu().clone() for k, v in last.state_dict().items()}
                    best_head = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
                print("STEP", step, "planning", val["correct"], "/", val["n"],
                      "balanced", round(val["balanced_accuracy"], 4), flush=True)
            if step >= steps:
                break
    save_json(out / "history.json", history)
    last.load_state_dict(best_block); head.load_state_dict(best_head)
    selected = {p: score(model, head, rows, 2, device) for p, rows in by_phase.items() if p != "train"}
    torch.save({"last_visual_block": best_block, "head": best_head, "base_encoder_sha256": sha256(ENCODER),
                "labels": LABELS, "selected_step": best_step, "not_deployed": True}, out / "candidate_delta.pt")
    last.load_state_dict(initial_block); head.load_state_dict(initial_head)
    baseline = {p: score(model, head, rows, 2, device) for p, rows in by_phase.items() if p != "train"}
    save_json(out / "results.json", {"selected_step": best_step, "baseline": baseline, "candidate": selected,
                                      "selection_used": "planning only", "deployed": False,
                                      "historic_test_is_not_blind": True})
    verify_backup(backup)
    for p, expected in protected.items():
        if sha256(p) != expected:
            raise RuntimeError("Protected source changed: " + p)
    save_json(out / "COMPLETE.json", {"backup_verified": True, "production_unchanged": True,
                                       "selected_step": best_step, "deployed": False})
    print("RESULT", json.dumps({"baseline": baseline, "candidate": selected, "selected_step": best_step}), flush=True)


MEAN = torch.tensor([.48145466, .4578275, .40821073], device="cuda")[None, :, None, None]
STD = torch.tensor([.26862954, .26130258, .27577711], device="cuda")[None, :, None, None]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=120)
    args = parser.parse_args()
    run(args.out.resolve(), args.steps)
