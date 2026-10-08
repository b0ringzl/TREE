"""Read-only production, source-label species-split probe for SSTv1.

The binary probes are experimental classifiers on frozen BioCLIP embeddings.
They do not create a deployable expanded softmax or imply street-view accuracy.
Every fit starts after a verified, run-specific physical backup.
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from distillation_core import digest, read_json, save_json
from training_safety import backup_before_training, deployed_artifacts, sha256, verify_backup

SOURCE = ROOT / "outputs/distillation/20260908_0045"
REPLAY = ROOT / "outputs/SSTv1_incremental_20260914_v1/accepted_replay_manifest.json"
H120 = ROOT / "tree_species_teacher_demo/models/SSTv1_Aquilaria_UserCrops_20260919_H120"
ENCODER = ROOT / "external_datasets/distillation_models/bioclip25/open_clip_model.safetensors"
PAIR_LABELS = {
    "ficus": ["Ficus benjamina", "Ficus microcarpa"],
    "araucaria": ["Araucaria columnaris", "Araucaria heterophylla"],
}


def canonical(pair, row):
    if pair == "ficus":
        return row.get("species") if row.get("species") in PAIR_LABELS[pair] else None
    if row.get("species") != "Araucaria columnaris(Araucaria heterophylla)":
        return None
    hits = re.findall(r"Araucaria (?:columnaris|heterophylla)", row.get("source_image", ""))
    return hits[0] if len(set(hits)) == 1 else None


def sample_rows(pair, phase, accepted_ids):
    rows = read_json(SOURCE / f"{phase}_rows.json")
    with np.load(SOURCE / f"teacher_cache/bioclip25/{phase}.npz", allow_pickle=False) as cache:
        if str(cache["rows_hash"]) != digest(rows):
            raise ValueError(f"Feature order mismatch: {phase}")
        indices = []
        records = []
        for i, row in enumerate(rows):
            label = canonical(pair, row)
            if label is None or row.get("auxiliary") or (phase == "train" and row["sample_id"] not in accepted_ids):
                continue
            records.append({
                "sample_id": row["sample_id"], "source_image": row["source_image"],
                "source_sha256": row.get("source_sha256"), "source_label": label,
                "group_id": row["group_id"], "phase": phase,
                "identity_status": row.get("identity_status"),
            })
            indices.append(i)
        x = cache["embeddings"][indices].astype(np.float32)
    return records, x


def evaluate(model, x, y, labels):
    pred = model.predict(x)
    matrix = confusion_matrix(y, pred, labels=[0, 1])
    return {
        "n": len(y), "correct": int((pred == y).sum()),
        "accuracy": float((pred == y).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "matrix_true_rows_pred_columns": matrix.tolist(),
        "per_class_recall": {labels[i]: float(matrix[i, i] / max(1, matrix[i].sum())) for i in range(2)},
    }


def run(out):
    out.mkdir(parents=True, exist_ok=False)
    accepted = read_json(REPLAY)
    accepted_ids = {r["sample_id"] for r in accepted}
    all_data = {}
    manifest = []
    for pair, labels in PAIR_LABELS.items():
        all_data[pair] = {}
        for phase in ("train", "planning", "selection", "test"):
            rows, x = sample_rows(pair, phase, accepted_ids)
            all_data[pair][phase] = (rows, x)
            manifest.extend({"pair": pair, **r} for r in rows)
        for a, b in (("train", "planning"), ("train", "selection"), ("train", "test"),
                     ("planning", "selection"), ("planning", "test"), ("selection", "test")):
            aa, bb = (all_data[pair][p][0] for p in (a, b))
            if {r["group_id"] for r in aa} & {r["group_id"] for r in bb}:
                raise ValueError(f"Cross-split observation overlap: {pair} {a} {b}")
            if {r["source_sha256"] for r in aa if r["source_sha256"]} & {r["source_sha256"] for r in bb if r["source_sha256"]}:
                raise ValueError(f"Cross-split exact-image overlap: {pair} {a} {b}")
    save_json(out / "accepted_data_manifest.json", manifest)
    save_json(out / "split_assignment.json", [{k: r[k] for k in ("pair", "sample_id", "source_label", "group_id", "phase")}
                                               for r in manifest])
    config = {
        "seed": 20260924, "candidate_type": "frozen_BioCLIP_binary_split_probes",
        "features": "verified original BioCLIP 2.5 1024-d embedding cache",
        "fit": "StandardScaler + balanced logistic regression",
        "regularization_C_grid": [0.001, 0.01, 0.1, 1.0, 10.0],
        "selection": "maximize planning balanced accuracy; deterministic lower-C tie-break",
        "selection_split": "historical selection; reported, not used to tune",
        "test_split": "previously inspected historical regression only",
        "source_labels": "iNaturalist/source species, not individually botanist-verified",
        "scope": "conditional pair disambiguation, not end-to-end Demo or expanded 153-class model",
    }
    save_json(out / "config.json", config)
    save_json(out / "label_mapping.json", PAIR_LABELS)
    baselines = {}
    for pair, phases in all_data.items():
        train = phases["train"][0]
        prior = Counter(r["source_label"] for r in train)
        majority = prior.most_common(1)[0][0]
        baselines[pair] = {
            "train_counts": dict(prior), "majority_label": majority,
            "split_counts": {phase: dict(Counter(r["source_label"] for r in rows))
                             for phase, (rows, _) in phases.items()},
        }
    save_json(out / "baseline_evaluation.json", baselines)
    inputs = [ENCODER, *H120.glob("member_*.pt"), H120 / "labels.json", H120 / "deployment.json",
              ROOT / "spacescan_trees_demo_v1/release.json",
              ROOT / "spacescan_trees_demo_v1/models/ptv2_seven_20260913_v1/model.joblib",
              ROOT / "outputs/SSTv1_Inference_Optimization_20260920_v2/regression_results.json",
              REPLAY, SOURCE / "classes.json", Path(__file__),
              out / "accepted_data_manifest.json", out / "split_assignment.json",
              out / "config.json", out / "label_mapping.json", out / "baseline_evaluation.json",
              *deployed_artifacts()]
    protected = {str(Path(p).resolve()): sha256(p) for p in inputs if Path(p).is_file()}
    backup = backup_before_training(out, inputs, {"seed": config["seed"], "kind": config["candidate_type"]})
    verify_backup(backup)
    print("BACKUP VERIFIED", backup, flush=True)
    results = {}
    for pair, phases in all_data.items():
        labels = PAIR_LABELS[pair]
        data = {phase: (x, np.array([labels.index(r["source_label"]) for r in rows]))
                for phase, (rows, x) in phases.items()}
        choices = []
        for C in config["regularization_C_grid"]:
            model = make_pipeline(StandardScaler(), LogisticRegression(
                C=C, class_weight="balanced", max_iter=1000, random_state=config["seed"]))
            model.fit(*data["train"])
            choices.append({"C": C, "planning": evaluate(model, *data["planning"], labels)})
        best = max(choices, key=lambda r: (r["planning"]["balanced_accuracy"], -r["C"]))
        model = make_pipeline(StandardScaler(), LogisticRegression(
            C=best["C"], class_weight="balanced", max_iter=1000, random_state=config["seed"]))
        model.fit(*data["train"])
        joblib.dump(model, out / f"{pair}_probe.joblib")
        results[pair] = {"labels": labels, "choices": choices, "selected_C": best["C"],
                         "train": evaluate(model, *data["train"], labels),
                         "planning": evaluate(model, *data["planning"], labels),
                         "selection": evaluate(model, *data["selection"], labels),
                         "historical_test": evaluate(model, *data["test"], labels)}
        print(pair, "selected", best["C"], "selection", results[pair]["selection"], flush=True)
    save_json(out / "results.json", results)
    verify_backup(backup)
    for path, expected in protected.items():
        if sha256(path) != expected:
            raise RuntimeError(f"Production or training input changed: {path}")
    save_json(out / "COMPLETE.json", {"backups_verified": True,
                                       "deployed_files_unchanged": True,
                                       "models_deployed": False,
                                       "unfrozen_backbone_tested": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.out.resolve())
