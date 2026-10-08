"""Evaluate frozen pair probes through the existing H120 dual-view image path.

No fitting and no production writes. Historic cohorts are regression diagnostics,
not a fresh Hong Kong street-view test.
"""

import argparse
import json
import re
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "outputs/artocarpus_preprocess_audit_20260917_v1"
OPT = ROOT / "outputs/SSTv1_Inference_Optimization_20260920_v2"
PREV = ROOT / "outputs/SSTv1_Artocarpus_User_20260918_v1"
CENTER = ROOT / "outputs/SSTv1_AQ_AR_Diagnostic_20260919_v1/exact_regression/center_features.npz"
LABELS = ROOT / "tree_species_teacher_demo/models/SSTv1_Aquilaria_UserCrops_20260919_H120/labels.json"
PAIRS = {
    "ficus": ("Ficus microcarpa / Ficus benjamina (榕樹)", ["Ficus benjamina", "Ficus microcarpa"]),
    "araucaria": ("Araucaria columnaris(Araucaria heterophylla)",
                  ["Araucaria columnaris", "Araucaria heterophylla"]),
}


def species(pair, row):
    if pair == "ficus":
        return row.get("species") if row.get("species") in PAIRS[pair][1] else None
    if row.get("species") != PAIRS[pair][0]:
        return None
    hits = re.findall(r"Araucaria (?:columnaris|heterophylla)", row.get("source_image", ""))
    return hits[0] if len(set(hits)) == 1 else None


def cohort(rows, x_whole, x_center, base, models, labels):
    answer = {}
    for pair, (pool_label, leaves) in PAIRS.items():
        model = models[pair]
        idx = [i for i, row in enumerate(rows) if species(pair, row)]
        yy = np.array([leaves.index(species(pair, rows[i])) for i in idx])
        base_pool = base[idx].argmax(axis=1) == labels.index(pool_label)
        split_p = .5 * (model.predict_proba(x_whole[idx]) + model.predict_proba(x_center[idx]))
        split_pred = split_p.argmax(axis=1)
        correct = base_pool & (split_pred == yy)
        answer[pair] = {
            "n": len(idx), "pool_detected": int(base_pool.sum()),
            "binary_correct_given_truth": int((split_pred == yy).sum()),
            "species_correct_end_to_end": int(correct.sum()),
            "by_true_species": {
                name: {"n": int((yy == i).sum()), "pool_detected": int((base_pool & (yy == i)).sum()),
                       "species_correct": int((correct & (yy == i)).sum())}
                for i, name in enumerate(leaves)
            },
            "conditional_failure_sample_ids": [rows[j]["sample_id"] for j, bad in zip(idx, ~correct) if bad],
        }
    return answer


def run(probe_dir):
    models = {pair: joblib.load(probe_dir / f"{pair}_probe.joblib") for pair in PAIRS}
    labels = json.loads(LABELS.read_text(encoding="utf-8"))
    dev_rows = json.loads((AUDIT / "rows.json").read_text(encoding="utf-8"))
    test_rows = json.loads((PREV / "test_rows.json").read_text(encoding="utf-8"))
    with np.load(AUDIT / "paired_features_predictions.npz", allow_pickle=False) as x, np.load(
            OPT / "development_components.npz", allow_pickle=False) as p:
        dev_base = .5 * (p["current_whole"] + p["current_center"])
        dev = cohort(dev_rows, x["demo_letterbox_x"], x["center_crop_x"], dev_base, models, labels)
    with np.load(PREV / "test_features.npz", allow_pickle=False) as whole, np.load(
            CENTER, allow_pickle=False) as center, np.load(
            OPT / "regression_predictions.npz", allow_pickle=False) as p:
        test = cohort(test_rows, whole["x"], center["x"], p["candidate"], models, labels)
    result = {"current_release": "spacescan-official-20260920-h120-dualview",
              "probe_dir": str(probe_dir), "development": dev, "historical_regression": test,
              "limitations": "Conditional pair model fitted to frozen original center-crop embeddings; adapted to deployed whole+center by fixed equal-probability mean, no view tuning. Previously inspected test is regression only; no new street-view evidence.",
              "deployed": False}
    (probe_dir / "deployment_path_regression.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name, block in (("development", dev), ("historical_regression", test)):
        print(name, {p: {k: v for k, v in b.items() if k not in ("conditional_failure_sample_ids", "by_true_species")}
                     for p, b in block.items()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.probe_dir.resolve())
