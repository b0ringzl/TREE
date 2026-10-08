"""Before/after checks for an explicitly authorized 148-class Demo switch.

Inference only. Never loads another GPU encoder, fits a model, edits weights,
changes review decisions, or restarts a service. Writes only this deployment's
new audit files. Expected distributions are staged from the locked candidate's
fixed, exact-Demo planning cache, independently CPU-checked by the staging tool.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
from spacescan_trees_demo_v1 import server, tree_detection as td
from training_safety import sha256, verify_backup

OUT = ROOT / "outputs/SSTv1_Artocarpus_deployment_20260918_v1"
RUN = ROOT / "outputs/SSTv1_Artocarpus_User_20260918_v1"
VERSION = "sstv1-artocarpus-user-20260918-v1"
OLD_VERSION = "sstv1-aquilaria-user-20260915-v1"
OLD_DEPLOYMENT = ROOT / "tree_species_teacher_demo/models/SSTv1_Aquilaria_User_20260915_v1/deployment.json"
BASE = ROOT / "outputs/SpaceScan_Trees_Demo_Files_20260913_56b49420/02_cloud_assist_gain"
URL = "http://127.0.0.1:8043"
UPSTREAM = "http://127.0.0.1:8039"
ATOL = 1e-4
UNCHANGED_ATOL = 1e-5


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_new(path, obj):
    path = Path(path)
    require(not path.exists(), "Refuse to overwrite audit: " + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def get_json(url):
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    return r.json()


def post_json(url, blob, *, filename="tree.jpg", fields=None, cloud=None):
    files = {"image": (filename, blob)}
    if cloud:
        files["cloud"] = cloud
    r = requests.post(url, files=files, data=fields or {}, timeout=180)
    require(r.status_code == 200, f"{url}: HTTP {r.status_code}: {r.text[:700]}")
    data = r.json()
    # Fail if the backend begins leaking NaN / Infinity values.
    json.dumps(data, allow_nan=False)
    return data


def smoke_rows():
    document = read(OUT / "expected.json")
    rows = document["smoke"]
    require(len(rows) == 3, "Expected three preselected smoke images")
    for row in rows:
        require(len(row["labels"]) == 148 and len(set(row["labels"])) == 148, "Expected output contract must have 148 unique labels")
        require(len(row["expected_candidate_distribution"]) == 148, "Bad expected probability count")
        require(sha256(row["image"]) == row["sha256"], "Smoke source changed: " + row["image"])
    return rows


def array_distribution(response, key, labels):
    distribution = response[key]
    require(set(distribution) == set(labels), "Distribution label set mismatch: " + key)
    scores = np.asarray([distribution[label] for label in labels], dtype=np.float64)
    require(np.isfinite(scores).all() and (scores >= 0).all() and np.isclose(scores.sum(), 1, atol=1e-5), "Invalid normalized distribution: " + key)
    return scores


def compare(a, b, atol=ATOL):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    require(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all(), "Bad probability shape/values")
    delta = float(np.max(np.abs(a - b)))
    require(delta <= atol, f"Probability parity failed: max_abs={delta:.9g} > {atol}")
    return delta


def assert_image_result(result, labels):
    require(result["mode"] == "sstv1" and result["image_model_version"] == VERSION, "Wrong image model/version")
    values = array_distribution(result, "image_distribution", labels)
    require(len(result["top3"]) == 3, "Missing top-three candidates")
    require(all(row.get("scientific_name") and row.get("chinese_name") for row in result["top3"]), "Missing bilingual candidate name")
    require([row["species"] for row in result["top3"]] == [labels[int(i)] for i in np.argsort(-values)[:3]], "Top3 does not match full probabilities")
    return values


def protected_paths():
    paths = [OLD_DEPLOYMENT]
    old = read(OLD_DEPLOYMENT)
    paths.extend(OLD_DEPLOYMENT.parent / x["file"] for x in old["members"])
    experiment = ROOT / "outputs/distillation/bioclip_probe_20260908_0227"
    paths.extend(experiment / name for name in ("classes.json", "protocol.json", "teacher_choice.json", "selection_lock.json", "cache_metadata.json"))
    protocol = read(experiment / "protocol.json")
    paths.append(Path(protocol["teacher_weight"]))
    selection = read(experiment / "selection_lock.json")
    paths.extend(Path(x["run"]) / "best.pt" for x in selection["selected_runs"])
    point_manifest = ROOT / "spacescan_trees_demo_v1/models/ptv2_seven_20260913_v1/deployment.json"
    paths.append(point_manifest)
    point = read(point_manifest)
    paths.extend(ROOT / point[k] for k in ("model", "encoder"))
    paths.extend(ROOT / name for name in ("spacescan_trees_demo_v1/species_names.json", "spacescan_trees_demo_v1/point_recognition.py"))
    return list(dict.fromkeys(path.resolve() for path in paths))


def pair_cases():
    result = []
    for folder in sorted(BASE.iterdir()):
        if not folder.is_dir():
            continue
        candidates = sorted(folder.glob("cloud_for_replay.*"))
        require(len(candidates) == 1 and (folder / "image.jpg").is_file(), "Incomplete packaged pair: " + str(folder))
        result.append((folder, candidates[0]))
    require(len(result) >= 2, "Expected at least the two existing packaged image-point cases")
    return result


def call_pair(folder, cloud):
    return post_json(URL + "/api/predict-with-cloud", (folder / "image.jpg").read_bytes(),
                     filename="input.jpg", cloud=(cloud.name, cloud.read_bytes()),
                     fields={"projection": "photo", "paired": "true", "units": "metres", "z_up": "true", "scenario": "street"})


def capture_before():
    require(not (OUT / "BEFORE.json").exists(), "Before audit already exists; never recapture after switching")
    rows = smoke_rows()
    health, upstream_health = get_json(URL + "/api/health"), get_json(UPSTREAM + "/api/health")
    require(health["ready"] and health["single_image_version"] == OLD_VERSION, "capture-before requires the original live Demo")
    require(upstream_health["status"] == "ready" and upstream_health["sstv1_version"] == OLD_VERSION and upstream_health["class_count"] == 148, "Original upstream not ready/148")
    capture_dir = OUT / ("before_capture_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    capture_dir.mkdir(exist_ok=False)
    protected = []
    for path in protected_paths():
        protected.append({"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size})
    singles = []
    for i, row in enumerate(rows):
        blob = Path(row["image"]).read_bytes()
        prepared = server.prepare(blob)[0]
        old = {}
        for mode in ("teacher", "adapted", "sstv1"):
            result = post_json(UPSTREAM + "/api/predict", prepared, fields={"mode": mode})
            array_distribution(result, mode + "_distribution", row["labels"])
            require(result["class_count"] == 148, "Old upstream catalog changed")
            old[mode] = result
            save_new(capture_dir / f"smoke_{i}_{mode}.json", result)
        demo = post_json(URL + "/api/predict", blob, fields={"scenario": "street", "projection": "photo"})
        require(demo["image_model_version"] == OLD_VERSION, "Old 8043 is not the captured original model")
        compare(array_distribution(demo, "image_distribution", row["labels"]), array_distribution(old["sstv1"], "sstv1_distribution", row["labels"]), UNCHANGED_ATOL)
        cache_delta = compare(array_distribution(demo, "image_distribution", row["labels"]), row["baseline_cached_distribution"], ATOL)
        save_new(capture_dir / f"smoke_{i}_demo.json", demo)
        singles.append({"index": i, "image": row["image"], "sha256": row["sha256"], "prepared_sha256": hashlib.sha256(prepared).hexdigest(), "cached_old_max_abs_difference": cache_delta})
        print("CAPTURED OLD IMAGE", i, flush=True)
    pairs = []
    for folder, cloud in pair_cases():
        result = call_pair(folder, cloud)
        require(result["fusion_used"] and result["mode"] == "teacher" and len(result["class_scope"]) == 7, "Old fusion contract invalid")
        save_new(capture_dir / (folder.name + "_dual.json"), result)
        pairs.append({"case": folder.name, "image_sha256": sha256(folder / "image.jpg"), "cloud_sha256": sha256(cloud)})
        print("CAPTURED OLD DUAL", folder.name, flush=True)
    save_new(OUT / "BEFORE.json", {"captured_at": datetime.now(timezone.utc).isoformat(), "version": OLD_VERSION,
             "health": health, "upstream_health": upstream_health, "expected_sha256": sha256(OUT / "expected.json"),
             "images": singles, "pairs": pairs, "capture_dir": str(capture_dir), "protected_files": protected, "training_performed": False})


def test_selected_routes(blob, scenario, prefix, labels):
    # HMAC SECRET is process-local. Use the unchanged real route in a TestClient
    # and that same module's signer; its inference still calls live port 8039.
    # This deliberately does not pretend to test the live detector or its recall.
    from fastapi.testclient import TestClient
    client = TestClient(server.app)
    width, height = td.decode(blob).size
    meta = {"expires": time.time() + 300, "sha256": hashlib.sha256(blob).hexdigest(), "crop_box": [0, 0, width, height]}
    token = td.sign(meta)
    r = client.post("/api/identify-tree", files={"image": ("tree.jpg", blob)}, data={"token": token, "scenario": scenario})
    require(r.status_code == 200, r.text[:600])
    result = r.json()
    scores = assert_image_result(result, labels)
    require(result["tree_detection"] and result["scenario"] == scenario, "Selected crop did not use required route")
    crop, _, _ = td.selected(blob, token)
    direct = post_json(URL + "/api/predict", crop, fields={"scenario": scenario, "projection": "photo"})
    delta = compare(scores, assert_image_result(direct, labels), UNCHANGED_ATOL)
    save_new(OUT / "after" / f"{prefix}_selected_{scenario}.json", result)
    return delta


def test_perspective(labels):
    from fastapi.testclient import TestClient
    client = TestClient(server.app)
    r = requests.get(URL + "/api/test-panorama", timeout=30)
    r.raise_for_status()
    blob = r.content
    spec = td.panorama.view_spec(td.decode(blob).size, 0, 30)
    meta = {"expires": time.time() + 300, "sha256": hashlib.sha256(blob).hexdigest(),
            "crop_box": [0, 0, min(600, spec["width"]), min(600, spec["height"])],
            "projection_view": spec, "projection": {"inferred_from_ratio": False}}
    token = td.sign(meta)
    r = client.post("/api/identify-tree", files={"image": ("pano.jpg", blob)}, data={"token": token, "scenario": "conservation"})
    require(r.status_code == 200, r.text[:600])
    result = r.json()
    scores = assert_image_result(result, labels)
    require(result["tree_detection"] and "perspective" in result["preprocessing"].lower(), "Perspective crop path missing")
    crop, _, _ = td.selected(blob, token)
    direct = post_json(URL + "/api/predict", crop, fields={"scenario": "conservation", "projection": "photo"})
    delta = compare(scores, assert_image_result(direct, labels), UNCHANGED_ATOL)
    save_new(OUT / "after" / "perspective_selected.json", result)
    return {"max_abs_difference": delta, "panorama_sha256": hashlib.sha256(blob).hexdigest(), "route": "local TestClient /api/identify-tree with live 8039; not live detector test"}


def test_manual_polygon(blob, labels):
    # Browser polygon cropping has no separate server API. It emits a native-size
    # gray-masked PNG, sets projection=photo, and sends /api/predict directly.
    # Test that transport contract without claiming browser interaction coverage.
    page = requests.get(URL, timeout=15)
    page.raise_for_status()
    require("polygon-crop.js" in page.text, "Polygon UI script missing")
    js = requests.get(URL + "/static/polygon-crop.js", timeout=15)
    js.raise_for_status()
    require("manualPolygon" in js.text and "image/png" in js.text, "Polygon UI contract missing")
    routing = requests.get(URL + "/static/tree-selection.js", timeout=15)
    routing.raise_for_status()
    require("!item.manualPolygon" in routing.text, "Manual target unexpectedly enters detector")
    with Image.open(io.BytesIO(blob)) as source:
        im = ImageOps.exif_transpose(source).convert("RGB")
    w, h = im.size
    vertices = [(round(.1*w), round(.12*h)), (round(.8*w), round(.05*h)), (round(.92*w), round(.86*h)), (round(.2*w), round(.94*h))]
    mask = Image.new("L", im.size, 0)
    ImageDraw.Draw(mask).polygon(vertices, fill=255)
    masked = Image.composite(im, Image.new("RGB", im.size, (128, 128, 128)), mask)
    box = (min(p[0] for p in vertices), min(p[1] for p in vertices), max(p[0] for p in vertices), max(p[1] for p in vertices))
    buf = io.BytesIO(); masked.crop(box).save(buf, format="PNG")
    crop = buf.getvalue()
    result = post_json(URL + "/api/predict", crop, filename="manual-polygon.png", fields={"scenario": "conservation", "projection": "photo"})
    scores = assert_image_result(result, labels)
    prepared = server.prepare(crop)[0]
    upstream = post_json(UPSTREAM + "/api/predict", prepared, fields={"mode": "sstv1"})
    delta = compare(scores, array_distribution(upstream, "sstv1_distribution", labels), UNCHANGED_ATOL)
    save_new(OUT / "after" / "manual_polygon_transport.json", result)
    return {"max_abs_difference": delta, "crop_sha256": hashlib.sha256(crop).hexdigest(), "vertices": vertices,
            "coverage": "Browser source contract plus real masked-PNG HTTP transport; no actual browser click/canvas raster parity claimed"}


def audit_backup(before):
    directory = OUT / "pre_training_backup"
    info = verify_backup(directory, verify_sources=False)
    copied = []
    for row in info["files"]:
        backup = directory / row["backup"]
        source = Path(row["source"])
        require(backup.stat().st_size == row["bytes"], "Backup size mismatch")
        require(backup.stat().st_nlink == 1 and not os.path.samefile(source, backup), "Backup is not a distinct physical file")
        # Code may be intentionally switched, old weight bytes may not.
        if source.suffix.lower() in {".pt", ".pth", ".joblib", ".safetensors"}:
            require(sha256(source) == row["sha256"], "Original weights were modified: " + str(source))
        copied.append(str(backup))
    for row in before["protected_files"]:
        require(Path(row["path"]).stat().st_size == row["bytes"] and sha256(row["path"]) == row["sha256"], "Protected original changed: " + row["path"])
    return {"directory": str(directory), "files_verified": len(copied), "physical_copy_not_hardlink": True,
            "old_weights_and_point_contract_unchanged": True, "protected_originals_verified": len(before["protected_files"])}


def verify_after():
    require(not (OUT / "VERIFIED.json").exists(), "After audit already exists")
    before = read(OUT / "BEFORE.json")
    capture_dir = Path(before["capture_dir"])
    require(capture_dir.resolve().is_relative_to(OUT.resolve()), "Before capture directory escapes deployment output")
    require(before["expected_sha256"] == sha256(OUT / "expected.json"), "Expected parity data changed after baseline capture")
    rows = smoke_rows()
    health, upstream_health = get_json(URL + "/api/health"), get_json(UPSTREAM + "/api/health")
    require(health["ready"] and health["single_image_version"] == VERSION and health["dual_image_mode"] == "teacher", "Demo is not ready on the intended split contract")
    require(upstream_health["status"] == "ready" and upstream_health["sstv1_status"] == "ready" and upstream_health["sstv1_version"] == VERSION and upstream_health["class_count"] == 148, "Upstream not compatible/ready")
    catalog = get_json(UPSTREAM + "/api/catalog")
    require([row["species"] for row in catalog] == rows[0]["labels"], "Live catalog order differs from candidate")
    singles = []
    for i, row in enumerate(rows):
        blob = Path(row["image"]).read_bytes()
        prepared = server.prepare(blob)[0]
        require(hashlib.sha256(prepared).hexdigest() == before["images"][i]["prepared_sha256"], "Image preparation changed")
        unchanged = {}
        for mode in ("teacher", "adapted"):
            after = post_json(UPSTREAM + "/api/predict", prepared, fields={"mode": mode})
            old = read(capture_dir / f"smoke_{i}_{mode}.json")
            unchanged[mode] = compare(array_distribution(after, mode + "_distribution", row["labels"]), array_distribution(old, mode + "_distribution", row["labels"]), UNCHANGED_ATOL)
            save_new(OUT / "after" / f"smoke_{i}_{mode}.json", after)
        errors = {}
        for scenario in ("street", "conservation"):
            after = post_json(URL + "/api/predict", blob, fields={"scenario": scenario, "projection": "photo"})
            scores = assert_image_result(after, row["labels"])
            require(after["scenario"] == scenario, "Scenario propagation failed")
            errors[scenario] = compare(scores, row["expected_candidate_distribution"], ATOL)
            save_new(OUT / "after" / f"smoke_{i}_{scenario}.json", after)
        selected = test_selected_routes(blob, "conservation", f"smoke_{i}", row["labels"])
        singles.append({"index": i, "image": row["image"], "sha256": row["sha256"],
                        "candidate_cached_max_abs_difference": errors, "unchanged_mode_max_abs_difference": unchanged,
                        "selected_crop_live_direct_max_abs_difference": selected})
        print("VERIFIED NEW IMAGE", i, errors, flush=True)
    pairs = []
    for folder, cloud in pair_cases():
        row = next(x for x in before["pairs"] if x["case"] == folder.name)
        require(sha256(folder / "image.jpg") == row["image_sha256"] and sha256(cloud) == row["cloud_sha256"], "Pair input changed")
        actual, old = call_pair(folder, cloud), read(capture_dir / (folder.name + "_dual.json"))
        require(actual["fusion_used"] and actual["mode"] == "teacher" and actual["class_scope"] == old["class_scope"], "Dual routing/class scope changed")
        deltas = {}
        for branch in ("image", "point", "fusion"):
            require([x["species"] for x in actual["comparison"][branch]] == [x["species"] for x in old["comparison"][branch]], "Dual top3 order changed: " + branch)
            deltas[branch] = compare([x["score"] for x in actual["comparison"][branch]], [x["score"] for x in old["comparison"][branch]], UNCHANGED_ATOL)
        deltas["full_image"] = compare(array_distribution(actual, "image_distribution", rows[0]["labels"]), array_distribution(old, "image_distribution", rows[0]["labels"]), UNCHANGED_ATOL)
        for key in ("model_sha256", "encoder_sha256", "version", "classes"):
            require(actual["point_model"][key] == old["point_model"][key], "Point model contract changed: " + key)
        require(actual["fusion_policy"] == old["fusion_policy"], "Fusion weights/policy changed")
        save_new(OUT / "after" / (folder.name + "_dual.json"), actual)
        pairs.append({"case": folder.name, "max_abs_differences": deltas})
        print("VERIFIED UNCHANGED DUAL", folder.name, flush=True)
    perspective = test_perspective(rows[0]["labels"])
    polygon = test_manual_polygon(Path(rows[0]["image"]).read_bytes(), rows[0]["labels"])
    backup = audit_backup(before)
    result = {"verified_at": datetime.now(timezone.utc).isoformat(), "deployed_version": VERSION, "health": health,
              "upstream_health": upstream_health, "class_count": 148, "images": singles, "dual_cases": pairs,
              "perspective_selected_crop": perspective, "manual_polygon_transport": polygon,
              "backup": backup, "candidate_parity_tolerance": ATOL, "unchanged_branch_tolerance": UNCHANGED_ATOL,
              "training_performed": False, "additional_gpu_encoder_loaded": False,
              "not_tested": ["detector recall/quality", "physical browser polygon clicking", "new independent species accuracy"],
              "verifier_sha256": sha256(Path(__file__)), "expected_sha256": sha256(OUT / "expected.json")}
    save_new(OUT / "VERIFIED.json", result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["capture-before", "verify-after"])
    args = parser.parse_args()
    (capture_before if args.stage == "capture-before" else verify_after)()


if __name__ == "__main__":
    main()
