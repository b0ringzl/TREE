"""Stage the current, inference-only Demo over the archived portable runtime.

This creates a *private evaluation folder*, not a redistributable installer.
It never writes to source assets or deployed model weights.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "desktop_app/dist/SpaceScan-Trees-Desktop-Private-20261023"
APP = DEST / "app"
OLD = Path("C:/SST_Pack_20260920_v1/SpaceScan-Trees-H120-Windows")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def put(relative: str, *, verify: str | None = None) -> Path:
    source = ROOT / relative
    target = APP / relative
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    if source.stat().st_size != target.stat().st_size or digest(source) != digest(target):
        raise RuntimeError(f"Copy verification failed: {relative}")
    if verify and digest(target) != verify:
        raise RuntimeError(f"Source checksum changed: {relative}")
    return target


def put_tree(relative: str, *, suffixes: set[str] | None = None) -> None:
    source = ROOT / relative
    for path in source.rglob("*"):
        if path.is_file() and (suffixes is None or path.suffix.lower() in suffixes):
            put(path.relative_to(ROOT).as_posix())


def rewrite(path: Path, old: str, new: str) -> None:
    content = path.read_text(encoding="utf-8")
    if new in content:
        return
    if old not in content:
        raise RuntimeError(f"Expected portability anchor missing in {path}: {old}")
    path.write_text(content.replace(old, new, 1), encoding="utf-8")


def main(destination: Path = DEST) -> None:
    global DEST, APP
    DEST = destination.resolve()
    APP = DEST / "app"
    if not (DEST / "runtime/python.exe").is_file() or not (DEST / "launch.py").is_file():
        raise RuntimeError("First copy the archived portable base to the new destination")
    if (DEST / "user_data").exists():
        raise RuntimeError("Staged folder has already been launched; use a fresh copy")

    # Current application logic. Do not copy generated jobs, logs, test images or
    # personal files from the working project.
    for directory in ("spacescan_trees_demo_v1", "tree_species_teacher_demo"):
        for path in (ROOT / directory).glob("*.py"):
            if not path.name.startswith("test_") and not path.name.startswith("collect_"):
                put(path.relative_to(ROOT).as_posix())
        put_tree(f"{directory}/static")
        if (ROOT / directory / "templates").is_dir():
            put_tree(f"{directory}/templates")
    for relative in (
        "spacescan_trees_demo_v1/release.json",
        "spacescan_trees_demo_v1/species_names.json",
        "tree_species_teacher_demo/models/SSTv1_Aquilaria_UserCrops_20260919_H120/deployment.json",
        "tools/demo_runtime_identity.py",
        "tools/predict_sam_polygon.py",
        "hk-tree-vmms/outputs/coordinates/frame_coordinates.csv",
        "hk-tree-vmms/derived/tree_detector_iteration2_full_20260906/images/val/base_000187_jianshazui__jianshazui_pano_1__002971.jpg",
    ):
        put(relative)
    # SAM2 Python code/config is already present in the archived Conda runtime.
    put("LabelPaw-web-images/weights/sam_weights/sam2.1_hiera_tiny.pt")

    probe = ROOT / "outputs/distillation/bioclip_probe_20260908_0227"
    protocol = json.loads((probe / "protocol.json").read_text(encoding="utf-8"))
    selection = json.loads((probe / "selection_lock.json").read_text(encoding="utf-8"))
    deployment = json.loads((ROOT / "tree_species_teacher_demo/models/SSTv1_Aquilaria_UserCrops_20260919_H120/deployment.json").read_text(encoding="utf-8"))
    flora = json.loads((ROOT / "external_datasets/bioclip25_flora_20261008_v1/manifest.json").read_text(encoding="utf-8"))
    point = json.loads((ROOT / "spacescan_trees_demo_v1/models/ptv2_seven_20260913_v1/deployment.json").read_text(encoding="utf-8"))

    teacher_relative = Path(protocol["teacher_weight"]).relative_to(ROOT).as_posix()
    put(teacher_relative, verify=protocol["teacher_sha256"])
    for row in selection["selected_runs"]:
        original = row["run"]
        relative = Path(original).relative_to(ROOT).as_posix()
        put(relative + "/best.pt", verify=selection["weights_sha256"][original])
        row["run"] = relative
    # Preserve checksum keys but make them relocatable as well.
    selection["weights_sha256"] = {
        Path(run).relative_to(ROOT).as_posix(): value
        for run, value in selection["weights_sha256"].items()
    }
    protocol["teacher_weight"] = teacher_relative
    stage_probe = APP / "outputs/distillation/bioclip_probe_20260908_0227"
    (stage_probe / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
    (stage_probe / "selection_lock.json").write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8")

    model_dir = "tree_species_teacher_demo/models/SSTv1_Aquilaria_UserCrops_20260919_H120"
    for member in deployment["members"]:
        put(f"{model_dir}/{member['file']}", verify=member["sha256"])
    put(f"{model_dir}/labels.json", verify=deployment["labels_sha256"])
    put("outputs/distillation/20260908_0045/aliases.json")
    for relative in (
        "hk-tree-vmms/derived/training_runs/combined_tree_segmentation/yolo11m_seg_1024_20260901_104757/weights/best.pt",
        "outputs/tree_detector_benchmark_20260909_v1/assets/clip/ViT-B-32.pt",
        "outputs/tree_detector_benchmark_20260909_v1/assets/grounding_dino_base/model.safetensors",
        point["model"], point["encoder"],
    ):
        put(relative)
    for row in flora["derived_files"]:
        put("external_datasets/bioclip25_flora_20261008_v1/" + row["file"], verify=row["sha256"])
    put("external_datasets/bioclip25_flora_20261008_v1/manifest.json")
    release = json.loads((ROOT / "spacescan_trees_demo_v1/release.json").read_text(encoding="utf-8"))
    android_dir = release["android_delivery"]
    for name, expected in (
        (release["android_apk"], release["android_apk_sha256"]),
        (release["android_model"], release["android_model_sha256"]),
        ("安装与验收说明.md", None),
    ):
        put(f"{android_dir}/{name}", verify=expected)

    engine = APP / "tree_species_teacher_demo/engine.py"
    rewrite(engine, "ROOT / 'external_datasets/distillation_runtime'", "ROOT / 'vendor/bioclip'")
    rewrite(engine, "weight = Path(protocol['teacher_weight'])", "weight = Path(protocol['teacher_weight'])\n        if not weight.is_absolute():\n            weight = ROOT / weight")
    rewrite(engine, "path = Path(run['run']) / 'best.pt'", "path = Path(run['run'])\n            if not path.is_absolute():\n                path = ROOT / path\n            path = path / 'best.pt'")
    rewrite(APP / "spacescan_trees_demo_v1/tree_detection.py",
            "RUNTIME = PROJECT / 'outputs/tree_detector_benchmark_20260909_v1/runtime'",
            "RUNTIME = PROJECT / 'vendor/detection'")
    rewrite(APP / "spacescan_trees_demo_v1/vmms_corridor.py",
            'VMMS_ROOT = PROJECT.parent / "vmms"',
            'VMMS_ROOT = Path(os.environ.get("SPACESCAN_VMMS_ROOT", str(PROJECT.parent / "vmms"))).resolve()')
    rewrite(APP / "spacescan_trees_demo_v1/server.py",
            "if origin not in {'http://localhost:8043', 'http://127.0.0.1:8043'}:",
            "port = request.url.port\n    if request.client.host not in {'127.0.0.1', '::1'} or origin not in {\n        f'http://localhost:{port}', f'http://127.0.0.1:{port}'}:")
    launch = DEST / "launch.py"
    shutil.copy2(ROOT / "desktop_app/portable_launch.py", launch)

    host = ROOT / "desktop_app/dist/host-white-icon-20261024"
    if not (host / "SpaceScan.Trees.Desktop.exe").is_file():
        raise FileNotFoundError("Publish the desktop host first")
    for source in host.rglob("*"):
        if source.is_file():
            target = DEST / source.relative_to(host)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    shutil.copy2(ROOT / "desktop_app/SpaceScan.Trees.Desktop/Assets/SpaceScanTrees-white.ico",
                 DEST / "SpaceScanTrees-white.ico")
    settings = DEST / "desktop-settings.json"
    if not settings.exists():
        vmms = ROOT.parent / "vmms"
        settings.write_text(json.dumps({
            "vmms_root": str(vmms) if vmms.is_dir() else "",
            "dropbox_app_key": "", "onedrive_client_id": "",
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    # Rebuild the package's read-only file manifest after refreshing application
    # code and data. The archive itself remains untouched.
    old_entries = json.loads((OLD / "model_manifest.json").read_text(encoding="utf-8"))
    paths = {row["file"] for row in old_entries}
    paths.update("app/" + path.relative_to(APP).as_posix() for path in APP.rglob("*") if path.is_file())
    entries = []
    for relative in sorted(paths):
        path = DEST / relative
        if not path.is_file():
            raise FileNotFoundError(f"Manifest file absent: {relative}")
        entries.append({"file": relative, "bytes": path.stat().st_size, "sha256": digest(path)})
    (DEST / "model_manifest.json").write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
    (DEST / "BUILD_INFO.json").write_text(json.dumps({
        "kind": "private-windows-desktop-evaluation", "source": str(ROOT),
        "base_package": str(OLD), "current_features": ["Identify", "Video", "VMMS", "world catalogue", "SAM"],
        "training_performed": False, "model_weights_changed": False,
        "file_manifest_count": len(entries), "external_vmms_folder_not_bundled": True,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"STAGED {DEST} files={len(entries)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", type=Path, default=DEST)
    args = parser.parse_args()
    main(args.dest)
