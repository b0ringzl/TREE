"""Record the actual official-Demo HTTP path for Stubbs tree-1 candidates."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/stubbs_tree1_clean_20260922_v1/evaluation_v1"
IMAGES = {
    "annotated_001557": ROOT / "hk-tree-vmms/derived/vmms_species_cls_20260906/train/10_Ficus_microcarpa/stubbs_road__stubbs_pano_1__001557__00.jpg",
    "neighbor_001562": ROOT / "hk-tree-vmms/derived/all_routes_coordinate_labeler/view_cache/stubbs_pano_1__RTR0004701/view_2_frame_001562.jpg",
}
CLOUDS = {
    "depth_1m": ROOT / "outputs/stubbs_tree1_clean_20260922_v1/visible_depth_1m_v1/tree1_demo_upload.npz",
    "depth_2m": ROOT / "outputs/stubbs_tree1_clean_20260922_v1/visible_depth_2m_v1/tree1_demo_upload.npz",
}
BASE = "http://127.0.0.1:8043"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def concise(rows: list[dict]) -> list[dict]:
    return [{"species": r.get("scientific_name", r.get("species")), "score": r.get("score")} for r in rows]


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation: {OUT}")
    OUT.mkdir(parents=True)
    health = requests.get(BASE + "/api/health", timeout=15).json()
    if not health.get("ready") or not health.get("fusion"):
        raise RuntimeError("Official Demo or fusion model not ready")
    (OUT / "health.json").write_text(json.dumps(health, indent=2), encoding="utf-8")
    summary = []
    for image_name, image in IMAGES.items():
        with image.open("rb") as stream:
            response = requests.post(BASE + "/api/predict", files={"image": (image.name, stream, "image/jpeg")},
                                     data={"scenario": "street", "source": "image", "projection": "photo"}, timeout=180)
        response.raise_for_status()
        image_only = response.json()
        (OUT / f"{image_name}_image_only.json").write_text(json.dumps(image_only, ensure_ascii=False, indent=2), encoding="utf-8")
        for cloud_name, cloud in CLOUDS.items():
            with image.open("rb") as image_stream, cloud.open("rb") as cloud_stream:
                response = requests.post(
                    BASE + "/api/predict-with-cloud",
                    files={"image": (image.name, image_stream, "image/jpeg"),
                           "cloud": (cloud.name, cloud_stream, "application/octet-stream")},
                    data={"paired": "true", "units": "metres", "z_up": "true",
                          "scenario": "street", "projection": "photo"}, timeout=240,
                )
            response.raise_for_status()
            dual = response.json()
            (OUT / f"{image_name}_{cloud_name}_dual.json").write_text(json.dumps(dual, ensure_ascii=False, indent=2), encoding="utf-8")
            comparison = dual.get("comparison", {})
            row = {
                "image": image_name, "image_sha256": sha(image),
                "cloud": cloud_name, "cloud_sha256": sha(cloud),
                "image_only_sstv1_top3": concise(image_only.get("top3", [])),
                "dual_teacher_image_top3": concise(comparison.get("image", [])),
                "point_top3": concise(comparison.get("point", [])),
                "fusion_top3": concise(comparison.get("fusion", [])),
                "fusion_used": dual.get("fusion_used"),
                "image_catalog_mass": dual.get("image_catalog_mass"),
                "point_support_state": dual.get("point_evidence", {}).get("support_state"),
                "warning": "The official single-image route uses SSTv1; the dual-source route uses the teacher image branch. Compare teacher-image vs fusion within the same dual response, not SSTv1 vs fusion as an accuracy delta.",
            }
            summary.append(row)
            print(image_name, cloud_name, "teacher", row["dual_teacher_image_top3"][:2],
                  "point", row["point_top3"][:2], "fusion", row["fusion_top3"][:2], flush=True)
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
