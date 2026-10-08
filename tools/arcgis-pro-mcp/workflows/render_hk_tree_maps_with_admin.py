"""Render the integrated Hong Kong tree maps over official 18-district boundaries."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import arcpy

import integrate_hk_tree_sources as tree_workflow

TREE_ROOT = Path(__file__).resolve().parents[3]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--integration-dir", required=True)
    parser.add_argument("--boundary-json", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    integration_dir = Path(args.integration_dir).resolve()
    boundary_json = Path(args.boundary_json).resolve()
    output_dir = Path(args.output_dir).resolve()
    allowed_root = (TREE_ROOT / "gis_output").resolve()
    for path in (integration_dir, boundary_json, output_dir):
        if allowed_root != path and allowed_root not in path.parents:
            raise ValueError(f"Path must be under {allowed_root}: {path}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    arcpy.env.overwriteOutput = False

    summary = json.loads(
        (integration_dir / "integration_summary.json").read_text(encoding="utf-8")
    )
    local_json = output_dir / "hksar_18_district_boundary.json"
    shutil.copy2(boundary_json, local_json)
    boundary_gdb = output_dir / "hong_kong_admin_boundary.gdb"
    arcpy.management.CreateFileGDB(str(output_dir), boundary_gdb.name)
    boundary_wgs84 = str(boundary_gdb / "districts_wgs84")
    boundary_hk1980 = str(boundary_gdb / "districts_hk1980")
    arcpy.conversion.JSONToFeatures(str(local_json), boundary_wgs84, "POLYGON")
    transformations = arcpy.ListTransformations(
        arcpy.SpatialReference(4326), arcpy.SpatialReference(2326)
    )
    transformation = transformations[0] if transformations else None
    arcpy.management.Project(
        boundary_wgs84,
        boundary_hk1980,
        arcpy.SpatialReference(2326),
        transformation,
    )
    district_count = int(arcpy.management.GetCount(boundary_hk1980)[0])
    if district_count != 18:
        raise RuntimeError(f"Expected 18 districts, got {district_count}")

    maps = tree_workflow.render_maps(output_dir, summary, boundary_hk1980)
    result = {
        "status": "complete",
        "district_count": district_count,
        "boundary_source": "Hong Kong Home Affairs Department",
        "boundary_source_url": (
            "https://www.had.gov.hk/psi/hong-kong-administrative-boundaries/"
            "hksar_18_district_boundary.json"
        ),
        "boundary_transformation": transformation,
        "boundary_feature_class": boundary_hk1980,
        "tree_integration_dir": str(integration_dir),
        "maps": maps,
    }
    (output_dir / "admin_map_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                "# 带香港十八区行政区背景的树木分布图",
                "",
                "行政区界线来源：香港特别行政区政府民政事务总署。",
                f"行政区数量：{district_count}。",
                "行政区与树木点统一为 Hong Kong 1980 Grid（EPSG:2326）。",
                "树木点数据引用既有三来源整合成果，未重新去重或修改属性。",
                "",
                f"- 总图：`{maps['all_png']}`",
                f"- 三树种图：`{maps['shared_png']}`",
                f"- ArcGIS Pro 工程：`{maps['project']}`",
                f"- 行政区要素：`{boundary_hk1980}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
