#!/usr/bin/env python
"""Export Hong Kong tree layers to browser-friendly GIS formats."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
import zipfile
from pathlib import Path

import arcpy


WGS84 = arcpy.SpatialReference(4326)
HK_TO_WGS84 = "Hong_Kong_1980_To_WGS_1984_1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--integration-dir", type=Path, required=True)
    parser.add_argument("--admin-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def field_mappings(in_features: str, fields: list[tuple[str, str, int | None]]):
    mappings = arcpy.FieldMappings()
    for source_name, output_name, output_length in fields:
        mapping = arcpy.FieldMap()
        mapping.addInputField(in_features, source_name)
        output_field = mapping.outputField
        output_field.name = output_name
        output_field.aliasName = output_name
        if output_length and output_field.type == "String":
            output_field.length = output_length
        mapping.outputField = output_field
        mappings.addFieldMap(mapping)
    return mappings


def export_fields(
    in_features: str,
    out_features: str,
    fields: list[tuple[str, str, int | None]],
) -> None:
    arcpy.conversion.ExportFeatures(
        in_features,
        out_features,
        field_mapping=field_mappings(in_features, fields),
    )


def export_geojson(in_features: str, out_path: Path) -> None:
    arcpy.conversion.FeaturesToJSON(
        in_features,
        str(out_path),
        format_json="NOT_FORMATTED",
        include_z_values="NO_Z_VALUES",
        include_m_values="NO_M_VALUES",
        geoJSON="GEOJSON",
        outputToWGS84="WGS84",
        use_field_alias="USE_FIELD_NAME",
    )


def combine_geojson(
    districts_path: Path, shared_trees_path: Path, output_path: Path
) -> int:
    districts = json.loads(districts_path.read_text(encoding="utf-8"))
    shared_trees = json.loads(shared_trees_path.read_text(encoding="utf-8"))
    features = []
    for feature in districts["features"]:
        feature["properties"]["MAP_LAYER"] = "Hong Kong 18 Districts"
        features.append(feature)
    for feature in shared_trees["features"]:
        feature["properties"]["MAP_LAYER"] = "HK-Nanjing Shared Species"
        features.append(feature)
    output_path.write_text(
        json.dumps(
            {"type": "FeatureCollection", "features": features},
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    return len(features)


def export_csv(in_features: str, out_path: Path) -> int:
    count = 0
    with out_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(["longitude", "latitude", "source", "species", "shared3"])
        fields = ["SHAPE@XY", "SOURCE", "SPECIES_STD", "SHARED3"]
        with arcpy.da.SearchCursor(in_features, fields) as rows:
            for (longitude, latitude), source, species, shared3 in rows:
                writer.writerow(
                    [
                        f"{longitude:.8f}",
                        f"{latitude:.8f}",
                        source or "",
                        species or "",
                        int(shared3 or 0),
                    ]
                )
                count += 1
    return count


def zip_shapefiles(shapefile_dir: Path, output_zip: Path) -> None:
    extensions = {".shp", ".shx", ".dbf", ".prj", ".cpg"}
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(shapefile_dir.iterdir()):
            if path.suffix.lower() in extensions:
                archive.write(path, arcname=path.name)


def feature_count(path: str | Path) -> int:
    return int(arcpy.management.GetCount(str(path))[0])


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume:
        raise RuntimeError(f"Output directory is not empty: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    shapefile_dir = output_dir / "shapefiles"
    shapefile_dir.mkdir(exist_ok=True)

    source_gdb = args.integration_dir / "hong_kong_trees.gdb"
    all_source = source_gdb / "trees_all_sources"
    shared_source = source_gdb / "trees_hk_nanjing_shared3"
    district_json = args.admin_dir / "hksar_18_district_boundary.json"

    for required in (all_source, shared_source, district_json):
        if not arcpy.Exists(str(required)) and not required.exists():
            raise FileNotFoundError(required)

    arcpy.env.overwriteOutput = True
    work_gdb = output_dir / "online_preview.gdb"
    if not arcpy.Exists(str(work_gdb)):
        arcpy.management.CreateFileGDB(str(output_dir), work_gdb.name)

    all_wgs84 = work_gdb / "trees_all_wgs84"
    shared_wgs84 = work_gdb / "trees_shared3_wgs84"
    districts_wgs84 = work_gdb / "districts_wgs84"

    print("[1/6] Projecting all tree points to WGS 84...", flush=True)
    if not arcpy.Exists(str(all_wgs84)):
        arcpy.management.Project(
            str(all_source), str(all_wgs84), WGS84, HK_TO_WGS84
        )
    else:
        print("      Reusing completed all-tree projection.", flush=True)
    print("[2/6] Projecting shared-species tree points to WGS 84...", flush=True)
    if not arcpy.Exists(str(shared_wgs84)):
        arcpy.management.Project(
            str(shared_source), str(shared_wgs84), WGS84, HK_TO_WGS84
        )
    else:
        print("      Reusing completed shared-species projection.", flush=True)
    print("[3/6] Importing the 18-district boundary...", flush=True)
    if not arcpy.Exists(str(districts_wgs84)):
        arcpy.conversion.JSONToFeatures(str(district_json), str(districts_wgs84))
    else:
        print("      Reusing completed district import.", flush=True)

    all_fields = [
        ("SOURCE", "SOURCE", 40),
        ("SPECIES_STD", "SPECIES", 160),
        ("SHARED3", "SHARED3", None),
    ]
    shared_fields = [
        ("SOURCE", "SOURCE", 40),
        ("SPECIES_STD", "SPECIES", 160),
    ]
    district_fields = [
        ("District", "DIST_EN", 80),
        ("\u5730\u5340", "DIST_ZH", 80),
    ]

    all_shp = shapefile_dir / "hk_trees_all.shp"
    shared_shp = shapefile_dir / "hk_trees_shared3.shp"
    district_shp = shapefile_dir / "hk_districts_18.shp"

    print("[4/6] Writing online-preview Shapefiles...", flush=True)
    export_fields(str(all_wgs84), str(all_shp), all_fields)
    export_fields(str(shared_wgs84), str(shared_shp), shared_fields)
    export_fields(str(districts_wgs84), str(district_shp), district_fields)

    shared_online = work_gdb / "trees_shared3_online"
    districts_online = work_gdb / "districts_online"
    export_fields(str(shared_wgs84), str(shared_online), shared_fields)
    export_fields(str(districts_wgs84), str(districts_online), district_fields)

    print("[5/6] Writing CSV and GeoJSON files...", flush=True)
    all_csv = output_dir / "hk_trees_all_wgs84.csv"
    csv_count = export_csv(str(all_wgs84), all_csv)
    shared_geojson = output_dir / "hk_trees_shared3.geojson"
    districts_geojson = output_dir / "hk_districts_18.geojson"
    export_geojson(str(shared_online), shared_geojson)
    export_geojson(str(districts_online), districts_geojson)
    combined_geojson = output_dir / "hk_shared3_with_districts.geojson"
    combined_count = combine_geojson(
        districts_geojson, shared_geojson, combined_geojson
    )

    print("[6/6] Packaging and validating outputs...", flush=True)
    bundle_zip = output_dir / "hk_tree_mapshaper_bundle.zip"
    zip_shapefiles(shapefile_dir, bundle_zip)

    counts = {
        "all_trees": feature_count(all_shp),
        "shared_species_trees": feature_count(shared_shp),
        "districts": feature_count(district_shp),
        "csv_rows": csv_count,
    }
    expected = {
        "all_trees": feature_count(all_source),
        "shared_species_trees": feature_count(shared_source),
        "districts": 18,
    }
    if counts["all_trees"] != expected["all_trees"]:
        raise RuntimeError("All-tree count changed during export")
    if counts["shared_species_trees"] != expected["shared_species_trees"]:
        raise RuntimeError("Shared-species count changed during export")
    if counts["districts"] != expected["districts"]:
        raise RuntimeError("District count changed during export")
    if counts["csv_rows"] != expected["all_trees"]:
        raise RuntimeError("CSV row count does not match all-tree count")
    if combined_count != counts["districts"] + counts["shared_species_trees"]:
        raise RuntimeError("Combined GeoJSON count is incomplete")

    readme_text = (
        "# \u9999\u6e2f\u6811\u6728\u6570\u636e\u5728\u7ebf\u9884\u89c8\n\n"
        "\u6240\u6709\u5bfc\u51fa\u56fe\u5c42\u90fd\u5df2\u8f6c\u6362\u4e3a WGS 84 (EPSG:4326)\u3002\n\n"
        "## \u63a8\u8350\u65b9\u5f0f 1: Kepler.gl\n\n"
        "1. \u6253\u5f00 https://kepler.gl/demo\n"
        "2. \u9009\u62e9 Load Files\u3002\n"
        "3. \u52a0\u8f7d `hk_trees_all_wgs84.csv` \u67e5\u770b 307,293 \u68f5\u6811\u3002\n"
        "4. \u53ef\u518d\u52a0\u8f7d `hk_trees_shared3.geojson` \u548c `hk_districts_18.geojson`\u3002\n\n"
        "## \u63a8\u8350\u65b9\u5f0f 2: Mapshaper\n\n"
        "1. \u6253\u5f00 https://mapshaper.org\n"
        "2. \u628a `hk_tree_mapshaper_bundle.zip` \u6574\u4e2a\u62d6\u5165\u9875\u9762\u3002\n"
        "3. ZIP \u5305\u542b\u5168\u91cf\u6811\u6728\u3001\u4e09\u4e2a\u5171\u6709\u6811\u79cd\u548c 18 \u533a\u8fb9\u754c\u56fe\u5c42\u3002\n\n"
        "## \u6587\u4ef6\u8bf4\u660e\n\n"
        "- `hk_trees_all_wgs84.csv`: \u5168\u91cf\u70b9\u4f4d\uff0c\u9002\u5408 Kepler.gl\u3002\n"
        "- `hk_trees_shared3.geojson`: \u9999\u6e2f\u4e0e\u5357\u4eac\u5171\u6709\u4e09\u4e2a\u6811\u79cd\u3002\n"
        "- `hk_districts_18.geojson`: \u9999\u6e2f 18 \u533a\u884c\u653f\u533a\u8fb9\u754c\u3002\n"
        "- `hk_shared3_with_districts.geojson`: \u5355\u6587\u4ef6\u76f4\u63a5\u62d6\u5165\u7248\uff0c\u540c\u65f6\u5305\u542b 18 \u533a\u8fb9\u754c\u548c\u4e09\u4e2a\u5171\u6709\u6811\u79cd\u3002\n"
        "- `hk_tree_mapshaper_bundle.zip`: Mapshaper \u53ef\u76f4\u63a5\u8bfb\u53d6\u7684 SHP \u538b\u7f29\u5305\u3002\n"
        "- `shapefiles/`: \u89e3\u538b\u540e\u7684 SHP \u7ec4\u4ef6\uff0c\u4e5f\u53ef\u5728 ArcGIS/QGIS \u6253\u5f00\u3002\n\n"
        "\u5bfc\u51fa\u672a\u5305\u542b TREE_ID\u3001\u8be6\u7ec6\u4f4d\u7f6e\u6587\u672c\u6216\u6811\u6728\u5c3a\u5bf8\u7b49\u975e\u5fc5\u8981\u5c5e\u6027\u3002\n"
    )
    (output_dir / "README_ONLINE_PREVIEW.md").write_text(
        readme_text, encoding="utf-8"
    )

    summary = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "coordinate_system": "WGS 84 (EPSG:4326)",
        "counts": counts,
        "files": {
            path.name: path.stat().st_size
            for path in [
                all_csv,
                shared_geojson,
                districts_geojson,
                combined_geojson,
                bundle_zip,
            ]
        },
    }
    (output_dir / "export_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        print(arcpy.GetMessages(), file=sys.stderr)
        raise
