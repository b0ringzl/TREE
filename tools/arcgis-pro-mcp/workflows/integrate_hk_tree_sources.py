"""Integrate the three Hong Kong tree point sources and export two map layouts.

The workflow is intentionally non-destructive: source shapefiles are read only,
all outputs are created under a new run directory, and no cross-source records
are deduplicated.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from collections import Counter, defaultdict
import datetime as dt
from pathlib import Path
from typing import Any

import arcpy

TREE_ROOT = Path(__file__).resolve().parents[3]

SHARED_SPECIES = (
    "Cinnamomum camphora",
    "Lagerstroemia indica",
    "Magnolia grandiflora",
)

SOURCES = (
    {
        "key": "tree20250604",
        "label": "Tree Register 2025",
        "path": str(TREE_ROOT / "arcgis_dd" / "tree20250604_converted.shp"),
        "tree_id": "Tree_Regis",
        "species": "Tree_Speic",
        "dbh": "DBH_MM",
        "height": "Height_M",
        "crown": "Crown_Spre",
        "location": ("Location_E", "Location_T"),
    },
    {
        "key": "major_parks",
        "label": "Major Parks",
        "path": str(TREE_ROOT / "arcgis_dd" / "Trees_Major_Parks_converted.shp"),
        "tree_id": None,
        "species": "Scientific",
        "dbh": None,
        "height": None,
        "crown": None,
        "location": ("Location_i", "Location00"),
    },
    {
        "key": "csdi_20260317",
        "label": "CSDI 2026-03-17",
        "path": str(TREE_ROOT / "arcgis_dd" / "VIS_INV_TREE_CSDI_202603171359_converted.shp"),
        "tree_id": "TREE_ID",
        "species": "SPECIES_NA",
        "dbh": "DBH",
        "height": None,
        "crown": None,
        "location": ("ROAD_NAME",),
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).replace("\x00", " ").split())


def clean_number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def shared_species_name(value: Any) -> str:
    text = clean_text(value)
    folded = text.casefold()
    for species in SHARED_SPECIES:
        pattern = r"(?<![a-z])" + re.escape(species.casefold()) + r"(?![a-z])"
        if re.search(pattern, folded):
            return species
    return ""


def unique_existing_fields(source: dict[str, Any]) -> list[str]:
    candidates = [
        source["tree_id"],
        source["species"],
        source["dbh"],
        source["height"],
        source["crown"],
        *source["location"],
    ]
    available = {field.name.casefold(): field.name for field in arcpy.ListFields(source["path"])}
    fields: list[str] = []
    for candidate in candidates:
        if candidate and candidate.casefold() in available:
            actual = available[candidate.casefold()]
            if actual not in fields:
                fields.append(actual)
    return fields


def value(row: tuple[Any, ...], field_indexes: dict[str, int], field: str | None) -> Any:
    if not field:
        return None
    index = field_indexes.get(field.casefold())
    return None if index is None else row[index]


def first_text(row: tuple[Any, ...], field_indexes: dict[str, int], fields: tuple[str, ...]) -> str:
    for field in fields:
        candidate = clean_text(value(row, field_indexes, field))
        if candidate:
            return candidate
    return ""


def preflight(output_dir: Path) -> dict[str, Any]:
    report: dict[str, Any] = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "shared_species": list(SHARED_SPECIES),
        "sources": [],
    }
    for source in SOURCES:
        path = source["path"]
        if not arcpy.Exists(path):
            raise FileNotFoundError(path)
        description = arcpy.Describe(path)
        species_counts: Counter[str] = Counter()
        matched_raw: dict[str, Counter[str]] = defaultdict(Counter)
        missing_species = 0
        with arcpy.da.SearchCursor(path, [source["species"]]) as rows:
            for (raw_value,) in rows:
                raw = clean_text(raw_value)
                if not raw:
                    missing_species += 1
                    continue
                species_counts[raw] += 1
                matched = shared_species_name(raw)
                if matched:
                    matched_raw[matched][raw] += 1
        sr = description.spatialReference
        source_report = {
            "key": source["key"],
            "label": source["label"],
            "path": path,
            "feature_count": int(arcpy.management.GetCount(path)[0]),
            "spatial_reference": {"name": sr.name, "wkid": sr.factoryCode},
            "species_field": source["species"],
            "distinct_species_values": len(species_counts),
            "missing_species": missing_species,
            "shared_species_matches": {
                species: {
                    "count": sum(matched_raw[species].values()),
                    "raw_values": dict(matched_raw[species].most_common()),
                }
                for species in SHARED_SPECIES
            },
            "top_species_values": species_counts.most_common(15),
        }
        report["sources"].append(source_report)
        print(
            f"Preflight {source['label']}: {source_report['feature_count']:,} features, "
            f"{len(species_counts):,} species values",
            flush=True,
        )

    output_path = output_dir / "preflight_species.json"
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def add_standard_fields(feature_class: str) -> None:
    field_specs = (
        ("SOURCE", "TEXT", 40),
        ("SOURCE_LBL", "TEXT", 80),
        ("SOURCE_OID", "LONG", None),
        ("TREE_ID", "TEXT", 80),
        ("SPECIES_RAW", "TEXT", 254),
        ("SPECIES_STD", "TEXT", 128),
        ("SHARED3", "SHORT", None),
        ("DBH_RAW", "DOUBLE", None),
        ("HEIGHT_M", "DOUBLE", None),
        ("CROWN_RAW", "DOUBLE", None),
        ("LOCATION", "TEXT", 254),
    )
    for name, field_type, length in field_specs:
        kwargs = {"field_length": length} if length else {}
        arcpy.management.AddField(feature_class, name, field_type, **kwargs)


def integrate(output_dir: Path, preflight_report: dict[str, Any]) -> dict[str, Any]:
    geodatabase = output_dir / "hong_kong_trees.gdb"
    arcpy.management.CreateFileGDB(str(output_dir), geodatabase.name)
    output_sr = arcpy.SpatialReference(2326)
    all_trees = str(geodatabase / "trees_all_sources")
    arcpy.management.CreateFeatureclass(
        str(geodatabase),
        "trees_all_sources",
        "POINT",
        spatial_reference=output_sr,
    )
    add_standard_fields(all_trees)

    insert_fields = [
        "SHAPE@",
        "SOURCE",
        "SOURCE_LBL",
        "SOURCE_OID",
        "TREE_ID",
        "SPECIES_RAW",
        "SPECIES_STD",
        "SHARED3",
        "DBH_RAW",
        "HEIGHT_M",
        "CROWN_RAW",
        "LOCATION",
    ]
    source_counts: Counter[str] = Counter()
    blank_id_counts: Counter[str] = Counter()
    shared_counts: dict[str, Counter[str]] = defaultdict(Counter)
    skipped_null_geometry: Counter[str] = Counter()
    transformations: dict[str, str | None] = {}

    with arcpy.da.InsertCursor(all_trees, insert_fields) as insert_cursor:
        for source in SOURCES:
            existing_fields = unique_existing_fields(source)
            search_fields = ["OID@", "SHAPE@", *existing_fields]
            field_indexes = {name.casefold(): index + 2 for index, name in enumerate(existing_fields)}
            source_sr = arcpy.Describe(source["path"]).spatialReference
            transform = None
            if source_sr.factoryCode != output_sr.factoryCode:
                transformations_found = arcpy.ListTransformations(source_sr, output_sr)
                transform = transformations_found[0] if transformations_found else None
            transformations[source["key"]] = transform
            total = int(arcpy.management.GetCount(source["path"])[0])
            print(
                f"Integrating {source['label']}: {total:,} features"
                + (f" using {transform}" if transform else ""),
                flush=True,
            )
            with arcpy.da.SearchCursor(source["path"], search_fields) as rows:
                for row_number, row in enumerate(rows, start=1):
                    source_oid, geometry = row[0], row[1]
                    if geometry is None or getattr(geometry, "pointCount", 0) == 0:
                        skipped_null_geometry[source["key"]] += 1
                        continue
                    if source_sr.factoryCode != output_sr.factoryCode:
                        geometry = geometry.projectAs(output_sr, transform) if transform else geometry.projectAs(output_sr)
                    raw_species = clean_text(value(row, field_indexes, source["species"]))
                    shared_name = shared_species_name(raw_species)
                    species_std = shared_name or raw_species
                    tree_id = clean_text(value(row, field_indexes, source["tree_id"]))
                    if not tree_id:
                        blank_id_counts[source["key"]] += 1
                    location = first_text(row, field_indexes, source["location"])
                    insert_cursor.insertRow(
                        (
                            geometry,
                            source["key"],
                            source["label"],
                            int(source_oid),
                            tree_id,
                            raw_species,
                            species_std,
                            1 if shared_name else 0,
                            clean_number(value(row, field_indexes, source["dbh"])),
                            clean_number(value(row, field_indexes, source["height"])),
                            clean_number(value(row, field_indexes, source["crown"])),
                            location,
                        )
                    )
                    source_counts[source["key"]] += 1
                    if shared_name:
                        shared_counts[source["key"]][shared_name] += 1
                    if row_number % 25000 == 0 or row_number == total:
                        print(f"  {source['label']}: {row_number:,}/{total:,}", flush=True)

    for field_name in ("SOURCE", "SPECIES_STD", "SHARED3", "TREE_ID"):
        arcpy.management.AddIndex(all_trees, field_name, f"IX_{field_name}")

    shared_trees = str(geodatabase / "trees_hk_nanjing_shared3")
    arcpy.analysis.Select(all_trees, shared_trees, "SHARED3 = 1")
    arcpy.management.AddIndex(shared_trees, "SPECIES_STD", "IX_SPECIES_STD")

    all_count = int(arcpy.management.GetCount(all_trees)[0])
    shared_count = int(arcpy.management.GetCount(shared_trees)[0])
    expected_count = sum(item["feature_count"] for item in preflight_report["sources"])
    if all_count != expected_count - sum(skipped_null_geometry.values()):
        raise RuntimeError(
            f"Integrated count check failed: expected {expected_count:,}, got {all_count:,}"
        )

    summary = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "coordinate_system": {"name": output_sr.name, "wkid": output_sr.factoryCode},
        "deduplication": "none; all source records retained",
        "all_feature_count": all_count,
        "shared3_feature_count": shared_count,
        "source_counts": dict(source_counts),
        "blank_tree_id_counts": dict(blank_id_counts),
        "skipped_null_geometry": dict(skipped_null_geometry),
        "transformations": transformations,
        "shared_species_counts": {
            source["key"]: {species: shared_counts[source["key"]][species] for species in SHARED_SPECIES}
            for source in SOURCES
        },
        "outputs": {
            "geodatabase": str(geodatabase),
            "all_trees": all_trees,
            "shared3_trees": shared_trees,
        },
    }
    (output_dir / "integration_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv_reports(output_dir, summary)
    return summary


def write_csv_reports(output_dir: Path, summary: dict[str, Any]) -> None:
    with (output_dir / "source_summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source", "feature_count", "blank_tree_id_count"])
        for source in SOURCES:
            key = source["key"]
            writer.writerow(
                [key, summary["source_counts"].get(key, 0), summary["blank_tree_id_counts"].get(key, 0)]
            )

    with (output_dir / "shared_species_counts.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["source", "species", "feature_count"])
        for source in SOURCES:
            key = source["key"]
            for species in SHARED_SPECIES:
                writer.writerow([key, species, summary["shared_species_counts"][key][species]])
        for species in SHARED_SPECIES:
            total = sum(summary["shared_species_counts"][source["key"]][species] for source in SOURCES)
            writer.writerow(["ALL_SOURCES", species, total])


def rectangle(x: float, y: float, width: float, height: float) -> arcpy.Polygon:
    points = arcpy.Array(
        [
            arcpy.Point(x, y),
            arcpy.Point(x + width, y),
            arcpy.Point(x + width, y + height),
            arcpy.Point(x, y + height),
            arcpy.Point(x, y),
        ]
    )
    return arcpy.Polygon(points)


def apply_unique_value_symbols(layer: Any, field: str, colors: dict[str, tuple[int, int, int]], size: float) -> None:
    symbology = layer.symbology
    symbology.updateRenderer("UniqueValueRenderer")
    symbology.renderer.fields = [field]
    for group in symbology.renderer.groups:
        for item in group.items:
            raw_value = item.values[0][0]
            key = "" if raw_value is None else str(raw_value)
            color = colors.get(key, (100, 100, 100))
            item.symbol.color = {"RGB": [*color, 78]}
            item.symbol.size = size
            item.label = key
    layer.symbology = symbology


def style_admin_boundary(layer: Any) -> None:
    symbology = layer.symbology
    symbology.updateRenderer("SimpleRenderer")
    symbology.renderer.symbol.color = {"RGB": [232, 236, 238, 82]}
    symbology.renderer.symbol.outlineColor = {"RGB": [88, 96, 101, 100]}
    symbology.renderer.symbol.outlineWidth = 0.8
    layer.symbology = symbology
    try:
        layer.showLabels = False
    except (AttributeError, RuntimeError):
        pass


def add_layout_text(layout: Any, project: Any, x: float, y: float, text: str, size: float, name: str) -> Any:
    element = project.createTextElement(
        layout,
        arcpy.Point(x, y),
        "POINT",
        text,
        size,
        "Microsoft YaHei",
        "Regular",
        None,
        name,
    )
    element.setAnchor("TOP_LEFT_CORNER")
    return element


def create_map_layout(
    project: Any,
    map_name: str,
    layout_name: str,
    feature_class: str,
    unique_field: str,
    colors: dict[str, tuple[int, int, int]],
    title: str,
    subtitle: str,
    note: str,
    point_size: float,
    output_png: Path,
    output_pdf: Path,
    background_feature_class: str | None = None,
) -> None:
    map_object = project.createMap(map_name, "MAP")
    for layer in map_object.listLayers():
        map_object.removeLayer(layer)
    if background_feature_class:
        boundary_layer = map_object.addDataFromPath(background_feature_class)
        boundary_layer.name = "香港十八区行政界线"
        style_admin_boundary(boundary_layer)
    layer = map_object.addDataFromPath(feature_class)
    layer.name = "数据来源" if unique_field == "SOURCE" else "共有树种"
    apply_unique_value_symbols(layer, unique_field, colors, point_size)

    layout = project.createLayout(11.69, 8.27, "INCH", layout_name)
    map_frame = layout.createMapFrame(rectangle(0.45, 0.68, 8.95, 6.78), map_object, "Map Frame")
    extent = arcpy.Describe(background_feature_class or feature_class).extent
    map_frame.camera.setExtent(extent)
    map_frame.camera.scale *= 1.08

    title_element = add_layout_text(layout, project, 0.48, 8.02, title, 20, "Title")
    title_element.fontStyleName = "Bold"
    add_layout_text(layout, project, 0.50, 7.62, subtitle, 9.5, "Subtitle")
    add_layout_text(
        layout,
        project,
        0.55,
        0.48,
        "投影：Hong Kong 1980 Grid（EPSG:2326），坐标单位：米",
        7.2,
        "Coordinate System",
    )
    add_layout_text(layout, project, 9.58, 1.35, note, 7.5, "Method Note")

    legend = layout.createMapSurroundElement(
        arcpy.Point(9.55, 6.75), "LEGEND", map_frame, None, "Legend"
    )
    legend.elementWidth = 1.7
    legend.elementHeight = 2.5
    try:
        legend.title = "图例"
        legend.syncLayerOrder = True
    except (AttributeError, RuntimeError):
        pass

    north_arrow = layout.createMapSurroundElement(
        arcpy.Point(10.35, 1.95), "NORTH_ARROW", map_frame, None, "North Arrow"
    )
    north_arrow.elementHeight = 0.55

    layout.exportToPNG(str(output_png), resolution=220)
    layout.exportToPDF(str(output_pdf), resolution=220)


def render_maps(
    output_dir: Path,
    summary: dict[str, Any],
    background_feature_class: str | None = None,
) -> dict[str, str]:
    template = Path(
        r"C:\Program Files\ArcGIS\Pro\Resources\ArcToolBox\Services\routingservices\data\Blank.aprx"
    )
    if not template.exists():
        raise FileNotFoundError(template)
    working_template = output_dir / f"_working_blank_{dt.datetime.now():%H%M%S%f}.aprx"
    shutil.copy2(template, working_template)
    project = arcpy.mp.ArcGISProject(str(working_template))
    for map_object in list(project.listMaps()):
        project.deleteItem(map_object)

    outputs = {
        "all_png": str(output_dir / "hong_kong_trees_all_sources.png"),
        "all_pdf": str(output_dir / "hong_kong_trees_all_sources.pdf"),
        "shared_png": str(output_dir / "hong_kong_nanjing_shared3.png"),
        "shared_pdf": str(output_dir / "hong_kong_nanjing_shared3.pdf"),
        "project": str(output_dir / "hong_kong_tree_maps.aprx"),
    }
    create_map_layout(
        project,
        "All Hong Kong Trees",
        "All Sources Layout",
        summary["outputs"]["all_trees"],
        "SOURCE",
        {
            "tree20250604": (36, 113, 163),
            "major_parks": (230, 126, 34),
            "csdi_20260317": (33, 145, 112),
        },
        "香港树木三来源整合总图",
        f"共 {summary['all_feature_count']:,} 条记录｜香港 1980 Grid（EPSG:2326）",
        "三个调查来源全部保留\n未执行跨来源去重",
        1.4,
        Path(outputs["all_png"]),
        Path(outputs["all_pdf"]),
        background_feature_class,
    )
    create_map_layout(
        project,
        "HK-Nanjing Shared Species",
        "Shared Species Layout",
        summary["outputs"]["shared3_trees"],
        "SPECIES_STD",
        {
            "Cinnamomum camphora": (0, 114, 188),
            "Lagerstroemia indica": (213, 94, 0),
            "Magnolia grandiflora": (0, 143, 87),
        },
        "香港与南京共有三树种分布图",
        f"共 {summary['shared3_feature_count']:,} 条记录｜樟、紫薇、广玉兰",
        "按学名匹配，兼容学名后的\n命名人等附加文本",
        3.0,
        Path(outputs["shared_png"]),
        Path(outputs["shared_pdf"]),
        background_feature_class,
    )
    project.saveACopy(outputs["project"])
    del project
    try:
        working_template.unlink(missing_ok=True)
    except PermissionError:
        # ArcGIS Pro may retain an APRX file lock briefly after export.
        pass
    return outputs


def write_readme(output_dir: Path, summary: dict[str, Any], maps: dict[str, str]) -> None:
    species_totals = {
        species: sum(summary["shared_species_counts"][source["key"]][species] for source in SOURCES)
        for species in SHARED_SPECIES
    }
    lines = [
        "# 香港树木三来源整合结果",
        "",
        f"生成时间：{summary['generated_at']}",
        "",
        "## 数据处理",
        "",
        "- 三个来源统一投影到 Hong Kong 1980 Grid（EPSG:2326）。",
        "- 原始 Shapefile 只读，输出保存在本目录内。",
        "- 不执行跨来源去重；不同调查来源的记录全部保留。",
        f"- 整合总数：{summary['all_feature_count']:,} 条。",
        f"- 香港与南京共有三树种：{summary['shared3_feature_count']:,} 条。",
        "",
        "## 三个共有树种",
        "",
        *[f"- {species}: {species_totals[species]:,} 条" for species in SHARED_SPECIES],
        "",
        "## 主要成果",
        "",
        f"- File Geodatabase：`{summary['outputs']['geodatabase']}`",
        f"- 全部树木要素类：`{summary['outputs']['all_trees']}`",
        f"- 三树种要素类：`{summary['outputs']['shared3_trees']}`",
        f"- ArcGIS Pro 工程：`{maps['project']}`",
        f"- 总图 PNG/PDF：`{maps['all_png']}` / `{maps['all_pdf']}`",
        f"- 三树种图 PNG/PDF：`{maps['shared_png']}` / `{maps['shared_pdf']}`",
        "",
        "## 字段说明",
        "",
        "`SOURCE` 为来源代码，`SOURCE_OID` 为来源内原对象号，`TREE_ID` 为原树木编号，",
        "`SPECIES_RAW` 保留原始树种文本，`SPECIES_STD` 对三个目标树种统一为标准学名，",
        "`SHARED3=1` 表示属于香港与南京共有三树种。Major Parks 来源本身没有树木 ID。",
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    allowed_root = (TREE_ROOT / "gis_output").resolve()
    if allowed_root != output_dir and allowed_root not in output_dir.parents:
        raise ValueError(f"Output must be under {allowed_root}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    arcpy.env.overwriteOutput = False

    report = preflight(output_dir)
    if args.preflight_only:
        print(json.dumps({"status": "preflight_complete", "output_dir": str(output_dir)}))
        return 0

    summary = integrate(output_dir, report)
    print("Rendering map layouts...", flush=True)
    maps = render_maps(output_dir, summary)
    summary["maps"] = maps
    (output_dir / "integration_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_readme(output_dir, summary, maps)
    print(
        json.dumps(
            {
                "status": "complete",
                "output_dir": str(output_dir),
                "all_feature_count": summary["all_feature_count"],
                "shared3_feature_count": summary["shared3_feature_count"],
                "maps": maps,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        import traceback

        traceback.print_exc()
        raise
