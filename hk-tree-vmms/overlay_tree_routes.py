from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
from shapely.geometry import mapping, shape


TREE_SOURCES = (
    {
        "source_id": "tree20250604",
        "filename": "tree20250604_converted.shp",
        "id_fields": ("TMCP_Tree_", "Tree_Regis"),
        "species_field": "Tree_Speic",
        "location_fields": ("Location_E", "Location_T"),
        "dbh_field": "DBH_MM",
        "height_field": "Height_M",
    },
    {
        "source_id": "major_parks",
        "filename": "Trees_Major_Parks_converted.shp",
        "id_fields": (),
        "species_field": "Scientific",
        "location_fields": ("Location00", "Location_i"),
        "dbh_field": None,
        "height_field": None,
    },
    {
        "source_id": "csdi_roadside",
        "filename": "VIS_INV_TREE_CSDI_202603171359_converted.shp",
        "id_fields": ("TREE_ID",),
        "species_field": "SPECIES_NA",
        "location_fields": ("ROAD_NAME",),
        "dbh_field": "DBH",
        "height_field": None,
    },
)


CSV_COLUMNS = (
    "source_dataset",
    "source_tree_id",
    "source_feature_index",
    "species_name",
    "location_name",
    "dbh_source_value",
    "height_source_value",
    "nearest_route_id",
    "distance_to_route_m",
    "distance_band",
    "within_50m",
    "hk80_easting",
    "hk80_northing",
    "wgs84_latitude",
    "wgs84_longitude",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay Hong Kong tree inventories on VMMS acquisition routes."
    )
    parser.add_argument("--route-geojson", type=Path, required=True)
    parser.add_argument("--shapefile-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-distance-m", type=float, default=100.0)
    parser.add_argument("--primary-distance-m", type=float, default=50.0)
    return parser.parse_args()


def clean_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if math.isnan(float(value)):
            return None
        return float(value)
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    return text or None


def first_value(row: Any, fields: tuple[str, ...]) -> Any:
    for field in fields:
        value = clean_value(row.get(field))
        if value is not None:
            return value
    return None


def distance_band(distance: float) -> str:
    for threshold, label in (
        (10.0, "0-10m"),
        (20.0, "10-20m"),
        (30.0, "20-30m"),
        (50.0, "30-50m"),
        (100.0, "50-100m"),
    ):
        if distance <= threshold:
            return label
    return ">100m"


def load_routes(path: Path) -> tuple[dict[str, Any], gpd.GeoDataFrame]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    route_features = [
        feature
        for feature in raw["features"]
        if feature.get("geometry", {}).get("type") == "LineString"
    ]
    if not route_features:
        raise ValueError(f"No LineString routes found in {path}")
    routes = gpd.GeoDataFrame(
        [feature.get("properties", {}) for feature in route_features],
        geometry=[shape(feature["geometry"]) for feature in route_features],
        crs="EPSG:4326",
    ).to_crs("EPSG:2326")
    if "stream_id" not in routes.columns:
        raise ValueError(f"Route features do not contain stream_id: {path}")
    return raw, routes


def normalize_source(
    source: dict[str, Any],
    path: Path,
    routes: gpd.GeoDataFrame,
    max_distance_m: float,
    primary_distance_m: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    frame = gpd.read_file(path)
    if frame.crs is None:
        raise ValueError(f"Shapefile has no CRS: {path}")
    original_count = len(frame)
    frame = frame[frame.geometry.notna() & ~frame.geometry.is_empty].copy()
    frame_hk80 = frame.to_crs("EPSG:2326")

    distances = np.column_stack(
        [frame_hk80.geometry.distance(route).to_numpy() for route in routes.geometry]
    )
    nearest_route_index = distances.argmin(axis=1)
    nearest_distance = distances.min(axis=1)
    keep = nearest_distance <= max_distance_m

    selected = frame_hk80.loc[keep].copy()
    selected["_nearest_route_index"] = nearest_route_index[keep]
    selected["_nearest_distance"] = nearest_distance[keep]
    selected_wgs84 = selected.to_crs("EPSG:4326")

    rows: list[dict[str, Any]] = []
    for position, ((index, row), geometry_wgs84) in enumerate(
        zip(selected.iterrows(), selected_wgs84.geometry)
    ):
        route_index = int(row["_nearest_route_index"])
        distance = float(row["_nearest_distance"])
        source_tree_id = first_value(row, source["id_fields"])
        if source_tree_id is None:
            source_tree_id = f"{source['source_id']}_{index}"
        record = {
            "source_dataset": source["source_id"],
            "source_tree_id": source_tree_id,
            "source_feature_index": clean_value(index),
            "species_name": clean_value(row.get(source["species_field"])),
            "location_name": first_value(row, source["location_fields"]),
            "dbh_source_value": (
                clean_value(row.get(source["dbh_field"]))
                if source["dbh_field"]
                else None
            ),
            "height_source_value": (
                clean_value(row.get(source["height_field"]))
                if source["height_field"]
                else None
            ),
            "nearest_route_id": clean_value(routes.iloc[route_index]["stream_id"]),
            "distance_to_route_m": round(distance, 3),
            "distance_band": distance_band(distance),
            "within_50m": distance <= primary_distance_m,
            "hk80_easting": round(float(row.geometry.x), 3),
            "hk80_northing": round(float(row.geometry.y), 3),
            "wgs84_latitude": round(float(geometry_wgs84.y), 9),
            "wgs84_longitude": round(float(geometry_wgs84.x), 9),
        }
        rows.append(record)

    by_threshold = {
        str(int(threshold)): int((nearest_distance <= threshold).sum())
        for threshold in (5, 10, 15, 20, 30, 50, 100)
    }
    summary = {
        "source_dataset": source["source_id"],
        "source_path": str(path),
        "source_crs": str(frame.crs),
        "source_feature_count": original_count,
        "nearby_feature_count": len(rows),
        "counts_within_distance_m": by_threshold,
        "nearby_by_route": dict(Counter(row["nearest_route_id"] for row in rows)),
    }
    return rows, summary


def tree_feature(row: dict[str, Any]) -> dict[str, Any]:
    properties = {key: clean_value(value) for key, value in row.items()}
    properties["feature_kind"] = "tree"
    return {
        "type": "Feature",
        "properties": properties,
        "geometry": {
            "type": "Point",
            "coordinates": [row["wgs84_longitude"], row["wgs84_latitude"]],
        },
    }


def main() -> int:
    args = parse_args()
    route_path = args.route_geojson.resolve()
    shapefile_dir = args.shapefile_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_routes, routes = load_routes(route_path)
    all_rows: list[dict[str, Any]] = []
    source_summaries: list[dict[str, Any]] = []
    for source in TREE_SOURCES:
        rows, summary = normalize_source(
            source,
            shapefile_dir / source["filename"],
            routes,
            args.max_distance_m,
            args.primary_distance_m,
        )
        all_rows.extend(rows)
        source_summaries.append(summary)

    combined_route_features = []
    for feature in raw_routes["features"]:
        copied = json.loads(json.dumps(feature))
        copied.setdefault("properties", {})["feature_kind"] = (
            "route"
            if copied.get("geometry", {}).get("type") == "LineString"
            else "route_endpoint"
        )
        combined_route_features.append(copied)

    combined = {
        "type": "FeatureCollection",
        "name": "VMMS routes with nearby Hong Kong tree inventories",
        "crs_note": "RFC 7946 WGS84 longitude/latitude",
        "selection_note": (
            f"Trees are included when their 2D HK80 distance to the nearest route "
            f"is <= {args.max_distance_m:g} m. within_50m uses "
            f"{args.primary_distance_m:g} m."
        ),
        "features": combined_route_features + [tree_feature(row) for row in all_rows],
    }
    overlay_path = output_dir / "route_with_nearby_trees.geojson"
    overlay_path.write_text(
        json.dumps(combined, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )

    with (output_dir / "nearby_trees.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(all_rows)

    by_route: dict[str, dict[str, int]] = defaultdict(lambda: {"within_50m": 0, "within_100m": 0})
    for row in all_rows:
        route = str(row["nearest_route_id"])
        by_route[route]["within_100m"] += 1
        if row["within_50m"]:
            by_route[route]["within_50m"] += 1

    species_counter = Counter(
        row["species_name"] for row in all_rows if row["species_name"]
    )
    summary = {
        "route_geojson": str(route_path),
        "output_geojson": str(overlay_path),
        "distance_crs": "EPSG:2326 Hong Kong 1980 Grid System",
        "output_crs": "EPSG:4326 WGS 84",
        "max_distance_m": args.max_distance_m,
        "primary_distance_m": args.primary_distance_m,
        "route_count": len(routes),
        "nearby_tree_count": len(all_rows),
        "within_primary_distance_count": sum(
            bool(row["within_50m"]) for row in all_rows
        ),
        "counts_by_distance_band": dict(Counter(row["distance_band"] for row in all_rows)),
        "counts_by_route": dict(by_route),
        "top_species": [
            {"species_name": name, "count": count}
            for name, count in species_counter.most_common(30)
        ],
        "sources": source_summaries,
    }
    (output_dir / "tree_route_overlay_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Routes: {len(routes)}")
    print(f"Trees within {args.max_distance_m:g} m: {len(all_rows)}")
    print(
        f"Trees within {args.primary_distance_m:g} m: "
        f"{sum(bool(row['within_50m']) for row in all_rows)}"
    )
    print(f"Output: {overlay_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

