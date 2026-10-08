from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

TREE_ROOT = Path(__file__).resolve().parent.parent


WHU_BENCHMARK_SPECIES = (
    (0, "Platanus x acerifolia", "PA", 5260, 30.7),
    (1, "Cinnamomum camphora", "CC", 3935, 23.0),
    (2, "Lagerstroemia indica", "LI", 1805, 10.5),
    (3, "Osmanthus fragrans", "OF", 868, 5.1),
    (4, "Koelreuteria paniculata", "KP", 609, 3.6),
    (5, "Prunus cerasifera 'atropurpurea'", "PC", 480, 2.8),
    (6, "Zelkova serrata", "ZS", 534, 3.1),
    (7, "Cedrus deodara", "CD", 526, 3.1),
    (8, "Ginkgo biloba", "GB", 250, 1.5),
    (9, "Acer palmatum var. atropurpureum", "AA", 180, 1.1),
    (10, "Prunus serrulata", "PS", 309, 1.8),
    (11, "Ligustrum lucidum", "LL", 339, 2.0),
    (12, "Malus x micromalus", "MM", 328, 1.9),
    (13, "Magnolia grandiflora", "MG", 245, 1.4),
    (14, "Metasequoia glyptostroboides", "MS", 310, 1.8),
    (15, "Sophora japonica", "SJ", 166, 1.0),
    (16, "Acer pictum subsp. mono", "AM", 231, 1.3),
    (17, "Photinia x fraseri", "PF", 163, 1.0),
    (18, "Others", "Others", 596, 3.5),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a lightweight WHU-STree index without reading point payloads."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=TREE_ROOT / "lidar data" / "whu",
    )
    parser.add_argument(
        "--hk-species-root",
        type=Path,
        default=TREE_ROOT / "选取树种",
    )
    parser.add_argument(
        "--merge-map",
        type=Path,
        default=TREE_ROOT / "tools" / "tree_species_label_merge_map.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <data-root>/metadata.",
    )
    return parser.parse_args()


def read_ply_header(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "format": "",
        "vertex_count": 0,
        "properties": [],
    }
    with path.open("rb") as stream:
        first_line = stream.readline().decode("ascii", errors="replace").strip()
        if first_line != "ply":
            raise ValueError(f"Not a PLY file: {path}")

        current_element = ""
        while True:
            raw_line = stream.readline()
            if not raw_line:
                raise ValueError(f"PLY header has no end_header marker: {path}")
            line = raw_line.decode("ascii", errors="replace").strip()
            if line == "end_header":
                break
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "format" and len(parts) >= 2:
                result["format"] = parts[1]
            elif parts[0] == "element" and len(parts) >= 3:
                current_element = parts[1]
                if current_element == "vertex":
                    result["vertex_count"] = int(parts[2])
            elif parts[0] == "property" and current_element == "vertex":
                result["properties"].append(" ".join(parts[1:]))
    return result


def load_test_reference_map(data_root: Path) -> dict[tuple[str, str], Path]:
    split_path = data_root / "test_split.txt"
    reference_dir = data_root / "reference_data"
    references: dict[tuple[str, str], Path] = {}
    if not split_path.exists():
        return references

    with split_path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            road_id = row["road_id"].strip()
            ply_id = row["ply_id"].strip()
            references[(road_id, ply_id)] = reference_dir / f"{road_id}_{ply_id}.npy"
    return references


def read_trajectory_image_names(path: Path) -> list[str]:
    names: list[str] = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.reader(stream):
            if row and row[0].strip():
                names.append(row[0].strip())
    return names


def build_trajectory_manifest(data_root: Path) -> list[dict[str, Any]]:
    references = load_test_reference_map(data_root)
    rows: list[dict[str, Any]] = []

    road_dirs = sorted(
        path for path in data_root.iterdir() if path.is_dir() and path.name.isdigit()
    )
    for road_dir in road_dirs:
        pcd_dir = road_dir / "PCD"
        for ply_path in sorted(pcd_dir.glob("*.ply"), key=lambda path: path.stem):
            trajectory_id = ply_path.stem
            traj_path = road_dir / "hdi" / trajectory_id / "traj.csv"
            image_dir = road_dir / "image" / trajectory_id
            reference_path = references.get((road_dir.name, trajectory_id))

            header = read_ply_header(ply_path)
            image_paths = sorted(image_dir.glob("*.jpg")) if image_dir.exists() else []
            image_names = {path.name.casefold() for path in image_paths}
            trajectory_names = (
                read_trajectory_image_names(traj_path) if traj_path.exists() else []
            )
            referenced_names = {name.casefold() for name in trajectory_names}
            missing_images = referenced_names - image_names
            extra_images = image_names - referenced_names

            status_parts: list[str] = []
            if not traj_path.exists():
                status_parts.append("missing_traj")
            if not image_dir.exists():
                status_parts.append("missing_image_dir")
            if missing_images:
                status_parts.append("missing_images")
            if extra_images:
                status_parts.append("extra_images")
            if reference_path is not None and not reference_path.exists():
                status_parts.append("missing_reference")

            rows.append(
                {
                    "road_id": road_dir.name,
                    "trajectory_id": trajectory_id,
                    "ply_path": str(ply_path),
                    "ply_size_bytes": ply_path.stat().st_size,
                    "ply_format": header["format"],
                    "ply_vertex_count": header["vertex_count"],
                    "ply_properties": "|".join(header["properties"]),
                    "traj_csv_path": str(traj_path),
                    "traj_record_count": len(trajectory_names),
                    "image_dir": str(image_dir),
                    "image_count": len(image_paths),
                    "image_size_bytes": sum(path.stat().st_size for path in image_paths),
                    "missing_image_count": len(missing_images),
                    "extra_image_count": len(extra_images),
                    "is_test_reference": reference_path is not None,
                    "reference_path": str(reference_path) if reference_path else "",
                    "reference_size_bytes": (
                        reference_path.stat().st_size
                        if reference_path is not None and reference_path.exists()
                        else 0
                    ),
                    "structural_status": ";".join(status_parts) or "ok",
                }
            )
    return rows


def load_hk_canonical_species(species_root: Path, merge_map_path: Path) -> set[str]:
    merge_map: dict[str, str] = {}
    with merge_map_path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            merge_map[row["original_species"].strip()] = row["label_species"].strip()

    canonical_species: set[str] = set()
    for path in species_root.iterdir():
        if path.is_dir():
            canonical_species.add(merge_map.get(path.name, path.name))
    return canonical_species


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_species_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label_id, name, abbreviation, instances, percentage in WHU_BENCHMARK_SPECIES:
        rows.append(
            {
                "benchmark_label_id": label_id,
                "scientific_name": name,
                "abbreviation": abbreviation,
                "published_instance_count": instances,
                "published_percentage": percentage,
                "mapping_scope": (
                    "benchmark aggregate for raw labels greater than 17"
                    if label_id == 18
                    else "published main class"
                ),
            }
        )
    return rows


def write_summary(
    path: Path,
    manifest: list[dict[str, Any]],
    shared_species: list[dict[str, Any]],
) -> None:
    status_counts: dict[str, int] = {}
    for row in manifest:
        status = str(row["structural_status"])
        status_counts[status] = status_counts.get(status, 0) + 1

    summary = {
        "scope": "lightweight metadata index; point payloads were not scanned",
        "road_count": len({row["road_id"] for row in manifest}),
        "trajectory_count": len(manifest),
        "ply_file_count": len(manifest),
        "ply_total_bytes": sum(int(row["ply_size_bytes"]) for row in manifest),
        "ply_vertex_count_from_headers": sum(
            int(row["ply_vertex_count"]) for row in manifest
        ),
        "image_count": sum(int(row["image_count"]) for row in manifest),
        "image_total_bytes": sum(int(row["image_size_bytes"]) for row in manifest),
        "test_reference_count": sum(bool(row["is_test_reference"]) for row in manifest),
        "reference_total_bytes": sum(
            int(row["reference_size_bytes"]) for row in manifest
        ),
        "structural_status_counts": status_counts,
        "confirmed_hk_shared_species_count": len(shared_species),
        "confirmed_hk_shared_species": [
            row["scientific_name"] for row in shared_species
        ],
        "label_note": (
            "WHU publishes names for labels 0-17. Raw labels greater than 17 are "
            "grouped as Others in the benchmark; their complete scientific-name "
            "mapping is not present in the public repository."
        ),
    }
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    output_dir = (args.output_dir or data_root / "metadata").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_trajectory_manifest(data_root)
    species_rows = build_species_rows()
    hk_species = load_hk_canonical_species(
        args.hk_species_root.resolve(), args.merge_map.resolve()
    )
    shared_species = [
        row
        for row in species_rows
        if row["benchmark_label_id"] != 18
        and row["scientific_name"] in hk_species
    ]

    write_csv(output_dir / "trajectory_manifest.csv", manifest)
    write_csv(output_dir / "whu_benchmark_species.csv", species_rows)
    write_csv(output_dir / "whu_hk_shared_species.csv", shared_species)
    write_summary(output_dir / "dataset_summary.json", manifest, shared_species)

    print(f"Indexed {len(manifest)} trajectories")
    print(f"Output: {output_dir}")
    print(f"Structural issues: {sum(row['structural_status'] != 'ok' for row in manifest)}")
    print(f"Confirmed HK shared species: {len(shared_species)}")


if __name__ == "__main__":
    main()
