#!/usr/bin/env python3
"""Build master CSV catalogs for dated iNaturalist image batches."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any


def slug(value: str, max_len: int = 32) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").upper()
    return (text or "SPECIES")[:max_len]


def iter_metadata_rows(root: Path) -> list[tuple[Path, dict[str, Any]]]:
    rows: list[tuple[Path, dict[str, Any]]] = []
    if not root.exists():
        return rows
    for metadata_path in sorted(root.rglob("metadata.jsonl")):
        for line in metadata_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rows.append((metadata_path, json.loads(line)))
            except json.JSONDecodeError:
                continue
    return rows


def read_label_map(path: Path | None) -> dict[str, str]:
    if not path or not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return {
            str(row["original_species"]).strip(): str(row["label_species"]).strip()
            for row in reader
            if row.get("original_species") and row.get("label_species")
        }


def row_to_catalog(
    metadata_path: Path,
    row: dict[str, Any],
    root: Path,
    batch_date: str,
    sequence: int,
    label_map: dict[str, str],
) -> dict[str, Any]:
    local_path = str(row.get("local_path") or "")
    absolute_path = root / local_path if local_path else Path("")
    original_species = str(row.get("original_species") or row.get("target_species") or metadata_path.parent.name)
    label_species = label_map.get(original_species, original_species)
    image_id = str(row.get("image_id") or "")
    if not image_id:
        image_id = f"INAT-{batch_date.replace('-', '')}-{slug(original_species)}-{sequence:05d}"
    source_page_url = row.get("source_page_url") or row.get("inat_uri") or ""
    return {
        "image_id": image_id,
        "batch_date": row.get("batch_date", batch_date),
        "batch_root": str(root.resolve()),
        "source": row.get("source", "iNaturalist"),
        "target_species": label_species,
        "original_species": original_species,
        "label_species": label_species,
        "query_name": row.get("query_name", original_species),
        "inat_taxon_name": row.get("inat_taxon_name", ""),
        "taxon_exact_match": row.get("taxon_exact_match", ""),
        "photo_id": row.get("photo_id", ""),
        "inat_observation_id": row.get("inat_observation_id", ""),
        "inat_observed_on": row.get("inat_observed_on", ""),
        "inat_quality_grade": row.get("inat_quality_grade", ""),
        "photo_license_code": row.get("photo_license_code", ""),
        "photo_attribution": row.get("photo_attribution", ""),
        "source_page_url": source_page_url,
        "photo_url": row.get("photo_url", ""),
        "local_path": local_path,
        "absolute_path": str(absolute_path.resolve()) if local_path else "",
        "file_sha256": row.get("file_sha256", ""),
        "file_size_bytes": row.get("file_size_bytes", ""),
        "place_guess": row.get("place_guess", ""),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["batch_date"]), str(row["target_species"]))
        item = grouped.setdefault(
            key,
            {
                "batch_date": row["batch_date"],
                "target_species": row["target_species"],
                "image_count": 0,
                "license_codes": set(),
                "original_species": set(),
            },
        )
        item["image_count"] += 1
        if row.get("original_species"):
            item["original_species"].add(str(row["original_species"]))
        if row.get("photo_license_code"):
            item["license_codes"].add(str(row["photo_license_code"]))
    summary = []
    for item in grouped.values():
        summary.append(
            {
                "batch_date": item["batch_date"],
                "target_species": item["target_species"],
                "image_count": item["image_count"],
                "original_species": "|".join(sorted(item["original_species"], key=str.casefold)),
                "license_codes": ",".join(sorted(item["license_codes"])),
            }
        )
    return sorted(summary, key=lambda item: (str(item["batch_date"]), str(item["target_species"]).casefold()))


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Create master catalogs for iNaturalist datasets.")
    parser.add_argument("--legacy-root", type=Path, default=Path("external_datasets/inat_max150"))
    parser.add_argument("--legacy-date", default="2026-05-24")
    parser.add_argument("--dated-root", type=Path, default=Path("external_datasets/inaturalist_dated"))
    parser.add_argument("--output", type=Path, default=Path("external_datasets/inaturalist_master_catalog.csv"))
    parser.add_argument("--summary-output", type=Path, default=Path("external_datasets/inaturalist_master_summary.csv"))
    parser.add_argument("--label-map", type=Path, default=Path("tools/tree_species_label_merge_map.csv"))
    args = parser.parse_args(argv)

    label_map = read_label_map(args.label_map)
    rows: list[dict[str, Any]] = []
    sequence = 1
    for metadata_path, row in iter_metadata_rows(args.legacy_root):
        rows.append(row_to_catalog(metadata_path, row, args.legacy_root, args.legacy_date, sequence, label_map))
        sequence += 1

    if args.dated_root.exists():
        for batch_dir in sorted([item for item in args.dated_root.iterdir() if item.is_dir()], key=lambda item: item.name):
            batch_date = batch_dir.name
            for metadata_path, row in iter_metadata_rows(batch_dir):
                rows.append(row_to_catalog(metadata_path, row, batch_dir, batch_date, sequence, label_map))
                sequence += 1

    fields = [
        "image_id",
        "batch_date",
        "batch_root",
        "source",
        "target_species",
        "original_species",
        "label_species",
        "query_name",
        "inat_taxon_name",
        "taxon_exact_match",
        "photo_id",
        "inat_observation_id",
        "inat_observed_on",
        "inat_quality_grade",
        "photo_license_code",
        "photo_attribution",
        "source_page_url",
        "photo_url",
        "local_path",
        "absolute_path",
        "file_sha256",
        "file_size_bytes",
        "place_guess",
    ]
    write_csv(args.output, rows, fields)
    write_csv(args.summary_output, summarize(rows), ["batch_date", "target_species", "image_count", "original_species", "license_codes"])
    print(f"catalog_rows={len(rows)} output={args.output}", flush=True)
    print(f"summary_output={args.summary_output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
