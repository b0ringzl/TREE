#!/usr/bin/env python3
"""Aggregate dated batch species reports with the project label merge map."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_label_map(path: Path | None) -> dict[str, str]:
    if not path or not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return {
            row["original_species"].strip(): row["label_species"].strip()
            for row in reader
            if row.get("original_species") and row.get("label_species")
        }


def int_field(row: dict[str, str], field: str) -> int:
    try:
        return int(row.get(field) or 0)
    except ValueError:
        return 0


def reason(row: dict[str, object]) -> str:
    if row["errors"]:
        return "api_error"
    if int(row["today_images"]) >= int(row["target_cap"]):
        return "complete_or_capped"
    if int(row["skipped_taxon_mismatch"]) > 0:
        return "strict_taxon_filter_limited_results"
    if int(row["skipped_license"]) > 0:
        return "license_filter_limited_results"
    if int(row["skipped_duplicate"]) > 0:
        return "mostly_duplicate_with_existing_dataset"
    return "insufficient_verified_results"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build canonical per-species dated batch reports.")
    parser.add_argument("--batch-root", type=Path, default=Path("external_datasets/inaturalist_dated/2026-06-07"))
    parser.add_argument("--label-map", type=Path, default=Path("tools/tree_species_label_merge_map.csv"))
    parser.add_argument("--target-cap", type=int, default=150)
    args = parser.parse_args()

    label_map = read_label_map(args.label_map)
    summary_path = args.batch_root / "summary.csv"
    with summary_path.open("r", encoding="utf-8-sig", newline="") as handle:
        summary_rows = list(csv.DictReader(handle))

    grouped: dict[str, dict[str, object]] = {}
    for row in summary_rows:
        original = row["species"]
        label = label_map.get(original, original)
        item = grouped.setdefault(
            label,
            {
                "species": label,
                "original_species": set(),
                "query_name": set(),
                "today_images": 0,
                "target_cap": args.target_cap,
                "deficit_to_150": 0,
                "new_downloads": 0,
                "existing_in_batch": 0,
                "attempted_downloads": 0,
                "skipped_duplicate": 0,
                "skipped_license": 0,
                "skipped_taxon_mismatch": 0,
                "failed_downloads": 0,
                "errors": [],
                "output_dirs": [],
            },
        )
        item["original_species"].add(original)
        if row.get("query_name"):
            item["query_name"].add(row["query_name"])
        today = int_field(row, "new_downloads") + int_field(row, "existing_in_batch")
        item["today_images"] += today
        for field in (
            "new_downloads",
            "existing_in_batch",
            "attempted_downloads",
            "skipped_duplicate",
            "skipped_license",
            "skipped_taxon_mismatch",
            "failed_downloads",
        ):
            item[field] += int_field(row, field)
        if row.get("error"):
            item["errors"].append(f"{original}: {row['error']}")
        if row.get("output_dir"):
            item["output_dirs"].append(row["output_dir"])

    output_rows = []
    for item in grouped.values():
        today_images = int(item["today_images"])
        item["deficit_to_150"] = max(0, args.target_cap - today_images)
        output_rows.append(
            {
                "species": item["species"],
                "original_species": "|".join(sorted(item["original_species"], key=str.casefold)),
                "query_name": "|".join(sorted(item["query_name"], key=str.casefold)),
                "today_images": today_images,
                "target_cap": item["target_cap"],
                "deficit_to_150": item["deficit_to_150"],
                "new_downloads": item["new_downloads"],
                "existing_in_batch": item["existing_in_batch"],
                "attempted_downloads": item["attempted_downloads"],
                "skipped_duplicate": item["skipped_duplicate"],
                "skipped_license": item["skipped_license"],
                "skipped_taxon_mismatch": item["skipped_taxon_mismatch"],
                "failed_downloads": item["failed_downloads"],
                "status_reason": reason(item),
                "error": " | ".join(item["errors"]),
                "output_dir": "|".join(item["output_dirs"]),
            }
        )

    output_rows.sort(key=lambda row: row["species"].casefold())
    fields = list(output_rows[0].keys()) if output_rows else []
    counts_path = args.batch_root / "species_counts_canonical.csv"
    deficit_path = args.batch_root / "deficit_species_canonical.csv"
    with counts_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)
    with deficit_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([row for row in output_rows if int(row["today_images"]) < args.target_cap or row["error"]])
    print(f"canonical_species={len(output_rows)}")
    print(f"counts={counts_path}")
    print(f"deficits={deficit_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
