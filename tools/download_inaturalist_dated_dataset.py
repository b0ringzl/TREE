#!/usr/bin/env python3
"""Download a dated iNaturalist image batch with per-image catalog records."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any

from download_inaturalist_dataset import (
    DEFAULT_LICENSES,
    INAT_OBSERVATIONS_URL,
    download_file,
    image_url,
    photo_extension,
    read_species,
    read_species_map,
    request_json,
    safe_dir_name,
)


PERMISSIVE_LICENSES = ("cc0", "cc-by", "cc-by-sa")


def slug(value: str, max_len: int = 32) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").upper()
    return (text or "SPECIES")[:max_len]


def load_query_overrides(path: Path | None) -> dict[str, str]:
    if not path:
        return {}
    return {target: query for target, query in read_species_map(path)}


def build_species_pairs(species_root: Path, query_map: Path | None, only_species: list[str]) -> list[tuple[str, str]]:
    overrides = load_query_overrides(query_map)
    if only_species:
        species = only_species
    else:
        species = read_species(species_root)
    return [(item, overrides.get(item, item)) for item in species]


def iter_metadata_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not root.exists():
        return rows
    for metadata_path in sorted(root.rglob("metadata.jsonl")):
        for line in metadata_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows.append(row)
    return rows


def load_excluded_photo_ids(roots: list[Path]) -> set[str]:
    excluded: set[str] = set()
    for root in roots:
        for row in iter_metadata_rows(root):
            photo_id = str(row.get("photo_id") or "").strip()
            if photo_id:
                excluded.add(photo_id)
    return excluded


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def existing_batch_state(output_root: Path, batch_date: str) -> tuple[set[str], dict[str, int], list[dict[str, Any]]]:
    existing_ids: set[str] = set()
    species_counts: dict[str, int] = {}
    catalog_rows: list[dict[str, Any]] = []
    for row in iter_metadata_rows(output_root):
        photo_id = str(row.get("photo_id") or "").strip()
        if photo_id:
            existing_ids.add(photo_id)
        species = str(row.get("target_species") or "").strip()
        if species:
            species_counts[species] = species_counts.get(species, 0) + 1
        if row.get("image_id"):
            catalog_rows.append(catalog_row_from_metadata(row, output_root, batch_date))
    return existing_ids, species_counts, catalog_rows


def catalog_row_from_metadata(row: dict[str, Any], root: Path, fallback_batch_date: str) -> dict[str, Any]:
    local_path = str(row.get("local_path") or "")
    absolute_path = root / local_path if local_path else Path("")
    return {
        "image_id": row.get("image_id", ""),
        "batch_date": row.get("batch_date", fallback_batch_date),
        "source": row.get("source", "iNaturalist"),
        "target_species": row.get("target_species", ""),
        "query_name": row.get("query_name", ""),
        "inat_taxon_name": row.get("inat_taxon_name", ""),
        "taxon_exact_match": row.get("taxon_exact_match", ""),
        "photo_id": row.get("photo_id", ""),
        "inat_observation_id": row.get("inat_observation_id", ""),
        "inat_observed_on": row.get("inat_observed_on", ""),
        "inat_quality_grade": row.get("inat_quality_grade", ""),
        "photo_license_code": row.get("photo_license_code", ""),
        "photo_attribution": row.get("photo_attribution", ""),
        "source_page_url": row.get("source_page_url") or row.get("inat_uri", ""),
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


def taxon_matches(obs: dict[str, Any], query_name: str, allow_descendant_taxa: bool) -> tuple[bool, str]:
    taxon_name = str((obs.get("taxon") or {}).get("name") or "").strip()
    if allow_descendant_taxa:
        return True, taxon_name
    return taxon_name.casefold() == query_name.casefold(), taxon_name


def collect_for_species(
    species: str,
    query_name: str,
    output_root: Path,
    batch_date: str,
    max_new: int,
    licenses: set[str],
    excluded_photo_ids: set[str],
    existing_batch_ids: set[str],
    existing_species_count: int,
    allow_descendant_taxa: bool,
    image_size: str,
    request_delay: float,
    timeout: int,
    retries: int,
    network_wait_seconds: int,
    max_network_wait_minutes: int,
    dry_run: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    species_dir = output_root / safe_dir_name(species)
    images_dir = species_dir / "images"
    metadata_path = species_dir / "metadata.jsonl"
    species_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    new_downloads = 0
    attempted = 0
    skipped_license = 0
    skipped_duplicate = 0
    skipped_taxon = 0
    failed = 0
    page = 1
    per_page = 200
    seen_this_run: set[str] = set()
    rows: list[dict[str, Any]] = []
    next_index = existing_species_count + 1

    with metadata_path.open("a", encoding="utf-8") as metadata_file:
        while new_downloads < max_new:
            params = {
                "taxon_name": query_name,
                "photos": "true",
                "quality_grade": "research",
                "per_page": per_page,
                "page": page,
                "order_by": "observed_on",
                "order": "desc",
                "locale": "en",
            }
            data = request_json(
                INAT_OBSERVATIONS_URL,
                params,
                timeout,
                retries,
                network_wait_seconds,
                max_network_wait_minutes,
            )
            results = data.get("results") or []
            if not results:
                break

            for obs in results:
                matches, taxon_name = taxon_matches(obs, query_name, allow_descendant_taxa)
                if not matches:
                    skipped_taxon += len(obs.get("photos") or [])
                    continue

                for photo in obs.get("photos") or []:
                    if new_downloads >= max_new:
                        break
                    photo_id = str(photo.get("id") or "").strip()
                    if not photo_id:
                        continue
                    if photo_id in seen_this_run or photo_id in existing_batch_ids or photo_id in excluded_photo_ids:
                        skipped_duplicate += 1
                        continue
                    seen_this_run.add(photo_id)

                    license_code = str(photo.get("license_code") or "").lower()
                    if license_code not in licenses:
                        skipped_license += 1
                        continue

                    url = image_url(photo, image_size)
                    if not url:
                        failed += 1
                        continue

                    image_id = f"INAT-{batch_date.replace('-', '')}-{slug(species)}-{next_index:04d}"
                    filename = f"{image_id}_photo_{photo_id}{photo_extension(url)}"
                    image_path = images_dir / filename
                    relative_path = image_path.relative_to(output_root).as_posix()
                    source_page_url = str(obs.get("uri") or f"https://www.inaturalist.org/observations/{obs.get('id')}")
                    record = {
                        "image_id": image_id,
                        "batch_date": batch_date,
                        "source": "iNaturalist",
                        "source_api_url": INAT_OBSERVATIONS_URL,
                        "source_page_url": source_page_url,
                        "target_species": species,
                        "query_name": query_name,
                        "inat_observation_id": obs.get("id"),
                        "inat_observed_on": obs.get("observed_on"),
                        "inat_quality_grade": obs.get("quality_grade"),
                        "inat_uri": source_page_url,
                        "inat_taxon_name": taxon_name,
                        "taxon_exact_match": taxon_name.casefold() == query_name.casefold(),
                        "inat_taxon_preferred_common_name": (obs.get("taxon") or {}).get("preferred_common_name"),
                        "photo_id": photo_id,
                        "photo_license_code": license_code,
                        "photo_attribution": photo.get("attribution"),
                        "photo_url": url,
                        "local_path": relative_path,
                        "place_guess": obs.get("place_guess"),
                        "geojson": obs.get("geojson"),
                    }

                    attempted += 1
                    if dry_run:
                        print(json.dumps(record, ensure_ascii=False), flush=True)
                        new_downloads += 1
                        next_index += 1
                        rows.append(catalog_row_from_metadata(record, output_root, batch_date))
                        continue

                    if download_file(
                        url,
                        image_path,
                        timeout,
                        retries,
                        network_wait_seconds,
                        max_network_wait_minutes,
                    ):
                        record["file_sha256"] = sha256_file(image_path)
                        record["file_size_bytes"] = image_path.stat().st_size
                        metadata_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                        metadata_file.flush()
                        rows.append(catalog_row_from_metadata(record, output_root, batch_date))
                        new_downloads += 1
                        next_index += 1
                    else:
                        failed += 1

                if new_downloads >= max_new:
                    break

            total_results = int(data.get("total_results") or 0)
            if page * per_page >= total_results:
                break
            page += 1
            time.sleep(request_delay)

    summary = {
        "species": species,
        "query_name": query_name,
        "new_downloads": new_downloads,
        "existing_in_batch": existing_species_count,
        "attempted_downloads": attempted,
        "skipped_duplicate": skipped_duplicate,
        "skipped_license": skipped_license,
        "skipped_taxon_mismatch": skipped_taxon,
        "failed_downloads": failed,
        "output_dir": str(species_dir),
    }
    return summary, rows


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Download a dated, cataloged iNaturalist tree image batch.")
    parser.add_argument("--species-root", type=Path, default=Path("选取树种"))
    parser.add_argument("--query-map", type=Path, default=Path("tools/inaturalist_parent_name_supplement.csv"))
    parser.add_argument("--output-root", type=Path, default=Path("external_datasets/inaturalist_dated"))
    parser.add_argument("--batch-date", default=date.today().isoformat())
    parser.add_argument("--max-new-per-species", type=int, default=150)
    parser.add_argument("--species-limit", type=int, default=0)
    parser.add_argument("--only-species", action="append", default=[])
    parser.add_argument("--exclude-root", type=Path, action="append", default=[Path("external_datasets/inat_max150")])
    parser.add_argument("--licenses", default=",".join(PERMISSIVE_LICENSES))
    parser.add_argument("--allow-descendant-taxa", action="store_true")
    parser.add_argument("--image-size", default="large", choices=["small", "medium", "large", "original"])
    parser.add_argument("--request-delay", type=float, default=0.75)
    parser.add_argument("--timeout", type=int, default=45)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--network-wait-seconds", type=int, default=60)
    parser.add_argument("--max-network-wait-minutes", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    output_root = args.output_root / args.batch_date
    output_root.mkdir(parents=True, exist_ok=True)
    species_pairs = build_species_pairs(args.species_root, args.query_map, args.only_species)
    if args.species_limit:
        species_pairs = species_pairs[: args.species_limit]

    licenses = {item.strip().lower() for item in args.licenses.split(",") if item.strip()}
    excluded_photo_ids = load_excluded_photo_ids([path for path in args.exclude_root if path])
    existing_batch_ids, existing_species_counts, existing_catalog_rows = existing_batch_state(output_root, args.batch_date)

    summary_rows: list[dict[str, Any]] = []
    catalog_rows: list[dict[str, Any]] = list(existing_catalog_rows)
    catalog_fields = [
        "image_id",
        "batch_date",
        "source",
        "target_species",
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
    summary_fields = [
        "species",
        "query_name",
        "new_downloads",
        "existing_in_batch",
        "attempted_downloads",
        "skipped_duplicate",
        "skipped_license",
        "skipped_taxon_mismatch",
        "failed_downloads",
        "output_dir",
        "error",
    ]

    for index, (species, query_name) in enumerate(species_pairs, 1):
        print(f"[{index}/{len(species_pairs)}] {species} -> {query_name}", flush=True)
        try:
            existing_for_species = existing_species_counts.get(species, 0)
            remaining_for_species = max(0, args.max_new_per_species - existing_for_species)
            summary, rows = collect_for_species(
                species=species,
                query_name=query_name,
                output_root=output_root,
                batch_date=args.batch_date,
                max_new=remaining_for_species,
                licenses=licenses,
                excluded_photo_ids=excluded_photo_ids,
                existing_batch_ids=existing_batch_ids,
                existing_species_count=existing_for_species,
                allow_descendant_taxa=args.allow_descendant_taxa,
                image_size=args.image_size,
                request_delay=args.request_delay,
                timeout=args.timeout,
                retries=args.retries,
                network_wait_seconds=args.network_wait_seconds,
                max_network_wait_minutes=args.max_network_wait_minutes,
                dry_run=args.dry_run,
            )
            for row in rows:
                photo_id = str(row.get("photo_id") or "")
                if photo_id:
                    existing_batch_ids.add(photo_id)
                catalog_rows.append(row)
        except Exception as exc:
            summary = {
                "species": species,
                "query_name": query_name,
                "new_downloads": 0,
                "existing_in_batch": existing_species_counts.get(species, 0),
                "attempted_downloads": 0,
                "skipped_duplicate": 0,
                "skipped_license": 0,
                "skipped_taxon_mismatch": 0,
                "failed_downloads": 0,
                "output_dir": str(output_root / safe_dir_name(species)),
                "error": str(exc),
            }
            print(f"ERROR {species}: {exc}", file=sys.stderr, flush=True)
        summary_rows.append(summary)
        print(
            f"  new={summary.get('new_downloads')} duplicate={summary.get('skipped_duplicate')} "
            f"license_skip={summary.get('skipped_license')} taxon_skip={summary.get('skipped_taxon_mismatch')} "
            f"failed={summary.get('failed_downloads')}",
            flush=True,
        )
        write_csv(output_root / "image_catalog.csv", catalog_rows, catalog_fields)
        write_csv(output_root / "summary.csv", summary_rows, summary_fields)

    write_csv(output_root / "image_catalog.csv", catalog_rows, catalog_fields)
    write_csv(output_root / "summary.csv", summary_rows, summary_fields)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
