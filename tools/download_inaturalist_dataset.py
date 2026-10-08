#!/usr/bin/env python3
"""Download iNaturalist photos for the tree species in this workspace.

The downloader writes to a separate output directory and keeps one metadata
record per downloaded photo so the dataset can be reviewed before use.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


INAT_OBSERVATIONS_URL = "https://api.inaturalist.org/v1/observations"
DEFAULT_LICENSES = ("cc0", "cc-by", "cc-by-sa", "cc-by-nc", "cc-by-nc-sa")
INVALID_PATH_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
TRANSIENT_HTTP_STATUS = {408, 425, 429, 500, 502, 503, 504}


def safe_dir_name(name: str) -> str:
    cleaned = INVALID_PATH_CHARS.sub("_", name).strip().rstrip(".")
    return cleaned or "unknown_species"


def read_species(species_root: Path) -> list[str]:
    if not species_root.exists():
        raise FileNotFoundError(f"species root not found: {species_root}")
    species = [p.name for p in species_root.iterdir() if p.is_dir()]
    return sorted(species, key=str.casefold)


def read_species_map(path: Path) -> list[tuple[str, str]]:
    """Read target species and query species pairs from a CSV file."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"target_species", "query_name"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"species map must contain columns: {', '.join(sorted(required))}")
        rows = []
        for row in reader:
            target = (row.get("target_species") or "").strip()
            query = (row.get("query_name") or "").strip()
            if target and query:
                rows.append((target, query))
    return rows


def is_transient_network_error(exc: Exception) -> bool:
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in TRANSIENT_HTTP_STATUS
    return isinstance(
        exc,
        (
            ConnectionError,
            TimeoutError,
            OSError,
            urllib.error.URLError,
        ),
    )


def wait_for_network(
    context: str,
    exc: Exception,
    started_at: float,
    wait_seconds: int,
    max_wait_minutes: int,
) -> bool:
    elapsed_minutes = (time.monotonic() - started_at) / 60
    if max_wait_minutes > 0 and elapsed_minutes >= max_wait_minutes:
        print(
            f"NETWORK GIVEUP {context}: waited {elapsed_minutes:.1f} minutes; last error: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return False
    print(
        f"NETWORK WAIT {context}: {exc}; retrying in {wait_seconds}s",
        file=sys.stderr,
        flush=True,
    )
    time.sleep(wait_seconds)
    return True


def request_json(
    url: str,
    params: dict[str, Any],
    timeout: int,
    retries: int,
    network_wait_seconds: int,
    max_network_wait_minutes: int,
) -> dict[str, Any]:
    query = urllib.parse.urlencode(params, doseq=True)
    req = urllib.request.Request(
        f"{url}?{query}",
        headers={
            "Accept": "application/json",
            "User-Agent": "TREE-dataset-builder/1.0 (local research dataset preparation)",
        },
    )
    attempt = 0
    network_wait_started_at = time.monotonic()
    while True:
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            if not is_transient_network_error(exc):
                raise RuntimeError(f"request failed with non-retryable error: {exc}") from exc
            if attempt < retries:
                time.sleep(min(30, 2 * (attempt + 1)))
                attempt += 1
                continue
            context = urllib.parse.urlparse(url).netloc
            if not wait_for_network(
                context,
                exc,
                network_wait_started_at,
                network_wait_seconds,
                max_network_wait_minutes,
            ):
                raise RuntimeError(f"request failed after network wait: {exc}") from exc
            attempt = 0


def download_file(
    url: str,
    path: Path,
    timeout: int,
    retries: int,
    network_wait_seconds: int,
    max_network_wait_minutes: int,
) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "TREE-dataset-builder/1.0 (local research dataset preparation)"},
    )
    attempt = 0
    network_wait_started_at = time.monotonic()
    while True:
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                data = response.read()
            if len(data) < 1024:
                raise RuntimeError(f"download too small ({len(data)} bytes)")
            tmp_path.write_bytes(data)
            tmp_path.replace(path)
            return True
        except Exception as exc:
            if not is_transient_network_error(exc):
                if tmp_path.exists():
                    tmp_path.unlink()
                return False
            if attempt < retries:
                time.sleep(min(30, 2 * (attempt + 1)))
                attempt += 1
                continue
            context = urllib.parse.urlparse(url).netloc or "image download"
            if not wait_for_network(
                context,
                exc,
                network_wait_started_at,
                network_wait_seconds,
                max_network_wait_minutes,
            ):
                if tmp_path.exists():
                    tmp_path.unlink()
                return False
            attempt = 0


def image_url(photo: dict[str, Any], size: str) -> str | None:
    url = photo.get("url")
    if not isinstance(url, str):
        return None
    return re.sub(r"/(square|small|medium|large|original)\.", f"/{size}.", url)


def photo_extension(url: str) -> str:
    suffix = Path(urllib.parse.urlparse(url).path).suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".webp"}:
        return ".jpg" if suffix == ".jpeg" else suffix
    return ".jpg"


def collect_for_species(
    species: str,
    query_name: str,
    output_root: Path,
    max_count: int,
    licenses: set[str],
    quality_grade: str,
    image_size: str,
    request_delay: float,
    timeout: int,
    retries: int,
    network_wait_seconds: int,
    max_network_wait_minutes: int,
    dry_run: bool,
) -> dict[str, Any]:
    species_dir = output_root / safe_dir_name(species)
    images_dir = species_dir / "images"
    metadata_path = species_dir / "metadata.jsonl"
    species_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    existing_ids: set[str] = set()
    if metadata_path.exists():
        for line in metadata_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            photo_id = str(row.get("photo_id", ""))
            image_path = row.get("local_path")
            if photo_id and image_path and (output_root / image_path).exists():
                existing_ids.add(photo_id)

    downloaded = len(existing_ids)
    attempted = 0
    skipped_license = 0
    failed = 0
    page = 1
    per_page = 200
    seen_photos = set(existing_ids)

    with metadata_path.open("a", encoding="utf-8") as metadata_file:
        while downloaded < max_count:
            params = {
                "taxon_name": query_name,
                "photos": "true",
                "quality_grade": quality_grade,
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
                photos = obs.get("photos") or []
                for photo in photos:
                    if downloaded >= max_count:
                        break
                    photo_id = str(photo.get("id") or "")
                    if not photo_id or photo_id in seen_photos:
                        continue
                    seen_photos.add(photo_id)

                    license_code = str(photo.get("license_code") or "").lower()
                    if license_code not in licenses:
                        skipped_license += 1
                        continue

                    url = image_url(photo, image_size)
                    if not url:
                        failed += 1
                        continue

                    attempted += 1
                    filename = f"{downloaded + 1:03d}_inat_photo_{photo_id}{photo_extension(url)}"
                    image_path = images_dir / filename
                    relative_path = image_path.relative_to(output_root).as_posix()
                    record = {
                        "target_species": species,
                        "query_name": query_name,
                        "inat_observation_id": obs.get("id"),
                        "inat_observed_on": obs.get("observed_on"),
                        "inat_quality_grade": obs.get("quality_grade"),
                        "inat_uri": obs.get("uri"),
                        "inat_taxon_name": (obs.get("taxon") or {}).get("name"),
                        "inat_taxon_preferred_common_name": (obs.get("taxon") or {}).get("preferred_common_name"),
                        "photo_id": photo_id,
                        "photo_license_code": license_code,
                        "photo_attribution": photo.get("attribution"),
                        "photo_url": url,
                        "local_path": relative_path,
                        "place_guess": obs.get("place_guess"),
                        "geojson": obs.get("geojson"),
                    }

                    if dry_run:
                        downloaded += 1
                        print(json.dumps(record, ensure_ascii=False))
                    elif download_file(
                        url,
                        image_path,
                        timeout,
                        retries,
                        network_wait_seconds,
                        max_network_wait_minutes,
                    ):
                        metadata_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                        metadata_file.flush()
                        downloaded += 1
                    else:
                        failed += 1

                if downloaded >= max_count:
                    break

            total_results = int(data.get("total_results") or 0)
            if page * per_page >= total_results:
                break
            page += 1
            time.sleep(request_delay)

    return {
        "species": species,
        "query_name": query_name,
        "downloaded_or_existing": downloaded,
        "attempted_downloads": attempted,
        "skipped_license": skipped_license,
        "failed_downloads": failed,
        "output_dir": str(species_dir),
    }


def write_summary(output_root: Path, rows: list[dict[str, Any]]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "summary.csv"
    fields = [
        "species",
        "query_name",
        "downloaded_or_existing",
        "attempted_downloads",
        "skipped_license",
        "failed_downloads",
        "output_dir",
        "error",
    ]
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Download iNaturalist tree photos by species.")
    parser.add_argument("--species-root", type=Path, default=Path("选取树种"))
    parser.add_argument(
        "--species-map",
        type=Path,
        help="CSV with target_species and query_name columns. Useful for querying parent species names.",
    )
    parser.add_argument("--output", type=Path, default=Path("external_datasets/inat_max150"))
    parser.add_argument("--max-per-species", type=int, default=150)
    parser.add_argument("--species-limit", type=int, default=0, help="Only process the first N species.")
    parser.add_argument("--only-species", action="append", default=[], help="Process one species; may be repeated.")
    parser.add_argument("--licenses", default=",".join(DEFAULT_LICENSES))
    parser.add_argument("--quality-grade", default="research", choices=["research", "needs_id", "casual"])
    parser.add_argument("--image-size", default="medium", choices=["small", "medium", "large", "original"])
    parser.add_argument("--request-delay", type=float, default=0.5)
    parser.add_argument("--timeout", type=int, default=45)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument(
        "--network-wait-seconds",
        type=int,
        default=60,
        help="When network-like errors persist after retries, wait this many seconds before retrying again.",
    )
    parser.add_argument(
        "--max-network-wait-minutes",
        type=int,
        default=0,
        help="Maximum minutes to wait for network recovery per request; 0 waits indefinitely.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.species_map:
        species_pairs = read_species_map(args.species_map)
    elif args.only_species:
        species_pairs = [(item, item) for item in args.only_species]
    else:
        species_pairs = [(item, item) for item in read_species(args.species_root)]
    if args.species_limit:
        species_pairs = species_pairs[: args.species_limit]
    licenses = {item.strip().lower() for item in args.licenses.split(",") if item.strip()}

    output_root = args.output
    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for index, (item, query_name) in enumerate(species_pairs, 1):
        query_note = f" -> {query_name}" if query_name != item else ""
        print(f"[{index}/{len(species_pairs)}] {item}{query_note}", flush=True)
        try:
            row = collect_for_species(
                species=item,
                query_name=query_name,
                output_root=output_root,
                max_count=args.max_per_species,
                licenses=licenses,
                quality_grade=args.quality_grade,
                image_size=args.image_size,
                request_delay=args.request_delay,
                timeout=args.timeout,
                retries=args.retries,
                network_wait_seconds=args.network_wait_seconds,
                max_network_wait_minutes=args.max_network_wait_minutes,
                dry_run=args.dry_run,
            )
        except Exception as exc:
            row = {
                "species": item,
                "query_name": query_name,
                "downloaded_or_existing": 0,
                "attempted_downloads": 0,
                "skipped_license": 0,
                "failed_downloads": 0,
                "output_dir": str(output_root / safe_dir_name(item)),
                "error": str(exc),
            }
            print(f"ERROR {item}: {exc}", file=sys.stderr, flush=True)
        rows.append(row)
        print(
            f"  saved/existing={row.get('downloaded_or_existing')} "
            f"failed={row.get('failed_downloads')} skipped_license={row.get('skipped_license')}",
            flush=True,
        )
        write_summary(output_root, rows)

    write_summary(output_root, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
