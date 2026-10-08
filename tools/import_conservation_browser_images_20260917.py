"""Offline import of browser-downloaded source-page images; never fit or approve.

Source records are provenance assertions, not botanical labels or permission grants.
Images are copied byte-for-byte into a NEW directory; historical manifests and
review databases are read only. No network requests or model calls are made.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qsl, unquote, urlencode, urlsplit, urlunsplit

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy.fft import dctn

ROOT = Path(__file__).resolve().parents[1]
SPECIES = {
    "Artocarpus hypargyreus": "白桂木",
    "Rhodoleia championii": "紅花荷",
    "Keteleeria fortunei": "油杉",
}
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}
SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__", ".pytest_cache"}
HASH_KEYS = ("sha256", "source_sha256", "image_sha256")
PATH_KEYS = ("source_image", "local_file", "image_path", "local_path", "path", "file", "image")
URL_KEYS = ("page_url", "image_url", "source_url", "source_page", "photo_url", "url")


def save(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def phash(image):
    values = dctn(np.asarray(image.convert("L").resize((32, 32)), dtype=float), norm="ortho")[:8, :8].flatten()
    return int("".join("1" if v else "0" for v in values > np.median(values[1:])), 2)


def walk_rows(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            if isinstance(child, (dict, list)):
                yield from walk_rows(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_rows(child)


def canonical_url(value):
    try:
        parsed = urlsplit(str(value))
        if parsed.scheme not in {"https", "http"} or not parsed.netloc:
            return None
        query = [(k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=True)
                 if not k.lower().startswith("utm_")]
        return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), unquote(parsed.path), urlencode(query), ""))
    except ValueError:
        return None


def identities(row):
    """Legacy unnamespaced numeric observation/photo fields in this project are iNat."""
    observations, photos, urls = set(), set(), set()
    for key in ("inat_observation_id", "observation_id"):
        if str(row.get(key, "")).isdigit():
            observations.add("inat:" + str(row[key]))
    for key in ("inat_photo_id", "photo_id"):
        if str(row.get(key, "")).isdigit():
            photos.add("inat:" + str(row[key]))
    for key in URL_KEYS + ("group_id", "source_group", "id", "item_id", "path", "source_image"):
        value = row.get(key)
        if not isinstance(value, str):
            continue
        canonical = canonical_url(value)
        if canonical:
            urls.add(canonical)
        for matched in re.findall(r"(?:/observations/|(?:inat_)?observation[_:])(\d+)", value):
            observations.add("inat:" + matched)
        for matched in re.findall(r"(?:inaturalist[^\s]*/photos/|inat_photo_|(?:conserve|sstv1)_inat_)(\d+)", value):
            photos.add("inat:" + matched)
    return observations, photos, urls


def normalized_species_path(path):
    flattened = re.sub(r"[^a-z]", "", str(path).lower())
    return any(re.sub(r"[^a-z]", "", name.lower()) in flattened for name in SPECIES)


class HistoryIndex:
    def __init__(self):
        self.hashes = defaultdict(set)
        self.visual = defaultdict(set)
        self.observations = defaultdict(set)
        self.photos = defaultdict(set)
        self.urls = defaultdict(set)
        self.paths = defaultdict(set)
        self.image_rows = {}
        self.path_cache = {}
        self.audit = []
        self.errors = []
        self.visual_scope = Counter()
        self.species_inventory = defaultdict(set)
        self.rejected_hashes = set()

    def add(self, row, origin, root):
        label = str(origin)
        decisions = [row.get(key) for key in ("status", "decision", "review_status", "human_review_status", "human_review")]
        decisions = [value.get("decision", value.get("status")) if isinstance(value, dict) else value for value in decisions]
        rejected = any(isinstance(value, str) and ("reject" in value.lower() or value.lower() in {"deleted", "delete", "exclude_visual_mismatch"}) for value in decisions)
        for key in HASH_KEYS:
            value = str(row.get(key, "")).lower()
            if re.fullmatch(r"[0-9a-f]{64}", value):
                self.hashes[value].add(label)
                if rejected:
                    self.rejected_hashes.add(value)
        value = str(row.get("phash", row.get("p_hash", "")))
        if re.fullmatch(r"[0-9a-fA-F]{16}", value):
            self.visual[int(value, 16)].add(label)
        observations, photos, urls = identities(row)
        for values, mapping in ((observations, self.observations), (photos, self.photos), (urls, self.urls)):
            for value in values:
                mapping[value].add(label)
        for key in PATH_KEYS:
            value = row.get(key)
            if not isinstance(value, str) or "://" in value:
                continue
            path = Path(value)
            if path.suffix.lower() not in EXTENSIONS:
                continue
            cache_key = str(path) if path.is_absolute() else (str(origin.parent), str(path))
            if cache_key not in self.path_cache:
                candidates = [path] if path.is_absolute() else [origin.parent / path, root / path]
                self.path_cache[cache_key] = next((p.resolve() for p in candidates if p.is_file()), None)
            path = self.path_cache[cache_key]
            if path:
                self.paths[str(path).casefold()].add(label)
                # Existing comprehensive pHash indices avoid re-decoding tens of thousands of images.
                known = self.image_rows.setdefault(path, {"hash": False, "phash": False, "target_species": False, "rejected": False, "sha256_values": []})
                known["hash"] |= any(re.fullmatch(r"[0-9a-fA-F]{64}", str(row.get(k, ""))) is not None for k in HASH_KEYS)
                known["phash"] |= bool(re.fullmatch(r"[0-9a-fA-F]{16}", str(row.get("phash", ""))))
                known["target_species"] |= normalized_species_path(path) or any(row.get(k) in SPECIES for k in
                    ("source_species", "scientific_name", "species", "label", "label_name", "class_name") if isinstance(row.get(k), str))
                known["rejected"] |= rejected
                for hash_key in HASH_KEYS:
                    h = str(row.get(hash_key, "")).lower()
                    if re.fullmatch(r"[0-9a-f]{64}", h) and h not in known["sha256_values"]:
                        known["sha256_values"].append(h)

    def build(self, root, excluded, checkpoint=None):
        raw_targets = []
        for folder, dirs, filenames in os.walk(root):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not (Path(folder) / d).is_symlink()
                       and not any((Path(folder) / d).resolve().is_relative_to(p) for p in excluded)]
            for filename in filenames:
                path = Path(folder) / filename
                if path in excluded:
                    continue
                lower = filename.lower()
                if path.suffix.lower() in EXTENSIONS and path.is_relative_to(root / "external_datasets") and normalized_species_path(path):
                    raw_targets.append(path.resolve())
                selected = (path.suffix.lower() in {".json", ".jsonl"} and
                            ("manifest" in lower or lower.endswith("rows.json") or "image_index" in lower
                             or "review" in lower or "decision" in lower))
                database = lower.endswith((".sqlite3", ".db")) and "review" in lower
                if not selected and not database:
                    continue
                try:
                    entry = {"path": str(path.resolve()), "sha256": digest(path), "bytes": path.stat().st_size}
                    rows = 0
                    if database:
                        # uri mode=ro plus query_only: do not create DBs, tables, decisions or checkpoints.
                        with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as conn:
                            conn.execute("PRAGMA query_only=ON")
                            conn.row_factory = sqlite3.Row
                            tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
                            for table in tables:
                                if table not in {"decisions", "history", "events"}:
                                    continue
                                for record in conn.execute('SELECT * FROM "' + table + '"'):
                                    self.add(dict(record), path, root)
                                    rows += 1
                    else:
                        with path.open(encoding="utf-8-sig") as stream:
                            records = (json.loads(line) for line in stream if line.strip()) if path.suffix == ".jsonl" else [json.load(stream)]
                            for record in records:
                                for row in walk_rows(record):
                                    self.add(row, path, root)
                                    rows += 1
                    entry["rows_examined"] = rows
                    self.audit.append(entry)
                    if len(self.audit) % 100 == 0:
                        print(f"History: {len(self.audit)} files indexed; {len(self.image_rows)} referenced images", flush=True)
                except (OSError, ValueError, sqlite3.Error) as exc:
                    self.errors.append({"path": str(path), "error": str(exc)})
        for path in raw_targets:
            self.paths[str(path).casefold()].add("historical_species_directory")
            self.image_rows.setdefault(path, {"hash": False, "phash": False, "target_species": True, "rejected": False, "sha256_values": []})["target_species"] = True
        for known in self.image_rows.values():
            known["rejected"] |= bool(set(known["sha256_values"]) & self.rejected_hashes)
        if checkpoint:
            # Persist the complete identity-key sets before opening any legacy
            # raster. Representative origin only keeps this recovery cache small;
            # the in-memory index and audit retain all origins for this run.
            save(checkpoint, {"root": str(root), "excluded_paths": sorted(map(str, excluded)),
                "historical_files": self.audit, "parse_errors": self.errors,
                "identity_maps": {name: {str(k): sorted(values)[:1] for k, values in getattr(self, name).items()}
                    for name in ("hashes", "visual", "observations", "photos", "urls", "paths")},
                "image_rows": {str(p): values for p, values in self.image_rows.items()},
                "reuse_rule": "Verify historical file fingerprints and index new/changed manifests before reuse. Current-run exclusions must remain explicit; never load this as accepted data."})
            print(f"History metadata checkpoint saved; {len(self.image_rows)} referenced images", flush=True)
        for index, (path, known) in enumerate(self.image_rows.items(), 1):
            if index % 10000 == 0:
                print(f"History: checked {index}/{len(self.image_rows)} image hash records", flush=True)
            for species in SPECIES:
                if re.sub(r"[^a-z]", "", species.lower()) in re.sub(r"[^a-z]", "", str(path).lower()):
                    self.species_inventory[species].add(str(path))
            if known["hash"] and known["phash"]:
                self.visual_scope["existing_record_hash_and_phash_reused"] += 1
                continue
            try:
                if not known["hash"]:
                    self.hashes[digest(path)].add(str(path))
                    self.visual_scope["missing_sha256_computed_all_species"] += 1
                if not known["phash"] and (known["target_species"] or known["rejected"]):
                    with Image.open(path) as image:
                        self.visual[phash(ImageOps.exif_transpose(image))].add(str(path))
                    self.visual_scope["missing_phash_computed_target_species_or_linked_rejection"] += 1
                elif not known["phash"]:
                    self.visual_scope["non_target_missing_phash_not_decoded"] += 1
            except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
                self.errors.append({"path": str(path), "error": str(exc)})
        return self

    def matches(self, row, local, near_distance):
        matches = {}
        observations, photos, _ = identities(row)
        # A gallery/species page can host genuinely different pictures; page URL
        # equality alone must not reject every later image from that gallery.
        urls = {u for key in ("image_url", "photo_url") if (u := canonical_url(row.get(key)))}
        for tag, values, mapping in (("historical_observation", observations, self.observations),
                                     ("historical_photo", photos, self.photos), ("historical_url", urls, self.urls)):
            hit = sorted({p for value in values for p in mapping.get(value, ())})
            if hit:
                matches[tag] = hit[:20]
        for tag, value, mapping in (("historical_exact_sha256", row["sha256"], self.hashes),
                                    ("historical_local_file", str(local).casefold(), self.paths)):
            if value in mapping:
                matches[tag] = sorted(mapping[value])[:20]
        hit = [p for h, paths in self.visual.items() if (h ^ int(row["phash"], 16)).bit_count() <= near_distance for p in paths]
        if hit:
            matches["historical_near_duplicate"] = sorted(set(hit))[:20]
        return matches


def contact_sheets(rows, out):
    for species in SPECIES:
        group = [r for r in rows if r["source_species"] == species and r["status"] == "imported_pending"]
        for start in range(0, len(group), 12):
            canvas = Image.new("RGB", (1440, 1080), "white")
            draw = ImageDraw.Draw(canvas)
            for j, row in enumerate(group[start:start + 12]):
                with Image.open(row["source_image"]) as image:
                    preview = ImageOps.exif_transpose(image).convert("RGB")
                    preview.thumbnail((350, 265))
                x, y = (j % 4) * 360, (j // 4) * 360
                canvas.paste(preview, (x + (350 - preview.width) // 2, y))
                draw.text((x + 3, y + 270), row["id"], fill="black")
                draw.text((x + 3, y + 288), species, fill="black")
                draw.text((x + 3, y + 306), str(row["dimensions"]) + " " + row["quality_group"], fill="black")
                draw.text((x + 3, y + 324), "PENDING / " + row["rights_status"][:42], fill="black")
            canvas.save(out / "contact_sheets" / f"{species.replace(' ', '_')}_{start // 12:03d}.jpg", quality=92)


def main(records_path, out, history_root=ROOT, near_distance=6, exclude_current_run=None):
    records_path, out, history_root = records_path.resolve(), out.resolve(), history_root.resolve()
    payload = json.loads(records_path.read_text(encoding="utf-8-sig"))
    records = payload if isinstance(payload, list) else payload.get("items", payload.get("records"))
    if not isinstance(records, list):
        raise ValueError("Input must be a list or {items:[...]} / {records:[...]}")
    if not all(isinstance(record, dict) for record in records):
        raise ValueError("Every source record must be an object")
    if not 0 <= near_distance <= 10:
        raise ValueError("near_distance must be 0..10")
    current_context = []
    context_paths = set()
    for raw_path in exclude_current_run or []:
        path = Path(raw_path).resolve(strict=True)
        if not path.is_dir() or not path.is_relative_to(history_root) or len(path.relative_to(history_root).parts) < 2:
            raise ValueError("Current-run context exclusion must name a specific nested project directory, never a broad root")
        files = [{"path": str(p.resolve()), "sha256": digest(p), "bytes": p.stat().st_size}
                 for p in sorted(path.rglob("*")) if p.is_file()]
        current_context.append({"path": str(path), "reason": "Explicitly authorized current-run acquisition/parallel-QA evidence, not historical data. Avoid indexing these same incoming assets as pre-existing images.", "existing_files_at_index_start": files})
        context_paths.add(path)
    out.mkdir(parents=True, exist_ok=False)
    (out / "images").mkdir()
    (out / "contact_sheets").mkdir()
    save(out / "input_source_records.json", payload)
    print("Indexing historical manifests, review DBs and existing species images (read only)", flush=True)
    history = HistoryIndex().build(history_root, {out, records_path} | context_paths, out / "history_metadata_cache.json")
    save(out / "prior_index_audit.json", {"files": history.audit, "errors": history.errors,
        "current_run_context_exclusions": current_context,
        "hashes": len(history.hashes), "phashes": len(history.visual), "observations": len(history.observations),
        "photos": len(history.photos), "scope": "All project historical manifest/rows/image-index/review JSON(L), read-only human-review DB decisions/events, and three species external image directories; no decision filtering.",
        "visual_scope": dict(history.visual_scope),
        "limitations": "All recorded historical pHashes are reused; missing pHashes are newly computed ONLY for the three target species (path or explicit row association) or a linked historical rejection, not unrelated panoramas. Missing SHA is still computed for all referenced images. This is not an exhaustive all-image visual deduplication. Unrecorded same-tree identities, mirrored photos and large crops can evade pHash. Source labels are not botanical confirmation.",
        "existing_species_path_inventory": {k: len(v) for k, v in history.species_inventory.items()}})
    # A malformed historical file must not silently relax rejected-image exclusion.
    if history.errors:
        save(out / "BLOCKED.json", {"reason": "Historical index incomplete; resolve prior_index_audit.errors before importing.", "training_performed": False})
        raise RuntimeError("Historical index incomplete; no images imported. See prior_index_audit.json")
    rows, seen_hashes, seen_visual, seen_urls = [], {}, {}, {}
    for sequence, record in enumerate(records, 1):
        row = {"id": f"cons_browser_{sequence:04d}", "source_record": record,
               "source_record_index": sequence,
               "source_species": record.get("species"), "source_species_zh_hant": SPECIES.get(record.get("species")),
               "source": "browser_source_page", "source_url": record.get("page_url"), "page_url": record.get("page_url"),
               "image_url": record.get("image_url"), "organ": record.get("organ", "not_reviewed"),
               "license": record.get("license", "unknown"), "attribution": record.get("attribution"),
               "source_evidence": record.get("evidence"), "rights_status": "rights_unverified",
               "input_rights_assertion": record.get("rights_status"), "review_status": "pending_agent_review",
               "human_review_status": "pending", "training_eligible": False, "split": "quarantine",
               "status": "excluded_invalid_source", "quality_flags": [], "duplicate_matches": {}}
        for field in ("source_group", "group_id", "identity_status", "observer_id", "observed_on", "location_public"):
            if field in record:
                row[field] = record[field]
        try:
            if row["source_species"] not in SPECIES:
                raise ValueError("Unsupported exact species; do not silently map synonyms")
            if not all(canonical_url(record.get(key)) for key in ("page_url", "image_url")):
                raise ValueError("Valid public page_url and image_url required")
            if record.get("asset_kind") not in {"original", "source_page_image"}:
                raise ValueError("asset_kind must be original or source_page_image; thumbnails/screenshots prohibited")
            if record.get("acquisition_method") != "browser_page_asset":
                raise ValueError("acquisition_method must explicitly be browser_page_asset")
            host = urlsplit(record["image_url"]).hostname or ""
            if "encrypted-tbn" in host or host in {"tse1.mm.bing.net", "tse2.mm.bing.net", "tse3.mm.bing.net", "tse4.mm.bing.net"}:
                raise ValueError("Search-engine thumbnail is not a source-page image")
            local = Path(record["local_file"])
            if not local.is_absolute():
                local = records_path.parent / local
            local = local.resolve(strict=True)
            if not local.is_file() or local.is_relative_to(out):
                raise ValueError("Input must be an existing file outside this new output directory")
            if local.stat().st_size > 80_000_000:
                raise ValueError("Image larger than 80 MB; inspect separately")
            with Image.open(local) as image:
                if image.format not in {"JPEG", "PNG", "WEBP", "TIFF", "BMP"}:
                    raise ValueError("Unsupported source image format")
                image.load()
                fmt = image.format
                native = list(image.size)
                rgb = ImageOps.exif_transpose(image).convert("RGB")
                gray = np.asarray(rgb.convert("L"))
                row.update(sha256=digest(local), phash=f"{phash(rgb):016x}", dimensions=list(rgb.size), native_dimensions=native,
                           bytes=local.stat().st_size, format=fmt, input_local_file=str(local),
                           asset_kind=record["asset_kind"], acquisition_method=record["acquisition_method"],
                           quality={"dark_fraction": float(np.mean(gray < 25)), "bright_fraction": float(np.mean(gray > 245))})
                row["id"] = "cons_browser_" + row["sha256"][:20]
            row["duplicate_matches"] = history.matches({**record, **row}, local, near_distance)
            if row["sha256"] in seen_hashes:
                row["duplicate_matches"]["batch_exact_duplicate"] = [seen_hashes[row["sha256"]]]
            near = [ident for h, ident in seen_visual.items() if (h ^ int(row["phash"], 16)).bit_count() <= near_distance]
            if near:
                row["duplicate_matches"]["batch_near_duplicate"] = near[:20]
            image_url = canonical_url(row["image_url"])
            if image_url in seen_urls:
                row["duplicate_matches"]["batch_same_image_url"] = [seen_urls[image_url]]
            if row["duplicate_matches"]:
                row["status"] = "excluded_historical_or_batch_duplicate"
                row["review_status"] = "not_reintroduced"
            else:
                smallest = min(row["dimensions"])
                row["quality_group"] = "high_resolution" if smallest >= 768 else "standard_resolution" if smallest >= 400 else "limited_resolution" if smallest >= 224 else "low_resolution"
                if smallest < 224:
                    row["quality_flags"].append("low_resolution")
                if row["quality"]["dark_fraction"] > .65:
                    row["quality_flags"].append("mostly_dark")
                if row["quality"]["bright_fraction"] > .85:
                    row["quality_flags"].append("mostly_bright")
                if record.get("rights_status") == "verified_open_license" and canonical_url(record.get("license_url")):
                    row["rights_status"] = "source_claimed_open_license_pending_verification"
                suffix = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp", "TIFF": ".tif", "BMP": ".bmp"}[fmt]
                destination = out / "images" / (row["id"] + suffix)
                with local.open("rb") as src, destination.open("xb") as dst:
                    shutil.copyfileobj(src, dst)
                if digest(destination) != row["sha256"] or digest(local) != row["sha256"]:
                    raise RuntimeError("Byte-copy verification failed or input changed during import")
                row.update(status="imported_pending", source_image=str(destination), file=destination.relative_to(out).as_posix(),
                           copy_verified=True, license_url=record.get("license_url"),
                           candidate_lane="pending_quality_review" if row["quality_flags"] else "pending_species_and_rights_review")
                seen_hashes[row["sha256"]] = row["id"]
                seen_visual[int(row["phash"], 16)] = row["id"]
                seen_urls[image_url] = row["id"]
        except (OSError, ValueError, KeyError, Image.DecompressionBombError) as exc:
            row["error"] = str(exc)
        rows.append(row)
    save(out / "manifest.json", rows)
    pending = [r for r in rows if r["status"] == "imported_pending"]
    save(out / "review_manifest.json", pending)
    contact_sheets(rows, out)
    required_notices = []
    for record in records:
        evidence = record.get("evidence") or {}
        rights_evidence = evidence.get("rights_evidence", {}) if isinstance(evidence, dict) else {}
        filename = rights_evidence.get("required_notice_file")
        if filename:
            if Path(filename).name != filename or filename != "COPYRIGHT_GREening.txt":
                raise ValueError("Unsupported source notice filename; no arbitrary file copies")
            required_notices.append(filename)
    for filename in sorted(set(required_notices)):
        src, dst = records_path.parent / filename, out / filename
        with src.open("rb") as original, dst.open("xb") as copied:
            shutil.copyfileobj(original, copied)
        if digest(src) != digest(dst):
            raise RuntimeError("Required copyright notice copy mismatch")
    db_integrity = []
    for entry in history.audit:
        path = Path(entry["path"])
        if path.suffix.lower() in {".sqlite3", ".db"} and "review" in path.name.lower():
            after = digest(path)
            db_integrity.append({"path": str(path), "before_sha256": entry["sha256"],
                                 "after_sha256": after, "unchanged": after == entry["sha256"]})
    save(out / "historical_review_db_integrity.json", {"databases": db_integrity,
        "all_unchanged": all(d["unchanged"] for d in db_integrity),
        "note": "Read-only connections only. Any concurrent modification by another process is reported, not overwritten."})
    totals = {"input_records": len(records), "new_unique_source_images": len(pending),
              "excluded_duplicates": sum(r["status"] == "excluded_historical_or_batch_duplicate" for r in rows),
              "invalid_source_records": sum(r["status"] == "excluded_invalid_source" for r in rows),
              "by_species": {species: {"new_pending": sum(r["source_species"] == species for r in pending),
                  "quality_groups": dict(Counter(r["quality_group"] for r in pending if r["source_species"] == species))} for species in SPECIES},
              "quality_flags": dict(Counter(flag for r in pending for flag in r["quality_flags"])),
              "human_accepted": 0, "training_eligible": 0, "training_performed": False,
              "deployed_files_changed": False, "network_requests": 0, "created_at": datetime.now(timezone.utc).isoformat()}
    save(out / "totals.json", totals)
    save(out / "VERIFIED.json", {"input_records_sha256": digest(records_path), "manifest_sha256": digest(out / "manifest.json"),
        "review_manifest_sha256": digest(out / "review_manifest.json"), "byte_copies_verified": len(pending),
        "old_files_opened_read_only": True, "human_db_mutations": 0, "training_performed": False})
    print(json.dumps(totals, ensure_ascii=True), flush=True)
    return totals


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--history-root", type=Path, default=ROOT, help="Default: complete project. Override only for isolated fixtures.")
    parser.add_argument("--near-distance", type=int, default=6)
    parser.add_argument("--exclude-current-run", type=Path, action="append", default=[],
                        help="Explicit current acquisition/QA directory, not historical data. Repeatable; fully audited.")
    args = parser.parse_args()
    main(args.records, args.out, args.history_root, args.near_distance, args.exclude_current_run)
