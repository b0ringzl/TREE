#!/usr/bin/env python3
"""Audit local tree species image folders and build review contact sheets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import random
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    species: str
    source: str


def safe_name(name: str) -> str:
    for char in '<>:"/\\|?*':
        name = name.replace(char, "_")
    return name.strip().rstrip(".") or "unknown"


def iter_records(root: Path, source: str) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    if not root.exists():
        return records
    for species_dir in sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name.casefold()):
        image_dir = species_dir / "images"
        if not image_dir.exists():
            continue
        for path in sorted(image_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                records.append(ImageRecord(path=path.resolve(), species=species_dir.name, source=source))
    return records


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_info(record: ImageRecord, hash_files: bool) -> dict:
    row = {
        "species": record.species,
        "source": record.source,
        "path": str(record.path),
        "bytes": record.path.stat().st_size,
        "width": "",
        "height": "",
        "aspect_ratio": "",
        "sha256": "",
        "status": "ok",
        "issue": "",
    }
    try:
        with Image.open(record.path) as image:
            image.verify()
        with Image.open(record.path) as image:
            width, height = image.size
        row["width"] = width
        row["height"] = height
        row["aspect_ratio"] = round(width / height, 4) if height else ""
        if width < 128 or height < 128:
            row["status"] = "warning"
            row["issue"] = "small_image"
        elif width / height > 3.0 or height / width > 3.0:
            row["status"] = "warning"
            row["issue"] = "extreme_aspect_ratio"
    except (OSError, UnidentifiedImageError) as exc:
        row["status"] = "corrupt"
        row["issue"] = str(exc)
    if hash_files and row["status"] != "corrupt":
        row["sha256"] = sha256_file(record.path)
    return row


def make_contact_sheet(species: str, records: list[ImageRecord], output_path: Path, sample_count: int, seed: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(f"{seed}:{species}")
    chosen = records[:]
    rng.shuffle(chosen)
    chosen = chosen[:sample_count]
    thumb_w, thumb_h = 160, 160
    label_h = 34
    cols = 6
    rows = max(1, (len(chosen) + cols - 1) // cols)
    canvas = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + label_h) + 36), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((8, 8), f"{species} ({len(records)} images, sample {len(chosen)})", fill=(20, 32, 27), font=font)
    for index, record in enumerate(chosen):
        col = index % cols
        row = index // cols
        x = col * thumb_w
        y = 36 + row * (thumb_h + label_h)
        try:
            with Image.open(record.path) as image:
                image = ImageOps.exif_transpose(image.convert("RGB"))
                image.thumbnail((thumb_w, thumb_h))
                tile = Image.new("RGB", (thumb_w, thumb_h), (238, 244, 240))
                tile.paste(image, ((thumb_w - image.width) // 2, (thumb_h - image.height) // 2))
        except Exception:
            tile = Image.new("RGB", (thumb_w, thumb_h), (255, 230, 226))
            ImageDraw.Draw(tile).text((12, 70), "LOAD ERROR", fill=(180, 35, 24), font=font)
        canvas.paste(tile, (x, y))
        label = record.path.name[:28]
        draw.text((x + 4, y + thumb_h + 4), label, fill=(49, 66, 56), font=font)
    canvas.save(output_path, quality=88)


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit local tree species image dataset.")
    parser.add_argument("--data-root", type=Path, default=Path("external_datasets/inat_max150"))
    parser.add_argument("--supplement-root", type=Path, default=Path("external_datasets/inat_parent_name_supplement"))
    parser.add_argument("--output", type=Path, default=Path("models/data_quality_audit"))
    parser.add_argument("--sample-count", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hash-files", action="store_true")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    primary = iter_records(args.data_root, "inat_max150")
    supplement = iter_records(args.supplement_root, "inat_parent_name_supplement")

    primary_counts: dict[str, int] = {}
    for record in primary:
        primary_counts[record.species] = primary_counts.get(record.species, 0) + 1
    combined = primary + [record for record in supplement if primary_counts.get(record.species, 0) == 0]

    info_rows = [image_info(record, args.hash_files) for record in combined]
    fields = ["species", "source", "path", "bytes", "width", "height", "aspect_ratio", "sha256", "status", "issue"]
    write_csv(args.output / "image_audit.csv", info_rows, fields)

    grouped: dict[str, list[ImageRecord]] = {}
    for record in combined:
        grouped.setdefault(record.species, []).append(record)

    summary_rows = []
    for species, records in sorted(grouped.items(), key=lambda item: item[0].casefold()):
        species_rows = [row for row in info_rows if row["species"] == species]
        ok_count = sum(1 for row in species_rows if row["status"] == "ok")
        warning_count = sum(1 for row in species_rows if row["status"] == "warning")
        corrupt_count = sum(1 for row in species_rows if row["status"] == "corrupt")
        contact_sheet = Path("contact_sheets") / f"{safe_name(species)}.jpg"
        make_contact_sheet(species, records, args.output / contact_sheet, args.sample_count, args.seed)
        summary_rows.append({
            "species": species,
            "images": len(records),
            "ok": ok_count,
            "warnings": warning_count,
            "corrupt": corrupt_count,
            "source_mix": ",".join(sorted({record.source for record in records})),
            "contact_sheet": contact_sheet.as_posix(),
        })
    write_csv(
        args.output / "species_summary.csv",
        summary_rows,
        ["species", "images", "ok", "warnings", "corrupt", "source_mix", "contact_sheet"],
    )

    duplicates = []
    if args.hash_files:
        by_hash: dict[str, list[dict]] = {}
        for row in info_rows:
            if row["sha256"]:
                by_hash.setdefault(row["sha256"], []).append(row)
        for digest, rows in by_hash.items():
            if len(rows) > 1:
                for row in rows:
                    duplicates.append({"sha256": digest, "species": row["species"], "path": row["path"]})
    write_csv(args.output / "duplicate_files.csv", duplicates, ["sha256", "species", "path"])

    html_rows = []
    for row in summary_rows:
        sheet = html.escape(row["contact_sheet"])
        html_rows.append(
            "<tr>"
            f"<td>{html.escape(row['species'])}</td>"
            f"<td>{row['images']}</td>"
            f"<td>{row['warnings']}</td>"
            f"<td>{row['corrupt']}</td>"
            f"<td>{html.escape(row['source_mix'])}</td>"
            f"<td><a href='{sheet}'>contact sheet</a></td>"
            "</tr>"
        )
    index_html = (
        "<!doctype html><meta charset='utf-8'>"
        "<title>Tree Species Dataset Audit</title>"
        "<style>body{font-family:Arial,sans-serif;margin:24px;color:#17201b}"
        "table{border-collapse:collapse;width:100%}td,th{border:1px solid #dbe4df;padding:6px 8px;text-align:left}"
        "th{background:#f4f7f5;position:sticky;top:0}</style>"
        "<h1>Tree Species Dataset Audit</h1>"
        f"<p>Species: {len(summary_rows)} | Images: {len(combined)} | Duplicates: {len(duplicates)}</p>"
        "<table><thead><tr><th>Species</th><th>Images</th><th>Warnings</th><th>Corrupt</th><th>Source</th><th>Preview</th></tr></thead>"
        f"<tbody>{''.join(html_rows)}</tbody></table>"
    )
    (args.output / "index.html").write_text(index_html, encoding="utf-8")

    report = {
        "species": len(summary_rows),
        "images": len(combined),
        "warnings": sum(1 for row in info_rows if row["status"] == "warning"),
        "corrupt": sum(1 for row in info_rows if row["status"] == "corrupt"),
        "duplicates": len(duplicates),
        "output": str(args.output.resolve()),
    }
    (args.output / "audit_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
