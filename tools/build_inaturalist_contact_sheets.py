#!/usr/bin/env python3
"""Build per-species contact sheets for an iNaturalist dated image batch."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


def safe_name(value: str) -> str:
    allowed = []
    for ch in value.strip():
        if ch.isalnum() or ch in (" ", "_", "-", ".", "'", "(", ")"):
            allowed.append(ch)
        else:
            allowed.append("_")
    return "".join(allowed).strip(" .") or "unnamed"


def load_font(size: int) -> ImageFont.ImageFont:
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_text_fit(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: str,
    max_width: int,
) -> None:
    if not text:
        return
    clipped = text
    while len(clipped) > 3 and draw.textlength(clipped, font=font) > max_width:
        clipped = clipped[:-2]
    if clipped != text:
        clipped = clipped.rstrip() + "..."
    draw.text(xy, clipped, font=font, fill=fill)


def open_thumbnail(path: Path, size: int) -> Image.Image:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((size, size), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (size, size), "white")
        offset = ((size - image.width) // 2, (size - image.height) // 2)
        canvas.paste(image, offset)
        return canvas


def build_sheet(species: str, rows: list[dict], output: Path, limit: int, columns: int, thumb: int) -> None:
    title_font = load_font(24)
    meta_font = load_font(13)
    label_font = load_font(11)
    gutter = 12
    header = 72
    label_h = 34
    selected = rows[:limit]
    count = len(selected)
    sheet_rows = max(1, (count + columns - 1) // columns)
    width = columns * thumb + (columns + 1) * gutter
    height = header + sheet_rows * (thumb + label_h + gutter) + gutter
    sheet = Image.new("RGB", (width, height), "#f8fafc")
    draw = ImageDraw.Draw(sheet)
    draw.text((gutter, 12), species, font=title_font, fill="#111827")
    draw.text(
        (gutter, 42),
        f"showing {count} of {len(rows)} images | source: iNaturalist research-grade exact taxon",
        font=meta_font,
        fill="#475569",
    )

    for index, row in enumerate(selected):
        grid_x = index % columns
        grid_y = index // columns
        x = gutter + grid_x * (thumb + gutter)
        y = header + grid_y * (thumb + label_h + gutter)
        path = Path(row["absolute_path"])
        try:
            tile = open_thumbnail(path, thumb)
        except Exception:
            tile = Image.new("RGB", (thumb, thumb), "#fee2e2")
            err_draw = ImageDraw.Draw(tile)
            err_draw.text((10, 10), "read error", font=meta_font, fill="#991b1b")
        sheet.paste(tile, (x, y))
        draw.rectangle((x, y, x + thumb - 1, y + thumb - 1), outline="#cbd5e1", width=1)
        label = f"{row.get('image_id', '')} | {row.get('photo_license_code', '')}"
        draw_text_fit(draw, (x, y + thumb + 5), label, label_font, "#0f172a", thumb)
        obs = f"obs {row.get('inat_observation_id', '')}"
        draw_text_fit(draw, (x, y + thumb + 20), obs, label_font, "#64748b", thumb)

    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=88)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build iNaturalist image contact sheets by species.")
    parser.add_argument("--catalog", type=Path, default=Path("external_datasets/inaturalist_dated/2026-06-07/image_catalog.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("external_datasets/inaturalist_dated/2026-06-07/contact_sheets"))
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--columns", type=int, default=6)
    parser.add_argument("--thumb", type=int, default=180)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    grouped: dict[str, list[dict]] = defaultdict(list)
    with args.catalog.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            grouped[row["target_species"]].append(row)

    index_rows = []
    for species in sorted(grouped, key=str.casefold):
        rows = sorted(grouped[species], key=lambda item: item["image_id"])
        output = args.output_dir / f"{safe_name(species)}.jpg"
        if not (args.skip_existing and output.exists()):
            build_sheet(species, rows, output, args.limit, args.columns, args.thumb)
        index_rows.append({"species": species, "image_count": len(rows), "contact_sheet": str(output)})

    index_path = args.output_dir / "index.csv"
    html_path = args.output_dir / "index.html"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with index_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["species", "image_count", "contact_sheet"])
        writer.writeheader()
        writer.writerows(index_rows)
    with html_path.open("w", encoding="utf-8") as handle:
        handle.write("<!doctype html><html><head><meta charset='utf-8'>")
        handle.write("<title>iNaturalist Contact Sheets</title>")
        handle.write(
            "<style>body{font-family:Arial,sans-serif;margin:24px;background:#f8fafc;color:#0f172a}"
            "h1{font-size:24px}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:16px}"
            "a{display:block;background:#fff;border:1px solid #cbd5e1;border-radius:8px;padding:10px;color:inherit;text-decoration:none}"
            "img{width:100%;height:180px;object-fit:cover;border-radius:6px;border:1px solid #e2e8f0}"
            ".name{font-weight:700;margin-top:8px}.count{color:#64748b;font-size:13px}</style>"
        )
        handle.write("</head><body><h1>iNaturalist Contact Sheets - 2026-06-07</h1><div class='grid'>")
        for row in index_rows:
            image = Path(row["contact_sheet"]).name
            handle.write(f"<a href='{image}'><img src='{image}' alt=''>")
            handle.write(f"<div class='name'>{row['species']}</div><div class='count'>{row['image_count']} images</div></a>")
        handle.write("</div></body></html>")
    print(f"contact_sheets={len(index_rows)} output_dir={args.output_dir}")
    print(f"index={index_path}")
    print(f"html={html_path}")


if __name__ == "__main__":
    main()
