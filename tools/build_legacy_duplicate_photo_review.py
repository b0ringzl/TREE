#!/usr/bin/env python3
"""Build a visual review page for duplicate iNaturalist photo IDs in a batch."""

from __future__ import annotations

import argparse
import csv
import html
from collections import defaultdict
from pathlib import Path
from urllib.parse import quote


def read_rows(catalog: Path, batch_date: str) -> list[dict[str, str]]:
    with catalog.open("r", encoding="utf-8-sig", newline="") as handle:
        return [row for row in csv.DictReader(handle) if row.get("batch_date") == batch_date]


def relative_image_src(output_dir: Path, image_path: str) -> str:
    absolute = Path(image_path)
    try:
        relative = absolute.relative_to(output_dir.resolve())
    except ValueError:
        relative = Path("..") / absolute.relative_to(output_dir.parent.resolve())
    return quote(relative.as_posix(), safe="/:")


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fields = [
        "photo_id",
        "left_species",
        "right_species",
        "merged_label_species",
        "left_image_id",
        "right_image_id",
            "source_page_url",
        "left_absolute_path",
        "right_absolute_path",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_html(path: Path, rows: list[dict[str, str]], output_dir: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("<!doctype html><html><head><meta charset='utf-8'>")
        handle.write("<title>Legacy Duplicate iNaturalist Photo Review</title>")
        handle.write(
            "<style>"
            "body{margin:0;background:#f8fafc;color:#0f172a;font-family:Arial,sans-serif}"
            "header{position:sticky;top:0;background:#fff;border-bottom:1px solid #cbd5e1;padding:16px 24px;z-index:2}"
            "h1{margin:0 0 6px;font-size:24px}.sub{color:#475569;font-size:14px}"
            ".wrap{padding:20px 24px;display:grid;gap:18px}"
            ".card{background:#fff;border:1px solid #cbd5e1;border-radius:8px;overflow:hidden}"
            ".card-head{display:flex;justify-content:space-between;gap:12px;padding:12px 14px;border-bottom:1px solid #e2e8f0}"
            ".photo{font-weight:700}.link{color:#0369a1;text-decoration:none;font-size:13px}"
            ".pair{display:grid;grid-template-columns:1fr 1fr;gap:0}"
            ".side{padding:14px;border-right:1px solid #e2e8f0}.side:last-child{border-right:0}"
            ".species{font-size:18px;font-weight:700;margin-bottom:4px}.meta{font-size:12px;color:#64748b;margin-bottom:10px}"
            "img{width:100%;max-height:520px;object-fit:contain;background:#f1f5f9;border:1px solid #e2e8f0;border-radius:6px}"
            "@media(max-width:900px){.pair{grid-template-columns:1fr}.side{border-right:0;border-bottom:1px solid #e2e8f0}.side:last-child{border-bottom:0}}"
            "</style></head><body>"
        )
        handle.write("<header>")
        handle.write("<h1>Legacy Duplicate iNaturalist Photo Review</h1>")
        handle.write(
            f"<div class='sub'>旧批次 2026-05-24 内部跨物种重复 photo_id：{len(rows)} 组。"
            "左右两侧为不同物种目录下的同一张 iNaturalist 图片。</div>"
        )
        handle.write("</header><main class='wrap'>")
        for index, row in enumerate(rows, start=1):
            left_src = relative_image_src(output_dir, row["left_absolute_path"])
            right_src = relative_image_src(output_dir, row["right_absolute_path"])
            source_url = html.escape(row["source_page_url"])
            handle.write("<section class='card'>")
            handle.write("<div class='card-head'>")
            handle.write(f"<div class='photo'>#{index:03d} photo_id {html.escape(row['photo_id'])}</div>")
            if row.get("merged_label_species"):
                handle.write(f"<div class='sub'>merged label: {html.escape(row['merged_label_species'])}</div>")
            if source_url:
                handle.write(f"<a class='link' href='{source_url}' target='_blank'>iNaturalist observation</a>")
            handle.write("</div><div class='pair'>")
            for side, src in (("left", left_src), ("right", right_src)):
                species = row[f"{side}_species"]
                image_id = row[f"{side}_image_id"]
                absolute = row[f"{side}_absolute_path"]
                handle.write("<div class='side'>")
                handle.write(f"<div class='species'>{html.escape(species)}</div>")
                handle.write(f"<div class='meta'>{html.escape(image_id)}<br>{html.escape(absolute)}</div>")
                handle.write(f"<img src='{src}' alt='{html.escape(species)}'>")
                handle.write("</div>")
            handle.write("</div></section>")
        handle.write("</main></body></html>")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build visual review for duplicate photo IDs in a legacy batch.")
    parser.add_argument("--catalog", type=Path, default=Path("external_datasets/inaturalist_master_catalog.csv"))
    parser.add_argument("--batch-date", default="2026-05-24")
    parser.add_argument("--output-dir", type=Path, default=Path("external_datasets/inat_max150/duplicate_photo_review"))
    args = parser.parse_args()

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_rows(args.catalog, args.batch_date):
        photo_id = row.get("photo_id", "").strip()
        if photo_id:
            grouped[photo_id].append(row)

    pairs: list[dict[str, str]] = []
    for photo_id, group in sorted(grouped.items()):
        species = sorted({row.get("original_species") or row["target_species"] for row in group}, key=str.casefold)
        if len(group) < 2 or len(species) < 2:
            continue
        left = next(row for row in group if (row.get("original_species") or row["target_species"]) == species[0])
        right = next(row for row in group if (row.get("original_species") or row["target_species"]) == species[1])
        left_label = left.get("label_species") or left["target_species"]
        right_label = right.get("label_species") or right["target_species"]
        pairs.append(
            {
                "photo_id": photo_id,
                "left_species": left.get("original_species") or left["target_species"],
                "right_species": right.get("original_species") or right["target_species"],
                "merged_label_species": left_label if left_label == right_label else "",
                "left_image_id": left["image_id"],
                "right_image_id": right["image_id"],
                "source_page_url": left.get("source_page_url") or right.get("source_page_url", ""),
                "left_absolute_path": left["absolute_path"],
                "right_absolute_path": right["absolute_path"],
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "duplicate_pairs.csv", pairs)
    write_html(args.output_dir / "index.html", pairs, args.output_dir)
    print(f"duplicate_pairs={len(pairs)}")
    print(f"html={args.output_dir / 'index.html'}")
    print(f"csv={args.output_dir / 'duplicate_pairs.csv'}")


if __name__ == "__main__":
    main()
