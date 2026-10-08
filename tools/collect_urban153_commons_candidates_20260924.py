"""Collect licensed Commons originals as quarantine-only urban-153 candidates.

Category membership and captions are not ground-truth species confirmation.
No model fitting, no production writes, no automatic acceptance.
"""

import argparse
import hashlib
import io
import json
import re
from html import unescape
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
API = "https://commons.wikimedia.org/w/api.php"
UA = "SpaceScanTrees/1.0 (private tree classifier candidate review)"
CATEGORIES = {
    "Michelia chapensis": "Category:Magnolia chapensis",
    "Yulania x soulangeana": "Category:Magnolia × soulangeana",
}
ALLOW = ("CC0", "CC BY ", "CC BY-SA ", "Public domain")


def plain(value):
    return re.sub(r"<[^>]+>", " ", unescape(value or "")).strip()


def collect(out, species, cap):
    out.mkdir(parents=True, exist_ok=False)
    (out / "originals").mkdir()
    pages = []
    cursor = None
    while len(pages) < cap:
        params = {"action": "query", "generator": "categorymembers", "gcmtitle": CATEGORIES[species],
                  "gcmtype": "file", "gcmlimit": min(50, cap), "prop": "imageinfo",
                  "iiprop": "url|size|mime|extmetadata", "format": "json"}
        if cursor:
            params["gcmcontinue"] = cursor
        response = requests.get(API, params=params, headers={"User-Agent": UA}, timeout=30)
        response.raise_for_status()
        body = response.json()
        pages.extend(sorted(body.get("query", {}).get("pages", {}).values(), key=lambda p: p["title"]))
        cursor = body.get("continue", {}).get("gcmcontinue")
        if not cursor:
            break
    manifest = []
    rate_limited = False
    for page in pages[:cap]:
        title = page["title"]
        if species == "Michelia chapensis" and "chartacea" in title.lower():
            continue
        info = page.get("imageinfo", [{}])[0]
        meta = info.get("extmetadata", {})
        short = plain(meta.get("LicenseShortName", {}).get("value", ""))
        if not short.startswith(ALLOW) or info.get("mime") not in ("image/jpeg", "image/png", "image/webp"):
            continue
        uri = info.get("url")
        row = {"commons_title": title, "commons_pageid": page["pageid"],
               "source_page": "https://commons.wikimedia.org/wiki/" + title.replace(" ", "_"),
               "source_species": species, "category": CATEGORIES[species],
               "source_media_url": uri, "license": short,
               "license_url": plain(meta.get("LicenseUrl", {}).get("value", "")),
               "artist": plain(meta.get("Artist", {}).get("value", "")),
               "description": plain(meta.get("ImageDescription", {}).get("value", ""))[:1000],
               "date_time_original": plain(meta.get("DateTimeOriginal", {}).get("value", "")),
               "source": plain(meta.get("Credit", {}).get("value", ""))[:500],
               "review_status": "pending", "training_eligible": False,
               "review_reason": "Wikimedia category/caption is not an independently verified botanical label",
               "source_group_hint": "creator/date/location must be checked; same tree may have multiple files"}
        try:
            if rate_limited:
                row["download_status"] = "deferred_rate_limit"
            elif info.get("size", 0) > 30_000_000:
                row["download_status"] = "skipped_over_30MB"
            else:
                image = requests.get(uri, headers={"User-Agent": UA}, timeout=45)
                image.raise_for_status()
                raw = image.content
                with Image.open(io.BytesIO(raw)) as im:
                    im.verify()
                suffix = ".png" if info["mime"] == "image/png" else ".webp" if info["mime"] == "image/webp" else ".jpg"
                filename = f"commons_{page['pageid']}{suffix}"
                (out / "originals" / filename).write_bytes(raw)
                row.update(download_status="downloaded", file="originals/" + filename,
                           sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 429:
                rate_limited = True
            row.update(download_status="failed", error=str(exc))
        except Exception as exc:
            row.update(download_status="failed", error=str(exc))
        manifest.append(row)
        print(row["download_status"], title[:80], flush=True)
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    rows = [r for r in manifest if r.get("download_status") == "downloaded"]
    for start in range(0, len(rows), 16):
        sheet = Image.new("RGB", (1600, 1200), "white")
        draw = ImageDraw.Draw(sheet)
        for j, row in enumerate(rows[start:start + 16]):
            with Image.open(out / row["file"]) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                im.thumbnail((380, 255))
                x = (j % 4) * 400
                y = (j // 4) * 300
                sheet.paste(im, (x + (380 - im.width) // 2, y))
            draw.text((x + 3, y + 257), f"{row['commons_pageid']} {row['commons_title'][5:47]}", fill="black")
        sheet.save(out / f"contact_{start // 16 + 1:02d}.jpg", quality=90)
    print(json.dumps({"species": species, "downloaded": len(rows), "pending_review": len(rows),
                      "training_eligible": 0}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--species", choices=list(CATEGORIES), required=True)
    parser.add_argument("--cap", type=int, default=50)
    args = parser.parse_args()
    collect(args.out.resolve(), args.species, args.cap)
