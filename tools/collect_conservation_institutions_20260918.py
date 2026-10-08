"""Small, explicit institutional-photo acquisition; candidate quarantine only.

No training, human decisions or existing data are modified. Public API routes
are discovered from the saved species-page JavaScript/HTML, not guessed.
"""
import argparse
import hashlib
import io
import json
import re
import time
from collections import Counter
from datetime import datetime, timezone
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlsplit
from urllib.robotparser import RobotFileParser

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageOps
from scipy.fft import dctn

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "outputs/conservation_authoritative_collection_20260918_v1/tsoplants"
RBG = "https://hortflora.rbg.vic.gov.au"
MYBIS = "https://www.mybis.gov.my"
UA = "SpaceScanTreesResearch/1.0 (bounded institutional photo provenance audit)"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def save(path, value):
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")


def phash(image):
    values = dctn(np.asarray(image.convert("L").resize((32, 32)), dtype=float), norm="ortho")[:8, :8].flatten()
    return int("".join("1" if v else "0" for v in values > np.median(values[1:])), 2)


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.images = []

    def handle_starttag(self, tag, attrs):
        if tag == "img":
            self.images.append(dict(attrs))


class Fetch:
    def __init__(self, out):
        self.out = out
        self.log = []
        self.robots = {}
        self.stopped = set()
        self.last = 0
        self.session = requests.Session()
        self.session.headers["User-Agent"] = UA

    def get(self, url, name=None, is_robot=False):
        parts = urlsplit(url)
        domain = parts.netloc
        if domain in self.stopped:
            raise RuntimeError("Origin stopped: " + domain)
        if not is_robot and domain not in self.robots:
            robot_url = parts.scheme + "://" + domain + "/robots.txt"
            robot = self.get(robot_url, "robots_" + domain + ".txt", True)
            parser = RobotFileParser()
            if robot.status_code == 200:
                parser.parse(robot.text.splitlines())
            elif robot.status_code == 404:
                parser.parse([])
            else:
                self.stopped.add(domain)
                raise RuntimeError("Unresolved robots status: " + str(robot.status_code))
            self.robots[domain] = parser
        if not is_robot and not self.robots[domain].can_fetch(UA, url):
            raise RuntimeError("Disallowed by robots: " + url)
        time.sleep(max(0, 1.5 - (time.monotonic() - self.last)))
        response = self.session.get(url, timeout=25, allow_redirects=False)
        self.last = time.monotonic()
        record = {"url": url, "status": response.status_code, "retrieved_at": datetime.now(timezone.utc).isoformat(),
                  "bytes": len(response.content), "sha256": sha(response.content), "content_type": response.headers.get("Content-Type")}
        if name:
            path = self.out / "source_responses" / name
            with path.open("xb") as f:
                f.write(response.content)
            record["saved_response"] = str(path.resolve())
        self.log.append(record)
        print(response.status_code, url, flush=True)
        if response.status_code in {401, 403, 429}:
            self.stopped.add(domain)
            raise RuntimeError("Access restriction; no retries: " + url)
        if 300 <= response.status_code < 400:
            raise RuntimeError("Redirect requires independent destination review: " + str(response.headers.get("Location")))
        if not is_robot:
            response.raise_for_status()
        return response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT)
    parser.add_argument("--recover-rbg-once", action="store_true", help="One bounded independent official API attempt in a NEW directory")
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "source_responses").mkdir()
    (out / "images").mkdir()
    net = Fetch(out)
    if args.recover_rbg_once:
        rows, errors = [], []
        page = RBG + "/taxon/ad8b05e0-5340-11e7-b82b-005056b0018f"
        api = RBG + "/api/taxa/ad8b05e0-5340-11e7-b82b-005056b0018f/images?pageSize=20"
        try:
            response = net.get(api, "rbg_oil_gallery_metadata.json")
            for item in response.json()["data"]:
                if item["subtype"] != "Photograph":
                    continue
                assert item["scientificName"] == "Keteleeria fortunei" and item["license"] == "CC BY-NC-SA 4.0"
                asset = next(a for a in item["accessPoints"]["data"] if a["variant"] == "preview")
                image_url = asset["accessURI"]
                data = net.get(image_url).content
                with Image.open(io.BytesIO(data)) as im:
                    im.load()
                    rgb = ImageOps.exif_transpose(im).convert("RGB")
                    width, height = rgb.size
                    visual = phash(rgb)
                path = out / "images" / ("rbg_" + item["id"] + ".jpg")
                with path.open("xb") as f:
                    f.write(data)
                rows.append({"id": "rbg_" + item["id"], "source_species": "Keteleeria fortunei",
                             "source_image": str(path), "source_page": page, "image_url": image_url,
                             "domain": urlsplit(image_url).netloc, "source_site": "Royal Botanic Gardens Victoria HortFlora",
                             "license": "CC BY-NC-SA 4.0", "license_url": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
                             "license_evidence_url": api, "author": item["creator"],
                             "attribution": item["creator"] + ", HortFlora, CC BY-NC-SA 4.0",
                             "source_date": item["createDate"], "source_group": "RBGV_HortFlora:Keteleeria fortunei",
                             "sha256": sha(data), "phash": f"{visual:016x}", "width": width, "height": height,
                             "bytes": len(data), "retrieval_date": datetime.now(timezone.utc).isoformat(),
                             "source_evidence": item, "organ": "not_visually_reviewed", "asset_kind": "source_page_image",
                             "training_eligible": False, "human_review_status": "not_reviewed", "review_status": "candidate_quarantine",
                             "source_label_only": True, "independent_individual_verified": False,
                             "preserve_original_bytes": True, "global_historical_audit_required": True,
                             "duplicate_check_scope": "none_in_this_bounded_recovery; parent_full_audit_required",
                             "rights_scope": "CC BY-NC-SA4.0; noncommercial local candidate review, no model/deployment permission asserted"})
        except Exception as error:
            errors.append(str(error))
        save(out / "manifest.json", rows)
        save(out / "request_audit.json", net.log)
        save(out / "summary.json", {"photos_downloaded": len(rows), "errors": errors, "one_bounded_api_recovery": True,
                                   "endpoint_discovery_evidence": str(DEFAULT / "source_responses/hortflora_public_page_script.js")})
        with (out / "REPORT.md").open("x", encoding="utf-8") as f:
            f.write("# RBGV one bounded recovery\n\nPublic gallery endpoint previously discovered in HortFlora page JavaScript.\n"
                    f"Downloaded {len(rows)} photographs, candidate quarantine only. No training or deployment.\n"
                    "Full contemporary historical/source/near-duplicate audit pending. Illustration excluded.\n")
        print(json.dumps({"out": str(out), "photos": len(rows), "errors": errors}), flush=True)
        return
    rows, skipped, errors = [], [], []
    old = ROOT / "external_datasets/conservation_browser_20260917_v3/manifest.json"
    old_rows = json.loads(old.read_text(encoding="utf-8"))
    old_urls = {str(r.get("image_url")) for r in old_rows}
    cache_path = ROOT / "external_datasets/conservation_browser_20260917_v3/history_metadata_cache.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    history = cache["identity_maps"]
    old_hashes = set(history["hashes"]) | {str(r.get("sha256")) for r in old_rows}
    history_visual = [int(h) for h in history["visual"]]
    history_urls = set(history["urls"]) | old_urls

    def download(row):
        image_url = row["image_url"]
        if image_url in history_urls:
            skipped.append({**row, "reason": "known_historical_image_url_not_new"})
            return
        response = net.get(image_url)
        data = response.content
        with Image.open(io.BytesIO(data)) as im:
            im.load()
            rgb = ImageOps.exif_transpose(im).convert("RGB")
            width, height = rgb.size
            visual = phash(rgb)
        digest = sha(data)
        path = out / "images" / (row["id"] + ".jpg")
        with path.open("xb") as f:
            f.write(data)
        nearest = min((visual ^ h).bit_count() for h in history_visual) if history_visual else None
        row.update(source_image=str(path), domain=urlsplit(image_url).netloc, sha256=digest,
                   phash=f"{visual:016x}", width=width, height=height, bytes=len(data),
                   retrieval_date=datetime.now(timezone.utc).isoformat(),
                   training_eligible=False, human_review_status="not_reviewed",
                   review_status="candidate_quarantine", source_label_only=True,
                   independent_individual_verified=False, preserve_original_bytes=True,
                   historical_sha_match=digest in old_hashes, historical_min_phash_hamming=nearest,
                   duplicate_check_scope="Sept17 historical metadata cache + browser_v3; NOT final current full-history audit",
                   global_historical_audit_required=True)
        rows.append(row)

    try:
        net.get(RBG + "/js/hortflora.js", "hortflora_public_page_script.js")
        for species, taxon in [("Keteleeria fortunei", "ad8b05e0-5340-11e7-b82b-005056b0018f"),
                               ("Rhodoleia championii", "ad8d4742-5340-11e7-b82b-005056b0018f")]:
            page = RBG + "/taxon/" + taxon
            net.get(page, species.replace(" ", "_") + "_rbg.html")
            api = RBG + "/api/taxa/" + taxon + "/images?pageSize=20"
            data = net.get(api, species.replace(" ", "_") + "_rbg_images.json").json()
            if not data["data"]:
                skipped.append({"source_page": page, "source_species": species, "reason": "institution_gallery_has_zero_images"})
            for item in data["data"]:
                if item["subtype"] != "Photograph":
                    skipped.append({"source_page": page, "source_species": species, "source_id": item["id"], "reason": "illustration_not_training_photo"})
                    continue
                assert item["scientificName"] == species and item["license"] == "CC BY-NC-SA 4.0"
                photo = next(x for x in item["accessPoints"]["data"] if x["variant"] == "preview")
                download({"id": "rbg_" + item["id"], "source_species": species, "source_page": page,
                          "image_url": photo["accessURI"], "source_site": "Royal Botanic Gardens Victoria HortFlora",
                          "license": item["license"], "license_url": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
                          "license_evidence_url": api, "attribution": item["creator"] + ", Royal Botanic Gardens Victoria, HortFlora",
                          "author": item["creator"], "source_date": item["createDate"],
                          "source_group": "RBGV_HortFlora:" + species, "organ": "not_visually_reviewed",
                          "source_evidence": item, "asset_kind": "source_page_image",
                          "rights_scope": "CC BY-NC-SA 4.0; noncommercial local candidate review; not cleared for demo publication"})
    except Exception as error:
        errors.append({"source": "RBGV", "error": str(error)})

    try:
        page = MYBIS + "/sp/3056"
        html = net.get(page, "mybis_Rhodoleia_species.html").text
        assert "profile_gallery.php?sid=" in html and "Rhodoleia championii" in html
        gallery_url = MYBIS + "/one/profile_gallery.php?sid=3056"
        html = net.get(gallery_url, "mybis_Rhodoleia_gallery.html").text
        for image_id, author, date in [("1521", "Syahida Emiza Suhaimi", "17/12/2015"),
                                       ("1991", "Lim Chung Lu", "22/12/2015"),
                                       ("4572", "Saw Leng Guan", "30/09/2016")]:
            start = html.index('id="f' + image_id + '"')
            stop = html.find('<div id="f', start + 1)
            fragment = html[start:stop if stop != -1 else len(html)]
            assert all(x in fragment for x in ["Rhodoleia championii", "CC BY-NC-SA", author, "Flower"])
            image_url = MYBIS + "/img/" + image_id
            assert image_url in fragment
            download({"id": "mybis_" + image_id, "source_species": "Rhodoleia championii",
                      "source_page": page, "image_url": image_url, "source_site": "Malaysia Biodiversity Information System (MyBIS)",
                      "license": "CC BY-NC-SA (version unspecified on item)", "license_url": None,
                      "license_evidence_url": gallery_url, "attribution": "© " + author + ", MyBIS SSN 3056, image " + image_id,
                      "author": author, "source_date": date, "source_group": "MyBIS:3056:" + author,
                      "organ": "flower_source_caption", "asset_kind": "source_page_image",
                      "source_evidence": {"species_id": 3056, "image_id": image_id, "caption": "Rhodoleia championii / Flower",
                                          "gallery_endpoint_discovered_from": page,
                                          "taxon_caution": "Malaysian taxon concept; requires specialist comparison with Hong Kong concept, no model-made label"},
                      "rights_scope": "Item explicitly CC BY-NC-SA; version unspecified. Local noncommercial candidate review only; no training or publication clearance asserted"})
    except Exception as error:
        errors.append({"source": "MyBIS", "error": str(error)})

    try:
        base = "https://www.treesandshrubsonline.org"
        net.get(base + "/about/licence/", "tso_licence.html")
        page = base + "/articles/keteleeria/keteleeria-fortunei/"
        html = net.get(page, "tso_Keteleeria_fortunei.html").text
        links = Links()
        links.feed(html)
        for item in links.images:
            url = urljoin(base, item.get("data-src", item.get("src", "")))
            if "/5949/" in url:
                skipped.append({"source_species": "Keteleeria fortunei", "source_page": page, "image_url": url,
                                "license": "CC BY-SA 4.0", "author": "John Anderson", "known_historical_url": url in history_urls,
                                "reason": "already_downloaded_previous_browser_batch; not a new independent source"})
        net.get(base + "/articles/rhodoleia/", "tso_Rhodoleia.html")
        skipped.append({"source_page": base + "/articles/rhodoleia/", "reason": "photos_captioned_Rhodoleia_henryi_not_championii"})
    except Exception as error:
        errors.append({"source": "TSO", "error": str(error)})

    skipped += [
        {"source_page": "https://jcra.ncsu.edu/resources/photographs/", "reason": "prior_permission_required_no_image_download", "promising_image_ids": [462055, 634441, 735825]},
        {"source_page": "https://hos.ifas.ufl.edu/woody/Pages/rhocha/rhocha.shtml", "reason": "live_page_and_rights_page_404; cached_search_not_enough_for_acquisition"},
        {"source_page": "https://phytokeys.pensoft.net/article/55778", "reason": "403_access_restricted_no_retry_or_bypass"},
        {"source_page": "https://www.naplesgarden.org/summer-plantings/", "reason": "third_party_Kaiyanwong223_CC_photo; exact_license_and_original_unresolved; not_independent_institution_photo"}]
    save(out / "manifest.json", rows)
    save(out / "skipped_sources.json", skipped)
    save(out / "request_audit.json", net.log)
    save(out / "summary.json", {"photos_downloaded": len(rows), "by_species": dict(Counter(r["source_species"] for r in rows)),
                               "historical_sha_hits": sum(r["historical_sha_match"] for r in rows),
                               "global_audit_pending": True, "training_eligible": 0, "errors": errors,
                               "history_cache_sha256": sha(cache_path.read_bytes()), "historical_browser_manifest_sha256": sha(old.read_bytes())})
    if rows:
        sheet = Image.new("RGB", (450 * min(4, len(rows)), 420), "#f4f4ef")
        draw = ImageDraw.Draw(sheet)
        for i, row in enumerate(rows[:4]):
            with Image.open(row["source_image"]) as im:
                image = ImageOps.exif_transpose(im).convert("RGB")
                image.thumbnail((430, 345))
                sheet.paste(image, (i * 450 + (450-image.width)//2, 10))
            draw.text((i * 450 + 10, 362), row["id"] + "\n" + row["source_species"] + "\n" + row["author"], fill="black")
        sheet.save(out / "contact_sheet.jpg", quality=90)
    report = "# Institutional candidate collection\n\n" + f"Downloaded {len(rows)} original source-page photos. All candidate quarantine, training_eligible=false. No training, deployment or human database changes.\n\n"
    report += "RBGV: per-image CC BY-NC-SA 4.0. MyBIS: per-image CC BY-NC-SA, version not stated; raw bytes and item-specific attribution retained. MyBIS photos are flower views, not whole trees; Malaysian naming concept needs botanical cross-check.\n\n"
    report += "TSO 3 oil-fir photos are known previous downloads, not counted as new. Rhodoleia genus photos are R. henryi, excluded. RBGV illustration excluded. JCRA requires permission; UF live URLs unavailable; Pensoft access restriction respected.\n\n"
    report += "Dedup here covers a recorded Sept17 metadata snapshot plus browser_v3, not all current data. Parent must run complete contemporary history/rejected/split isolation before use. No independent individual count or accepted labels are asserted.\n\n"
    report += "See manifest.json for source URL, license, attribution, SHA, pHash, dimensions, source group and source identification evidence; request_audit.json records HTTP and robots responses.\n"
    with (out / "REPORT.md").open("x", encoding="utf-8") as f:
        f.write(report)
    print(json.dumps({"out": str(out), "photos": len(rows), "errors": errors}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
