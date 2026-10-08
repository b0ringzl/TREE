"""Small, conservative, metadata-only audit of official Taiwan plant sources.

No training, model operations, human-review edits, or image downloads.  This run
found no target ecological photograph with a verified suitable image license.
Exact species identity, robots restrictions, and lack of rights are recorded
rather than treating all government/university pictures as freely licensed.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import time
from collections import Counter
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import quote, urljoin, urlsplit
from urllib.robotparser import RobotFileParser

import requests


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "outputs/conservation_authoritative_collection_20260918_v1/taiwan"
TARGETS = ["Aquilaria sinensis", "Artocarpus hypargyreus", "Rhodoleia championii", "Keteleeria fortunei"]
AGENT = "SpaceScanTreesResearch/1.0 (metadata-only botanical-source audit)"


class PageParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.images, self.links, self.texts = [], [], []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in ("script", "style"):
            self.skip += 1
        if tag == "img":
            self.images.append(attrs)
        if tag == "a" and attrs.get("href"):
            self.links.append(attrs["href"])

    def handle_endtag(self, tag):
        if tag in ("script", "style") and self.skip:
            self.skip -= 1

    def handle_data(self, value):
        if not self.skip and value.strip():
            self.texts.append(value.strip())


class AuditClient:
    def __init__(self, out):
        self.out = out
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": AGENT})
        self.robots, self.last, self.pages = {}, {}, []

    def check_robots(self, origin):
        if origin in self.robots:
            return self.robots[origin]
        url = origin + "/robots.txt"
        entry = {"url": url, "fetched_at": datetime.now(timezone.utc).isoformat()}
        try:
            r = self.session.get(url, timeout=20)
            entry.update(status_code=r.status_code, final_url=r.url)
            if r.status_code == 200:
                body = r.content.decode("utf-8-sig", errors="replace")
                entry["body"] = body
                parser = RobotFileParser(url)
                parser.parse(body.splitlines())
                entry["parser"] = parser
                # Some sites repeat User-agent: * groups. urllib may stop at the
                # first group, so respect the maximum explicitly posted delay.
                explicit_delays = re.findall(r"(?im)^\s*crawl-delay\s*:\s*([0-9.]+)", body)
                entry["crawl_delay_seconds"] = max([1.0, float(parser.crawl_delay("*") or 1)] + [float(x) for x in explicit_delays])
                entry["state"] = "parsed"
            elif r.status_code in (404, 410):
                entry.update(state="no_robots_file", crawl_delay_seconds=1.0)
            else:
                entry["state"] = "unverified_stop"
        except requests.RequestException as exc:
            entry.update(state="unverified_stop", error=str(exc))
        self.robots[origin] = entry
        self.last[origin] = time.monotonic()
        return entry

    def fetch(self, url, purpose):
        parts = urlsplit(url)
        origin = parts.scheme + "://" + parts.netloc
        rules = self.check_robots(origin)
        entry = {"url": url, "purpose": purpose, "fetched_at": datetime.now(timezone.utc).isoformat()}
        if rules["state"] == "unverified_stop":
            entry.update(state="skipped_robots_unverified")
            self.pages.append(entry)
            return None
        if rules.get("parser") and not rules["parser"].can_fetch(AGENT, url):
            entry.update(state="skipped_robots_disallowed")
            self.pages.append(entry)
            return None
        delay = rules.get("crawl_delay_seconds", 1)
        time.sleep(max(0, delay - (time.monotonic() - self.last.get(origin, 0))))
        try:
            r = self.session.get(url, timeout=20)
            self.last[origin] = time.monotonic()
            entry.update(status_code=r.status_code, final_url=r.url,
                         response_sha256=hashlib.sha256(r.content).hexdigest(),
                         response_bytes=len(r.content), content_type=r.headers.get("Content-Type"))
            if r.status_code != 200:
                entry["state"] = "http_error"
                self.pages.append(entry)
                return None
            if "image/" in r.headers.get("Content-Type", ""):
                raise RuntimeError("Metadata-only collector must not be pointed at an image URL")
            body = r.content.decode("utf-8-sig", errors="replace")
            p = PageParser()
            p.feed(body)
            entry["state"] = "metadata_fetched"
            entry["links"] = [urljoin(r.url, u) for u in p.links]
            entry["text_sha256"] = hashlib.sha256(" ".join(p.texts).encode("utf-8")).hexdigest()
            # Limited factual excerpts, not full copied articles or image pixels.
            entry["target_mentions"] = {name: name.lower() in body.lower() for name in TARGETS}
            entry["no_search_result"] = any(s in " ".join(p.texts) for s in ("沒有搜尋結果", "找不到符合條件"))
            self.pages.append(entry)
            print(purpose + ": " + str(r.status_code), flush=True)
            return {"html": body, "parser": p, "evidence": entry}
        except requests.RequestException as exc:
            self.last[origin] = time.monotonic()
            entry.update(state="connection_failed", error=str(exc))
            self.pages.append(entry)
            return None


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit("Refuse to overwrite a prior collection. Choose a new versioned --out.")
    out.mkdir(parents=True, exist_ok=True)
    client = AuditClient(out)
    collected = datetime.now(timezone.utc).isoformat()

    # Official site policies; no inference from institution identity to image rights.
    client.fetch("https://tai2.ntu.edu.tw/about", "NTU official usage policy")
    client.fetch("https://taicol.tw/en-us/policy", "TaiCOL text-data policy")
    client.fetch("https://taicol.tw/en-us/api", "TaiCOL public API documentation landing page")
    client.fetch("https://www.tbn.org.tw/about/terms", "TBN official data license")
    client.fetch("https://www.tbn.org.tw/about/OGDL-1.0", "TBN government open-data license")
    client.fetch("https://tpbg.tfri.gov.tw/js/vue/components/PageFooter.js", "TPBG official copyright notice")
    client.fetch("https://tpbg.tfri.gov.tw/js/vue/config.js", "TPBG public search API host discovery")
    client.fetch("https://portal.taibif.tw/", "TaiBIF official portal availability")
    # This host is explicitly referenced by the TPBG site's public Vue config.
    client.fetch("https://bgip.tfri.gov.tw/tpbg2012/API/plant/Search.php?keyword=Rhodoleia&p=1", "TPBG official search, subject to API-host robots")

    sources = []
    for name in TARGETS:
        page = client.fetch("https://tai2.ntu.edu.tw/search/1/" + quote(name), "NTU exact-name search " + name)
        sources.append({"institution": "National Taiwan University / Plants of Taiwan", "scientific_name": name,
                        "source_page": "https://tai2.ntu.edu.tw/search/1/" + quote(name),
                        "search_state": None if page is None else page["evidence"]["state"],
                        "server_no_result_message": None if page is None else page["evidence"]["no_search_result"],
                        "conclusion": "Search result is not proof of taxon absence; direct indexed records checked separately."})

    # An indexed, exact-label NTUF herbarium record page, explicitly non-Taiwanese.
    species_url = "https://tai2.ntu.edu.tw/species/999%20990%20323%2033"
    aq = client.fetch(species_url, "NTU Aquilaria sinensis exact indexed species page")
    rows = []
    collectors = {"F00003777": "H. Fung", "F00003778": "Tsang, Wai-Tak, L.U.", "F00003779": "C. I. Lei",
                  "F00003780": "F. A. McClure", "F00003845": "Lau S. K."}
    if aq:
        for image in aq["parser"].images:
            src = image.get("src", "")
            match = re.search(r"(F\d{8})\.jpg$", src)
            if not match or "/NTUFimage/" not in src:
                continue
            record_id = match.group(1)
            rows.append({
                "id": "NTUF_" + record_id, "record_id": record_id,
                "scientific_name": "Aquilaria sinensis", "source_scientific_name": "Aquilaria sinensis (Lour.) Spreng.",
                "vernacular_name_in_source": "牙香樹", "institution": "National Taiwan University / NTUF",
                "source": "NTUF_PlantsofTaiwan", "source_page": species_url,
                "image_url": urljoin(species_url, src), "image_url_kind": "public_page_specimen_thumbnail",
                "original_image_url": None, "original_image_url_reason": "Not exposed by inspected page; no guessed larger URL",
                "creator": None, "collector": collectors.get(record_id),
                "creator_note": "Specimen collector is not automatically image photographer/copyright holder.",
                "license": None, "license_url": "https://tai2.ntu.edu.tw/about",
                "rights_status": "metadata_only_image_reuse_not_explicitly_cleared",
                "rights_note": "Site states academic attribution and personal/noncommercial use; no explicit image training license verified.",
                "identification_basis": "Official NTU species page assigns NTUF specimen to exact species; per-specimen determiner not inspected.",
                "label_authority": "institutional_species_page_not_model_or_human_project_review",
                "media_type": "herbarium_specimen", "organ": "not_visually_reviewed",
                "image_path": None, "sha256": None, "width": None, "height": None,
                "download_status": "not_downloaded_rights_unresolved", "candidate_status": "metadata_only",
                "source_group": "NTUF_Aquilaria_sinensis_species_gallery", "independent_tree_count": None,
                "human_review_status": "not_requested", "training_eligible": False,
                "collected_at": collected,
            })

    tbn_taxa = [
        ("Rhodoleia", "https://www.tbn.org.tw/taxa/748615e1-e2d0-4d1a-8032-13504de04cc4", "genus_only_not_Rhodoleia_championii"),
        ("Keteleeria fortunei", "https://www.tbn.org.tw/taxa/81b42dc8-f001-4f3c-bfad-a00e961624bf", "exact_species_taxonomy_no_target_media_found"),
        ("Keteleeria fortunei var. cyclolepis", "https://www.tbn.org.tw/taxa/5c2e32df-f03c-422c-a0d9-645cfa966962", "excluded_infraspecific_not_automatically_merged"),
        ("Keteleeria fortunei var. oblonga", "https://www.tbn.org.tw/taxa/e2bfb4c8-e1be-4d8a-b5c8-646eb302f153", "excluded_infraspecific_not_automatically_merged"),
    ]
    taxonomy = []
    for name, url, decision in tbn_taxa:
        page = client.fetch(url, "TBN taxonomic boundary " + name)
        taxonomy.append({"name": name, "source_page": url, "record_id": url.rsplit("/", 1)[-1],
                         "decision": decision, "state": page["evidence"]["state"] if page else "unavailable",
                         "training_eligible": False})
    for name in TARGETS[:3]:
        url = "https://www.tbn.org.tw/taxasearch/result?name=" + quote(name)
        page = client.fetch(url, "TBN exact-name search " + name)
        sources.append({"institution": "Taiwan Biodiversity Research Institute / TBN", "scientific_name": name,
                        "source_page": url, "server_no_result_message": None if page is None else page["evidence"]["no_search_result"],
                        "conclusion": "No exact target record returned by this query; not a claim that the entire taxon is absent."})

    # Exact names matter more than misleading common names.
    for name, url in [
        ("Excoecaria agallocha", "https://tai2.ntu.edu.tw/species/415%20016%2001%200"),
        ("Excoecaria formosana", "https://tai2.ntu.edu.tw/species/415%20016%2004%200"),
    ]:
        client.fetch(url, "NTU common-name collision exclusion " + name)
        taxonomy.append({"name": name, "source_page": url, "decision": "excluded_not_Aquilaria_sinensis_despite_TuChenXiang_common_name", "training_eligible": False})

    policy = {
        "no_training_or_deployment": True, "downloaded_image_count": 0,
        "NTU": {"policy_url": "https://tai2.ntu.edu.tw/about", "decision": "metadata_only_no_explicit_suitable_image_license_verified"},
        "TaiCOL": {"policy_url": "https://taicol.tw/en-us/policy", "text_license": "CC BY 4.0", "decision": "Text license is not an image license; robots-disallowed catalogue/taxon/static/media paths not fetched."},
        "TBN": {"policy_url": "https://www.tbn.org.tw/about/terms", "default_license": "Open Government Data License 1.0 unless otherwise stated", "decision": "No exact target ecological photo with verified per-image provenance found; do not reuse images of varieties or genus-only records."},
        "TPBG": {"policy_url": "https://tpbg.tfri.gov.tw/js/vue/components/PageFooter.js", "decision": "All-rights-reserved notice; API-host robots could not be verified, search collection stopped."},
        "TaiBIF": {"source_page": "https://portal.taibif.tw/", "decision": "Portal robots explicitly disallows all generic crawlers; no bypass, no bot impersonation, and no collection."},
    }
    robots_serializable = {k: {a: b for a, b in v.items() if a != "parser"} for k, v in client.robots.items()}
    write_json(out / "manifest.json", {"schema_version": 1, "collected_at": collected,
               "targets": TARGETS, "rows": rows, "downloaded_image_count": 0,
               "training_eligible_count": 0, "scope": "official Taiwan source investigation; metadata-only findings, not a training manifest"})
    write_json(out / "robots_checks.json", robots_serializable)
    write_json(out / "source_requests.json", client.pages)
    write_json(out / "source_searches.json", sources)
    write_json(out / "taxonomy_exclusions.json", taxonomy)
    write_json(out / "rights_audit.json", policy)
    report = """# Taiwan official-source collection audit

No new training images were downloaded or admitted. The result is **5 metadata-only NTUF herbarium records**, not 5 usable photographs, trees, or independent test cases. No weights, deployed files, historical manifests, or review decisions were changed.

| Target | Verified finding | This collection |
|---|---|---|
| Aquilaria sinensis | NTU exact species page lists 5 NTUF specimens; its digital-photograph section is not populated | 5 specimen-thumbnail metadata records; no image download |
| Artocarpus hypargyreus | No exact record returned by the inspected NTU/TBN queries | 0 photos; database-wide absence is not established |
| Rhodoleia championii | TBN has a Rhodoleia genus record, not an exact identified target photograph | 0 photos; genus-only identity not promoted to species |
| Keteleeria fortunei | TBN exact species record is present, plus separate cyclolepis/oblonga variety records | 0 photos; variants not silently merged |

## Sources and restrictions

- [NTU Aquilaria sinensis](https://tai2.ntu.edu.tw/species/999%20990%20323%2033): F00003777, F00003778, F00003779, F00003780, F00003845. Source lists collectors; these are not assumed to be photographers. The inspected page supplies thumbnail URLs; no unobserved original URL was fabricated.
- [NTU usage policy](https://tai2.ntu.edu.tw/about): attribution and academic/noncommercial conditions are stated, but this audit did not establish an explicit suitable image-training license. Metadata only, pending permission/clarification. Even licensed specimens would be a different visual domain from street-tree photographs.
- [TaiCOL policy](https://taicol.tw/en-us/policy): text data are CC BY 4.0; this is not a blanket image license. Its robots restrictions disallow catalogue/taxon/static/media paths. They were not crawled.
- [TBN Keteleeria fortunei](https://www.tbn.org.tw/taxa/81b42dc8-f001-4f3c-bfad-a00e961624bf) and [Rhodoleia genus](https://www.tbn.org.tw/taxa/748615e1-e2d0-4d1a-8032-13504de04cc4): useful taxonomic boundaries, but no exact target ecological-photo batch established here. [TBN terms](https://www.tbn.org.tw/about/terms) give default OGDL-1.0 unless otherwise specified. The occurrence and download paths disallowed by robots were not fetched.
- [Taipei Botanical Garden database](https://tpbg.tfri.gov.tw/PlantList.html): its public page uses an API on bgip.tfri.gov.tw. API-host robots failed to load; no API or imagery was crawled afterward. The site footer says all rights reserved, not a verified open image grant.
- [TaiBIF](https://portal.taibif.tw/): the final robots fetch succeeded and explicitly blocks generic crawlers. Collection stopped; no impersonation of an allowed search-engine bot. This is an access-policy limitation, not proof that it has no target data.

## Taxonomic exclusions

Taiwan common names must not become labels: NTU's [土沉香](https://tai2.ntu.edu.tw/species/415%20016%2001%200) is Excoecaria agallocha, and its [台灣土沉香](https://tai2.ntu.edu.tw/species/415%20016%2004%200) is Excoecaria formosana, neither the target Aquilaria sinensis. Likewise Keteleeria davidiana (including Taiwan var. formosana) is not K. fortunei. K. fortunei var. cyclolepis/oblonga require a separately agreed taxonomic mapping; they are not part of this target batch.

## Reuse / next action

manifest.json is metadata-only and has training_eligible=false on every record. Obtain explicit suitable image permission from NTU if those specimens are needed; for street imagery prioritize source-identified living trees from other institutions. Do not count records, views, or duplicate specimens as independent trees. New licensed ecological images must still undergo historical deduplication and real human-review exclusion checks before training. No content-based image hash or dimensions are invented for undownloaded assets.

Machine-readable request states, robots evidence, rights decisions and exclusions are stored alongside this report. This audit is bounded to the inspected sources and does not claim an exhaustive absence of official photographs anywhere in Taiwan.
"""
    (out / "REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps({"out": str(out), "metadata_rows": len(rows), "downloaded_images": 0,
                      "request_states": dict(Counter(x["state"] for x in client.pages))}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
