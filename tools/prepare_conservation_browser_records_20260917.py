"""Prepare records from eight real browser pageAssets bundles, offline only."""
import argparse
import json
from pathlib import Path

from import_conservation_browser_images_20260917 import ROOT, digest, save

ASSET_BASE = Path("C:/Users/250417~1/AppData/Local/Temp/browser-use/assets")
HK_PAGE = "https://www.herbarium.gov.hk/en/hk-plant-database/plant-detail/index.html?pType=species&oID="
HK_TERMS = "https://www.herbarium.gov.hk/en/important-notices/index.html"
GREEN_PAGE = "https://www.greening.gov.hk/en/community-outreach/qrcode-tree-labels/index_id_"
GREEN_TERMS = "https://www.greening.gov.hk/en/important-notice/index.html"
GREEN_NOTICE = """Source: https://www.greening.gov.hk/en/important-notice/index.html
© 2021 Development Bureau. All rights reserved.

Copyright Notices:
The content available on this website, including but not limited to all text, graphics, drawings, diagrams, photographs and compilation of data or other materials are subject to copyright owned by the Government of the Hong Kong Special Administrative Region or other entities. Except as expressly permitted herein or where prior written authorization is obtained from the Development Bureau - Works Branch, any reproduction, adaptation, distribution, dissemination or making available of such copyright works to the public is strictly prohibited. Permission is granted for users to download the materials herein to store them in local computers, provided that this is solely for personal or non-commercial internal use, and provided further that this copyright notice is downloaded at the same time. Users should note that this permission only applies to Government copyright materials. Where third party copyrights are involved, an appropriate notice will appear in this website.

Project note: These source-page images remain quarantined. No ML-training or public-Demo authorization is inferred from local-download permission. Keep originals and watermarks unchanged.
"""
BATCHES = [
    ("b88b56a9-cb8a-4a6b-b486-c95a99f8e1eb", "Artocarpus hypargyreus", 13, "HKHerbarium", HK_PAGE + "9207"),
    ("f0e196b2-a5a9-4990-963f-40ebca45c676", "Artocarpus hypargyreus", 7, "IISc", "https://indiaflora-ces.iisc.ac.in/herbsheet.php?cat=13&id=1045"),
    ("d87718c0-c25a-4ec4-b394-1dfb2b9a032b", "Rhodoleia championii", 14, "HKHerbarium", HK_PAGE + "5131"),
    ("af182956-ade2-4f94-9d95-57f450b21171", "Keteleeria fortunei", 16, "HKHerbarium", HK_PAGE + "9347"),
    ("21c7deed-31d9-41e5-a861-919993de586e", "Keteleeria fortunei", 3, "TreesAndShrubsOnline", "https://www.treesandshrubsonline.org/articles/keteleeria/keteleeria-fortunei/"),
    ("a8d8edfc-c64f-4c94-95b7-80afde863d65", "Artocarpus hypargyreus", 5, "Greening", GREEN_PAGE + "184.html"),
    ("cc5c150f-534a-438b-92b9-9c3f86800cb1", "Rhodoleia championii", 5, "Greening", GREEN_PAGE + "1193.html"),
    ("041ccc59-01f7-450d-b6ab-7001e08ce4b7", "Rhodoleia championii", 3, "Pukeiti", "https://pukeiti.gardenexplorer.org/taxon-148030.aspx"),
]


def prepare(out):
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    records, batches = [], []
    for uuid, species, expected, source, page in BATCHES:
        manifest = ASSET_BASE / uuid / "manifest.json"
        data = json.loads(manifest.read_text(encoding="utf-8"))
        if data.get("failures") or len(data["assets"]) != expected:
            raise ValueError(f"Unexpected pageAssets count/failure: {uuid}")
        batches.append({"bundle_manifest": str(manifest.resolve()), "sha256": digest(manifest), "source": source,
                        "species": species, "assets": expected, "page_url": page})
        for asset in data["assets"]:
            file = Path(asset["path"]).resolve(strict=True)
            if asset.get("kind") != "image" or not file.is_file():
                raise ValueError("Bundle asset is not an image file")
            terms = {"reviewer": "root browser inspection 2026-09-17", "ml_training_rights": "not_established",
                     "public_demo_distribution_rights": "not_granted_by_this_import"}
            license, license_url, attribution, rights_status = "unknown", None, source, "rights_unverified"
            if source == "HKHerbarium":
                license = "Source-specific non-commercial reproduction terms; ML use not established"
                attribution = "Hong Kong Herbarium / Agriculture, Fisheries and Conservation Department"
                terms.update(terms_url=HK_TERMS, summary="Non-commercial reproduction requires source attribution and preservation of watermarked originals; no ML permission inferred.")
            elif source == "IISc":
                license, license_url, attribution = "CC BY-NC-SA 4.0", "https://creativecommons.org/licenses/by-nc-sa/4.0/", "K. Sankara Rao / India Flora Online, IISc"
                terms.update(terms_url=page, summary="Source page states non-commercial share-alike license; do not treat as unrestricted training/public-demo permission.")
            elif source == "TreesAndShrubsOnline":
                license, license_url, attribution = "CC BY-SA 4.0", "https://creativecommons.org/licenses/by-sa/4.0/", "John Anderson / Trees and Shrubs Online"
                rights_status = "verified_open_license"
                terms.update(terms_url="https://www.treesandshrubsonline.org/about/licence/", location="Hillier Gardens",
                             captions="Gallery includes 2008 foliage, 2011 tree and 2012 bark. Per-image organ/date not assigned without further checking.")
            elif source == "Greening":
                license, attribution = "Personal or non-commercial internal download with required copyright notice", "© 2021 Development Bureau. All rights reserved."
                terms.update(terms_url=GREEN_TERMS, required_notice_file="COPYRIGHT_GREening.txt",
                             summary="Local download for personal/non-commercial internal use only; applies only to government-copyright material; keep the copyright notice. ML use remains unverified.")
            else:
                terms.update(terms_url=page, summary="Image rights not verified; quarantine only.")
            stem = Path(asset.get("name", "")).stem.lower()
            organ = {"overview": "whole_tree", "leaf": "leaf", "flower": "flower", "fruit": "fruit", "bark": "bark"}.get(stem, "not_reviewed") if source == "Greening" else "not_reviewed"
            group = "source_gallery:" + source + ":" + species.replace(" ", "_")
            records.append({"local_file": str(file), "species": species, "page_url": page, "image_url": asset["url"],
                "asset_kind": "source_page_image", "acquisition_method": "browser_page_asset", "organ": organ,
                "license": license, "license_url": license_url, "rights_status": rights_status, "attribution": attribution,
                "source_group": group, "group_id": group, "identity_status": "Unknown individual identity; gallery is a conservative grouping proxy, not independent trees.",
                "evidence": {"source_site": source, "source_species_assertion": species, "bundle_manifest": str(manifest.resolve()),
                             "bundle_asset_id": asset.get("id"), "source_filename": asset.get("name"), "rights_evidence": terms},
                "training_eligible": False})
    save(out / "source_records.json", records)
    save(out / "browser_bundles_audit.json", {"batches": batches, "actual_downloaded_assets": len(records),
        "network_requests_by_builder": 0, "training_performed": False, "human_accepted": 0})
    with (out / "COPYRIGHT_GREening.txt").open("x", encoding="utf-8") as stream:
        stream.write(GREEN_NOTICE)
    print(json.dumps({"records": len(records), "path": str(out / "source_records.json")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/conservation_browser_collection_20260917_v1")
    prepare(parser.parse_args().out)
