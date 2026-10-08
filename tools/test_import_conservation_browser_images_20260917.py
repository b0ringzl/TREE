"""Synthetic, offline fixtures only. No live review service or real data changes."""
import hashlib
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import import_conservation_browser_images_20260917 as importer


class BrowserImportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="sst_browser_import_fixture_")
        self.base = Path(self.temp.name)
        self.history = self.base / "history"
        self.history.mkdir()
        self.sources = self.base / "source"
        self.sources.mkdir()

    def tearDown(self):
        self.temp.cleanup()

    def make_image(self, name, seed, size=(500, 600)):
        # Unique spatial patterns, not just flat colours (which share a pHash).
        pixels = np.random.default_rng(seed).integers(20, 230, (17, 17, 3), dtype=np.uint8)
        image = Image.fromarray(pixels).resize(size)
        ImageDraw.Draw(image).text((10, 10), f"SYNTHETIC FIXTURE {seed}", fill="white")
        path = self.sources / name
        image.save(path)
        return path

    def record(self, path, number, **extras):
        return {"local_file": str(path), "species": "Artocarpus hypargyreus",
                "page_url": "https://example.test/species/shared-gallery",
                "image_url": f"https://example.test/source/{number}.png",
                "asset_kind": "source_page_image", "acquisition_method": "browser_page_asset", **extras}

    def run_import(self, records, name="out"):
        input_path = self.base / (name + "_source_records.json")
        input_path.write_text(json.dumps(records), encoding="utf-8")
        out = self.base / name
        result = importer.main(input_path, out, self.history)
        return result, out, input_path

    def test_new_copy_unknown_rights_quality_and_batch_dedup(self):
        first = self.make_image("first.png", 1)
        second = self.make_image("second.png", 2, (200, 300))
        records = [self.record(first, 1), self.record(first, 1), self.record(second, 2),
                   self.record(first, 3, asset_kind="screenshot")]
        original = first.read_bytes()
        result, out, _ = self.run_import(records)
        self.assertEqual(result["new_unique_source_images"], 2)
        self.assertEqual(result["excluded_duplicates"], 1)
        self.assertEqual(result["invalid_source_records"], 1)
        rows = json.loads((out / "review_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(Path(rows[0]["source_image"]).read_bytes(), original)
        self.assertEqual(first.read_bytes(), original)
        self.assertEqual(rows[1]["quality_group"], "low_resolution")
        self.assertIn("low_resolution", rows[1]["quality_flags"])
        self.assertTrue(all(r["rights_status"] == "rights_unverified" and not r["training_eligible"] for r in rows))
        self.assertEqual(len(list((out / "contact_sheets").glob("*.jpg"))), 1)

    def test_rejected_manifest_near_duplicate_and_observation_are_excluded(self):
        prior = self.make_image("prior.png", 5)
        with Image.open(prior) as image:
            h = f"{importer.phash(image):016x}"
            compressed = self.sources / "same_reencoded.jpg"
            image.save(compressed, quality=97)
        manifest = self.history / "rejected_manifest.json"
        manifest.write_text(json.dumps([{"source_image": str(prior), "status": "rejected", "sha256": importer.digest(prior),
                                         "phash": h, "observation_id": 12345}]), encoding="utf-8")
        before = importer.digest(manifest)
        other = self.make_image("new_same_observation.png", 8)
        result, out, _ = self.run_import([self.record(compressed, 4), self.record(other, 5, observation_id=12345)])
        self.assertEqual(result["new_unique_source_images"], 0)
        self.assertEqual(result["excluded_duplicates"], 2)
        rows = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        self.assertIn("historical_near_duplicate", rows[0]["duplicate_matches"])
        self.assertIn("historical_observation", rows[1]["duplicate_matches"])
        self.assertEqual(before, importer.digest(manifest))

    def test_rejected_database_sha_is_read_only_and_excluded(self):
        image = self.make_image("db_rejected.png", 22)
        database = self.history / "human_review.sqlite3"
        with sqlite3.connect(database) as conn:
            conn.execute("CREATE TABLE decisions (item_id TEXT, sha256 TEXT, decision TEXT)")
            conn.execute("INSERT INTO decisions VALUES (?,?,?)", ("old", importer.digest(image), "reject_source"))
        before = importer.digest(database)
        result, out, _ = self.run_import([self.record(image, 8)])
        self.assertEqual(result["excluded_duplicates"], 1)
        self.assertEqual(before, importer.digest(database))
        self.assertEqual(json.loads((out / "VERIFIED.json").read_text())["human_db_mutations"], 0)

    def test_prior_gallery_does_not_exclude_new_different_photo(self):
        prior, current = self.make_image("gallery_old.png", 43), self.make_image("gallery_new.png", 44)
        (self.history / "manifest.json").write_text(json.dumps([self.record(prior, 9)]))
        result, _, _ = self.run_import([self.record(current, 10)])
        self.assertEqual(result["new_unique_source_images"], 1)

    def test_output_exists_is_never_overwritten(self):
        source = self.make_image("new.png", 25)
        _, out, records = self.run_import([self.record(source, 9)])
        before = importer.digest(out / "manifest.json")
        with self.assertRaises(FileExistsError):
            importer.main(records, out, self.history)
        self.assertEqual(before, importer.digest(out / "manifest.json"))

    def test_bad_history_fails_closed_and_search_thumbnails_are_invalid(self):
        source = self.make_image("new.png", 29)
        (self.history / "bad_manifest.json").write_text("{BAD")
        records = self.base / "source_records.json"
        records.write_text(json.dumps([self.record(source, 7)]))
        with self.assertRaisesRegex(RuntimeError, "Historical index incomplete"):
            importer.main(records, self.base / "blocked", self.history)
        self.assertTrue((self.base / "blocked" / "BLOCKED.json").is_file())
        self.assertEqual(len(list((self.base / "blocked" / "images").iterdir())), 0)

    def test_invalid_thumbnail_license_and_species(self):
        source = self.make_image("new.png", 19)
        records = [self.record(source, 1, image_url="https://encrypted-tbn0.gstatic.com/images?q=fixture"),
                   self.record(source, 2, species="Other species"),
                   self.record(source, 3, asset_kind="thumbnail")]
        result, _, _ = self.run_import(records)
        self.assertEqual(result["invalid_source_records"], 3)
        self.assertEqual(result["new_unique_source_images"], 0)

    def test_current_run_qa_is_not_history_but_old_rejection_remains(self):
        incoming, rejected = self.make_image("current.png", 51), self.make_image("reject.png", 52)
        qa = self.history / "outputs" / "current_run_qa"
        qa.mkdir(parents=True)
        (qa / "image_index.json").write_text(json.dumps([self.record(incoming, 1)]))
        prior = self.history / "old_rejected_manifest.json"
        prior.write_text(json.dumps([{"source_image": str(rejected), "sha256": importer.digest(rejected), "status": "rejected"}]))
        before = importer.digest(prior)
        records = self.base / "inputs.json"
        records.write_text(json.dumps([self.record(incoming, 1), self.record(rejected, 2)]))
        out = self.base / "new_out"
        result = importer.main(records, out, self.history, exclude_current_run=[qa])
        self.assertEqual(result["new_unique_source_images"], 1)
        self.assertEqual(result["excluded_duplicates"], 1)
        audit = json.loads((out / "prior_index_audit.json").read_text())
        self.assertEqual(audit["current_run_context_exclusions"][0]["existing_files_at_index_start"][0]["sha256"], importer.digest(qa / "image_index.json"))
        self.assertEqual(before, importer.digest(prior))
        with self.assertRaises(ValueError):
            importer.main(records, self.base / "broad_exclusion", self.history, exclude_current_run=[self.history])


if __name__ == "__main__":
    unittest.main(verbosity=2)
