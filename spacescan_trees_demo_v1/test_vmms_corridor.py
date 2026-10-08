import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
import numpy as np
from PIL import Image

from . import corridor_identity, server, vmms_corridor


class VmmsCorridorTests(unittest.TestCase):
    @staticmethod
    def _tracking_frame(sample, candidates):
        rows = []
        for yaw, vector, box in candidates:
            embedding = np.asarray(vector, dtype=np.float32)
            embedding /= np.linalg.norm(embedding)
            rows.append({
                "relative_yaw": yaw, "embedding": embedding, "quality": .8,
                "score": .9, "box": box, "crop": Image.new("RGB", (16, 16), "green"),
            })
        return {"sample_index": sample, "chainage_m": float(sample), "frame_id": sample,
                "heading_deg": 0.0, "latitude": None, "longitude": None,
                "candidates": rows}

    def test_low_confidence_preview_boxes_are_yellow(self):
        self.assertEqual(
            vmms_corridor._preview_style(.299)["outline"],
            vmms_corridor.LOW_CONFIDENCE_BOX["outline"],
        )
        self.assertEqual(
            vmms_corridor._preview_style(.30)["outline"],
            vmms_corridor.STANDARD_CONFIDENCE_BOX["outline"],
        )

    def test_tsim_sha_tsui_catalog_and_200m_sampling(self):
        catalog = vmms_corridor.catalog()
        self.assertEqual(catalog["survey_id"], "2025-02-10_jianshazui")
        self.assertIn(catalog["default_segment_id"], {row["id"] for row in catalog["segments"]})
        samples = vmms_corridor.sample_route(catalog["default_segment_id"], 0, 200, 5)
        self.assertEqual(len(samples), 41)
        self.assertEqual(len({frame.frame_id for _, frame in samples}), 41)
        self.assertTrue(all(frame.source.is_file() for _, frame in samples))
        self.assertAlmostEqual(samples[0][0], 0, delta=.3)
        self.assertAlmostEqual(samples[-1][0], 200, delta=.6)
        gaps = [right[0] - left[0] for left, right in zip(samples, samples[1:])]
        self.assertTrue(all(4.4 <= gap <= 5.6 for gap in gaps))

    def test_page_and_catalog_endpoint(self):
        client = TestClient(server.app)
        self.assertEqual(client.get("/corridor").status_code, 200)
        response = client.get("/api/vmms/corridor/catalog")
        self.assertEqual(response.status_code, 200)
        self.assertGreaterEqual(len(response.json()["segments"]), 1)

    def test_selected_folder_supports_one_metre_sampling(self):
        catalog = vmms_corridor.catalog()
        row = next(item for item in catalog["segments"]
                   if item["stream_id"] == "jianshazui_folder_1")
        self.assertEqual(row["source_frames"], 438)
        self.assertEqual(row["start_frame"], 1514)
        self.assertEqual(row["end_frame"], 1951)
        self.assertEqual(row["repaired_coordinate_frames"], [1563, 1653, 1728])
        length = int(row["continuous_length_m"])
        samples = vmms_corridor.sample_route(row["id"], 0, length, 1)
        self.assertGreaterEqual(len(samples), 100)
        self.assertEqual(len(samples), len({frame.frame_id for _, frame in samples}))
        self.assertTrue(all("ladybug\\1" in str(frame.source) for _, frame in samples))
        gaps = [right[0] - left[0] for left, right in zip(samples, samples[1:])]
        self.assertTrue(all(.5 <= gap <= 1.5 for gap in gaps))

    def test_invalid_range_is_rejected_before_job_start(self):
        client = TestClient(server.app)
        segment_id = vmms_corridor.catalog()["default_segment_id"]
        response = client.post("/api/vmms/corridor/jobs", json={
            "segment_id": segment_id, "offset_m": 300, "length_m": 200, "spacing_m": 5,
        })
        self.assertEqual(response.status_code, 400)

    def test_user_folder_is_filtered_naturally_and_sampled_without_copying(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            for name in ("VMMS_Panoramic_000010_test.jpg", "VMMS_Panoramic_000002_test.jpg",
                         "VMMS_Panoramic_000001_test.jpg"):
                Image.new("RGB", (1024, 512), (40, 110, 70)).save(folder / name)
            Image.new("RGB", (800, 600), (40, 110, 70)).save(folder / "ordinary.jpg")
            (folder / "notes.txt").write_text("not an image", encoding="utf-8")

            report = vmms_corridor.public_folder_inspection(folder)
            self.assertEqual(report["valid_image_count"], 3)
            self.assertEqual(report["rejected_image_count"], 1)
            self.assertEqual(report["non_image_count"], 1)
            self.assertEqual(report["first_file"], "VMMS_Panoramic_000001_test.jpg")
            self.assertEqual(report["image_names"], ["VMMS_Panoramic_000001_test.jpg",
                "VMMS_Panoramic_000002_test.jpg", "VMMS_Panoramic_000010_test.jpg"])
            samples, _ = vmms_corridor.sample_panorama_folder(folder, 0, 2, 2)
            self.assertEqual([frame.frame_id for _, frame in samples], [1, 10])
            self.assertTrue(all(frame.source.parent == folder.resolve() for _, frame in samples))

    def test_selected_consecutive_middle_range_and_endpoint_names(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            for i in range(12):
                Image.new("RGB", (512, 256), "green").save(folder / f"pano_{i:03d}.jpg")
            samples, inspection = vmms_corridor.sample_panorama_folder(folder, 4, 1, 5)
            self.assertEqual([int(position) + 1 for position, _ in samples], [5,6,7,8,9])
            self.assertEqual([frame.source.name for _, frame in samples], inspection["image_names"][4:9])
            self.assertEqual(len(list(folder.iterdir())),12)

    def test_api_transmits_selected_300_image_range_without_starting_test_job(self):
        with patch.object(server.CORRIDOR_JOBS, "create_folder", return_value="a"*32) as create:
            response = TestClient(server.app).post("/api/vmms/corridor/jobs", json={
                "source_mode":"folder", "folder":"C:/vmms", "start_index":300,
                "frame_step":1,"max_frames":300,
            })
            self.assertEqual(response.status_code,200)
            self.assertEqual(create.call_args.args[:4],("C:/vmms",300,1,300))

    def test_more_than_300_and_one_image_ranges_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            for i in range(3):
                Image.new("RGB", (512,256), "green").save(folder/f"{i}.jpg")
            for kwargs in ({"max_frames":301},{"max_frames":1},{"start_index":2}):
                with self.assertRaises(ValueError):
                    vmms_corridor.sample_panorama_folder(folder,**kwargs)

    def test_folder_inspection_endpoint_rejects_non_panorama_folder(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            Image.new("RGB", (800, 600), "green").save(folder / "ordinary.jpg")
            response = TestClient(server.app).post(
                "/api/vmms/corridor/folder/inspect", json={"folder": str(folder)})
            self.assertEqual(response.status_code, 400)

    def test_tracker_reconnects_same_tree_after_short_occlusion(self):
        box = [.48, .2, .58, .8]
        frames = [self._tracking_frame(index, [(12, [1, 0], box)])
                  for index in (0, 1, 9, 10)]
        tracks = vmms_corridor._track_candidates(frames)
        self.assertEqual(len(tracks), 1)
        self.assertEqual(len(tracks[0]["observations"]), 4)
        self.assertEqual(tracks[0]["merged_fragment_count"], 1)

    def test_tracker_does_not_merge_simultaneous_distant_trees(self):
        frames = [self._tracking_frame(index, [
            (0, [1, 0], [.47, .2, .53, .8]),
            (60, [.995, .05], [.64, .2, .70, .8]),
        ]) for index in (0, 1, 2)]
        tracks = vmms_corridor._track_candidates(frames)
        self.assertEqual(len(tracks), 2)
        self.assertTrue(all(len(track["observations"]) == 3 for track in tracks))

    def test_tracker_keeps_weakly_overlapping_crowns_separate(self):
        frames = [self._tracking_frame(index, [
            (4, [1, 0], [.49, .15, .54, .75]),
            (8, [.999, .02], [.53, .28, .58, .86]),
        ]) for index in (0, 1, 2)]
        tracks = vmms_corridor._track_candidates(frames)
        self.assertEqual(len(tracks), 2)
        self.assertTrue(all(track["merged_fragment_count"] == 0 for track in tracks))

    def test_tracker_keeps_close_but_separate_simultaneous_trees(self):
        frames = [self._tracking_frame(index, [
            (4, [1, 0], [.49, .15, .52, .75]),
            (12, [.999, .02], [.53, .15, .56, .75]),
        ]) for index in (0, 1, 2)]
        tracks = vmms_corridor._track_candidates(frames)
        self.assertEqual(len(tracks), 2)
        self.assertTrue(all(track["merged_fragment_count"] == 0 for track in tracks))

    def test_frame_assignment_preserves_both_targets_when_greedy_would_drop_one(self):
        pairs = [(.95, 0, 0), (.93, 0, 1), (.94, 1, 0)]
        self.assertEqual(set(vmms_corridor._optimal_pairs(pairs, 2, 2)), {(0, 1), (1, 0)})

    def test_non_tree_centre_is_not_rescued_by_greenery_at_crop_edge(self):
        self.assertFalse(vmms_corridor._usable_tree_crop(
            {"keep": True, "background_minus_tree": [-.0133, .0211]}))
        self.assertTrue(vmms_corridor._usable_tree_crop(
            {"keep": True, "background_minus_tree": [-.0149, .0047]}))

    def test_identical_appearance_does_not_rescue_vertical_object_switch(self):
        frames = [self._tracking_frame(i, [(0, [1, 0], [.48, .35, .54, .55])]) for i in (0, 1)]
        frames += [self._tracking_frame(i, [(0, [1, 0], [.48, .10, .54, .28])]) for i in (2, 3)]
        tracks = vmms_corridor._track_candidates(frames)
        self.assertEqual(len(tracks), 2)
        self.assertTrue(all(len(track["observations"]) == 2 for track in tracks))

    def test_identical_appearance_does_not_rescue_large_horizontal_jump(self):
        frames = [self._tracking_frame(i, [(0, [1, 0], [.48, .2, .54, .7])]) for i in (0, 1)]
        frames += [self._tracking_frame(i, [(50, [1, 0], [.64, .2, .70, .7])]) for i in (2, 3)]
        self.assertEqual(len(vmms_corridor._track_candidates(frames)), 2)

    def test_appearance_anchor_prevents_chaining_between_different_objects(self):
        box = [.48, .2, .54, .7]
        frames = [self._tracking_frame(i, [(0, [np.cos(a), np.sin(a)], box)])
                  for i, a in enumerate((0, .3, .6, .9, 1.2))]
        tracks = vmms_corridor._track_candidates(frames)
        self.assertEqual(len(tracks), 2)
        self.assertTrue(all(len(track["observations"]) <= 3 for track in tracks))

    def test_view_selection_does_not_average_semantically_different_crops(self):
        observations = []
        for i, vector in enumerate(([1, 0], [.7, .714], [0, 1])):
            row = self._tracking_frame(i, [(i * 20, vector, [.48, .2, .54, .7])])["candidates"][0]
            row.update(chainage_m=i * 5, sample_index=i)
            observations.append(row)
        selected = vmms_corridor._select_views({"observations": observations})
        self.assertEqual(len(selected), 1)

    def test_dark_camera_hardware_crop_is_not_a_useful_tree_view(self):
        pixels = np.zeros((100, 100, 3), dtype=np.uint8)
        pixels[:, :20] = [40, 130, 50]
        self.assertFalse(corridor_identity.crop_evidence(Image.fromarray(pixels))["usable"])
        self.assertTrue(corridor_identity.crop_evidence(Image.new("RGB", (100, 100), (40, 130, 50)))["usable"])

    def test_periodic_local_flow_predicts_across_panorama_seam(self):
        points = [[.99, y, 1.01, y] for y in (.30, .35, .40, .45, .50)]
        shift = corridor_identity._local_shift([.97, .2, 1.02, .7], {"points": points})
        self.assertIsNotNone(shift)
        self.assertAlmostEqual(shift[0], .02, places=5)

    def test_valid_local_flow_can_follow_large_camera_displacement(self):
        first = self._tracking_frame(0, [(0, [1, 0], [.48, .2, .54, .7])])
        second = self._tracking_frame(1, [(22, [1, 0], [.54, .2, .60, .7])])
        second["motion"] = {"points": [[.50, y, .56, y] for y in (.3, .35, .4, .45, .5)]}
        tracks = vmms_corridor._track_candidates([first, second])
        self.assertEqual(len(tracks), 1)
        self.assertEqual(len(tracks[0]["observations"]), 2)


if __name__ == "__main__":
    unittest.main()
