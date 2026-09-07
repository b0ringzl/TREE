from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path

from app.frame_manager import (
    ALSTONIA_LABEL,
    BANYAN_LABEL,
    CRAPE_MYRTLE_LABEL,
    FOXTAIL_PALM_LABEL,
    NORFOLK_ISLAND_PINE_LABEL,
    TRAVELLERS_PALM_LABEL,
    FrameAnnotationManager,
    PanoramaFrame,
    normalize_species_name,
)
from app.schemas import FrameAnnotationRequest, FrameTreeLabel


def frame(frame_id: str, distance: float, stream_id: str = "test_stream") -> PanoramaFrame:
    return PanoramaFrame(
        frame_key=f"{stream_id}__{frame_id}",
        stream_id=stream_id,
        frame_id=frame_id,
        seq_id=int(frame_id),
        utc_datetime="2023-10-27T00:00:00Z",
        hong_kong_datetime="2023-10-27T08:00:00+08:00",
        source_image_relpath=f"{frame_id}.jpg",
        easting=0.0,
        northing=0.0,
        latitude=22.3,
        longitude=114.1,
        heading_deg=0.0,
        route_distance_m=distance,
    )


class FrameManagerTests(unittest.TestCase):
    def test_corrected_foxtail_palm_label_is_normalized(self) -> None:
        self.assertEqual(
            normalize_species_name("Archontophoenix alexandrae 假檳榔"),
            FOXTAIL_PALM_LABEL,
        )

    def test_corrected_norfolk_island_pine_label_is_normalized(self) -> None:
        self.assertEqual(
            normalize_species_name("Araucaria heterophylla异叶南阳杉"),
            NORFOLK_ISLAND_PINE_LABEL,
        )
        self.assertEqual(
            normalize_species_name("Araucaria heterophylla异叶南洋杉"),
            NORFOLK_ISLAND_PINE_LABEL,
        )
        self.assertEqual(
            normalize_species_name("Archontophoenix alexandrae 假槟榔"),
            FOXTAIL_PALM_LABEL,
        )

    def test_note_species_labels_are_normalized(self) -> None:
        self.assertEqual(normalize_species_name("Alstonia scholaris"), ALSTONIA_LABEL)
        self.assertEqual(normalize_species_name("紫薇"), CRAPE_MYRTLE_LABEL)
        self.assertEqual(normalize_species_name("旅人蕉"), TRAVELLERS_PALM_LABEL)
        self.assertEqual(normalize_species_name("杉树"), "杉树")

    def test_fine_leaf_and_weeping_figs_share_one_banyan_label(self) -> None:
        for name in (
            "Ficus microcarpa",
            "Ficus microcarpa 榕樹(細葉榕)",
            "细叶榕",
            "Ficus benjamina",
            "Ficus benjamina 垂葉榕",
            "垂叶榕",
        ):
            self.assertEqual(normalize_species_name(name), BANYAN_LABEL)

    def test_frame_label_accepts_model_confidence(self) -> None:
        label = FrameTreeLabel(
            label_id="tree-confidence",
            species=BANYAN_LABEL,
            points=[(0.1, 0.2), (0.2, 0.2), (0.2, 0.4)],
            confidence=0.937,
        )

        self.assertAlmostEqual(label.confidence or 0.0, 0.937)

    def test_distance_sampling_keeps_first_last_and_spacing(self) -> None:
        manager = object.__new__(FrameAnnotationManager)
        manager.frames = [
            frame("0", 0.0),
            frame("1", 1.9),
            frame("2", 5.1),
            frame("3", 9.9),
            frame("4", 10.3),
            frame("5", 12.0),
        ]

        self.assertEqual(manager._build_anchor_indices(5.0), [0, 2, 4, 5])
        self.assertEqual(manager._build_anchor_indices(0.0), [0, 1, 2, 3, 4, 5])

    def test_distance_sampling_resets_for_each_stream(self) -> None:
        manager = object.__new__(FrameAnnotationManager)
        manager.frames = [
            frame("0", 0.0, "stream_a"),
            frame("1", 5.2, "stream_a"),
            frame("0", 0.0, "stream_b"),
            frame("1", 5.4, "stream_b"),
        ]

        self.assertEqual(manager._build_anchor_indices(5.0), [0, 1, 2, 3])

    def test_polygon_bbox_is_normalized(self) -> None:
        bbox = FrameAnnotationManager._bbox([(0.1, 0.2), (0.5, 0.3), (0.4, 0.8)])
        self.assertAlmostEqual(bbox["x_center"], 0.3)
        self.assertAlmostEqual(bbox["y_center"], 0.5)
        self.assertAlmostEqual(bbox["width"], 0.4)
        self.assertAlmostEqual(bbox["height"], 0.6)

    def test_frame_annotation_schema_accepts_multi_species_labels(self) -> None:
        request = FrameAnnotationRequest(
            frame_id="000001",
            labels=[
                FrameTreeLabel(
                    label_id="tree-1",
                    species="Ficus microcarpa",
                    points=[(0.1, 0.2), (0.2, 0.2), (0.2, 0.4)],
                ),
                FrameTreeLabel(
                    label_id="tree-2",
                    species="Unknown / 待定",
                    points=[(0.5, 0.2), (0.7, 0.2), (0.6, 0.6)],
                    visibility="uncertain",
                ),
            ],
        )

        self.assertEqual(len(request.labels), 2)
        self.assertNotEqual(request.labels[0].species, request.labels[1].species)

    def test_draft_autosave_persists_new_species_in_class_list(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager = object.__new__(FrameAnnotationManager)
            manager.frames_by_id = {
                "test_stream__000001": frame("000001", 0.0)
            }
            manager.species = [BANYAN_LABEL, "Unknown / 待定"]
            manager.class_names_path = Path(directory) / "classes.json"
            manager.drafts_path = Path(directory) / "frame_drafts.json"
            manager.drafts = {}
            manager._lock = threading.RLock()
            payload = FrameAnnotationRequest(
                frame_id="test_stream__000001",
                labels=[
                    FrameTreeLabel(
                        label_id="new-species",
                        species="Ceiba pentandra",
                        points=[(0.1, 0.2), (0.2, 0.2), (0.2, 0.4)],
                    )
                ],
            )

            manager.save_draft(payload)

            persisted = json.loads(manager.class_names_path.read_text(encoding="utf-8"))
            self.assertEqual(persisted[-1], "Ceiba pentandra")
            self.assertEqual(
                manager.drafts["test_stream__000001"]["labels"][0]["species"],
                "Ceiba pentandra",
            )


if __name__ == "__main__":
    unittest.main()
