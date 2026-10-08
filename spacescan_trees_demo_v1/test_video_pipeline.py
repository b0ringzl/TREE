import tempfile
import unittest
from pathlib import Path

import numpy as np

from . import video_pipeline as video


def distribution(aquilaria, artocarpus):
    labels = ['Aquilaria sinensis', 'Artocarpus hypargyreus',
              'Bauhinia purpurea', 'Bauhinia variegata', 'Bauhinia x blakeana']
    labels += [f'Test species {index}' for index in range(143)]
    remaining = 1.0 - aquilaria - artocarpus
    row = {label: remaining / 146 for label in labels}
    row['Aquilaria sinensis'] = aquilaria
    row['Artocarpus hypargyreus'] = artocarpus
    return row


class VideoPipelineTests(unittest.TestCase):
    def test_video_upload_limit_is_two_gibibytes(self):
        self.assertEqual(video.MAX_VIDEO_BYTES, 2 * 1024 * 1024 * 1024)
        self.assertIn('2 GB', video.video_size_error())

    def test_video_profiles_use_high_detail_sampling_rates(self):
        self.assertEqual(
            {name: row['tracking_fps'] for name, row in video.PROFILES.items()},
            {'fast': 10.0, 'balanced': 15.0, 'detail': 20.0},
        )
        for row in video.PROFILES.values():
            self.assertGreaterEqual(row['max_processed'],
                                    row['tracking_fps'] * video.MAX_DURATION_SECONDS)

    def test_rotation_metadata_is_applied_clockwise(self):
        frame = np.zeros((2, 3, 3), dtype=np.uint8)
        frame[0, 0] = [10, 20, 30]
        rotated = video._rotate_frame(frame, 90)
        self.assertEqual(rotated.shape, (3, 2, 3))
        np.testing.assert_array_equal(rotated[0, 1], [10, 20, 30])

    def test_orientation_value_is_normalised(self):
        self.assertEqual(video._orientation_degrees(89.8), 90)
        self.assertEqual(video._orientation_degrees(-90), 270)
        self.assertEqual(video._orientation_degrees(360), 0)

    def test_new_tracker_id_reuses_geometric_handoff(self):
        mapping = {10: 1}
        display, next_id = video._handoff_display_ids([11], [1], mapping, 2)
        self.assertEqual(display, [1])
        self.assertEqual(mapping[11], 1)
        self.assertGreaterEqual(next_id, 2)

    def test_two_live_tracker_ids_never_share_public_id(self):
        mapping = {10: 1}
        display, _ = video._handoff_display_ids([10, 11], [2, 1], mapping, 2)
        self.assertEqual(len(set(display)), 2)

    def test_fallback_tracker_keeps_nearby_tree_id(self):
        tracker = video.FallbackTracker(max_age=3)
        first = tracker.update([[10, 10, 100, 180], [300, 20, 390, 190]], 1, 500, 300)
        second = tracker.update([[14, 12, 104, 182], [295, 22, 385, 192]], 2, 500, 300)
        self.assertEqual(first, second)
        self.assertEqual(len(set(first)), 2)

    def test_species_label_window_follows_classified_view_evidence(self):
        first, last = video._evidence_window(
            0.0, 9.27, [{'second': 4.07}, {'second': 8.47}])
        self.assertAlmostEqual(first, 3.57)
        self.assertAlmostEqual(last, 8.97)

    def test_multiview_uses_complete_distributions_and_quality_weights(self):
        first = distribution(.70, .10)
        second = distribution(.20, .60)
        _, top3, consensus = video.aggregate_predictions([
            {'image_distribution': first, 'view_quality': 1.0},
            {'image_distribution': second, 'view_quality': .25},
        ])
        expected = (.70 * .8) + (.20 * .2)
        self.assertEqual(top3[0]['species'], 'Aquilaria sinensis')
        self.assertAlmostEqual(top3[0]['score'], expected, places=6)
        self.assertEqual(consensus, .5)

    def test_similar_view_only_replaces_with_better_quality(self):
        same_hash = np.zeros(64, dtype=bool)
        views = [{'hash': same_hash, 'quality': .4}]
        video._retain_view(views, {'hash': same_hash.copy(), 'quality': .3}, 4)
        self.assertEqual(views[0]['quality'], .4)
        video._retain_view(views, {'hash': same_hash.copy(), 'quality': .8}, 4)
        self.assertEqual(views[0]['quality'], .8)

    def test_identifiable_view_rejects_ground_level_and_non_vegetated_crop(self):
        usable = {'box': [100, 100, 700, 800], 'quality_metrics': {
            'vegetation_fraction': .12, 'vegetation_central_fraction': .10,
            'vegetation_grid_coverage': 12}}
        ground = {'box': [100, 850, 700, 1000],
                  'quality_metrics': {'vegetation_fraction': .30,
                                      'vegetation_central_fraction': .20,
                                      'vegetation_grid_coverage': 16}}
        wall = {'box': [100, 100, 700, 800], 'quality_metrics': {
            'vegetation_fraction': .09, 'vegetation_central_fraction': 0.0,
            'vegetation_grid_coverage': 4}}
        self.assertTrue(video._view_is_identifiable(usable, 1000, 1000))
        self.assertFalse(video._view_is_identifiable(ground, 1000, 1000))
        self.assertFalse(video._view_is_identifiable(wall, 1000, 1000))

    def test_duplicate_tracks_require_species_time_and_spatial_agreement(self):
        def track(species, box):
            return {
                'first_second': 0.0, 'last_second': 3.0, 'consensus': 1.0,
                'top3': [{'species': species, 'score': .95}],
                'trajectory': [{'second': 1.0, 'box': box}],
            }

        left = track('Callistemon viminalis', [0, 100, 450, 800])
        right = track('Callistemon viminalis', [480, 120, 1000, 820])
        other = track('Ficus microcarpa', [480, 120, 1000, 820])
        self.assertTrue(video._tracks_are_duplicate(left, right, 1000, 1000))
        self.assertFalse(video._tracks_are_duplicate(left, other, 1000, 1000))

    def test_manager_rejects_non_video_extension(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = video.VideoJobManager(Path(directory), Path(directory) / 'detector.pt')
            with self.assertRaises(ValueError):
                manager.reserve('notes.txt')


if __name__ == '__main__':
    unittest.main()
