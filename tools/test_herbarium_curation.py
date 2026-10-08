import unittest
from PIL import Image
from collect_herbarium_candidates import perceptual_hash
from curate_herbarium_candidates import LEGEND, candidate_lane, organ_tags


class CurationTests(unittest.TestCase):
    def test_annotation_coverage(self):
        tags = organ_tags()
        self.assertEqual(len(tags), 197)
        self.assertTrue(all(tags))

    def test_identical_phash(self):
        im = Image.new('RGB', (300, 200), (40, 120, 60))
        self.assertEqual(perceptual_hash(im), perceptual_hash(im.copy()))
        self.assertEqual(len(perceptual_hash(im)), 16)

    def test_quarantine_overrides_content(self):
        self.assertEqual(candidate_lane([LEGEND['T']], cross_match=True), 'cross_dataset_match_quarantine')
        self.assertEqual(candidate_lane([LEGEND['T']], duplicate=True), 'duplicate_reference_only')

    def test_macro_does_not_enter_street_training(self):
        self.assertEqual(candidate_lane([LEGEND['R'], LEGEND['M']]), 'macro_reference_only')

    def test_trunk_and_organ_distinct(self):
        self.assertEqual(candidate_lane([LEGEND['B']]), 'trunk_root_auxiliary_candidate')
        self.assertEqual(candidate_lane([LEGEND['L']]), 'organ_auxiliary_candidate')


if __name__ == '__main__':
    unittest.main()
