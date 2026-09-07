"""Synthetic geometric tests; no real annotations are written."""
import unittest
import numpy as np
from audit_hk_metric_height import measure, ground_candidate
from repair_hk_banyan_stems import balanced_ids

class HeightTests(unittest.TestCase):
    def setUp(self):
        x, y = np.meshgrid(np.arange(-6, 6, .15), np.arange(-6, 6, .15))
        self.ground = np.c_[x.ravel(), y.ravel(), np.zeros(x.size)]
        a, z = np.meshgrid(np.linspace(0, 2*np.pi, 24, endpoint=False), np.arange(.4, 6.1, .1))
        self.tree = np.r_[np.c_[.3*np.cos(a.ravel()), .3*np.sin(a.ravel()), z.ravel()],
                          np.random.default_rng(7).uniform([-2, -2, 6], [2, 2, 12], (3000, 3))]
        self.r = {'quality': {'axis_local_xy': [0, 0], 'crown_floor_z': 6}}
    def test_metric_height_and_missing_base(self):
        m = measure(self.tree, self.r, self.ground)
        self.assertTrue(m['height_feature_eligible'])
        self.assertFalse(m['total_height_verified'])
        self.assertAlmostEqual(m['estimated_tree_height_m'], 12, delta=.15)
        cut = measure(self.tree[self.tree[:, 2] > 3], self.r, self.ground)
        self.assertFalse(cut['height_feature_eligible'])
        self.assertIn('missing_basal_stem_or_wrong_ground', cut['flags'])
    def test_no_ground_means_no_total_height(self):
        m = measure(self.tree, self.r)
        self.assertIsNone(m['estimated_tree_height_m'])
        self.assertFalse(m['height_feature_eligible'])
    def test_translation_invariance(self):
        shift = np.array([1000, -300, 82.])
        r = {'quality': {'axis_local_xy': shift[:2].tolist(), 'crown_floor_z': 88.}}
        a = measure(self.tree, self.r, self.ground)
        b = measure(self.tree+shift, r, self.ground+shift)
        self.assertAlmostEqual(a['estimated_tree_height_m'], b['estimated_tree_height_m'])
    def test_preview_retains_real_stem_points(self):
        cloud = np.r_[self.tree, np.repeat(self.tree[-1000:], 30, axis=0)]
        ids = balanced_ids(cloud)
        self.assertEqual(len(ids), len(np.unique(ids)))
        self.assertLessEqual(len(ids), 8000)
        self.assertTrue((cloud[ids, 2] < 1).any())

if __name__ == '__main__': unittest.main()
