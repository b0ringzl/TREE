import tempfile
import unittest
from pathlib import Path

from tools.benchmark_tree_detectors import iou, nms, metrics, label_boxes


class DetectorBenchmarkTests(unittest.TestCase):
    def test_iou_and_nms(self):
        self.assertEqual(iou([0, 0, 1, 1], [0, 0, 1, 1]), 1)
        self.assertEqual(iou([0, 0, .2, .2], [.3, .3, .4, .4]), 0)
        p = [{'box': [0, 0, .2, .2], 'score': .9}, {'box': [0, 0, .2, .2], 'score': .7},
             {'box': [.3, .3, .4, .4], 'score': .6}]
        self.assertEqual(len(nms(p)), 2)

    def test_metrics_no_double_matching(self):
        r = {'id': 'test', 'reference': [[0, 0, .2, .2]], 'seconds': 1,
             'predictions': [{'box': [0, 0, .2, .2], 'score': .9}, {'box': [0, 0, .2, .2], 'score': .8}]}
        m = metrics([r])
        self.assertEqual(m['tp_reference'], 1)
        self.assertEqual(m['unmatched_predictions'], 1)
        self.assertEqual(m['reference_recall'], 1)
        self.assertEqual(m['reference_precision'], .5)
        self.assertEqual(m['reference_box_coverage90'], 1)

    def test_coverage_does_not_reward_whole_scene_box(self):
        r = {'id': 'test', 'reference': [[.1, .1, .2, .2]], 'seconds': 1,
             'predictions': [{'box': [0, 0, 1, 1], 'score': .99}]}
        self.assertEqual(metrics([r])['reference_box_coverage90'], 0)

    def test_polygon_and_box(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'label.txt'
            p.write_text('0 0.5 0.5 0.2 0.4\n0 0.1 0.2 0.3 0.2 0.3 0.4\n')
            b = label_boxes(p)
            self.assertEqual(b[0], [.4, .3, .6, .7])
            self.assertEqual(b[1], [.1, .2, .3, .4])


if __name__ == '__main__':
    unittest.main()
