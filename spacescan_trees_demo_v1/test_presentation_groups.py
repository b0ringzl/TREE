import io
import json
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image

from spacescan_trees_demo_v1 import official_release, presentation_groups, server


class BauhiniaGroupTests(unittest.TestCase):
    def setUp(self):
        labels = json.loads((Path(__file__).resolve().parents[1] /
                             'tree_species_teacher_demo/models/SSTv1_Aquilaria_UserCrops_20260919_H120/labels.json')
                            .read_text(encoding='utf-8'))
        self.assertEqual(len(labels), 148)
        self.scores = dict.fromkeys(labels, 0.0)
        self.scores.update({'Albizia lebbeck': .25, 'Aquilaria sinensis': .24,
                            'Ficus microcarpa / Ficus benjamina (榕樹)': .23,
                            'Bauhinia purpurea': .10, 'Bauhinia variegata': .10,
                            'Bauhinia x blakeana': .08})

    def test_three_probabilities_are_summed_once_and_raw_scores_remain(self):
        grouped = presentation_groups.grouped_distribution(self.scores)
        self.assertEqual(len(grouped), 146)
        self.assertAlmostEqual(grouped['Bauhinia group'], .28)
        self.assertNotIn('Bauhinia purpurea', grouped)
        self.assertNotIn('Bauhinia variegata', grouped)
        self.assertNotIn('Bauhinia x blakeana', grouped)
        self.assertEqual(presentation_groups.top_grouped(self.scores)[0][0], 'Bauhinia group')
        self.assertAlmostEqual(self.scores['Bauhinia x blakeana'], .08)

    def test_missing_or_truncated_distribution_fails_closed(self):
        with self.assertRaises(ValueError):
            presentation_groups.grouped_distribution({'Bauhinia x blakeana': .8})

    def test_web_result_is_grouped_but_export_retains_original_scores(self):
        data = io.BytesIO()
        Image.new('RGB', (300, 300), 'green').save(data, format='PNG')
        original_top = [{'species': key, 'score': value} for key, value in
                        sorted(self.scores.items(), key=lambda row: -row[1])[:3]]
        upstream = Mock()
        upstream.json.return_value = {
            'comparison': {'sstv1': original_top},
            'sstv1_distribution': self.scores,
            'teacher_distribution': self.scores,
            'quality': {'warnings': []},
            'inference_pipeline': official_release.RELEASE['web_pipeline'],
            'deployment': official_release.RELEASE['classifier_version'],
        }
        with patch.object(server.requests, 'post', return_value=upstream), \
             patch.object(server, 'references', return_value=[]):
            result = server.infer(data.getvalue(), 'street')
        first = result['top3'][0]
        self.assertEqual(first['species'], 'Bauhinia group')
        self.assertEqual(first['chinese_name'], '洋紫荊')
        self.assertAlmostEqual(first['score'], .28)
        self.assertFalse(first['group_confirmed'])
        self.assertEqual(len(first['group_members']), 3)
        self.assertEqual(result['raw_model_top3'][0]['species'], 'Albizia lebbeck')
        self.assertEqual(len(result['image_distribution']), 148)
        self.assertEqual(result['image_distribution']['Bauhinia x blakeana'], .08)
        self.assertEqual(result['display_class_count'], 146)


if __name__ == '__main__':
    unittest.main()
