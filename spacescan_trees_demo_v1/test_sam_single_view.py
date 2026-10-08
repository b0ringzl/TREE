import io
import json
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi import HTTPException
from PIL import Image

from spacescan_trees_demo_v1 import official_release, server


class SamSingleViewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        labels = json.loads((Path(__file__).resolve().parents[1] /
                             'tree_species_teacher_demo/models/SSTv1_Aquilaria_UserCrops_20260919_H120/labels.json')
                            .read_text(encoding='utf-8'))
        cls.scores = dict.fromkeys(labels, 0.0)
        cls.scores['Jacaranda mimosifolia'] = 1.0

    def image(self):
        stream = io.BytesIO()
        Image.new('RGB', (320, 420), 'green').save(stream, format='PNG')
        return stream.getvalue()

    def upstream(self, version=None):
        response = Mock()
        response.json.return_value = {
            'comparison': {'sstv1': [{'species': 'Jacaranda mimosifolia', 'score': 1.0},
                                     {'species': 'Albizia lebbeck', 'score': 0.0},
                                     {'species': 'Aquilaria sinensis', 'score': 0.0}]},
            'sstv1_distribution': self.scores,
            'quality': {'warnings': []},
            'deployment': version or official_release.RELEASE['classifier_version'],
        }
        return response

    def test_single_view_uses_one_pinned_model_call_without_detail(self):
        with patch.object(server.requests, 'post', return_value=self.upstream()) as post, \
             patch.object(server, 'references', return_value=[]):
            result = server.infer(self.image(), 'street', view_policy='single')
        self.assertTrue(post.call_args.args[0].endswith('/api/predict'))
        self.assertEqual(set(post.call_args.kwargs['files']), {'image'})
        self.assertEqual(post.call_args.kwargs['data']['mode'], 'sstv1')
        self.assertTrue(result['single_view'])
        self.assertIsNone(result['model_input_detail'])
        self.assertIsNone(result['view_evidence'])
        self.assertEqual(result['inference_pipeline'], 'sstv1-single-view-crop-v1')
        self.assertEqual(result['release_provenance']['pipeline'], result['inference_pipeline'])

    def test_wrong_classifier_version_is_rejected(self):
        with patch.object(server.requests, 'post', return_value=self.upstream('wrong-version')):
            with self.assertRaises(HTTPException) as caught:
                server.infer(self.image(), 'street', view_policy='single')
        self.assertEqual(caught.exception.status_code, 503)


if __name__ == '__main__':
    unittest.main()
