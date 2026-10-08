import hashlib
import io
import unittest
from unittest.mock import Mock, patch

from PIL import Image
from fastapi.testclient import TestClient
from spacescan_trees_demo_v1 import official_release as release, server


class OfficialReleaseTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(server.app)

    def health(self, version, status='ready'):
        response = Mock()
        response.json.return_value = {'status': 'ready', 'sstv1_status': status, 'sstv1_version': version}
        with patch.object(server.requests, 'get', return_value=response):
            return self.client.get('/api/health').json()

    def test_correct_version_is_ready(self):
        self.assertTrue(self.health(release.RELEASE['classifier_version'])['ready'])

    def test_wrong_or_missing_version_is_not_official_ready(self):
        for version in ('old', None):
            self.assertFalse(self.health(version)['ready'])

    def test_aligned_version_but_unready_engine(self):
        self.assertFalse(self.health(release.RELEASE['classifier_version'], 'error')['ready'])

    def test_public_manifest_does_not_overclaim_android_pipeline(self):
        record = release.public_release(release.RELEASE['classifier_version'])
        self.assertEqual(record['web_version_state'], 'aligned')
        self.assertFalse(record['full_android_pipeline_parity'])
        self.assertNotEqual(record['web_pipeline'], record['android_pipeline'])
        self.assertNotIn('C:', str(record))
        self.assertTrue(record['android_no_internet_permission'])

    def test_download_allowlist_and_range(self):
        self.assertEqual(self.client.get('/api/official-download/not-a-file').status_code, 404)
        with self.assertRaises(KeyError):
            release.download_path('../model')
        response = self.client.get('/api/official-download/apk', headers={'Range': 'bytes=0-7'})
        self.assertEqual(response.status_code, 206)
        self.assertEqual(response.content[:2], b'PK')
        self.assertEqual(len(response.content), 8)

    def test_apps_page_and_home_link(self):
        self.assertEqual(self.client.get('/apps').status_code, 200)
        self.assertIn('href="/apps"', self.client.get('/').text)
        self.assertIn('no-store', self.client.get('/apps').headers['cache-control'])

    def test_actual_model_and_input_hashes_not_just_expected_version(self):
        buf = io.BytesIO()
        Image.new('RGB', (280, 360), 'green').save(buf, format='PNG')
        data = buf.getvalue()
        scores = [{'species': s, 'score': v} for s, v in [('Aquilaria sinensis', .7), ('Celtis sinensis', .2), ('Acacia confusa', .1)]]
        response = Mock()
        response.json.return_value = {'comparison': {'sstv1': scores}, 'deployment': 'old'}
        with patch.object(server.requests, 'post', return_value=response), patch.object(server, 'references', return_value=[]):
            result = server.infer(data, 'street')
        self.assertEqual(result['release_provenance']['classifier_alignment'], 'mismatch')
        self.assertEqual(result['image_model_version'], 'old')
        self.assertEqual(result['release_provenance']['target_bytes_sha256'], hashlib.sha256(data).hexdigest())
        self.assertEqual(result['top3'][0]['score'], .7)

    def test_fusion_branch_retained_separately(self):
        result = release.result_provenance('teacher', 'bioclip25-original-148', b'x', b'y')
        self.assertEqual(result['classifier_alignment'], 'separate-evaluated-fusion-image-branch')
        self.assertNotEqual(result['pipeline'], release.RELEASE['web_pipeline'])


if __name__ == '__main__':
    unittest.main()
