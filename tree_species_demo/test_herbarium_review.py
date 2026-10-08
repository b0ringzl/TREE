import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from fastapi.testclient import TestClient
import herbarium_review_api as review


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / 'images').mkdir()
        raw = b'unchanged original'
        (self.root / 'images/a.jpg').write_bytes(raw)
        self.digest = hashlib.sha256(raw).hexdigest()
        self.rows = [{'id': 'a', 'sha256': self.digest, 'file': 'images/a.jpg', 'training_eligible': False}]
        (self.root / 'curated_manifest.json').write_text(json.dumps(self.rows), encoding='utf-8')
        self.patch = patch.object(review, 'DATA', self.root)
        self.patch.start()
        self.client = TestClient(review.app)

    def tearDown(self):
        self.client.close()
        self.patch.stop()
        self.tmp.cleanup()

    def write(self, status, revision=0):
        return self.client.post('/api/herbarium-review/decision/a', json={'status': status, 'expected_revision': revision})

    def test_reject_restore_and_persistence(self):
        self.assertEqual(self.write('rejected').status_code, 200)
        self.assertEqual(self.client.get('/api/herbarium-review').json()['counts']['rejected'], 1)
        self.assertEqual(self.write('pending', 1).status_code, 200)
        self.assertEqual(self.write('accepted', 2).status_code, 200)
        exported = self.client.get('/api/herbarium-review/export').json()
        self.assertEqual(exported['accepted_count'], 1)
        self.assertFalse(exported['accepted'][0]['training_eligible'])
        self.assertEqual(hashlib.sha256((self.root / 'images/a.jpg').read_bytes()).hexdigest(), self.digest)
        with review.connect() as db:
            self.assertEqual(db.execute('SELECT count(*) FROM events').fetchone()[0], 3)

    def test_stale_tab_is_rejected(self):
        self.write('accepted')
        self.assertEqual(self.write('rejected').status_code, 409)
        self.assertEqual(self.client.get('/api/herbarium-review/export').json()['accepted_count'], 1)

    def test_invalid_and_cross_origin_writes(self):
        self.assertEqual(self.write('delete-file').status_code, 422)
        self.assertEqual(self.write('accepted', -1).status_code, 422)
        response = self.client.post('/api/herbarium-review/decision/a', json={'status': 'rejected', 'expected_revision': 0}, headers={'Origin': 'https://unrelated.example'})
        self.assertEqual(response.status_code, 403)

    def test_changed_photo_is_not_silently_accepted(self):
        self.write('accepted')
        self.rows[0]['sha256'] = 'changed'
        (self.root / 'curated_manifest.json').write_text(json.dumps(self.rows), encoding='utf-8')
        self.assertEqual(self.write('rejected', 1).status_code, 409)
        self.assertEqual(self.client.get('/api/herbarium-review').status_code, 409)

    def test_image_path_restricted(self):
        self.assertEqual(self.client.get('/api/herbarium-review/image/a').content, b'unchanged original')
        self.assertEqual(self.client.get('/api/herbarium-review/image/unknown').status_code, 404)
        self.rows[0]['file'] = '../outside.txt'
        (self.root / 'curated_manifest.json').write_text(json.dumps(self.rows), encoding='utf-8')
        self.assertEqual(self.client.get('/api/herbarium-review/image/a').status_code, 404)


if __name__ == '__main__':
    unittest.main()
