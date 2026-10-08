import hashlib
import io
import time
import unittest
from unittest.mock import patch
from PIL import Image
from fastapi.testclient import TestClient
from spacescan_trees_demo_v1 import server, tree_detection as td


class TreeSelectionTests(unittest.TestCase):
    def setUp(self):
        b = io.BytesIO()
        Image.new('RGB', (2000, 1000), 'green').save(b, 'PNG')
        self.data = b.getvalue()
        self.client = TestClient(server.app)

    def token(self, **kw):
        return td.sign({'sha256': hashlib.sha256(self.data).hexdigest(), 'crop_box': [100, 100, 600, 900],
                        'expires': time.time()+100, **kw})

    def test_signed_native_crop(self):
        crop, meta, _ = td.selected(self.data, self.token())
        self.assertEqual(Image.open(io.BytesIO(crop)).size, (500, 800))
        padded, w, h = server.prepare(crop)
        self.assertEqual((w, h), (500, 800))
        self.assertEqual(Image.open(io.BytesIO(padded)).size, (800, 800))

    def test_tamper_expiration_and_wrong_image(self):
        for token in [self.token()+'x', self.token(expires=0), self.token(sha256='wrong'), 'bad']:
            with self.assertRaises(ValueError):
                td.selected(self.data, token)

    def test_exif_orientation(self):
        b = io.BytesIO(); im = Image.new('RGB', (400, 600)); exif = im.getexif(); exif[274] = 6
        im.save(b, 'JPEG', exif=exif)
        self.assertEqual(td.decode(b.getvalue()).size, (600, 400))

    def test_central_policy(self):
        middle = {'box': [.35, .1, .65, .9], 'score': .8}
        left = {'box': [.01, .1, .3, .9], 'score': .8}
        self.assertEqual(td.centered([middle]), 0)
        self.assertIsNone(td.centered([left]))
        self.assertIsNone(td.centered([left, middle]))
        self.assertIsNone(td.centered([]))
        self.assertEqual(td.centered([{'box':[0,0,1,1],'score':.8}]), 0)
        self.assertIsNone(td.centered([{'box':[0,0,1,1],'score':.8}], panorama=True))

    def test_filter_threshold_duplicate_group_and_tiny(self):
        rows = [{'box': b, 'score': s} for b, s in [([.1,.1,.8,.9],.9),
                ([.1,.1,.8,.9],.85), ([.12,.12,.43,.88],.8), ([.46,.12,.78,.88],.7),
                ([.8,.2,.9,.8],.399), ([.1,.1,.11,.11],.9)]]
        result = td.clean_boxes(rows, 2000, 1000)
        self.assertEqual(len(result), 2)
        self.assertTrue(all(r['score'] >= .4 for r in result))

    def test_black_candidate_is_rejected_before_tree_selection(self):
        black = Image.new('RGB', (100, 150), (5, 5, 5))
        visible = Image.new('RGB', (100, 150), (80, 100, 60))
        self.assertLess(td.visibility_p90(black), 24)
        self.assertGreaterEqual(td.visibility_p90(visible), 24)

    def test_invalid_and_dual_never_run_detector(self):
        with patch.object(td, 'detect') as fn:
            response = self.client.post('/api/detect-trees', files={'image': ('x.png', self.data)}, data={'source':'image_point'})
        self.assertEqual(response.status_code, 409); fn.assert_not_called()
        response = self.client.post('/api/detect-trees', files={'image': ('x.jpg', b'invalid')})
        self.assertEqual(response.status_code, 400)

    def test_selection_routes_to_actual_crop(self):
        with patch.object(server, 'infer', return_value={'warnings':[]}) as infer:
            r = self.client.post('/api/identify-tree', files={'image': ('x.png', self.data)}, data={'token':self.token()})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(Image.open(io.BytesIO(infer.call_args.args[0])).size, (500, 800))
        self.assertTrue(r.json()['tree_detection'])

    def test_busy_does_not_silently_classify(self):
        with patch.object(td, 'detect', side_effect=BlockingIOError('busy')), patch.object(server, 'infer') as infer:
            r = self.client.post('/api/detect-trees', files={'image': ('x.png', self.data)})
        self.assertEqual(r.status_code, 429); infer.assert_not_called()

    def test_export_is_exact_inference_crop_not_preview(self):
        token = self.token()
        crop, _, _ = td.selected(self.data, token)
        response = self.client.post('/api/export-target', files={'image': ('x.png', self.data)}, data={'token': token})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, crop)
        self.assertEqual(response.headers['x-target-sha256'], hashlib.sha256(crop).hexdigest())
        self.assertEqual(Image.open(io.BytesIO(response.content)).size, (500, 800))

    def test_export_rejects_stale_or_tampered_selection(self):
        for token in (self.token(expires=0), self.token() + 'x', self.token(sha256='wrong')):
            response = self.client.post('/api/export-target', files={'image': ('x.png', self.data)}, data={'token': token})
            self.assertEqual(response.status_code, 400)


if __name__ == '__main__':
    unittest.main()
