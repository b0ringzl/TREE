import unittest,io
from unittest.mock import patch
import torch
from PIL import Image
from fastapi.testclient import TestClient
from . import server,tree_detection as td

class JsonErrorTests(unittest.TestCase):
    def test_nonfinite_features_rejected(self):
        e=object.__new__(td.Engine);e.torch=torch
        for value in (float('nan'),float('inf'),0.):
            with self.assertRaises(RuntimeError):e.normalized_features(torch.full((2,4),value))
        result=e.normalized_features(torch.ones((2,4)))
        self.assertTrue(torch.allclose(result.norm(dim=-1),torch.ones(2)))
    def test_nonfinite_detector_result_is_json_error(self):
        b=io.BytesIO();Image.new('RGB',(300,500)).save(b,format='PNG')
        with patch.object(td,'detect',return_value={'candidates':[{'background_check':float('nan')}]}):
            r=TestClient(server.app).post('/api/detect-trees',files={'image':('p.png',b.getvalue())})
        self.assertEqual(r.status_code,503);self.assertIsInstance(r.json()['detail'],str)
    def test_unexpected_exception_is_json_error(self):
        b=io.BytesIO();Image.new('RGB',(300,500)).save(b,format='PNG')
        with patch.object(server,'infer',side_effect=RuntimeError('private details')):
            r=TestClient(server.app,raise_server_exceptions=False).post('/api/predict',files={'image':('p.png',b.getvalue())},data={'projection':'photo'})
        self.assertEqual(r.status_code,500);self.assertNotIn('private details',r.text)
        self.assertIsInstance(r.json()['detail'],str)
