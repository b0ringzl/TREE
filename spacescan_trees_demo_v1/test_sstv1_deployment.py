import io
import unittest
from unittest.mock import Mock,patch
from PIL import Image
from fastapi.testclient import TestClient
from spacescan_trees_demo_v1 import server

class SSTv1RoutingTests(unittest.TestCase):
    def setUp(self):
        self.client=TestClient(server.app)
        b=io.BytesIO();Image.new('RGB',(250,350),'green').save(b,format='PNG');self.image=b.getvalue()
    def test_missing_sstv1_not_silent_adapted_fallback(self):
        mock=Mock();mock.json.return_value={'comparison':{'adapted':[]}}
        with patch.object(server.requests,'post',return_value=mock):
            r=self.client.post('/api/predict',files={'image':('tree.png',self.image)})
        self.assertEqual(r.status_code,503)
    def test_health_requires_sstv1_ready(self):
        mock=Mock();mock.json.return_value={'status':'ready','sstv1_status':'error'}
        with patch.object(server.requests,'get',return_value=mock):
            health=self.client.get('/api/health').json()
        self.assertFalse(health['ready']);self.assertEqual(health['single_image_mode'],'sstv1')
        self.assertEqual(health['dual_image_mode'],'teacher')
    def test_about_describes_incremental_training(self):
        page=self.client.get('/').text
        self.assertIn('SSTv1',page)
        self.assertNotIn('No new training was performed',page)
        self.assertIn('Dual-source keeps the original teacher',page)

    def test_actual_upstream_model_version_is_reported(self):
        scores=[{'species':s,'score':p} for s,p in [('Aquilaria sinensis',.7),('Celtis sinensis',.2),('Lagerstroemia indica',.1)]]
        mock=Mock();mock.json.return_value={'comparison':{'sstv1':scores},'deployment':'sstv1-aquilaria-user-20260915-v1'}
        with patch.object(server.requests,'post',return_value=mock), patch.object(server,'references',return_value=[]):
            result=server.infer(self.image,'street')
        self.assertEqual(result['image_model_version'],'sstv1-aquilaria-user-20260915-v1')

if __name__=='__main__':unittest.main()
