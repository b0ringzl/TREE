"""Isolated regression tests; never submit a review of real user data."""
import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient
import hk_pairs_api as api
from engine import point_input

class HKPairTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='hk_pair_api_test_');self.root=Path(self.temp.name)
        self.patch=patch.object(api,'ROOT',self.root);self.patch.start()
        folder=self.root/'instances/example';folder.mkdir(parents=True)
        xyz=np.random.default_rng(7).normal(size=(2048,3)).astype(np.float32);xyz/=np.linalg.norm(xyz,axis=1).max()
        np.savez(folder/'point.npz',points_xyz=xyz,centroid_xyz=np.zeros(3),scale=1)
        r={'id':'example','status':'geometry_candidate_pending_review','species':api.CLASSES[0],'point':str(folder/'point.npz'),'image':str(folder/'image.jpg'),'quality':{'clean_points':2048},'tree_group':'tree1'}
        (self.root/'manifest.json').write_text(json.dumps([r]),encoding='utf-8')
        app=FastAPI();app.include_router(api.router);self.client=TestClient(app)
    def tearDown(self):self.patch.stop();self.temp.cleanup()
    def test_list_preview_and_no_automatic_acceptance(self):
        self.assertIsNone(self.client.get('/api/hk-pairs').json()['items'][0]['review'])
        self.assertEqual(len(self.client.get('/api/hk-pairs/example/preview').json()['points']),2048)
        self.assertFalse((self.root/'pair_reviews.json').exists())
    def test_save_and_reload_relabel(self):
        payload={'decision':'accept','species':api.CLASSES[2],'note':'synthetic test only'}
        self.assertEqual(self.client.post('/api/hk-pairs/example/review',json=payload).status_code,200)
        saved=self.client.get('/api/hk-pairs').json()['items'][0]['review'];self.assertEqual(saved['species'],api.CLASSES[2])
        self.assertEqual(json.loads((self.root/'manifest.json').read_text())[0]['species'],api.CLASSES[0])
    def test_invalid_targets_and_values(self):
        self.assertEqual(self.client.get('/api/hk-pairs/missing/preview').status_code,404)
        self.assertEqual(self.client.get('/hk-pair-file/example/secrets.txt').status_code,404)
        self.assertEqual(self.client.post('/api/hk-pairs/example/review',json={'decision':'accept','species':'invalid'}).status_code,400)
        self.assertEqual(self.client.get('/hk-stem-repair/example/secrets.txt').status_code,404)
        self.assertEqual(self.client.get('/hk-stem-repair/missing/report.json').status_code,404)
    def test_metric_overlay_does_not_change_review(self):
        folder=self.root/'metric_height_v1';folder.mkdir()
        (folder/'measurements.json').write_text(json.dumps({'example':{'estimated_tree_height_m':12,'total_height_verified':False}}),encoding='utf-8')
        item=self.client.get('/api/hk-pairs').json()['items'][0]
        self.assertEqual(item['metric_height']['estimated_tree_height_m'],12)
        self.assertIsNone(item['review'])
        self.assertFalse((self.root/'pair_reviews.json').exists())
    def test_hk_point_sampling_preserves_normalized_asset(self):
        path=self.root/'instances/example/point.npz';original=np.load(path)['points_xyz']
        sampled,quality,_=point_input(path.read_bytes(),'point.npz',target_points=2048)
        np.testing.assert_array_equal(sampled,original);self.assertEqual(quality['unique_count'],2048)
    def test_pilot_review_is_version_isolated(self):
        folder=self.root/'banyan_tracking_pilot_v1';folder.mkdir()
        (folder/'results.json').write_text(json.dumps([{'id':'example','status':'candidate_requires_review'}]),encoding='utf-8')
        response=self.client.post('/api/hk-stem-pilot/example/review',json={'decision':'accept','species':'榕树','note':'synthetic only'})
        self.assertEqual(response.status_code,200)
        self.assertTrue((folder/'reviews.json').exists())
        self.assertFalse((self.root/'pair_reviews.json').exists())
        self.assertEqual(self.client.get('/hk-stem-pilot-file/example/secrets.txt').status_code,404)
        self.assertEqual(self.client.post('/api/hk-stem-pilot/missing/review',json={'decision':'accept','species':'榕树'}).status_code,404)
    def test_batch_review_does_not_change_pilot(self):
        for name in ['banyan_tracking_pilot_v1','banyan_tracking_batch_v2']:
            folder=self.root/name;folder.mkdir()
            (folder/'results.json').write_text(json.dumps([{'id':'example','status':'candidate_requires_review'}]),encoding='utf-8')
        payload={'decision':'accept','species':'榕树'}
        self.assertEqual(self.client.post('/api/hk-stem-batch/example/review',json=payload).status_code,200)
        self.assertFalse((self.root/'banyan_tracking_pilot_v1/reviews.json').exists())
        saved=json.loads((self.root/'banyan_tracking_batch_v2/reviews.json').read_text(encoding='utf-8'))
        self.assertEqual(saved['example']['version'],'banyan_tracking_batch_v2')
        self.assertEqual(self.client.get('/api/hk-stem-pilot?version=invalid').status_code,400)

if __name__=='__main__':unittest.main()
