"""Isolated tests: never write production reviews."""
import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient
import delonix_refined_api as api

class RefinedTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.patch=patch.object(api,'ROOT',self.root);self.patch.start()
        self.original=json.dumps([{'id':'tree','status':'candidate_pending_review','review':{'identification':'delonix'}},{'id':'failed','status':'failed'}])
        (self.root/'manifest.json').write_text(self.original)
        (self.root/'agent_visual_qa.json').write_text(json.dumps({'tree':{'status':'review_crown_only','note':'synthetic'}}))
        folder=self.root/'instances/tree';folder.mkdir(parents=True)
        np.savez(folder/'point.npz',points_xyz=np.array([[0.,0.,0.],[1.,0.,0.]]))
        app=FastAPI();app.include_router(api.router);self.client=TestClient(app)
    def tearDown(self):self.patch.stop();self.temp.cleanup()
    def test_no_inherited_acceptance(self):
        item=self.client.get('/api/hk-delonix-refined').json()['items'][0]
        self.assertIsNone(item['pair_review']);self.assertEqual(item['agent_qa']['status'],'review_crown_only')
        self.assertFalse((self.root/'reviews.json').exists())
    def test_preview_and_asset_permissions(self):
        self.assertEqual(len(self.client.get('/api/hk-delonix-refined/tree/preview').json()['points']),2)
        self.assertEqual(self.client.get('/hk-delonix-refined-file/tree/point.npz').status_code,200)
        for url in ['/api/hk-delonix-refined/failed/preview','/hk-delonix-refined-file/tree/reviews.json','/hk-delonix-refined-file/missing/point.npz']:
            self.assertEqual(self.client.get(url).status_code,404)
    def test_explicit_review_is_separate(self):
        for decision in ['accept','reject','pending']:
            self.assertEqual(self.client.post('/api/hk-delonix-refined/tree/review',json={'decision':decision,'note':'synthetic'}).status_code,200)
            self.assertEqual(self.client.get('/api/hk-delonix-refined').json()['items'][0]['pair_review']['decision'],decision)
        self.assertEqual((self.root/'manifest.json').read_text(),self.original)
        self.assertFalse((self.root/'scene_reviews.json').exists())
    def test_invalid_review_cannot_write(self):
        for ident,body in [('tree',{'decision':'invalid'}),('tree',{'decision':'accept','note':'x'*2001}),('failed',{'decision':'accept'})]:
            self.assertEqual(self.client.post('/api/hk-delonix-refined/'+ident+'/review',json=body).status_code,400)
        self.assertFalse((self.root/'reviews.json').exists())

if __name__=='__main__':unittest.main()
