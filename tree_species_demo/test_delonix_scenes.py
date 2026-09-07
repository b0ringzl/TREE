"""Tests use only synthetic data in temporary folders."""
import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from fastapi import FastAPI
from fastapi.testclient import TestClient
import delonix_scenes_api as api

class SceneTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.patch=patch.object(api,'ROOT',self.root);self.patch.start()
        self.original=json.dumps([{'id':'frame','status':'ready','targets':[{'id':'target','original_species':'Delonix regia'}]}])
        (self.root/'manifest.json').write_text(self.original)
        app=FastAPI();app.include_router(api.router);self.client=TestClient(app)
    def tearDown(self):self.patch.stop();self.temp.cleanup()
    def test_list_has_no_automatic_decisions(self):
        self.assertEqual(self.client.get('/api/hk-delonix').json()['reviews'],{})
        self.assertFalse((self.root/'scene_reviews.json').exists())
    def test_species_and_alignment_independent(self):
        body={'target_id':'target','identification':'delonix','alignment':'background','note':'synthetic test'}
        self.assertEqual(self.client.post('/api/hk-delonix/frame/review',json=body).status_code,200)
        r=self.client.get('/api/hk-delonix').json()['reviews']['target']
        self.assertEqual(r['identification'],'delonix');self.assertEqual(r['alignment'],'background')
        self.assertEqual((self.root/'manifest.json').read_text(),self.original)
        self.assertFalse((self.root/'reviews.json').exists())
    def test_invalid_targets_and_inconsistent_values(self):
        body={'target_id':'wrong','identification':'delonix','alignment':'aligned'}
        self.assertEqual(self.client.post('/api/hk-delonix/frame/review',json=body).status_code,400)
        body['target_id']='target';body['corrected_species']='other'
        self.assertEqual(self.client.post('/api/hk-delonix/frame/review',json=body).status_code,400)
        self.assertEqual(self.client.get('/hk-delonix-file/frame/private.json').status_code,404)
        self.assertEqual(self.client.get('/hk-delonix-file/missing/panorama.jpg').status_code,404)

if __name__=='__main__':unittest.main()
