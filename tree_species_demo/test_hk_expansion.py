"""Synthetic review tests; no real user review is written."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from fastapi import FastAPI
from fastapi.testclient import TestClient
import hk_pairs_api as api

class ExpansionTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.patch=patch.object(api,'EXPANSION_ROOT',self.root);self.patch.start()
        self.rows=[{'id':'synthetic','species':api.EXPANSION_CLASSES[0],'original_species':'original',
                    'status':'geometry_candidate_pending_review'},
                   {'id':'failed','species':api.EXPANSION_CLASSES[0],'status':'rejected_geometry'}]
        (self.root/'manifest.json').write_text(json.dumps(self.rows))
        app=FastAPI();app.include_router(api.router);self.client=TestClient(app)
    def tearDown(self):self.patch.stop();self.tmp.cleanup()
    def test_no_auto_accept(self):
        self.assertIsNone(self.client.get('/api/hk-expansion').json()['items'][0]['review'])
        self.assertFalse((self.root/'reviews.json').exists())
    def test_save_relabel_reload_preserves_source(self):
        result=self.client.post('/api/hk-expansion/synthetic/review',json={'decision':'accept','species':'Unknown','note':'test'})
        self.assertEqual(result.status_code,200)
        self.assertEqual(self.client.get('/api/hk-expansion').json()['items'][0]['review']['species'],'Unknown')
        self.assertEqual(json.loads((self.root/'manifest.json').read_text()),self.rows)
    def test_reject_missing_invalid_and_failed(self):
        body={'decision':'accept','species':api.EXPANSION_CLASSES[0]}
        self.assertEqual(self.client.post('/api/hk-expansion/failed/review',json=body).status_code,400)
        self.assertEqual(self.client.post('/api/hk-expansion/missing/review',json=body).status_code,404)
        body['species']='invalid'
        self.assertEqual(self.client.post('/api/hk-expansion/synthetic/review',json=body).status_code,400)
        self.assertEqual(self.client.get('/hk-expansion-file/synthetic/private.txt').status_code,404)

if __name__=='__main__':unittest.main()
