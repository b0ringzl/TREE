"""Temporary fixtures only; no real review or model writes."""
import hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from fastapi import FastAPI
from fastapi.testclient import TestClient
import albizia_conflict_api as api

class ConflictTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.source=self.root/'source_reviews.json';self.source.write_text('{"untouched":true}')
        data={'items':[{'id':i,'role':'conflict','original_species':s,'tree_group':'g','human_review':{'decision':'accept'}} for i,s in [('a','Albizia lebbeck'),('b','Leucaena leucocephala')]],'source_review_sha256':hashlib.sha256(self.source.read_bytes()).hexdigest()}
        (self.root/'manifest.json').write_text(json.dumps(data));(self.root/'training_candidates.json').write_text('[]')
        self.patches=[patch.object(api,'ROOT',self.root),patch.object(api,'SOURCE',self.source)]
        for p in self.patches:p.start()
        app=FastAPI();app.include_router(api.router);self.client=TestClient(app)
    def tearDown(self):
        for p in reversed(self.patches):p.stop()
        self.temp.cleanup()
    def post(self,**kwargs):return self.client.post('/api/hk-label-conflicts/adjudicate',json=kwargs)
    def test_default_no_resolution(self):
        self.assertIsNone(self.client.get('/api/hk-label-conflicts').json()['adjudication'])
        self.assertEqual(self.client.get('/api/hk-label-conflicts/training-candidates').json()['count'],0)
        self.assertFalse((self.root/'adjudications.json').exists())
    def test_same_tree_resolution_and_reversible_history(self):
        self.assertEqual(self.post(identity='same_tree',species='Leucaena leucocephala').status_code,200)
        rows=self.client.get('/api/hk-label-conflicts/training-candidates').json()['items']
        self.assertEqual(len(rows),2);self.assertEqual({r['species'] for r in rows},{'Leucaena leucocephala'})
        self.assertEqual(len({r['split_group'] for r in rows}),1);self.assertFalse(any(r['training_eligible'] for r in rows))
        self.post(identity='uncertain')
        self.assertEqual(self.client.get('/api/hk-label-conflicts/training-candidates').json()['count'],0)
        self.assertEqual(len(json.loads((self.root/'adjudications.json').read_text())),2)
        self.assertEqual(self.source.read_text(),'{"untouched":true}')
    def test_different_and_mixed_remain_held(self):
        for identity in ['different_trees','mixed']:
            self.assertEqual(self.post(identity=identity).status_code,200)
            self.assertEqual(self.client.get('/api/hk-label-conflicts/training-candidates').json()['count'],0)
        self.assertEqual(self.post(identity='mixed',species='Albizia lebbeck').status_code,400)
    def test_other_not_silently_added_to_trained_classes(self):
        self.assertEqual(self.post(identity='same_tree',species='other').status_code,400)
        self.assertEqual(self.post(identity='same_tree',species='other',other_species='new species').status_code,200)
        self.assertEqual(self.client.get('/api/hk-label-conflicts/training-candidates').json()['count'],0)
    def test_stale_source_and_invalid_fields(self):
        self.assertEqual(self.post(identity='invalid').status_code,422)
        self.assertEqual(self.post(identity='same_tree',note='x'*2001).status_code,422)
        self.source.write_text('{}')
        self.assertEqual(self.post(identity='same_tree',species='Albizia lebbeck').status_code,409)
        self.assertEqual(self.client.get('/api/hk-label-conflicts/training-candidates').status_code,409)
        self.assertFalse((self.root/'adjudications.json').exists())
    def test_file_allowlist(self):
        self.assertEqual(self.client.get('/hk-label-conflict-file/a/source_reviews.json').status_code,404)
        self.assertEqual(self.client.get('/hk-label-conflict-file/missing/panorama.jpg').status_code,404)
    def test_old_resolution_not_reused_after_evidence_refresh(self):
        self.post(identity='same_tree',species='Albizia lebbeck')
        self.source.write_text('{"new_review":true}')
        p=self.root/'manifest.json';data=json.loads(p.read_text());data['source_review_sha256']=hashlib.sha256(self.source.read_bytes()).hexdigest();p.write_text(json.dumps(data))
        self.assertIsNone(self.client.get('/api/hk-label-conflicts').json()['adjudication'])
        self.assertEqual(self.client.get('/api/hk-label-conflicts/training-candidates').json()['count'],0)

if __name__=='__main__':unittest.main()
