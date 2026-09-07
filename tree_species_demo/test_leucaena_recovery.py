"""Tests in temporary directories only, including species/identity safeguards."""
import hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient
import leucaena_recovery_api as api

class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.patch=patch.object(api,'ROOT',self.root);self.patch.start()
        self.source=self.root/'source.json';self.source.write_text('{}')
        self.rows=[{'id':'partial','species':'Leucaena leucocephala','status':'candidate_pending_review'},
                   {'id':'linked','species':'Leucaena leucocephala','status':'candidate_pending_review','conflicting_reference_links':[{'id':'reference'}]},
                   {'id':'failed','species':'Leucaena leucocephala','status':'needs_rework'}]
        for name,data in [('manifest.json',self.rows),('agent_visual_qa.json',{}),('summary.json',{}),('protected_hashes.json',{str(self.source):hashlib.sha256(self.source.read_bytes()).hexdigest()})]:
            (self.root/name).write_text(json.dumps(data))
        folder=self.root/'instances/partial';folder.mkdir(parents=True);np.savez(folder/'point.npz',points_xyz=np.zeros((1,3)))
        app=FastAPI();app.include_router(api.router);self.client=TestClient(app)
    def tearDown(self):self.patch.stop();self.temp.cleanup()
    def post(self,ident='partial',**kwargs):
        return self.client.post('/api/hk-leucaena-recovery/'+ident+'/review',json={'decision':'accept','species':'Leucaena leucocephala',**kwargs})
    def test_no_inherited_review(self):
        self.assertTrue(all(x['review'] is None for x in self.client.get('/api/hk-leucaena-recovery').json()['items']))
        self.assertFalse((self.root/'reviews.json').exists())
    def test_accept_is_new_review_not_training(self):
        result=self.post();self.assertEqual(result.status_code,200);self.assertFalse(result.json()['review']['training_eligible'])
        self.assertEqual(json.loads((self.root/'manifest.json').read_text()),self.rows)
        self.assertEqual(self.source.read_text(),'{}')
    def test_linked_requires_identity_and_correct_species(self):
        self.assertEqual(self.post('linked').status_code,400)
        self.assertEqual(self.post('linked',reference_identity='same_adjudicated_tree').status_code,400)
        r=self.post('linked',reference_identity='same_adjudicated_tree',species='Albizia lebbeck')
        self.assertEqual(r.status_code,200);self.assertEqual(r.json()['review']['split_group'],'adjudicated_same_tree_4276_4287')
        self.assertEqual(self.post(reference_identity='same_adjudicated_tree',species='Albizia lebbeck').status_code,400)
    def test_reject_can_preserve_species_correction(self):
        self.assertEqual(self.post('failed').status_code,400)
        self.assertEqual(self.post('failed',decision='reject',species='Albizia lebbeck').status_code,200)
        self.assertEqual(self.client.get('/api/hk-leucaena-recovery').json()['items'][2]['review']['species'],'Albizia lebbeck')
    def test_invalid_stale_and_missing(self):
        self.assertEqual(self.post(species='Unknown').status_code,400)
        self.assertEqual(self.post(note='x'*2001).status_code,422)
        self.assertEqual(self.post('missing').status_code,404)
        self.source.write_text('{"changed":true}');self.assertEqual(self.post().status_code,409)
        self.assertFalse((self.root/'reviews.json').exists())
    def test_preview_and_file_allowlist(self):
        self.assertEqual(len(self.client.get('/api/hk-leucaena-recovery/partial/preview').json()['points']),1)
        self.assertEqual(self.client.get('/api/hk-leucaena-recovery/failed/preview').status_code,404)
        self.assertEqual(self.client.get('/hk-leucaena-recovery-file/partial/reviews.json').status_code,404)

if __name__=='__main__':unittest.main()
