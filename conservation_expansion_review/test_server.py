import base64
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import threading
import unittest
from urllib.request import Request, urlopen
from urllib.error import HTTPError

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from conservation_expansion_review.server import ReviewStore, ReviewServer, plantnet_reference, common

PNG=base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aZ1cAAAAASUVORK5CYII=')


class ExpansionReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.data=self.root/'data'
        self.data.mkdir()
        self.image=self.data/'photo.png'
        self.image.write_bytes(PNG)
        self.sha=hashlib.sha256(PNG).hexdigest()
        self.rows=[dict(id=f'tree_{i}',source_image='photo.png',sha256=self.sha,source_species=name,
                        source_url='https://example.org/source',license='CC BY',attribution='Fixture',
                        model_score=.99,model_prediction='DO NOT EXPOSE')
                   for i,name in enumerate(['Artocarpus hypargyreus','Rhodoleia championii','Keteleeria fortunei'])]
        self.rows[0]['agent_question']='Check source sign / 請核對樹牌 <script>bad()</script>'
        self.rows[0]['plantnet_reference']={'scientific_name':'Artocarpus hypargyreus','score':.85,'source_url':'https://identify.plantnet.org/','checked_at':'2026-09-17'}
        self.manifest=self.data/'review.json'
        self.manifest.write_text(json.dumps(self.rows),encoding='utf8')
        self.db=self.root/'reviews'/'human_review.sqlite3'
        self.store=ReviewStore(self.manifest,self.db)

    def tearDown(self):
        self.temp.cleanup()

    def payload(self,choice='confirm_source',revision=0,index=0,session='fixture-session-01'):
        return dict(id=f'tree_{index}',sha256=self.sha,source_species=self.rows[index]['source_species'],
                    decision=choice,note='Human fixture only',expected_revision=revision,session_id=session)

    def test_three_species_labels_and_reference_safety(self):
        rows=self.store.listing()
        self.assertEqual([r['source_species_zh_hant'] for r in rows],['白桂木','紅花荷','油杉'])
        self.assertIn('<script>',rows[0]['agent_question']) # client must use textContent, not HTML
        self.assertEqual(rows[0]['plantnet_reference']['score'],.85)
        self.assertIn('not ground truth',rows[0]['plantnet_reference']['meaning'])
        self.assertIsNone(rows[1]['plantnet_reference'])
        self.assertNotIn('model_score',rows[0])
        self.assertNotIn('model_prediction',rows[0])
        self.assertTrue(all(r['decision'] is None for r in rows))

    def test_generic_decision_stores_species_hash_and_keeps_originals(self):
        before=self.manifest.read_bytes()
        saved=self.store.save(self.payload())['decision']
        self.assertEqual(saved['decision'],'confirm_source')
        self.assertEqual(saved['source_species'],'Artocarpus hypargyreus')
        self.assertEqual(saved['source_species_zh_hant'],'白桂木')
        self.assertEqual(saved['sha256'],self.sha)
        self.assertEqual(saved['reviewer_identity'],'human')
        self.assertEqual(self.manifest.read_bytes(),before)
        self.assertEqual(self.image.read_bytes(),PNG)
        reopened=ReviewStore(self.manifest,self.db)
        self.assertEqual(reopened.listing()[0]['decision'],saved)

    def test_rejection_is_not_species_relabel(self):
        self.store.save(self.payload('reject_source',index=1))
        row=self.store.listing()[1]
        self.assertEqual(row['source_species'],'Rhodoleia championii')
        self.assertEqual(row['decision']['source_species'],'Rhodoleia championii')
        self.assertNotIn('replacement_species',row['decision'])
        export=self.store.export()
        self.assertEqual(export['schema_version'],'expansion-1')
        self.assertIn('does not assign another species',export['meaning'])

    def test_undo_restore_and_conflict(self):
        self.store.save(self.payload())
        self.store.save(self.payload('uncertain',1))
        self.assertEqual(self.store.undo({'session_id':'fixture-session-01'})['decision']['decision'],'confirm_source')
        self.store.undo({'session_id':'fixture-session-01'})
        self.assertIsNone(self.store.listing()[0]['decision'])
        self.store.save(self.payload())
        self.store.save(self.payload('reject_source',1,session='another-session-02'))
        with self.assertRaises(common.ReviewError) as failure:
            self.store.undo({'session_id':'fixture-session-01'})
        self.assertEqual(failure.exception.status,409)

    def test_old_choices_and_database_never_changed(self):
        old_choices=set(common.CHOICES)
        legacy=self.root/'old_aquilaria.sqlite3'
        common.ReviewStore(self.manifest,legacy)
        original=legacy.read_bytes()
        with self.assertRaises(ValueError):
            ReviewStore(self.manifest,legacy)
        self.assertEqual(legacy.read_bytes(),original)
        self.assertEqual(common.CHOICES,old_choices)
        self.assertEqual(common.CHOICES,{'aquilaria','not_aquilaria','uncertain','unusable'})

    def test_species_hash_revision_and_legacy_decision_rejected(self):
        for patch in [{'sha256':'0'*64},{'source_species':'Aquilaria sinensis'},{'decision':'aquilaria'},{'decision':[]}]:
            payload=self.payload()
            payload.update(patch)
            with self.assertRaises(common.ReviewError):
                self.store.save(payload)
        self.store.save(self.payload())
        with self.assertRaises(common.ReviewError) as failure:
            self.store.save(self.payload())
        self.assertEqual(failure.exception.status,409)

    def test_shared_image_guards(self):
        self.image.write_bytes(PNG+b'changed')
        with self.assertRaises(common.ReviewError):
            self.store.save(self.payload())
        outside=self.root/'outside.png'
        outside.write_bytes(PNG)
        self.rows[0]['source_image']='../outside.png'
        self.manifest.write_text(json.dumps(self.rows),encoding='utf8')
        denied=ReviewStore(self.manifest,self.root/'denied.sqlite3')
        self.assertFalse(denied.listing()[0]['integrity_ok'])

    def test_reference_url_and_score_allowlist(self):
        self.assertIsNone(plantnet_reference({'scientific_name':'X','source_url':'javascript:alert(1)'}))
        value=plantnet_reference({'scientific_name':'X','source_url':'https://example.org/','score':float('nan'),'training_label':'X'})
        self.assertIsNone(value['score'])
        self.assertNotIn('training_label',value)

    def test_http_security_generic_ui_and_export(self):
        service=ReviewServer(('127.0.0.1',0),self.store)
        worker=threading.Thread(target=service.serve_forever,daemon=True)
        worker.start()
        url=f'http://127.0.0.1:{service.server_port}'
        try:
            with urlopen(url+'/') as response:
                page=response.read().decode()
            self.assertIn('Confirm source species',page)
            self.assertNotIn('Is this Aquilaria',page)
            with urlopen(url+'/api/items') as response:
                initial=json.load(response)
            token=initial['csrf_token']
            with self.assertRaises(HTTPError) as err:
                urlopen(Request(url+'/api/decision',data=json.dumps(self.payload()).encode(),headers={'Content-Type':'application/json'}))
            self.assertEqual(err.exception.code,403)
            self.assertIn('error',json.load(err.exception))
            with urlopen(Request(url+'/api/decision',data=json.dumps(self.payload()).encode(),headers={'Content-Type':'application/json','X-Review-Token':token})) as response:
                self.assertEqual(json.load(response)['decision']['source_species'],'Artocarpus hypargyreus')
            for route in ['/style.css','/app.js','/api/health','/api/image/tree_0','/api/export']:
                with urlopen(url+route) as response:
                    self.assertEqual(response.status,200)
            with self.assertRaises(HTTPError) as err:
                urlopen(url+'/server.py')
            self.assertEqual(err.exception.code,404)
            with self.assertRaises(HTTPError) as err:
                urlopen(Request(url+'/api/decision',data=b'{}',headers={'Content-Type':'application/json','X-Review-Token':token,'Origin':'https://foreign.example'}))
            self.assertEqual(err.exception.code,403)
        finally:
            service.shutdown()
            service.server_close()
            worker.join()


if __name__=='__main__':
    unittest.main(verbosity=2)
