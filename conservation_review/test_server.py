import base64
import hashlib
import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from server import ReviewError, ReviewServer, ReviewStore

PNG = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aZ1cAAAAASUVORK5CYII=')


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.data = self.root / 'dataset'
        self.data.mkdir()
        self.image = self.data / 'photo.png'
        self.image.write_bytes(PNG)
        self.digest = hashlib.sha256(PNG).hexdigest()
        self.manifest = self.data / 'manifest.json'
        self.row = {'id':'sample_1','source_image':'photo.png','sha256':self.digest,
                    'source_species':'Source species stays unchanged','source_url':'https://example.org/record',
                    'license':'CC BY 4.0', 'model_score':0.999, 'predicted_species':'HIDDEN'}
        self.manifest.write_text(json.dumps([self.row]), encoding='utf8')
        self.store = ReviewStore(self.manifest, self.root/'review'/'human_review.sqlite3')

    def tearDown(self):
        self.tmp.cleanup()

    def payload(self, choice='aquilaria', revision=0, session='session-test-01'):
        return dict(id='sample_1', sha256=self.digest, decision=choice, note='Human note',
                    expected_revision=revision, session_id=session)

    def test_source_untouched_and_no_model_evidence(self):
        before = self.manifest.read_bytes()
        self.store.save(self.payload())
        self.assertEqual(self.image.read_bytes(), PNG)
        self.assertEqual(self.manifest.read_bytes(), before)
        public = self.store.listing()[0]
        self.assertNotIn('model_score', public)
        self.assertNotIn('predicted_species', public)
        self.assertEqual(public['decision']['reviewer_identity'], 'human')
        self.assertEqual(public['source_species'], self.row['source_species'])
        self.assertEqual(self.store.export()['decisions'][0]['source_species'], self.row['source_species'])

    def test_decision_save_revision_and_persistence(self):
        first = self.store.save(self.payload('not_aquilaria'))
        self.assertEqual(first['decision']['revision'],1)
        with self.assertRaises(ReviewError) as error:
            self.store.save(self.payload())
        self.assertEqual(error.exception.status,409)
        reopened = ReviewStore(self.manifest, self.store.database)
        self.assertEqual(reopened.listing()[0]['decision']['decision'],'not_aquilaria')
        reopened.save(self.payload('uncertain',1))
        self.assertEqual(reopened.listing()[0]['decision']['revision'],2)

    def test_undo_restore_delete_and_audit_trail(self):
        self.store.save(self.payload('aquilaria'))
        self.store.save(self.payload('unusable',1))
        restored = self.store.undo({'session_id':'session-test-01'})
        self.assertEqual(restored['decision']['decision'],'aquilaria')
        self.store.undo({'session_id':'session-test-01'})
        self.assertIsNone(self.store.listing()[0]['decision'])
        self.assertEqual(len(self.store.export()['history']),4)

    def test_undo_does_not_overwrite_other_tab(self):
        self.store.save(self.payload())
        self.store.save(self.payload('uncertain',1,'session-another-tab'))
        with self.assertRaises(ReviewError) as error:
            self.store.undo({'session_id':'session-test-01'})
        self.assertEqual(error.exception.status,409)
        self.assertEqual(self.store.listing()[0]['decision']['decision'],'uncertain')

    def test_path_allowlist_and_sha_guard(self):
        external = self.root/'outside.png'
        external.write_bytes(PNG)
        unsafe = dict(self.row, source_image='../outside.png')
        self.manifest.write_text(json.dumps([unsafe]),encoding='utf8')
        denied = ReviewStore(self.manifest,self.root/'denied.sqlite3')
        self.assertFalse(denied.listing()[0]['integrity_ok'])
        with self.assertRaises(ReviewError):
            denied.verified_image('sample_1')
        allowed = ReviewStore(self.manifest,self.root/'allowed.sqlite3',[self.root])
        self.assertEqual(allowed.verified_image('sample_1'),external.resolve())
        external.write_bytes(PNG+b'changed')
        with self.assertRaises(ReviewError):
            allowed.save(self.payload())

    def test_hash_mismatch_missing_and_unsafe_link(self):
        invalid=dict(self.row,sha256='0'*64,source_url='javascript:alert(1)')
        self.manifest.write_text(json.dumps([invalid]),encoding='utf8')
        store=ReviewStore(self.manifest,self.root/'invalid.sqlite3')
        self.assertFalse(store.listing()[0]['integrity_ok'])
        self.assertEqual(store.listing()[0]['source_url'],'')
        with self.assertRaises(ReviewError):
            store.save(self.payload())

    def test_invalid_decision_and_stale_hash(self):
        with self.assertRaises(ReviewError):
            self.store.save(self.payload('Pinus'))
        payload=self.payload()
        payload['sha256']='a'*64
        with self.assertRaises(ReviewError) as error:
            self.store.save(payload)
        self.assertEqual(error.exception.status,409)

    def test_optional_review_note_is_plain_data_only(self):
        self.assertEqual(self.store.listing()[0]['review_note'],'')
        supplied='Source sign differs / 樹牌物種不同 <img src=x onerror=alert(1)>'
        self.manifest.write_text(json.dumps([dict(self.row,review_note=supplied)]),encoding='utf8')
        store=ReviewStore(self.manifest,self.root/'note.sqlite3')
        public=store.listing()[0]
        self.assertEqual(public['review_note'],supplied)
        self.assertNotIn('model_score',public)
        self.assertIsNone(public['decision'])
        self.assertEqual(public['source_species'],self.row['source_species'])
        self.manifest.write_text(json.dumps([dict(self.row,review_note={'model_score':1})]),encoding='utf8')
        self.assertEqual(ReviewStore(self.manifest,self.root/'invalidnote.sqlite3').listing()[0]['review_note'],'')

    def test_http_json_csrf_static_export(self):
        server = ReviewServer(('127.0.0.1',0),self.store)
        worker=threading.Thread(target=server.serve_forever,daemon=True)
        worker.start()
        base=f'http://127.0.0.1:{server.server_port}'
        try:
            with urlopen(base+'/api/items') as response:
                result=json.load(response)
            self.assertEqual(len(result['items']),1)
            with self.assertRaises(HTTPError) as err:
                urlopen(Request(base+'/api/decision',data=json.dumps(self.payload()).encode(),headers={'Content-Type':'application/json'}))
            self.assertEqual(err.exception.code,403)
            self.assertIn('error',json.load(err.exception))
            headers={'Content-Type':'application/json','X-Review-Token':result['csrf_token']}
            with urlopen(Request(base+'/api/decision',data=json.dumps(self.payload()).encode(),headers=headers)) as response:
                self.assertTrue(json.load(response)['saved'])
            for route in ['/','/app.js','/style.css','/api/health','/api/image/sample_1','/api/export']:
                with urlopen(base+route) as response:
                    self.assertEqual(response.status,200)
            with self.assertRaises(HTTPError) as err:
                urlopen(base+'/server.py')
            self.assertEqual(err.exception.code,404)
            self.assertIn('error',json.load(err.exception))
            with self.assertRaises(HTTPError) as err:
                urlopen(Request(base+'/api/undo',data=b'{broken',headers=headers))
            self.assertEqual(err.exception.code,400)
            self.assertIn('error',json.load(err.exception))
            with self.assertRaises(HTTPError) as err:
                urlopen(Request(base+'/api/undo',data=b'{}',headers={**headers,'Origin':'https://malicious.example'}))
            self.assertEqual(err.exception.code,403)
            with self.assertRaises(HTTPError) as err:
                urlopen(Request(base+'/api/items',headers={'Host':'malicious.example'}))
            self.assertEqual(err.exception.code,403)
        finally:
            server.shutdown()
            server.server_close()
            worker.join()


if __name__=='__main__':
    unittest.main(verbosity=2)
