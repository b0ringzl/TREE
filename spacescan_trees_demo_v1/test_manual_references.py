import hashlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from PIL import Image
from fastapi.testclient import TestClient
from . import server, manual_references as manual, reference_library as library


class ManualReferencesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.patch = patch.object(manual,'ROOT',Path(self.temp.name))
        self.patch.start()
        self.assets_patch=patch.object(library,'ASSETS',{})
        self.assets_patch.start()
        self.client = TestClient(server.app)
        self.headers = {'X-Reference-Editor':'1'}
        self.species = 'Lagerstroemia speciosa'
        self.data = {'species':self.species,'slot':'1','author':'Test photographer','rights':'own','confirmed':'true'}

    def tearDown(self):
        self.assets_patch.stop();self.patch.stop();self.temp.cleanup()

    def image(self,color='green'):
        out=io.BytesIO();Image.new('RGB',(400,600),color).save(out,format='PNG');return out.getvalue()

    def upload(self,color='green',**changes):
        return self.client.post('/api/manual-references',headers=self.headers,data={**self.data,**changes},files={'image':('a.png',self.image(color),'image/png')})

    def test_save_persist_restore_and_prediction_reference(self):
        first=self.upload();self.assertEqual(first.status_code,200,first.text);one=first.json()
        self.assertFalse(one['training_eligible']);self.assertFalse(one['redistribution_approved'])
        self.assertEqual((manual.ROOT/'images'/one['id']/'original.bin').read_bytes(),self.image())
        self.assertEqual(self.client.get(one['url']).status_code,200)
        self.assertEqual(library.references(self.species)[0]['url'],one['url'])
        self.assertEqual(library.references(self.species,hashlib.sha256(self.image()).hexdigest()),[])
        second=self.upload('blue').json()
        self.assertEqual(len(manual.rows(self.species,history=True)),2)
        self.assertEqual(manual.rows(self.species)[0]['id'],second['id'])
        r=self.client.post('/api/manual-references/'+second['id']+'/status',headers=self.headers,data={'active':'false'})
        self.assertEqual(r.status_code,200);self.assertEqual(manual.rows(self.species),[])
        self.client.post('/api/manual-references/'+one['id']+'/status',headers=self.headers,data={'active':'true'})
        self.assertEqual(manual.rows(self.species)[0]['id'],one['id'])

    def test_validation_and_cross_origin(self):
        for args in ({'species':'invented'},{'slot':'3'},{'confirmed':'false'},{'rights':'cc-by'},{'rights':'cc-by','source_url':'javascript:alert(1)'},{'author':''}):
            self.assertIn(self.upload(**args).status_code,(400,422))
        bad=self.client.post('/api/manual-references',headers=self.headers,data=self.data,files={'image':('bad.png',b'bad','image/png')})
        self.assertEqual(bad.status_code,400)
        self.headers['Origin']='https://untrusted.example'
        self.assertEqual(self.upload().status_code,403)
        self.headers={};self.assertEqual(self.upload().status_code,403)
        self.assertEqual(manual.rows(),[])

    def test_slots_duplicates_and_allowlist(self):
        row=self.upload(slot='2').json()
        self.assertEqual(library.references(self.species)[0]['display_slot'],2)
        self.assertEqual(self.upload(slot='1').status_code,400)
        self.assertEqual(self.client.get('/api/manual-reference/'+('a'*32)).status_code,404)
        self.assertEqual(self.client.get('/api/reference-pool/not-allowed').status_code,404)
        self.assertEqual(self.client.get('/api/reference-pool?species=invented').status_code,400)
        self.assertEqual(self.client.get('/references/add').status_code,200)


if __name__=='__main__':unittest.main()
