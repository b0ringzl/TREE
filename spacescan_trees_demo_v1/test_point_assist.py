import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient
from spacescan_trees_demo_v1 import point_assist as pa,server

def cloud_bytes(xyz,**meta):
    f=io.BytesIO();np.savez(f,points_xyz=xyz,**meta);return f.getvalue()

class PointAssistTests(unittest.TestCase):
    def setUp(self):
        self.xyz=np.random.default_rng(2).normal(size=(1000,3))*[2,2,5]
        self.data=cloud_bytes(self.xyz);self.client=TestClient(server.app)
    def test_unknown_units_never_report_metres(self):
        d=pa.from_upload(self.data,'x.npz','unknown',True)
        self.assertIsNone(d['observed_vertical_span_m']);self.assertIsNone(d['total_tree_height_m'])
    def test_axis_unconfirmed_disables_height(self):
        d=pa.from_upload(self.data,'x.npz','metres',False)
        self.assertIsNone(d['observed_vertical_span_m']);self.assertEqual(d['height_profile'],[])
    def test_metrics_are_not_species_or_full_height(self):
        d=pa.from_upload(self.data,'x.npz','metres',True)
        self.assertGreater(d['observed_vertical_span_m'],10);self.assertFalse(d['model_scores_changed'])
        self.assertFalse(d['validated_accuracy_gain']);self.assertIsNone(d['total_tree_height_m'])
        self.assertNotIn('species',d);self.assertLessEqual(len(d['preview_points']),2048)
    def test_inverse_normalization(self):
        center=np.array([123.,20.,-2.]);scale=np.array(15.)
        data=cloud_bytes((self.xyz-center)/scale,centroid_xyz=center,scale=scale)
        x,meta=pa.decode(data,'x.npz','metres');np.testing.assert_allclose(x,self.xyz,atol=1e-12)
        self.assertTrue(meta['metric_restored_from_declared_metadata'])
    def test_duplicate_quality(self):
        xyz=np.tile(self.xyz[:40],(20,1));d=pa.analyze(xyz,'metres',True)
        self.assertEqual(d['unique_count'],40);self.assertEqual(d['support_state'],'insufficient_evidence')
    def test_planar_quality(self):
        x=self.xyz.copy();x[:,2]=0;d=pa.analyze(x,'metres',True)
        self.assertEqual(d['support_state'],'insufficient_evidence')
    def test_nonfinite(self):
        x=self.xyz.copy();x[0,0]=np.nan
        with self.assertRaises(ValueError):pa.decode(cloud_bytes(x),'x.npz')
    def test_object_array_rejected(self):
        with self.assertRaises(ValueError):pa.decode(cloud_bytes(np.array([object()],dtype=object)),'x.npz')
    def test_bad_archive_returns_400(self):
        r=self.client.post('/api/point-assist/analyze',files={'cloud':('x.npz',b'garbage')})
        self.assertEqual(r.status_code,400)
    def test_catalog_and_safe_paths(self):
        d=self.client.get('/api/point-assist/catalog').json();self.assertEqual(d['accepted_views'],95)
        self.assertEqual(self.client.get('/api/point-assist/asset/not-a-key/point').status_code,404)
        self.assertEqual(self.client.get('/point-assist').status_code,200)
    def test_real_record(self):
        row=pa.rows()[0];d=self.client.get('/api/point-assist/record/'+row['key'])
        self.assertEqual(d.status_code,200);self.assertFalse(d.json()['evidence']['total_height_verified'])
    def test_banyan_coverage_is_display_only_and_whole_extent(self):
        rows=[r for r in pa.rows() if r['species']=='Ficus microcarpa'];self.assertEqual(len(rows),16)
        for row in rows:
            cov=pa.coverage(row);self.assertFalse(cov['geometry_changed']);self.assertGreater(cov['upper_count'],0)
            self.assertEqual(cov['original_geometry_sha256'],row['sha256']['cleaned'])
            self.assertEqual(len(cov['preview_points']),2048);self.assertEqual(len(cov['preview_lower_mask']),2048)
            self.assertGreater(cov['preview_lower_count'],cov['old_preview_lower_count'])
            with np.load(cov['asset_paths']['point'],allow_pickle=False) as f:
                with np.load(row['asset_paths']['cleaned'],allow_pickle=False) as old:xyz=old['xyz']
                restored=f['points_xyz']*f['scale']+f['centroid_xyz']
                np.testing.assert_allclose(restored,xyz[f['source_indices']],atol=1e-5)
                np.testing.assert_allclose(restored.min(0),xyz.min(0),atol=1e-5)
                np.testing.assert_allclose(restored.max(0),xyz.max(0),atol=1e-5)
        row=rows[0];d=self.client.get('/api/point-assist/record/'+row['key']).json()
        self.assertFalse(d['evidence']['preview_geometry_changed']);self.assertFalse(d['evidence']['total_height_verified'])
        self.assertEqual(self.client.get('/api/point-assist/coverage/'+row['key']+'/all').status_code,200)
        self.assertEqual(self.client.get('/api/point-assist/coverage/'+row['key']+'/bad').status_code,404)
    def test_non_banyan_coverage_unchanged(self):
        for row in pa.rows():
            if row['species']!='Ficus microcarpa':self.assertIsNone(pa.coverage(row));self.assertEqual(pa.coverage_links(row),{})
    def test_dual_requires_same_tree_and_preserves_scores_without_vertical_axis(self):
        buf=io.BytesIO();Image.new('RGB',(300,500),'green').save(buf,format='PNG')
        files={'image':('tree.png',buf.getvalue()),'cloud':('point.npz',self.data)}
        with patch.object(server,'infer') as infer:
            self.assertEqual(self.client.post('/api/predict-with-cloud',files=files).status_code,400);infer.assert_not_called()
            infer.return_value=dict(top3=[dict(species='X',score=.77)],warnings=[])
            r=self.client.post('/api/predict-with-cloud',files=files,data={'paired':'true','units':'metres','z_up':'false','projection':'photo'})
            self.assertEqual(r.status_code,200);self.assertEqual(r.json()['top3'],[dict(species='X',score=.77)])
            self.assertFalse(r.json()['fusion_used']);self.assertTrue(r.json()['paired_user_confirmed'])
    def test_panorama_requires_crop_in_dual_mode(self):
        buf=io.BytesIO();Image.new('RGB',(600,300),'green').save(buf,format='PNG')
        with patch.object(server,'infer') as infer:
            r=self.client.post('/api/predict-with-cloud',files={'image':('p.png',buf.getvalue()),'cloud':('p.npz',self.data)},data={'paired':'true'})
            self.assertEqual(r.status_code,409);infer.assert_not_called()
    def test_review_is_separate_versioned_and_origin_checked(self):
        row=pa.rows()[0]
        with tempfile.TemporaryDirectory() as tmp,patch.object(pa,'DATA',Path(tmp)):
            (Path(tmp)/'catalog.json').write_text(json.dumps([row]),encoding='utf-8')
            url='/api/point-assist/review/'+row['key']
            self.assertEqual(self.client.post(url,json={}).status_code,403)
            self.assertEqual(self.client.post(url,json={},headers={'X-Point-Review':'1','Origin':'https://evil.example'}).status_code,403)
            r=self.client.post(url,json={'same_tree':True},headers={'X-Point-Review':'1'})
            self.assertEqual(r.status_code,200);self.assertEqual(r.json()['review']['asset_hashes'],row['sha256'])
            self.assertTrue((Path(tmp)/'geometry_review_history.json').exists())

if __name__=='__main__':unittest.main()
