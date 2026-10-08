import io
import json
import unittest
from unittest.mock import patch, Mock

from PIL import Image
from fastapi.testclient import TestClient
from spacescan_trees_demo_v1 import server
from spacescan_trees_demo_v1.species_names import bilingual_names


class PreviewTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(server.app)
        b = io.BytesIO()
        Image.new('RGB', (120, 360), 'green').save(b, format='PNG')
        self.image = b.getvalue()

    def test_page_and_assets(self):
        page = self.client.get('/')
        self.assertEqual(page.status_code, 200)
        self.assertIn('SpaceScan Trees', page.text)
        self.assertNotIn('Tree Atlas', page.text)
        self.assertNotIn('Slope trees', page.text)
        self.assertNotIn('value="slope"', page.text)
        self.assertNotIn('name="scenario"', page.text)
        self.assertNotIn('Recognition scenario', page.text)
        self.assertNotIn('Scenarios share one teacher', page.text)
        self.assertIn('id="modelVersion">SSTv1 · 2026-09-18</p>', page.text)
        self.assertNotIn('Artocarpus update', page.text)
        self.assertIn('value="image" checked', page.text)
        self.assertIn('value="image_point"', page.text)
        self.assertEqual(self.client.get('/static/app.js').status_code, 200)
        self.assertEqual(self.client.get('/api/reference/not-allowed').status_code, 404)

    def test_frontend_no_longer_reads_removed_scenario_input(self):
        script = self.client.get('/static/app.js').text
        self.assertNotIn('input[name=scenario]', script)
        self.assertNotIn('scenarioTip', script)
        self.assertNotIn('scenarioNames', script)
        self.assertIn("const scenario=()=> 'street';", script)

    def test_point_upload_does_not_add_a_header_evidence_link(self):
        script = self.client.get('/static/point-upload.js').text
        self.assertNotIn('evidenceLink', script)
        self.assertNotIn("document.querySelector('header nav')", script)
        self.assertIn('/api/predict-with-cloud', script)

    def test_padding_preserves_entire_rectangle(self):
        encoded, w, h = server.prepare(self.image)
        result = Image.open(io.BytesIO(encoded))
        self.assertEqual((w, h), (120, 360))
        self.assertEqual(result.size, (360, 360))

    def test_invalid_image_and_scenario(self):
        self.assertEqual(self.client.post('/api/predict', files={'image': ('tree.png', self.image)}, data={'scenario': 'slope'}).status_code, 400)
        self.assertEqual(self.client.post('/api/predict', files={'image': ('bad.jpg', b'not an image')}).status_code, 400)
        self.assertEqual(self.client.post('/api/predict', files={'image': ('tree.png', self.image)}, data={'scenario': 'invented'}).status_code, 400)

    def test_fusion_never_silently_falls_back(self):
        with patch.object(server.requests, 'post') as upstream:
            r = self.client.post('/api/predict', files={'image': ('tree.png', self.image)}, data={'source': 'image_point'})
            self.assertEqual(r.status_code, 409)
            upstream.assert_not_called()

    def test_true_scores_forwarded_without_renormalizing(self):
        scores = [{'species': n, 'name': n, 'score': v} for n, v in [('Lagerstroemia speciosa', .6), ('Ficus microcarpa', .15), ('Acacia confusa', .02)]]
        mock = Mock()
        mock.json.return_value = {'comparison': {'sstv1': scores}, 'quality': {'warnings': []}, 'inference_seconds': 1}
        with patch.object(server.requests, 'post', return_value=mock) as upstream:
            r = self.client.post('/api/predict', files={'image': ('tree.png', self.image)}, data={'scenario': 'conservation'})
        self.assertEqual(r.status_code, 200)
        d = r.json()
        self.assertEqual([x['score'] for x in d['top3']], [.6, .15, .02])
        self.assertEqual([x['species'] for x in d['top3']], [x['species'] for x in scores])
        self.assertEqual([x['chinese_name'] for x in d['top3']], ['大花紫薇', '榕樹（細葉榕）', '台灣相思'])
        self.assertEqual(d['top3'][0]['name'], 'Lagerstroemia speciosa / 大花紫薇')
        self.assertFalse(d['multiview_fusion'])
        self.assertFalse(d['specialist_model'])
        self.assertEqual(upstream.call_args.kwargs['data'], {'mode': 'sstv1'})
        self.assertEqual(d['image_model_version'], 'sstv1-version-unavailable')
        self.assertNotIn('truth', upstream.call_args.kwargs)

    def test_offline_service(self):
        with patch.object(server.requests, 'post', side_effect=server.requests.ConnectionError):
            r = self.client.post('/api/predict', files={'image': ('tree.png', self.image)})
        self.assertEqual(r.status_code, 503)

    def test_all_deployed_classes_have_bilingual_names(self):
        classes = json.loads((server.PROJECT / 'outputs/distillation/bioclip_probe_20260908_0227/classes.json').read_text(encoding='utf-8'))
        labels = classes['labels'] + ['Wodyetia bifurcata']
        self.assertEqual(len(labels), 148)
        for label in labels:
            with self.subTest(label=label):
                names = bilingual_names(label)
                self.assertRegex(names['chinese_name'], r'[\u4e00-\u9fff]')
                self.assertNotRegex(names['scientific_name'], r'[\u4e00-\u9fff]')
                self.assertEqual(names['name'], names['scientific_name'] + ' / ' + names['chinese_name'])

    def test_distinct_bauhinia_names_and_grouped_banyan(self):
        self.assertEqual(bilingual_names('Bauhinia x blakeana')['chinese_name'], '洋紫荊')
        self.assertEqual(bilingual_names('Bauhinia purpurea')['chinese_name'], '紅花羊蹄甲')
        self.assertEqual(bilingual_names('Bauhinia variegata')['chinese_name'], '宮粉羊蹄甲')
        names = bilingual_names('Ficus microcarpa / Ficus benjamina (榕樹)')
        self.assertEqual(names['chinese_name'], '榕樹（細葉榕）')
        self.assertEqual(names['scientific_name'], 'Ficus microcarpa')

    def test_pooled_araucaria_has_genus_level_display_name(self):
        label = 'Araucaria columnaris(Araucaria heterophylla)'
        names = bilingual_names(label)
        self.assertEqual(names['scientific_name'], 'Araucaria spp.')
        self.assertEqual(names['chinese_name'], '南洋杉')
        self.assertEqual(names['name'], 'Araucaria spp. / 南洋杉')

    def test_banyan_alias_in_results_catalog_and_references(self):
        label = 'Ficus microcarpa / Ficus benjamina (榕樹)'
        expected = 'Ficus microcarpa / 榕樹（細葉榕）'
        scores = [{'species': label, 'score': .7},
                  {'species': 'Lagerstroemia speciosa', 'score': .2},
                  {'species': 'Acacia confusa', 'score': .1}]
        mock = Mock()
        mock.json.return_value = {'comparison': {'sstv1': scores}}
        with patch.object(server.requests, 'post', return_value=mock):
            result = server.infer(self.image, 'street')['top3'][0]
        self.assertEqual(result['name'], expected)
        self.assertEqual(result['species'], label)
        self.assertEqual(result['score'], .7)
        entry = next(r for r in self.client.get('/api/reference-catalog').json() if r['species'] == label)
        self.assertEqual(entry['name'], expected)
        self.assertTrue(entry['references'])
        for ref in entry['references']:
            self.assertEqual(ref['name'], expected)

    def test_reference_names_are_bilingual(self):
        refs = server.references('Lagerstroemia speciosa')
        self.assertTrue(all(r['name'] == 'Lagerstroemia speciosa / 大花紫薇' for r in refs))

    def test_missing_reviewed_photos_never_falls_back_to_organ_test_images(self):
        with patch.object(server.reference_library, 'ASSETS', {}):
            self.assertEqual(server.references('Lagerstroemia speciosa'), [])

    def test_identical_input_is_not_a_reference(self):
        import hashlib
        p = server.DATA / 'test_data/hongkong_025/image.jpg'
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        refs = server.references('Lagerstroemia speciosa', digest)
        for ref in refs:
            content = self.client.get(ref['url']).content
            self.assertNotEqual(hashlib.sha256(content).hexdigest(), digest)

    def test_web_reference_paths_are_allowlisted(self):
        self.assertEqual(self.client.get('/api/web-reference/not-allowed').status_code, 404)

    def test_web_reference_has_license_and_bilingual_name(self):
        from spacescan_trees_demo_v1 import reference_library as lib
        from pathlib import Path
        import tempfile
        import hashlib
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'images').mkdir()
            (root / 'images/a.jpg').write_bytes(self.image)
            digest = hashlib.sha256(self.image).hexdigest()
            row = {'id': 'inat_test', 'species': 'Bauhinia purpurea', 'file': 'images/a.jpg',
                'sha256': digest, 'source_url': 'https://www.inaturalist.org/observations/1',
                'photo_page': 'https://www.inaturalist.org/photos/1', 'attribution': 'Test author, CC BY',
                'author': 'Test author', 'license': 'cc-by', 'license_url': 'https://creativecommons.org/licenses/by/4.0/',
                'date': '2026-01-01', 'observed_taxon': 'Bauhinia purpurea'}
            with patch.object(lib, 'ROOT', root), patch.object(lib, 'ASSETS', {'inat_test': row}):
                refs = server.references('Bauhinia purpurea')
                self.assertEqual(len(refs), 1)
                self.assertEqual(refs[0]['name'], 'Bauhinia purpurea / 紅花羊蹄甲')
                self.assertEqual(refs[0]['license'], 'CC-BY')
                self.assertEqual(self.client.get(refs[0]['url']).content, self.image)
                self.assertEqual(lib.references('Bauhinia purpurea', digest), [])
                row['file'] = '../outside.jpg'
                with self.assertRaises(KeyError):
                    lib.asset_path('inat_test')


if __name__ == '__main__':
    unittest.main()
