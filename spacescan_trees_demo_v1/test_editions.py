"""Edition isolation without changing classifiers or shared target selection."""
import unittest
from html.parser import HTMLParser

from fastapi.testclient import TestClient
from spacescan_trees_demo_v1 import server, official_release


class Elements(HTMLParser):
    def __init__(self, html):
        super().__init__()
        self.elements = []
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        self.elements.append((tag, dict(attrs)))


class EditionTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(server.app)

    def test_image_only_omits_cloud_controls_and_code(self):
        page = self.client.get('/image-only')
        self.assertEqual(page.status_code, 200)
        self.assertIn('no-store', page.headers['cache-control'])
        dom = Elements(page.text).elements
        self.assertTrue(any(t == 'body' and a.get('data-edition') == 'image-only' for t, a in dom))
        for _, attrs in dom:
            self.assertNotEqual(attrs.get('name'), 'source')
            self.assertNotIn(attrs.get('id'), {'cloudPanel', 'cloud', 'paired', 'cloudUnits', 'cloudZUp'})
            self.assertNotIn('/static/point-', attrs.get('src', '') + attrs.get('href', ''))
        self.assertNotIn('PTv2', page.text)
        self.assertNotIn('{%', page.text)
        self.assertIn('href="/image-only"', page.text)

    def test_complete_edition_retains_both_sources_and_point_feature(self):
        page = self.client.get('/multimodal').text
        self.assertIn('data-edition="multimodal"', page)
        self.assertIn('value="image" checked', page)
        self.assertIn('value="image_point"', page)
        self.assertIn('id="cloud"', page)
        self.assertIn('/static/point-upload.js', page)
        self.assertIn('href="/multimodal"', page)
        self.assertEqual(self.client.get('/').text, page)
        self.assertEqual(self.client.get('/static/index.html').text, page)

    def test_both_editions_use_identical_image_feature_scripts(self):
        scripts = []
        for path in ['/image-only', '/multimodal']:
            html = self.client.get(path).text
            self.assertIn('id="modelVersion">SSTv1 · 2026-09-18</p>', html)
            self.assertNotIn('name="scenario"', html)
            scripts.append([a['src'] for t, a in Elements(html).elements
                            if t == 'script' and '/point-' not in a.get('src', '')])
        self.assertEqual(scripts[0], scripts[1])
        for feature in ['app.js', 'tree-selection.js', 'polygon-crop.js', 'clipboard-upload.js', 'photo-fallback.js']:
            self.assertTrue(any(feature in s for s in scripts[0]))

    def test_official_hub_lists_both_editions(self):
        links = official_release.public_release()['demo_editions']
        self.assertEqual(links, {'image_only': '/image-only', 'image_and_cloud': '/multimodal'})
        page = self.client.get('/apps').text
        for link in links.values():
            self.assertIn('href="' + link + '"', page)


if __name__ == '__main__':
    unittest.main()
