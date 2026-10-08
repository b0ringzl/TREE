import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from . import flora_scope, server


class FloraScopeTests(unittest.TestCase):
    def test_low_score_is_not_a_claim_of_outside_species(self):
        result = flora_scope.advisory({'suggest_broader_scope': False}, {'score': .12}, [])
        self.assertEqual(result['action'], 'uncertain')
        self.assertFalse(result['auto_switched'])

    def test_quality_blocks_strong_scope_advice(self):
        broad = {'suggest_broader_scope': True, 'top3': [{'species': 'Helianthus annuus'}]}
        result = flora_scope.advisory(broad, {'score': .99}, ['Dark input: improve lighting.'])
        self.assertEqual(result['action'], 'improve_image')

    def test_overconfident_closed_set_can_still_receive_advice(self):
        broad = {'suggest_broader_scope': True, 'top3': [{'species': 'Helianthus annuus'}]}
        result = flora_scope.advisory(broad, {'score': .99}, [])
        self.assertEqual(result['action'], 'suggest_broad')
        self.assertFalse(result['auto_switched'])
        self.assertIn('世界模型', result['message'])
        self.assertNotIn('原始 BioCLIP', result['message'])

    def test_invalid_scope_rejected(self):
        with self.assertRaises(server.HTTPException) as context:
            server.infer(b'invalid', 'street', recognition_scope='all-secret-paths')
        self.assertEqual(context.exception.status_code, 400)

    def test_manual_broad_failure_does_not_substitute_sst(self):
        with patch.object(server, 'prepare', return_value=(b'pixels', 400, 400)), \
             patch.object(flora_scope, 'query', side_effect=ValueError('unavailable')):
            with self.assertRaises(server.HTTPException) as context:
                server.infer(b'image', 'street', recognition_scope='broad_plants')
        self.assertEqual(context.exception.status_code, 503)

    def test_two_editions_show_explicit_scope_selector(self):
        client = TestClient(server.app)
        for url in ['/image-only', '/multimodal']:
            page = client.get(url)
            self.assertEqual(page.status_code, 200)
            self.assertIn('id="floraScope"', page.text)
            self.assertIn('flora-scope.js?v=world-popup-20261008', page.text)
            self.assertIn('id="floraScopeDialog"', page.text)
            self.assertIn('aria-describedby="floraScopeDialogMessage"', page.text)
            self.assertIn('查看世界模型候選', page.text)
            self.assertNotIn('原始 BioCLIP', page.text)
        scopes = client.get('/api/flora-scopes').json()
        self.assertEqual(scopes['default'], 'hong_kong')
        self.assertEqual(scopes['scopes'][1]['name'], '世界模型')


if __name__ == '__main__':
    unittest.main()
