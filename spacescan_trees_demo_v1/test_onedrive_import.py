"""OneDrive entry is image-only and never grants the cloud branch access."""
import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from .server import app


class OneDrivePageTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_unconfigured_image_page_explains_setup_without_fake_login(self):
        with patch.dict(os.environ, {'SPACESCAN_ONEDRIVE_CLIENT_ID': ''}):
            response = self.client.get('/image-only')
        self.assertEqual(response.status_code, 200)
        self.assertIn('id="importOneDrive"', response.text)
        self.assertIn('data-client-id=""', response.text)
        self.assertIn('onedrive-import.bundle.js', response.text)

    def test_valid_client_id_only_in_image_edition(self):
        client_id = '11111111-2222-3333-4444-555555555555'
        with patch.dict(os.environ, {'SPACESCAN_ONEDRIVE_CLIENT_ID': client_id}):
            image = self.client.get('/image-only').text
            multimodal = self.client.get('/multimodal').text
        self.assertIn(f'data-client-id="{client_id}"', image)
        self.assertNotIn('id="importOneDrive"', multimodal)
        self.assertNotIn('onedrive-import.bundle.js', multimodal)

    def test_redirect_bridge_is_served_locally(self):
        response = self.client.get('/static/onedrive-redirect.html')
        self.assertEqual(response.status_code, 200)
        self.assertIn('onedrive-redirect.bundle.js', response.text)
        self.assertNotIn('Cross-Origin-Opener-Policy', response.headers)


if __name__ == '__main__':
    unittest.main()
