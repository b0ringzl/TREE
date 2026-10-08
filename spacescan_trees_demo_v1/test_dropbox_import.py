"""Dropbox Chooser appears only in the image-only edition and needs an App Key."""
import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from .server import app


class DropboxPageTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_unconfigured_button_explains_setup(self):
        with patch.dict(os.environ, {'SPACESCAN_DROPBOX_APP_KEY': ''}):
            image = self.client.get('/image-only').text
        self.assertIn('id="importDropbox"', image)
        self.assertIn('data-app-key=""', image)
        self.assertIn('/static/dropbox-import.js', image)
        self.assertNotIn('https://www.dropbox.com/static/api/2/dropins.js', image)

    def test_valid_key_loads_chooser_only_in_image_edition(self):
        with patch.dict(os.environ, {'SPACESCAN_DROPBOX_APP_KEY': 'testAppKey123'}):
            image = self.client.get('/image-only').text
            multimodal = self.client.get('/multimodal').text
        self.assertIn('data-app-key="testAppKey123"', image)
        self.assertIn('https://www.dropbox.com/static/api/2/dropins.js', image)
        self.assertNotIn('id="importDropbox"', multimodal)
        self.assertNotIn('/static/dropbox-import.js', multimodal)
        self.assertNotIn('https://www.dropbox.com/static/api/2/dropins.js', multimodal)

    def test_invalid_key_is_not_rendered_as_configuration(self):
        with patch.dict(os.environ, {'SPACESCAN_DROPBOX_APP_KEY': '<bad key>'}):
            image = self.client.get('/image-only').text
        self.assertIn('data-app-key=""', image)
        self.assertNotIn('https://www.dropbox.com/static/api/2/dropins.js', image)


if __name__ == '__main__':
    unittest.main()
