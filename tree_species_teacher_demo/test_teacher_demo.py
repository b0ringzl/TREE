import io
import unittest
from unittest.mock import patch
from PIL import Image
from fastapi.testclient import TestClient
from tree_species_teacher_demo.engine import decode, display_name, MAX_BYTES
from tree_species_teacher_demo.server import app, engine


def picture(size=(224, 224), color='green'):
    buffer = io.BytesIO()
    Image.new('RGB', size, color).save(buffer, format='PNG')
    return buffer.getvalue()


class TeacherTests(unittest.TestCase):
    def test_decode_and_crop(self):
        pixels, quality, preview = decode(picture((800, 400)))
        self.assertEqual(tuple(pixels.shape), (3, 224, 224))
        self.assertEqual(quality['crop_retained_fraction'], .5)
        self.assertTrue(quality['warnings'])
        self.assertTrue(preview.startswith('data:image/jpeg;base64,'))

    def test_bad_inputs(self):
        for data in (b'', b'not an image', b'a' * (MAX_BYTES + 1)):
            with self.assertRaises(ValueError):
                decode(data)

    def test_low_quality(self):
        _, quality, _ = decode(picture((20, 20), 'black'))
        self.assertGreaterEqual(len(quality['warnings']), 3)

    def test_names(self):
        for name in engine.classes['labels']:
            rendered = display_name(name)
            self.assertNotIn('Outside catalog', rendered)
            self.assertNotIn('其他', rendered)
        self.assertIn('榕樹', display_name('Ficus microcarpa / Ficus benjamina (榕樹)'))

    def test_api(self):
        # No lifespan here: unit tests must not reload a 4 GB GPU model.
        client = TestClient(app)
        self.assertEqual(client.get('/').status_code, 200)
        catalog = client.get('/api/catalog').json()
        self.assertEqual(len(catalog), 148)
        self.assertEqual(catalog[-1]['species'], 'Wodyetia bifurcata')
        self.assertFalse(client.get('/api/health').json()['palm_correction_loaded'])
        self.assertEqual(client.post('/api/predict', files={'image': ('bad.png', b'bad')}).status_code, 400)
        self.assertEqual(client.post('/api/predict', files={'image': ('a.png', picture())}, data={'mode': 'invalid'}).status_code, 400)
        with patch.object(engine, 'predict', return_value={'mode': 'teacher'}) as predict:
            r = client.post('/api/predict', files={'image': ('a.png', picture())}, data={'mode': 'teacher'})
            self.assertEqual(r.json()['mode'], 'teacher')
            self.assertEqual(predict.call_args.args[1], 'teacher')


if __name__ == '__main__':
    unittest.main()
