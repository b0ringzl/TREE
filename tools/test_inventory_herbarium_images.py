import io,unittest
from PIL import Image
from inventory_herbarium_images import norm,quality,PROJECT_ALIASES

class InventoryTests(unittest.TestCase):
    def test_hybrid_symbol_normalization(self):
        self.assertEqual(norm('Bauhinia × blakeana'),norm(' Bauhinia x blakeana '))
    def test_distinct_species_not_merged(self):
        self.assertNotEqual(norm('Leucaena leucocephala'),norm('Albizia lebbeck'))
        self.assertNotEqual(norm('Araucaria heterophylla'),norm('Araucaria columnaris'))
    def test_explicit_parent_mapping(self):
        self.assertEqual(PROJECT_ALIASES[norm('Macaranga tanarius var. tomentosa')],norm('Macaranga tanarius'))
        self.assertNotIn(norm('Macaranga lowii'),PROJECT_ALIASES)
    def test_dark_small_image_is_not_accepted(self):
        b=io.BytesIO();Image.new('RGB',(100,100),'black').save(b,format='PNG');q,_=quality(b.getvalue())
        self.assertIn('small_resolution',q['heuristic_flags']);self.assertIn('dark',q['heuristic_flags']);self.assertEqual(q['visual_review'],'pending')

if __name__=='__main__':unittest.main()
