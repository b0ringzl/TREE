import unittest
from train_sstv1_incremental import mapping,read_json,SOURCE

class MappingTests(unittest.TestCase):
    def setUp(self):self.labels=read_json(SOURCE/'classes.json')['labels']
    def test_unmapped_not_nearest_species(self):
        self.assertIsNone(mapping('Tabebuia rosea',self.labels)[0])
    def test_ficus_preserves_existing_pooled_output(self):
        a,n=mapping('Ficus benjamina',self.labels);b,_=mapping('Ficus microcarpa',self.labels)
        self.assertEqual(a,b);self.assertIn('pooled',n)
    def test_subspecies_parent_is_explicit(self):
        a,n=mapping('Melaleuca cajuputi subsp. cumingiana',self.labels)
        self.assertEqual(self.labels[a],'Melaleuca cajuputi');self.assertIn('parent',n)
    def test_exact(self):
        a,n=mapping('Livistona chinensis',self.labels)
        self.assertEqual(self.labels[a],'Livistona chinensis');self.assertEqual(n,'exact')

if __name__=='__main__':unittest.main()
