import unittest
from pathlib import Path
from collect_sstv1_incremental import SPECIES, LICENSES, norm

class SSTv1CollectionTests(unittest.TestCase):
    def test_official_catalog_is_not_83_classes(self):
        self.assertEqual(len(SPECIES),27)
        self.assertEqual(len({s for s,z in SPECIES}),27)

    def test_distinct_taxa_not_silently_merged(self):
        names={s for s,z in SPECIES}
        self.assertIn('Ficus benjamina',names)
        self.assertIn('Ficus microcarpa',names)
        self.assertIn('Tabebuia rosea',names)
        self.assertNotIn('Tabebuia impetiginosa',names)
        self.assertIn('Melaleuca cajuputi subsp. cumingiana',names)

    def test_synonym_spelling_normalization_not_fuzzy_relabel(self):
        self.assertEqual(norm('Yulania × soulangeana'),norm('Yulania x soulangeana'))
        self.assertNotEqual(norm('Araucaria columnaris'),norm('Araucaria heterophylla'))
        self.assertNotEqual(norm('Michelia chapensis'),norm('Michelia x alba'))
        self.assertEqual(norm('Melaleuca cajuputi subsp. cumingiana'),norm('Melaleuca cajuputi cumingiana'))
        self.assertNotEqual(norm('Melaleuca cajuputi subsp. cumingiana'),norm('Melaleuca cajuputi'))

    def test_license_allowlist(self):
        self.assertEqual(LICENSES,{'cc0','cc-by','cc-by-sa'})

if __name__=='__main__': unittest.main()
