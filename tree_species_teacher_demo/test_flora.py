import unittest

import torch

from .flora import FloraBank, normalize_name


class FloraGateTests(unittest.TestCase):
    def bank(self, outside_genus='Helianthus'):
        bank = FloraBank.__new__(FloraBank)
        bank.records = [{'species': 'Hibiscus tiliaceus', 'genus': 'Hibiscus'},
                        {'species': outside_genus + ' annuus', 'genus': outside_genus},
                        {'species': 'Monstera deliciosa', 'genus': 'Monstera'}]
        bank.hk_mask = torch.tensor([True, False, False])
        bank.hk_genera = {'hibiscus'}
        bank.manifest = {'version': 'test', 'encoder_sha256': 'test',
                         'derived_files': [{'sha256': 'a'}, {'sha256': 'b'}]}
        return bank

    def test_margin_and_consistent_outside_genus_trigger(self):
        result = self.bank().summarize([torch.tensor([0., 4., -2.]), torch.tensor([0., 3., -2.])])
        self.assertTrue(result['suggest_broader_scope'])

    def test_covered_genus_is_not_used_for_strong_scope_claim(self):
        result = self.bank('Hibiscus').summarize([torch.tensor([0., 4., -2.])])
        self.assertFalse(result['suggest_broader_scope'])

    def test_view_disagreement_suppresses_hint(self):
        result = self.bank().summarize([torch.tensor([0., 4., -2.]), torch.tensor([0., -2., 4.])])
        self.assertFalse(result['suggest_broader_scope'])

    def test_small_margin_suppresses_hint(self):
        result = self.bank().summarize([torch.tensor([0., 1., -2.])])
        self.assertFalse(result['suggest_broader_scope'])

    def test_name_normalization_preserves_hybrid_marker(self):
        self.assertEqual(normalize_name('Bauhinia × blakeana'), normalize_name('Bauhinia x blakeana'))


if __name__ == '__main__':
    unittest.main()
