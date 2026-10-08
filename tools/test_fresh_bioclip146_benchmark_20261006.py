import unittest
from tools import fresh_bioclip146_benchmark_20261006 as benchmark


class FreshBenchmarkTests(unittest.TestCase):
    def test_same_146_label_catalog(self):
        rows = benchmark.catalog()
        self.assertEqual(len(rows), 146)
        bauhinia = next(r for r in rows if r['label'] == 'Bauhinia group')
        self.assertEqual(len(bauhinia['query_terms']), 3)
        ficus = next(r for r in rows if 'Ficus microcarpa /' in r['label'])
        self.assertEqual(set(ficus['query_terms']), {'Ficus benjamina','Ficus microcarpa'})

    def test_synonyms_require_exact_scientific_evidence(self):
        data = {'results':[{'id':1,'is_active':True,'rank':'species','name':'Accepted plant','matched_term':'Old plant'}]}
        self.assertEqual(benchmark.resolve_taxon(data, 'Old plant')[0]['id'], 1)
        self.assertIsNone(benchmark.resolve_taxon(data, 'Old')[0])

    def test_hybrid_normalization(self):
        data = {'results':[{'is_active':True,'rank':'hybrid','name':'Bauhinia × blakeana'}]}
        self.assertIsNotNone(benchmark.resolve_taxon(data, 'Bauhinia x blakeana')[0])

    def test_phash_gate_is_same_threshold(self):
        self.assertEqual(benchmark.nearest_hash(0,{15:'old'})['distance'],4)
        self.assertIsNone(benchmark.nearest_hash(0,{31:'old'}))

    def test_metric_reports_regression_and_missing_categories(self):
        rows = [{'label':'a'},{'label':'a'},{'label':'b'}]
        modes = ('teacher_center','teacher_context','sstv1_center','sstv1_context')
        pred = [{'top3':{m:[x] for m in modes}} for x in ('a','b','b')]
        pred[0]['top3']['sstv1_context']=['b']
        metrics = benchmark.metrics(rows,pred,['a','b','missing'])
        self.assertEqual(metrics['covered_classes'],2)
        self.assertEqual(metrics['per_class'][-1]['n'],0)
        pair = metrics['paired_comparisons']['teacher_context_vs_sstv1_context']
        self.assertEqual(pair['regressed'],1)
        self.assertEqual(pair['corrected'],0)
        self.assertLess(pair['gain_percentage_points'],0)


if __name__ == '__main__': unittest.main()
