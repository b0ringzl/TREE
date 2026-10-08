"""Unit checks only; never train, infer images or change a deployed artifact."""
import io
import ast
import unittest
import warnings
from PIL import Image
import numpy as np
import ablate_sstv1_accuracy_20261007 as a


class AblationTests(unittest.TestCase):
    def test_grouped_matches_production(self):
        labels = a.load(a.PROBE / 'classes.json')['labels'] + ['Wodyetia bifurcata']
        p = np.random.default_rng(7).dirichlet(np.ones(148), size=2)
        out, names = a.grouped_array(p, labels)
        self.assertEqual(out.shape, (2, 146))
        self.assertTrue(np.allclose(out.sum(1), 1))
        for i in range(2):
            production = a.grouped_distribution(dict(zip(labels, p[i])))
            self.assertEqual(out[i].tolist(), [production[name] for name in names])

    def test_production_padding_byte_identity(self):
        # Execute only the real prepare() AST, without loading the web server.
        tree = ast.parse((a.ROOT / 'spacescan_trees_demo_v1/server.py').read_text(encoding='utf-8-sig'))
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'prepare')
        namespace = {'Image': Image, 'ImageOps': a.ImageOps, 'io': io, 'warnings': warnings, 'MAX_BYTES': 20*1024*1024}
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'production_prepare', 'exec'), namespace)
        prepare = namespace['prepare']
        for size in ((301, 120), (120, 301), (1400, 1400)):
            raw = io.BytesIO(); Image.new('RGB', size, (12, 210, 120)).save(raw, format='PNG')
            self.assertEqual(a.pad(raw.getvalue()), prepare(raw.getvalue())[0])

    def test_clustered_pair_difference(self):
        rows = [{'observer_id': 1}, {'observer_id': 1}, {'observer_id': 2}, {'observer_id': 3}]
        before = np.array([False, True, False, True]); after = np.array([True, True, True, False])
        r = a.paired(rows, before, after)
        self.assertEqual(r['corrected'], 2); self.assertEqual(r['regressed'], 1)
        self.assertEqual(r['gain_pp'], 25); self.assertEqual(r['clusters'], 3)

    def test_score_retains_unsupported_classes(self):
        rows = [{'label': 'a'}, {'label': 'b'}]
        metrics, correct = a.score(rows, np.array([[.8, .1, .1], [.4, .2, .4]]), ['a', 'b', 'c'])
        self.assertEqual(len(metrics['per_class']), 3)
        self.assertEqual(metrics['covered_classes'], 2)
        self.assertEqual(metrics['correct'], 1)
        self.assertEqual(correct.tolist(), [True, False])


if __name__ == '__main__': unittest.main()
