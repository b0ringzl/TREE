"""Read-only numeric and training-safety audits; never fits or changes deployment."""
import argparse
import ast
import inspect
import json
from pathlib import Path
import unittest

import numpy as np
import conservation_pilot as pilot
import training_safety


class ThresholdTests(unittest.TestCase):
    def test_float32_boundary_is_excluded(self):
        y = np.array([0, 0, 0, 1])
        s = np.array([.9, .8, .7, .85], dtype=np.float32)
        t = pilot.threshold(y, s, 1 / 3)
        self.assertGreater(t, float(s[1]))
        m = pilot.binary_metrics(y, s, t)
        self.assertEqual((m['fp'], m['tp'], m['tn'], m['fn']), (1, 1, 2, 0))
        self.assertEqual(m, pilot.binary_metrics(y, s.astype(np.float64), t))

    def test_tied_negatives_are_excluded_together(self):
        y = np.array([0] * 10 + [1, 1])
        s = np.array([.9, .8, .8, .8, .7, .6, .5, .4, .3, .2, .85, .4], np.float32)
        t = pilot.threshold(y, s, .2)
        m = pilot.binary_metrics(y, s, t)
        self.assertEqual(m['fp'], 1)  # Cannot retain only one of the three .8 negatives.
        self.assertLessEqual(m['fpr'], .2)

    def test_zero_budget_and_single_negative(self):
        y = np.array([0, 1, 1])
        s = np.array([.7, .8, .7], np.float32)
        m = pilot.binary_metrics(y, s, pilot.threshold(y, s, 0))
        self.assertEqual((m['fp'], m['tp'], m['fn'], m['negative']), (0, 1, 1, 1))

    def test_budget_under_random_scores_ties_and_dtypes(self):
        rng = np.random.default_rng(20260917)
        for dtype in (np.float32, np.float64):
            for n in (1, 2, 3, 19, 99, 100, 101, 2618):
                for budget in (0., .001, .01, .05, .2, 1.):
                    for rounded in (False, True):
                        with self.subTest(dtype=dtype, n=n, budget=budget, ties=rounded):
                            s = rng.random(n + 5)
                            if rounded:
                                s = s.round(1)
                            s = s.astype(dtype)
                            y = np.array([0] * n + [1] * 5)
                            t = pilot.threshold(y, s, budget)
                            m = pilot.binary_metrics(y, s, t)
                            self.assertLessEqual(m['fp'], int(np.floor(n * budget)))
                            self.assertEqual(m['fp'] + m['tn'], n)
                            self.assertEqual(m['tp'] + m['fn'], 5)
                            self.assertEqual(m['positive'] + m['negative'], m['n'])

    def test_negative_denominator_is_not_full_dataset(self):
        y = np.array([1] * 90 + [0] * 10)
        s = np.array([.9] * 90 + [.9] * 2 + [.1] * 8)
        m = pilot.binary_metrics(y, s, .5)
        self.assertEqual(m['fpr'], .2)
        self.assertEqual(m['precision'], 90 / 92)
        self.assertEqual(m['recall'], 1.)


def call_name(node):
    if not isinstance(node, ast.Call):
        return ''
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return ''


class StaticSafetyTests(unittest.TestCase):
    def test_backup_is_inside_each_recipe_before_optimizer_updates(self):
        source = inspect.getsource(pilot.train)
        tree = ast.parse(source)
        recipes = [n for n in ast.walk(tree) if isinstance(n, ast.For)
                   and 'recipes_l2' in ast.unparse(n.iter)]
        self.assertEqual(len(recipes), 1)
        loop = recipes[0]
        calls = [n for n in ast.walk(loop) if isinstance(n, ast.Call)]
        backups = [n for n in calls if call_name(n) == 'backup_before_training']
        steps = [n for n in calls if call_name(n) == 'step']
        self.assertEqual(len(backups), 1)
        self.assertTrue(steps)
        self.assertLess(backups[0].lineno, min(n.lineno for n in steps))
        self.assertIn('exist_ok=False', source)
        self.assertIn("run/'initial.pt'", source)
        self.assertIn("out/'train_features.npz'", source)
        self.assertIn("out/'planning_features.npz'", source)
        self.assertIn("out.glob('*.json')", source)
        self.assertIn('verify_backup(backup)', source)

    def test_backup_helper_uses_physical_copies_and_verifies(self):
        source = inspect.getsource(training_safety.backup_before_training)
        self.assertIn('shutil.copy2(source, target)', source)
        self.assertIn('exist_ok=False', source)
        self.assertIn('verify_backup(dest)', source)
        self.assertNotIn('os.link(', source)
        self.assertNotIn('hardlink', source)

    def test_heldout_extraction_requires_selection_lock(self):
        source = inspect.getsource(pilot.extract)
        self.assertIn("phase=='test'", source)
        self.assertIn("(out/'selection_lock.json').exists()", source)
        self.assertIn('Test requires selection lock', source)


def audit_run(run):
    """No refitting and no writes into the run. Verify immutable contracts and backups."""
    read = lambda p: json.loads(p.read_text(encoding='utf-8'))
    contract = read(run / 'contract.json')
    artifact_checks = []
    for item in contract['artifacts']:
        got = training_safety.sha256(item['path'])
        if got != item['sha256']:
            raise AssertionError('Protected source changed: ' + item['path'])
        artifact_checks.append({'path': item['path'], 'sha256': got, 'unchanged': True})
    cfg = read(run / 'config.json')
    rows = {p: read(run / f'{p}_rows.json') for p in ('train', 'planning', 'test')}
    key = lambda r: r.get('sha256') or r.get('source_sha256')
    split_checks = []
    for p, q in (('train', 'planning'), ('train', 'test'), ('planning', 'test')):
        hashes = {key(r) for r in rows[p]} & {key(r) for r in rows[q]}
        groups = {r['group_id'] for r in rows[p]} & {r['group_id'] for r in rows[q]}
        if hashes or groups:
            raise AssertionError(f'Exact image/group overlap: {p}/{q}')
        split_checks.append({'phases': [p, q], 'sha_overlap': 0, 'group_id_overlap': 0})
    backups = []
    for l2 in cfg['recipes_l2']:
        path = run / f'linear_l2_{l2}'
        if not path.exists():
            continue
        backup = path / 'pre_training_backup'
        if not backup.is_dir():
            raise AssertionError('Recipe exists without pre-training backup: ' + str(path))
        verified = training_safety.verify_backup(backup)
        sources = {str(Path(r['source']).resolve()) for r in verified['files']}
        for required in (path / 'initial.pt', path / 'config.json', run / 'config.json',
                         run / 'train_features.npz', run / 'planning_features.npz',
                         run / 'train_rows.json', run / 'planning_rows.json', run / 'test_rows.json',
                         run / 'baseline_evaluation.json'):
            if str(required.resolve()) not in sources:
                raise AssertionError('Missing snapshot in recipe backup: ' + str(required))
        if not {str(Path(a['path']).resolve()) for a in contract['artifacts']}.issubset(sources):
            raise AssertionError('Protected weights absent from backup')
        for item in verified['files']:
            if Path(item['source']).samefile(backup / item['backup']):
                raise AssertionError('Backup aliases source inode: ' + item['source'])
        backups.append({'recipe': path.name, 'backup': str(backup), 'files': len(verified['files']),
                        'physical_copies_verified': True, 'source_hashes_verified': True})
    return {'run': str(run), 'baseline': cfg['baseline_version'], 'source_artifacts': artifact_checks,
            'exact_split_checks': split_checks, 'recipe_backups': backups,
            'fully_backed_recipe_count': len(backups), 'configured_recipe_count': len(cfg['recipes_l2']),
            'note': 'Exact SHA/group checks do not establish unseen observer/site/tree independence. '
                    'No optimizer or inference run by this audit.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    suite = unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(c)
                               for c in (ThresholdTests, StaticSafetyTests)])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
    summary = {'unit_tests': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors)}
    if args.run:
        summary['run_audit'] = audit_run(args.run.resolve())
    if args.output:
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=True))
