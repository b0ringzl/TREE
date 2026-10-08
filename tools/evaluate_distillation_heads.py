"""Prospectively lock the best frozen KD head before any new test inference."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from distillation_core import read_json, save_json, collapse_logits, metrics, bootstrap_gain
from training_safety import sha256


def lock(out):
    target = out / 'head_final_lock.json'
    if target.exists():
        raise FileExistsError(target)
    if (out / 'student_cache/test.npz').exists() or (out / 'final_test.json').exists():
        raise RuntimeError('Cannot add head selection after test inference')
    trials = read_json(out / 'screen_results.json')
    kd = max((r for r in trials if r['lambda'] > 0), key=lambda r: r['selection_macro_f1'])
    control = next(r for r in trials if r['lambda'] == 0)
    save_json(target, {'created_at': datetime.now(timezone.utc).isoformat(), 'kd': kd, 'gt': control,
        'weights': {k: sha256(Path(r['run']) / 'best.pt') for k, r in [('kd', kd), ('gt', control)]},
        'policy': 'Supplementary matched frozen-backbone GT/KD contrast, selected by validation before any new test predictions. No test-selected head or seed.'})


def evaluate(out):
    locked = read_json(out / 'head_final_lock.json')
    cache = np.load(out / 'student_cache/test.npz')
    rows = read_json(out / 'test_rows.json')
    groups = read_json(out / 'classes.json')['groups']
    x = torch.tensor(cache['features'])
    results, predictions = {}, {}
    y = np.array([r['target'] for r in rows])
    for kind in ['gt', 'kd']:
        checkpoint = Path(locked[kind]['run']) / 'best.pt'
        if sha256(checkpoint) != locked['weights'][kind]:
            raise RuntimeError('Locked head weight changed')
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)['model_state']
        with torch.inference_mode():
            logits = torch.nn.functional.linear(x, state['weight'], state['bias'])
            p = collapse_logits(logits, groups).softmax(-1).numpy()
        results[kind] = metrics(y, p)
        predictions[kind] = p
        np.save(out / f'frozen_head_{kind}_test_probabilities.npy', p)
    results['kd_vs_matched_gt_accuracy_ci'] = bootstrap_gain(y, predictions['gt'], predictions['kd'], [r['group_id'] for r in rows])
    save_json(out / 'frozen_head_final_test.json', results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--stage', choices=['lock', 'evaluate'], required=True)
    args = parser.parse_args()
    (lock if args.stage == 'lock' else evaluate)(args.out.resolve())
