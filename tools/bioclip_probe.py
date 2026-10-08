"""Frozen BioCLIP head adaptation with prospective selection and per-fit backups.

The original calibrated zero-shot logits are a fixed offset. A zero-initialized
linear residual is equivalent to a warm-start linear classifier; an MLP residual
tests nonlinear adaptation without discarding the pretrained taxonomy prior.
No images are relabeled, no teacher backbone or production artifact is updated.
"""
import argparse
import gc
import os
import shutil
import sqlite3
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault('MPLBACKEND', 'Agg')
import numpy as np
import torch
import torch.nn.functional as F
from distillation_core import read_json, save_json, digest, collapse_logits, probabilities, metrics, bootstrap_gain
from training_safety import sha256, backup_before_training, deployed_artifacts, verify_backup

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'outputs/distillation/20260908_0045'
TEACHER = ROOT / 'external_datasets/distillation_models/bioclip25/open_clip_model.safetensors'
SEEDS = [20260909, 20260910, 20260911]
RECIPES = [
    {'name': 'linear_lr001', 'architecture': 'linear', 'lr': .001},
    {'name': 'linear_lr01', 'architecture': 'linear', 'lr': .01},
    {'name': 'residual_mlp', 'architecture': 'mlp', 'lr': .003},
]


class ResidualHead(torch.nn.Module):
    def __init__(self, architecture, features=1024, classes=148):
        super().__init__()
        if architecture == 'linear':
            self.net = torch.nn.Linear(features, classes)
            last = self.net
        elif architecture == 'mlp':
            self.net = torch.nn.Sequential(torch.nn.LayerNorm(features), torch.nn.Linear(features, 256),
                torch.nn.GELU(), torch.nn.Dropout(.1), torch.nn.Linear(256, classes))
            last = self.net[-1]
        else:
            raise ValueError(architecture)
        torch.nn.init.zeros_(last.weight)
        torch.nn.init.zeros_(last.bias)

    def forward(self, x, raw_teacher_logits):
        return raw_teacher_logits + self.net(x)


def status(out, phase, **extra):
    save_json(out / 'status.json', {'stage': phase, 'updated_at': datetime.now(timezone.utc).isoformat(),
                                  'pid': os.getpid(), 'deployed': False, **extra})
    print(phase, extra, flush=True)


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    cache = SOURCE / 'teacher_cache/bioclip25'
    source_protocol = read_json(SOURCE / 'protocol.json')
    metadata = read_json(cache / 'metadata.json')
    classes = read_json(SOURCE / 'classes.json')
    choice = read_json(SOURCE / 'teacher_choice.json')
    if sha256(TEACHER) != metadata['weight_sha256']:
        raise ValueError('Teacher encoder changed')
    if metadata['class_names'] != classes['original_names'] or metadata['class_hash'] != digest(classes['original_names']):
        raise ValueError('Teacher cache class order changed')
    if metadata['manifest_hash'] != sha256(SOURCE / 'rows.json'):
        raise ValueError('Teacher source manifest changed')
    for source, expected in source_protocol['protected_sources'].items():
        if sha256(source) != expected:
            raise ValueError('Production changed since baseline; do not compare silently')
    db_path = ROOT / 'external_datasets/herbarium_training_candidates_20260907/human_review.sqlite3'
    with sqlite3.connect(db_path.as_uri() + '?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        decisions = {r['id']: dict(r) for r in db.execute('SELECT * FROM decisions')}
    save_json(out / 'human_review_snapshot.json', decisions)
    source_hashes, train_indices = {}, []
    for phase in ['train', 'planning', 'selection', 'test', 'field', 'case']:
        path = SOURCE / f'{phase}_rows.json'
        rows = read_json(path)
        # Snapshot provenance now; test/field feature arrays are not read for selection.
        save_json(out / f'{phase}_rows.json', rows)
        source_hashes[str(path)] = sha256(path)
        source_hashes[str(cache / f'{phase}.npz')] = sha256(cache / f'{phase}.npz')
        if phase == 'train':
            for i, r in enumerate(rows):
                if r.get('auxiliary'):
                    decision = decisions.get(r['sample_id'].removeprefix('herbarium_'), {})
                    if decision.get('status') != 'accepted' or decision.get('sha256') != r['source_sha256']:
                        continue
                train_indices.append(i)
    train = read_json(out / 'train_rows.json')
    save_json(out / 'accepted_training_indices.json', train_indices)
    save_json(out / 'accepted_training_manifest.json', [train[i] for i in train_indices])
    save_json(out / 'classes.json', classes)
    save_json(out / 'teacher_choice.json', choice)
    save_json(out / 'cache_metadata.json', metadata)
    protocol = {
        'created_at': datetime.now(timezone.utc).isoformat(), 'source_experiment': str(SOURCE),
        'purpose': 'Preserve frozen BioCLIP representation; compare zero-shot, warm linear and residual MLP classifiers.',
        'recipes': RECIPES, 'seeds': SEEDS, 'epochs': 80, 'weight_decay': .01,
        'loss': 'Class-balanced GT cross entropy only; auxiliary accepted herbarium weight x3. No pseudo-labels.',
        'model': '148 raw outputs, Ficus merged via logsumexp to 147 labels; fixed teacher calibration T.',
        'input': 'Previously cached normalized BioCLIP embedding (float16 cache promoted to float32) and raw prompt logits.',
        'train_count': len(train_indices), 'removed_after_review_change': len(train) - len(train_indices),
        'selection': 'Each fit selects initial state or epoch on planning Macro-F1; recipe chosen by median of three seeds on selection Macro-F1. Zero-shot baseline may win. Fixed three-seed probability ensemble chosen before any test-array read.',
        'test': 'Historical regression only; previously observed metrics and possible teacher pretraining overlap prohibit fresh-blind claims. Test and 13-image field diagnostic never tune this run.',
        'teacher_weight': str(TEACHER), 'teacher_sha256': metadata['weight_sha256'],
        'source_hashes': source_hashes, 'protected_sources': source_protocol['protected_sources'],
        'safety': 'Every fit including smoke gets a physical backup of initial head, frozen teacher weight, deployed artifacts, labels and configs; no production deployment.',
        'deployment_cost': 'The classifier head is small but the BioCLIP Huge image encoder is still required; this does not make inference YOLO-sized.',
        'scope': 'No new unverified VMMS training labels or guessed tree IDs; LiDAR and existing fusion branches unchanged.'}
    save_json(out / 'protocol.json', protocol)
    status(out, 'prepared', train_count=len(train_indices))


def load_phase(out, phase, device='cuda'):
    if phase in ('test', 'field', 'case') and not (out / 'selection_lock.json').exists():
        raise RuntimeError('Heldout inference before selection lock')
    protocol = read_json(out / 'protocol.json')
    source_rows = SOURCE / f'{phase}_rows.json'
    source_cache = SOURCE / f'teacher_cache/bioclip25/{phase}.npz'
    for path in (source_rows, source_cache):
        if sha256(path) != protocol['source_hashes'][str(path)]:
            raise ValueError('Cached experiment source changed')
    rows = read_json(out / f'{phase}_rows.json')
    choice = read_json(out / 'teacher_choice.json')
    meta = read_json(out / 'cache_metadata.json')
    with np.load(source_cache, allow_pickle=False) as data:
        if str(data['rows_hash']) != digest(rows):
            raise ValueError('Cache sample order mismatch')
        x = data['embeddings'].astype(np.float32)
        z = data['similarities'][choice['prompt']].astype(np.float32) * meta['logit_scale']
    if phase == 'train':
        ids = read_json(out / 'accepted_training_indices.json')
        rows, x, z = [rows[i] for i in ids], x[ids], z[ids]
    if len(rows) != len(x) or x.shape[1] != 1024 or z.shape != (len(rows), 148):
        raise ValueError('Feature/label dimensions mismatch')
    if not np.isfinite(x).all() or not np.isfinite(z).all():
        raise ValueError('Nonfinite feature cache')
    return rows, torch.tensor(x, device=device), torch.tensor(z, device=device)


def scored(head, x, z, groups, temperature):
    head.eval()
    with torch.inference_mode():
        return (collapse_logits(head(x, z), groups) / temperature).softmax(-1).cpu().numpy()


def train_fit(out, recipe, seed, data, smoke=False):
    train, x, z, planning, xv, zv, selection, xs, zs = data
    config = {**recipe, 'seed': seed, 'epochs': 2 if smoke else 80, 'smoke': smoke,
              'teacher_calibration': read_json(out / 'teacher_choice.json')['calibration_temperature'],
              'backbone_trainable': False, 'optimizer': 'AdamW', 'weight_decay': .01}
    run = out / 'runs' / (('smoke_' if smoke else '') + recipe['name'] + '_' + str(seed) + '_' + datetime.now().strftime('%H%M%S_%f'))
    run.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(seed)
    head = ResidualHead(recipe['architecture']).cuda()
    torch.save(head.state_dict(), run / 'initial_head.pt')
    save_json(run / 'config.json', config)
    inputs = [run / 'initial_head.pt', TEACHER, *deployed_artifacts(), run / 'config.json',
        out / 'protocol.json', out / 'classes.json', out / 'teacher_choice.json', out / 'human_review_snapshot.json',
        out / 'accepted_training_manifest.json', out / 'planning_rows.json', out / 'selection_rows.json',
        out / 'baseline_selection.json', Path(__file__), ROOT / 'tools/distillation_core.py']
    backup = backup_before_training(run, inputs, config)
    if smoke:
        train, x, z = train[:128], x[:128], z[:128]
    y = torch.tensor([r['target'] for r in train], dtype=torch.long, device='cuda')
    counts = Counter(r['target'] for r in train)
    weights = torch.tensor([(3 if r.get('auxiliary') else 1)/counts[r['target']] for r in train], device='cuda')
    weights = weights / weights.mean()
    yp = np.array([r['target'] for r in planning]); ys = np.array([r['target'] for r in selection])
    groups = read_json(out / 'classes.json')['groups']; temp = config['teacher_calibration']
    optimizer = torch.optim.AdamW(head.parameters(), lr=recipe['lr'], weight_decay=.01)
    baseline = metrics(yp, scored(head, xv, zv, groups, temp))
    best = baseline['macro_f1']
    best_epoch = 0
    best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    history = []
    started = time.perf_counter()
    for epoch in range(1, config['epochs']+1):
        head.train(); optimizer.zero_grad(set_to_none=True)
        # Linear feature dropout is disabled to retain exactly the cached teacher view.
        logits = collapse_logits(head(x, z), groups) / temp
        loss = (F.cross_entropy(logits, y, reduction='none') * weights).mean()
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 10, error_if_nonfinite=True)
        optimizer.step()
        p = scored(head, xv, zv, groups, temp)
        m = metrics(yp, p)
        if m['macro_f1'] > best:
            best, best_epoch = m['macro_f1'], epoch
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        history.append({'epoch': epoch, 'loss': loss.item(), 'planning_macro_f1': m['macro_f1'], 'planning_top1': m['top1']})
        if epoch % 20 == 0 or smoke:
            status(out, 'training_head', recipe=recipe['name'], seed=seed, epoch=epoch, best_epoch=best_epoch, best_planning_f1=best)
    if not optimizer.state or max(int(v['step'].item()) for v in optimizer.state.values()) != config['epochs']:
        raise RuntimeError('Optimizer step count failed')
    torch.save({'model_state': head.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': config['epochs'],
        'config': config, 'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all()}, run / 'last.pt')
    torch.save({'model_state': best_state, 'epoch': best_epoch, 'config': config,
                'classes': read_json(out / 'classes.json'), 'encoder_sha256': sha256(TEACHER)}, run / 'best.pt')
    head.load_state_dict(torch.load(run / 'best.pt', weights_only=False)['model_state'])
    ps = scored(head, xs, zs, groups, temp)
    m = metrics(ys, ps)
    np.save(run / 'selection_probabilities.npy', ps)
    save_json(run / 'selection_metrics.json', m)
    save_json(run / 'history.json', history)
    verify_backup(backup)
    result = {'recipe': recipe['name'], 'architecture': recipe['architecture'], 'seed': seed, 'run': str(run),
              'best_epoch': best_epoch, 'selection_macro_f1': m['macro_f1'], 'selection_top1': m['top1'],
              'elapsed_fit_seconds': time.perf_counter()-started, 'backup': str(backup), 'smoke': smoke}
    save_json(run / 'result.json', result)
    status(out, 'fit_complete', **result)
    del head, optimizer
    gc.collect(); torch.cuda.empty_cache()
    return result


def train_all(out):
    train, x, z = load_phase(out, 'train')
    planning, xv, zv = load_phase(out, 'planning')
    selection, xs, zs = load_phase(out, 'selection')
    groups = read_json(out / 'classes.json')['groups']
    temp = read_json(out / 'teacher_choice.json')['calibration_temperature']
    p = (collapse_logits(zs, groups)/temp).softmax(-1).cpu().numpy()
    baseline = metrics([r['target'] for r in selection], p)
    save_json(out / 'baseline_selection.json', baseline)
    data = (train, x, z, planning, xv, zv, selection, xs, zs)
    train_fit(out, RECIPES[-1], SEEDS[0], data, smoke=True)
    results = []
    for recipe in RECIPES:
        for seed in SEEDS:
            results.append(train_fit(out, recipe, seed, data))
            save_json(out / 'fit_results.json', results)
    ranking = [{'recipe': recipe['name'], 'median_selection_macro_f1': float(np.median([
        r['selection_macro_f1'] for r in results if r['recipe'] == recipe['name']]))} for recipe in RECIPES]
    winner = max(ranking, key=lambda r: r['median_selection_macro_f1'])
    selected = [r for r in results if r['recipe'] == winner['recipe']]
    save_json(out / 'selection_lock.json', {'created_at': datetime.now(timezone.utc).isoformat(),
        'ranking': ranking, 'selected_recipe': winner['recipe'], 'selected_runs': selected,
        'baseline_selection_macro_f1': baseline['macro_f1'],
        'selected_beats_baseline': winner['median_selection_macro_f1'] > baseline['macro_f1'],
        'ensemble': 'Mean probabilities of all three predeclared seeds; no best-seed selection.',
        'weights_sha256': {r['run']: sha256(Path(r['run'])/'best.pt') for r in selected},
        'note': 'Evaluate selected recipe and unchanged zero-shot baseline; do not switch recipes after reading test/field results.'})
    status(out, 'selection_locked', winner=winner)


def evaluate(out):
    lock = read_json(out / 'selection_lock.json')
    groups = read_json(out / 'classes.json')['groups']
    temp = read_json(out / 'teacher_choice.json')['calibration_temperature']
    results = {}
    for phase in ['test', 'field', 'case']:
        rows, x, z = load_phase(out, phase)
        y = np.array([r['target'] for r in rows])
        basep = (collapse_logits(z, groups)/temp).softmax(-1).cpu().numpy()
        runs, predictions = [], []
        for selected in lock['selected_runs']:
            path = Path(selected['run']) / 'best.pt'
            if sha256(path) != lock['weights_sha256'][selected['run']]:
                raise ValueError('Locked checkpoint changed')
            state = torch.load(path, map_location='cpu', weights_only=False)
            head = ResidualHead(state['config']['architecture']).cuda()
            head.load_state_dict(state['model_state'])
            p = scored(head, x, z, groups, temp)
            predictions.append(p)
            runs.append({'seed': selected['seed'], **metrics(y, p)})
        ensemble = np.mean(predictions, axis=0)
        results[phase] = {'baseline': metrics(y, basep), 'seeds': runs, 'ensemble': metrics(y, ensemble),
            'macro_f1_mean': float(np.mean([r['macro_f1'] for r in runs])),
            'macro_f1_std': float(np.std([r['macro_f1'] for r in runs]))}
        if phase == 'test':
            results[phase]['accuracy_difference_ci'] = bootstrap_gain(y, basep, ensemble, [r['group_id'] for r in rows])
        np.savez(out / f'{phase}_predictions.npz', baseline=basep, ensemble=ensemble, seeds=np.stack(predictions))
        save_json(out / f'{phase}_results.json', results[phase])
    for source, expected in read_json(out / 'protocol.json')['protected_sources'].items():
        if sha256(source) != expected:
            raise ValueError('Protected deployment changed')
    save_json(out / 'final_results.json', results)
    status(out, 'complete', production_unchanged=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--stage', choices=['prepare', 'train', 'evaluate', 'all'], required=True)
    args = p.parse_args(); out = args.out.resolve()
    try:
        if args.stage in ['prepare', 'all']: prepare(out)
        if args.stage in ['train', 'all']: train_all(out)
        if args.stage in ['evaluate', 'all']: evaluate(out)
    except Exception:
        import traceback
        out.mkdir(parents=True, exist_ok=True)
        with (out/'FAILURES.md').open('a', encoding='utf-8') as f:
            f.write('\n```\n'+traceback.format_exc()+'\n```\n')
        status(out, 'failed')
        raise


if __name__ == '__main__': main()
