"""Preliminary closed-eight BioCLIP residual probe. No production deployment.

Uses existing web observations and currently accepted auxiliary photos only.
New authoritative-source candidates are NOT used by this experiment.
Historical web regression is not an independent Hong Kong street benchmark.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from bioclip_probe import ResidualHead, SOURCE, TEACHER
from distillation_core import read_json, save_json, digest, metrics
from focus8_data import ROOT, TAXA, LABELS, HERB, review_snapshot
from training_safety import backup_before_training, verify_backup, deployed_artifacts, sha256


def allowed(row, decisions, managed):
    ident = row['sample_id'].removeprefix('herbarium_') if row.get('auxiliary') else managed.get(row['source_sha256'])
    if ident is None:
        return True
    decision = decisions.get(ident, {})
    return decision.get('status') == 'accepted' and decision.get('sha256') == row['source_sha256']


def probabilities(head, x, z, temperature):
    head.eval()
    with torch.inference_mode():
        return (head(x, z) / temperature).softmax(-1).cpu().numpy()


def main(out):
    out.mkdir(parents=True, exist_ok=False)
    decisions = review_snapshot()
    save_json(out / 'human_review_snapshot.json', decisions)
    managed = {r['sha256']: r['id'] for r in read_json(HERB / 'curated_manifest.json')}
    meta = read_json(SOURCE / 'teacher_cache/bioclip25/metadata.json')
    choice = read_json(SOURCE / 'teacher_choice.json')
    if sha256(TEACHER) != meta['weight_sha256']:
        raise ValueError('Teacher differs from cached embeddings')
    original_names = read_json(SOURCE / 'classes.json')['original_names']
    if meta['class_names'] != original_names:
        raise ValueError('Cache class order mismatch')
    columns = [original_names.index(s) for s in LABELS]
    config = {'seed': 20260909, 'architecture': 'linear', 'epochs': 100, 'lr': .01,
        'weight_decay': .01, 'frozen_backbone': True, 'features': 1024,
        'temperature': choice['calibration_temperature'], 'prompt': choice['prompt'],
        'selection': 'Best planning macro F1, earliest tie, including unchanged epoch zero; no test selection',
        'task': 'closed-eight classification, not detection or full-catalog replacement',
        'data_scope': 'Existing source-labeled web data + SQLite-accepted herbarium auxiliaries. No newly collected photos.',
        'field_training': False, 'deployment': False,
        'preprocessing': read_json(SOURCE / 'protocol.json')['preprocessing']}
    save_json(out / 'config.json', config)
    save_json(out / 'classes.json', {'labels': LABELS, 'taxa': TAXA, 'original_columns': columns,
        'ficus_policy': 'Strict raw Ficus microcarpa only; no merged banyan labels, no Ficus benjamina',
        'melaleuca_policy': 'Web label species level; does not prove subsp. cumingiana identity',
        'bauhinia_policy': 'x/× alias only, never variegata/purpurea'})
    arrays, rows_by_phase, hashes, exclusions = {}, {}, {}, []
    phase_groups, phase_hashes = {}, {}
    for phase in ('train', 'planning', 'selection', 'test'):
        src = SOURCE / f'{phase}_rows.json'
        rows = read_json(src)
        cache = SOURCE / 'teacher_cache/bioclip25' / (phase + '.npz')
        hashes[str(src)] = sha256(src)
        hashes[str(cache)] = sha256(cache)
        ids = []
        for i, row in enumerate(rows):
            if row['species'] not in LABELS:
                continue
            if not allowed(row, decisions, managed):
                exclusions.append({'sample_id': row['sample_id'], 'reason': 'human decision not accepted or SHA mismatch'})
                continue
            if sha256(row['source_image']) != row['source_sha256']:
                raise ValueError('Source image changed: ' + row['sample_id'])
            ids.append(i)
        selected = [{**rows[i], 'previous_target': rows[i]['target'], 'target': LABELS.index(rows[i]['species']),
            'source_trust': 'human_accepted_official_photo' if rows[i].get('auxiliary') else 'existing_web_source_label_not_expert_verified'} for i in ids]
        rows_by_phase[phase] = selected
        save_json(out / f'{phase}_rows.json', selected)
        phase_groups[phase] = {r['group_id'].split('observation:')[-1] for r in selected}
        phase_hashes[phase] = {r['source_sha256'] for r in selected}
        # Cache association verified even for the locked regression set. Its predictions
        # are not computed or used for any fitting or checkpoint choice until the lock.
        with np.load(cache, allow_pickle=False) as data:
            if str(data['rows_hash']) != digest(rows):
                raise ValueError('Feature-row hash mismatch')
            x = data['embeddings'][ids].astype(np.float32)
            z = data['similarities'][choice['prompt']][ids][:, columns].astype(np.float32) * meta['logit_scale']
        if x.shape != (len(ids), 1024) or not np.isfinite(x).all() or not np.isfinite(z).all():
            raise ValueError('Invalid cache dimensions/values')
        arrays[phase] = (torch.tensor(x, device='cuda'), torch.tensor(z, device='cuda'))
    for a in phase_groups:
        for b in phase_groups:
            if a < b and (phase_groups[a] & phase_groups[b] or phase_hashes[a] & phase_hashes[b]):
                raise ValueError('Cross-split source group or exact image leakage')
    save_json(out / 'data_audit.json', {'counts': {p: dict(Counter(r['species'] for r in rows)) for p, rows in rows_by_phase.items()},
        'excluded': exclusions, 'cross_split_exact_or_observation_overlap': False,
        'tree_identity_verified': False, 'teacher_pretraining_overlap_unknown': True,
        'near_duplicate_policy': 'Inherited historical source split; not a fresh tree-level or exhaustive near-duplicate benchmark'})
    save_json(out / 'cache_hashes.json', hashes)
    torch.manual_seed(config['seed'])
    head = ResidualHead('linear', classes=8).cuda()
    torch.save(head.state_dict(), out / 'initial_head.pt')
    baseline = {p: metrics([r['target'] for r in rows_by_phase[p]], probabilities(head, *arrays[p], config['temperature']))
        for p in ('planning', 'selection')}
    save_json(out / 'baseline_validation.json', baseline)
    previous = ROOT / 'outputs/distillation/bioclip_probe_20260908_0227'
    heads = [Path(r['run']) / 'best.pt' for r in read_json(previous / 'selection_lock.json')['selected_runs']]
    inputs = [TEACHER, out / 'initial_head.pt', *deployed_artifacts(), *heads,
        ROOT / 'tree_species_teacher_demo/engine.py', previous / 'selection_lock.json',
        Path(__file__), ROOT / 'tools/focus8_data.py', ROOT / 'tools/bioclip_probe.py', ROOT / 'tools/distillation_core.py',
        *[out / f'{p}_rows.json' for p in rows_by_phase], out / 'human_review_snapshot.json',
        out / 'config.json', out / 'classes.json', out / 'cache_hashes.json', out / 'baseline_validation.json', out / 'data_audit.json']
    # Any active teacher extension is protected as well, although this experiment never uses it.
    for candidate in (ROOT / 'tree_species_teacher_demo').glob('*.json'):
        inputs.append(candidate)
    inputs.extend((ROOT / 'outputs').glob('palm_extension*/**/best.pt'))
    save_json(out / 'status.json', {'stage': 'physical_backup_before_fit', 'deployed': False})
    backup = backup_before_training(out, inputs, config)
    print('Backup verified: ' + str(backup), flush=True)
    # Recheck the authoritative review after copying, before any optimizer step.
    if review_snapshot() != decisions:
        raise RuntimeError('Human reviews changed during preparation; start a new run')
    x, z = arrays['train']
    y = torch.tensor([r['target'] for r in rows_by_phase['train']], device='cuda')
    counts = Counter(y.tolist())
    weights = torch.tensor([(3 if r.get('auxiliary') else 1) / counts[r['target']] for r in rows_by_phase['train']], device='cuda')
    weights /= weights.mean()
    optimizer = torch.optim.AdamW(head.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    best = baseline['planning']['macro_f1']
    best_epoch = 0
    best_state = {k: v.detach().cpu().clone() for k,v in head.state_dict().items()}
    history = []
    for epoch in range(1, config['epochs'] + 1):
        head.train()
        optimizer.zero_grad(set_to_none=True)
        loss = (F.cross_entropy(head(x,z)/config['temperature'], y, reduction='none') * weights).mean()
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 10, error_if_nonfinite=True)
        optimizer.step()
        m = metrics([r['target'] for r in rows_by_phase['planning']], probabilities(head, *arrays['planning'], config['temperature']))
        if m['macro_f1'] > best:
            best, best_epoch = m['macro_f1'], epoch
            best_state = {k: v.detach().cpu().clone() for k,v in head.state_dict().items()}
        history.append({'epoch': epoch, 'loss': loss.item(), 'planning_macro_f1': m['macro_f1']})
    torch.save({'model_state': head.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': config['epochs'],
        'config': config, 'rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all()}, out / 'last.pt')
    torch.save({'model_state': best_state, 'epoch': best_epoch, 'config': config, 'labels': LABELS,
        'encoder_sha256': meta['weight_sha256']}, out / 'best.pt')
    save_json(out / 'history.json', history)
    save_json(out / 'selection_lock.json', {'checkpoint_sha256': sha256(out / 'best.pt'), 'epoch': best_epoch,
        'planning_macro_f1': best, 'locked_at': datetime.now(timezone.utc).isoformat(), 'no_test_based_selection': True})
    head.load_state_dict(best_state)
    initial = ResidualHead('linear', classes=8).cuda()
    results = {}
    for phase in ('planning', 'selection', 'test'):
        truth = [r['target'] for r in rows_by_phase[phase]]
        before = probabilities(initial, *arrays[phase], config['temperature'])
        after = probabilities(head, *arrays[phase], config['temperature'])
        results[phase] = {'baseline': metrics(truth, before), 'candidate': metrics(truth, after)}
        np.savez(out / f'{phase}_predictions.npz', baseline=before, candidate=after)
    save_json(out / 'results.json', results)
    verify_backup(backup)
    save_json(out / 'status.json', {'stage': 'completed_preliminary_web_experiment', 'deployed': False,
        'production_unchanged': True, 'best_epoch': best_epoch,
        'project_validation': 'pending separate positive-only evaluation; eight-class street accuracy not established'})
    print({p: {k: round(v['top1'],4) for k,v in r.items()} for p,r in results.items()}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    main(parser.parse_args().out.resolve())
