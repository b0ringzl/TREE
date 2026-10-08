"""Isolated BioCLIP -> YOLO classification experiments; production never modified.

Stages are independently executable. A training retry always gets a new directory.
Run with the existing yolo Python environment. Optional OpenCLIP is loaded from a
project-local dependency directory, not installed into the Demo environment.
"""
import argparse
import csv
import gc
import os
import platform
import re
import sqlite3
import sys
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('PYTHONUTF8', '1')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'external_datasets/distillation_runtime'))
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO

from distillation_core import (save_json, read_json, digest, class_groups, collapse_logits,
    canonical_image, probabilities, confidence_weight, kd_loss, metrics, grouped_metrics,
    bootstrap_gain)
from training_safety import backup_before_training, deployed_artifacts, verify_backup, sha256
from species_scale_experiment import source_rows, partition_validation, INITIAL, BASE

MODEL_DIR = ROOT / 'external_datasets/distillation_models/bioclip25'
REVISION = '6e3d04e3d6522012c88181085c5ae666e14c45cd'
SEED = 20260908


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def status(out, stage, **kwargs):
    save_json(out / 'status.json', {'stage': stage, 'updated_at': datetime.now().isoformat(),
                                  'pid': os.getpid(), 'production_deployed': False, **kwargs})
    print(stage, kwargs, flush=True)


def audit(out):
    out.mkdir(parents=True, exist_ok=False)
    (out / 'audit').mkdir()
    status(out, 'audit')
    model = YOLO(str(INITIAL))
    if model.task != 'classify':
        raise RuntimeError('This adapter supports classification only')
    model_output_names = [model.names[i] for i in range(len(model.names))]
    dataset_summary = read_json(BASE / 'datasets/full/dataset_summary.json')
    original_names = dataset_summary['class_names']
    labels, groups = class_groups(original_names)
    del model
    folder_names = [dataset_summary['class_folders'][str(i)] for i in range(len(original_names))]
    if model_output_names not in (original_names, folder_names):
        raise RuntimeError('Original model / dataset class order differs')
    rows = source_rows()
    partition_validation(rows)
    db_path = ROOT / 'external_datasets/herbarium_training_candidates_20260907/human_review.sqlite3'
    with sqlite3.connect(db_path.as_uri() + '?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        decisions = {r['id']: dict(r) for r in db.execute('SELECT * FROM decisions')}
    save_json(out / 'audit/human_review_snapshot.json', decisions)
    removed = []
    accepted = []
    for row in rows:
        if row['auxiliary']:
            decision = decisions.get(row['sample_id'].removeprefix('herbarium_'), {})
            if decision.get('status') != 'accepted' or decision.get('sha256') != row['source_sha256']:
                removed.append(row['sample_id'])
                continue
        row['target'] = labels.index(row['label'])
        row['tree_id'] = None
        row['identity_status'] = 'source_observation_group_not_verified_single_tree'
        accepted.append(row)
    rows = accepted
    # Check all image bytes, not just a path-only manifest. Preserve exclusions in audit.
    with ThreadPoolExecutor(max_workers=8) as pool:
        hashes = list(pool.map(lambda r: sha256(r['materialized_path']), rows))
    for r, actual in zip(rows, hashes):
        if actual != r['source_sha256']:
            raise RuntimeError('Source image changed: ' + r['materialized_path'])
    hash_splits, group_splits = defaultdict(set), defaultdict(set)
    for r in rows:
        hash_splits[r['source_sha256']].add(r['phase'])
        group_splits[r['group_id']].add(r['phase'])
    leaks = [{'kind': kind, 'id': key, 'phases': '|'.join(sorted(phases))}
             for kind, table in [('source_sha256', hash_splits), ('group_id', group_splits)]
             for key, phases in table.items() if len(phases) > 1]
    write_csv(out / 'audit/source_group_leakage.csv', leaks)
    if leaks:
        raise RuntimeError('Cross-split exact-image/source-group leakage; do not train')
    save_json(out / 'rows.json', rows)
    for phase in ['train', 'planning', 'selection', 'test']:
        save_json(out / f'{phase}_rows.json', [r for r in rows if r['phase'] == phase])
    save_json(out / 'classes.json', {'original_names': original_names, 'labels': labels, 'groups': groups})
    # Existing aliases are project label definitions, not newly asserted botanical synonyms.
    aliases = []
    for i, name in enumerate(original_names):
        terms = [x.strip() for x in re.split(r'[()]', name) if x.strip()]
        aliases.append({'original_index': i, 'original_label': name, 'prompt_terms': terms,
                        'validation': 'existing_project_mapping_not_independently_taxonomy_verified'})
    save_json(out / 'aliases.json', aliases)
    vmms_split = ROOT / 'hk-tree-vmms/derived/reviewed_five_species_pilot_v1_20260907/split.json'
    vmms = read_json(vmms_split)
    links = read_json(vmms_split.parent / 'leakage_links.json')
    explicit = read_json(ROOT / 'hk-tree-vmms/derived/albizia_conflict_review_v1_20260907/adjudications.json')
    save_json(out / 'audit/vmms_identity_audit.json', {
        'five_species_split_snapshot': vmms, 'leakage_links_snapshot': links,
        'explicit_adjudications': explicit,
        'finding': 'Spatial leakage blocks are not verified tree identities. Existing links explicitly mark identity_verified=false. One manually adjudicated same-tree pair is insufficient for independent multiview training/validation.',
        'multiview_status': 'BLOCKED_BY_INSUFFICIENT_VERIFIED_TREE_IDS'})
    (out / 'audit/MULTIVIEW_BLOCKED.md').write_text(
        '# Multi-view identity gate\n\nWeb observation IDs are not confirmed individual trees. '
        'VMMS spatial/point-overlap blocks prevent leakage but do not prove identity; inspected links '
        'explicitly have identity_verified=false. One explicit same-tree adjudication exists, but is '
        'insufficient for separate train/validation tree cohorts. E2/E4 are not run. No guessed Tree-ID.\n', encoding='utf-8')
    street = []
    for r in vmms:
        if r['split'] != 'heldout' or r['species'] not in labels:
            continue
        if r.get('accepted_review', {}).get('decision') != 'accept':
            continue
        if sha256(r['image']) != r['image_sha256']:
            raise RuntimeError('VMMS accepted image changed')
        street.append({'sample_id': r['id'], 'materialized_path': r['image'],
                       'source_sha256': r['image_sha256'], 'group_id': r['leakage_group'],
                       'target': labels.index(r['species']), 'phase': 'street_diagnostic',
                       'label': r['species'], 'tree_id': None})
    save_json(out / 'street_rows.json', street)
    protocol = {
        'seed': SEED, 'task': 'classification', 'architecture': 'YOLO11s-cls',
        'original_outputs': len(original_names), 'supervised_and_evaluated_labels': len(labels),
        'label_policy': 'Merge Ficus outputs using logsumexp before CE/KD; preserve 148-output architecture. No new Other prompt. No silent inclusion of Wodyetia/Leucaena absent from output catalog.',
        'initial_weight': str(INITIAL), 'initial_sha256': sha256(INITIAL),
        'teacher': 'imageomics/bioclip-2.5-vith14', 'teacher_revision': REVISION,
        'class_hash': digest(original_names), 'manifest_sha256': sha256(out / 'rows.json'),
        'phases': dict(Counter(r['phase'] for r in rows)), 'rejected_since_previous_snapshot': removed,
        'preprocessing': 'EXIF RGB; bicubic shortest-side resize 224; center crop 224. Identical pixels; YOLO /255; BioCLIP official channel normalization. No stochastic geometric augmentation.',
        'split_policy': 'Reuse existing source groups and planning/selection partitions. No observed exact hash or source group overlap. Not a verified tree-level benchmark.',
        'selection': 'Planning chooses teacher prompt/calibration and head epochs; selection compares KD hyperparameters and full-fit epochs. Final configurations locked before test. Historical validation exposure disclosed.',
        'test_policy': 'Historical regression test, not fresh blind evidence. Teacher pretraining overlap with internet images is unknown. No test inference until lock.',
        'teacher_temperature': 'Calibration on planning NLL, separate from KD softening T. Cache raw similarities/logit_scale then calibrated probabilities.',
        'kd': 'Confidence-weighted KL teacher||student plus primary GT CE. Teacher never replaces GT. Matched GT-only continuation control.',
        'sampling': 'Class-balanced weighted loss; accepted herbarium multiplier 3. Deterministic shuffling. All accepted current auxiliary images included.',
        'screen': {'epochs': 40, 'temperatures': [2., 4.], 'lambda': [.1, .3, .5], 'backbone': 'frozen'},
        'full': {'epochs': 3, 'batch': 64, 'lr': 0.00001, 'optimizer': 'AdamW', 'backbone': 'trainable'},
        'deployment': 'Never automatic; existing image, point and fusion weights protected',
        'environment': {'python': platform.python_version(), 'torch': torch.__version__,
                        'gpu': torch.cuda.get_device_name(0), 'bf16': torch.cuda.is_bf16_supported()},
        'protected_sources': {str(p): sha256(p) for p in deployed_artifacts()}}
    save_json(out / 'protocol.json', protocol)
    write_csv(out / 'audit/dataset_statistics.csv', [
        {'species': label, **{phase: sum(r['label'] == label and r['phase'] == phase for r in rows)
          for phase in ['train', 'planning', 'selection', 'test']}} for label in labels])
    save_json(out / 'audit/task_type.json', {'task': 'classification', 'num_classes': len(labels),
        'model_outputs': len(original_names), 'multiview': 'BLOCKED', 'exact_hash_group_leakage': False,
        'verified_tree_leakage': 'UNKNOWN', 'street_diagnostic_images': len(street)})
    status(out, 'audit_complete', counts=protocol['phases'])


def load_batch(pool, rows):
    return torch.stack(list(pool.map(lambda r: canonical_image(r['materialized_path']), rows)))


def teacher(out, phases):
    import open_clip
    protocol = read_json(out / 'protocol.json')
    aliases = read_json(out / 'aliases.json')
    classes = read_json(out / 'classes.json')
    weight = MODEL_DIR / 'open_clip_model.safetensors'
    if not weight.exists():
        raise FileNotFoundError('BioCLIP weight download not complete')
    status(out, 'teacher_loading')
    precision = 'bf16' if torch.cuda.is_bf16_supported() else 'fp16'
    model, _, _ = open_clip.create_model_and_transforms('local-dir:' + str(MODEL_DIR.resolve()),
                    precision=precision, device='cuda')
    model.eval().requires_grad_(False)
    tokenizer = open_clip.get_tokenizer('ViT-H-14')
    templates = ['{}', 'a photo of {}', 'a photo of the tree species {}']
    banks = []
    with torch.inference_mode():
        for template in templates + [None]:
            bank = []
            for a in aliases:
                prompts = ([template.format(name) for name in a['prompt_terms']] if template else
                    [t.format(name) for name in a['prompt_terms'] for t in
                     ['{}', 'a photo of {}', 'a street tree of {}', 'a photo of the tree species {}']])
                embedded = model.encode_text(tokenizer(prompts).to('cuda'), normalize=True).float()
                bank.append(F.normalize(embedded.mean(0), dim=0))
            banks.append(torch.stack(bank))
        banks = torch.stack(banks)
        scale = model.logit_scale.exp().float().item()
    cache = out / 'teacher_cache/bioclip25'
    cache.mkdir(parents=True, exist_ok=True)
    meta = {'model': protocol['teacher'], 'revision': REVISION, 'weight_sha256': sha256(weight),
            'class_names': classes['original_names'], 'class_hash': protocol['class_hash'],
            'manifest_hash': protocol['manifest_sha256'], 'preprocessing': protocol['preprocessing'],
            'precision': precision, 'logit_scale': scale, 'aliases_sha256': sha256(out / 'aliases.json'),
            'format': 'raw per-prompt similarities, uncalibrated logits = similarities * logit_scale'}
    existing = cache / 'metadata.json'
    if existing.exists() and read_json(existing) != meta:
        raise RuntimeError('Teacher cache metadata mismatch')
    save_json(existing, meta)
    mean = torch.tensor([.48145466, .4578275, .40821073], device='cuda')[None, :, None, None]
    std = torch.tensor([.26862954, .26130258, .27577711], device='cuda')[None, :, None, None]
    dtype = torch.bfloat16 if precision == 'bf16' else torch.float16
    with ThreadPoolExecutor(max_workers=8) as pool, torch.inference_mode():
        for phase in phases:
            if phase == 'test' and not (out / 'final_lock.json').exists():
                raise RuntimeError('Test teacher inference before final configuration lock')
            rows = read_json(out / f'{phase}_rows.json')
            if not rows:
                continue
            path = cache / f'{phase}.npz'
            if path.exists():
                z = np.load(path)
                if str(z['rows_hash']) != digest(rows):
                    raise RuntimeError('Cache row order mismatch')
                continue
            chunks = cache / (phase + '_chunks')
            chunks.mkdir(exist_ok=True)
            similarities, embeddings, start, batch = [], [], 0, 16
            tic = time.perf_counter()
            while start < len(rows):
                stop = min(start + batch, len(rows))
                chunk = chunks / f'{start:06d}_{stop:06d}.npz'
                if chunk.exists():
                    z = np.load(chunk)
                    if str(z['rows_hash']) != digest(rows[start:stop]):
                        raise RuntimeError('Chunk cache mismatch')
                    sim, emb = z['similarities'], z['embeddings']
                else:
                    try:
                        x = ((load_batch(pool, rows[start:stop]).to('cuda') - mean) / std).to(dtype)
                        emb_t = model.encode_image(x, normalize=True).float()
                        sim = torch.einsum('bd,pcd->pbc', emb_t, banks).cpu().numpy()
                        emb = emb_t.cpu().numpy().astype(np.float16)
                        if not np.isfinite(sim).all() or not np.isfinite(emb).all():
                            raise RuntimeError('Teacher NaN/Inf')
                        np.savez(chunk, similarities=sim, embeddings=emb, rows_hash=digest(rows[start:stop]))
                    except torch.cuda.OutOfMemoryError:
                        if batch <= 1:
                            raise
                        batch = max(1, batch // 2)
                        gc.collect(); torch.cuda.empty_cache()
                        continue
                similarities.append(sim); embeddings.append(emb); start = stop
                if start % 256 == 0 or start == len(rows):
                    status(out, 'teacher_inference', phase=phase, completed=start, total=len(rows),
                           elapsed_seconds=round(time.perf_counter() - tic, 1))
            np.savez(path, similarities=np.concatenate(similarities, axis=1),
                     embeddings=np.concatenate(embeddings), rows_hash=digest(rows))
            save_json(cache / f'{phase}_timing.json', {'elapsed_seconds_including_chunk_reuse': time.perf_counter() - tic,
                'images': len(rows), 'batch': batch, 'benchmark': 'pipeline wall time, not isolated GPU latency'})
    del model, banks
    gc.collect(); torch.cuda.empty_cache()


def student_features(out, phases):
    model = YOLO(str(INITIAL)).model.float().cuda().eval()
    head = model.model[-1]
    cache = out / 'student_cache'
    cache.mkdir(exist_ok=True)
    initial = cache / 'initial_head.pt'
    if not initial.exists():
        torch.save(head.linear.state_dict(), initial)
    groups = read_json(out / 'classes.json')['groups']
    captured = []
    hook = head.drop.register_forward_hook(lambda m, i, o: captured.append(o.detach()))
    try:
        with ThreadPoolExecutor(max_workers=8) as pool, torch.inference_mode():
            for phase in phases:
                rows = read_json(out / f'{phase}_rows.json')
                if not rows:
                    continue
                path = cache / f'{phase}.npz'
                if path.exists():
                    if str(np.load(path)['rows_hash']) != digest(rows):
                        raise RuntimeError('Student feature rows mismatch')
                    continue
                features, predictions = [], []
                for start in range(0, len(rows), 128):
                    captured.clear()
                    result = model(load_batch(pool, rows[start:start+128]).cuda())
                    logits = result[1]
                    features.append(captured[0].float().cpu().numpy())
                    predictions.append(collapse_logits(logits, groups).softmax(-1).cpu().numpy())
                    if start % 2048 == 0:
                        status(out, 'student_feature_extraction', phase=phase, completed=start, total=len(rows))
                np.savez(path, features=np.concatenate(features), probabilities=np.concatenate(predictions), rows_hash=digest(rows))
    finally:
        hook.remove()
    del model
    gc.collect(); torch.cuda.empty_cache()


def teacher_prob(out, phase, choice=None):
    choice = choice or read_json(out / 'teacher_choice.json')
    z = np.load(out / f'teacher_cache/bioclip25/{phase}.npz')
    meta = read_json(out / 'teacher_cache/bioclip25/metadata.json')
    raw = torch.tensor(z['similarities'][choice['prompt']] * meta['logit_scale'])
    logits = collapse_logits(raw, read_json(out / 'classes.json')['groups'])
    return probabilities(logits.numpy() / choice['calibration_temperature'])


def calibrate(out):
    rows = read_json(out / 'planning_rows.json')
    y = np.array([r['target'] for r in rows])
    trials = []
    for prompt in range(4):
        for temp in np.geomspace(.25, 8, 25):
            choice = {'prompt': prompt, 'calibration_temperature': float(temp)}
            m = metrics(y, teacher_prob(out, 'planning', choice))
            trials.append({**choice, **{k: m[k] for k in ['macro_f1', 'top1', 'nll']}})
    # Prompt uses Macro-F1; calibrated T uses NLL (T cannot change argmax).
    prompt = max(range(4), key=lambda p: max(t['macro_f1'] for t in trials if t['prompt'] == p))
    chosen = min((t for t in trials if t['prompt'] == prompt), key=lambda t: t['nll'])
    save_json(out / 'teacher_calibration_trials.json', trials)
    save_json(out / 'teacher_choice.json', chosen)
    for phase in ['planning', 'selection', 'street']:
        rows = read_json(out / f'{phase}_rows.json')
        if not rows:
            continue
        tp = teacher_prob(out, phase)
        sp = np.load(out / f'student_cache/{phase}.npz')['probabilities']
        y = np.array([r['target'] for r in rows])
        save_json(out / f'metrics/teacher_{phase}.json', metrics(y, tp))
        save_json(out / f'metrics/baseline_{phase}.json', metrics(y, sp))
        outcomes = []
        for i, r in enumerate(rows):
            a, b = sp[i].argmax(), tp[i].argmax()
            kind = ('both_correct' if a == b == y[i] else 'student_wrong_teacher_correct' if b == y[i]
                    else 'student_correct_teacher_wrong' if a == y[i] else
                    'both_wrong_same' if a == b else 'both_wrong_different')
            outcomes.append({'sample_id': r['sample_id'], 'path': r['materialized_path'], 'GT': int(y[i]),
                'student': int(a), 'teacher': int(b), 'teacher_confidence': float(tp[i, b]), 'kind': kind})
        write_csv(out / f'analysis/{phase}_hard_samples.csv', outcomes)
        save_json(out / f'analysis/{phase}_complementarity.json', dict(Counter(r['kind'] for r in outcomes)))
    status(out, 'teacher_calibrated', choice=chosen)


def weights_for(rows):
    counts = Counter(r['target'] for r in rows)
    w = np.array([(3 if r.get('auxiliary') else 1) / counts[r['target']] for r in rows], dtype=np.float32)
    return w / w.mean()


def backup_fit(out, run, initial_files, config):
    files = [*initial_files, *deployed_artifacts(), out / 'protocol.json', out / 'rows.json',
             out / 'classes.json', out / 'teacher_choice.json', out / 'audit/human_review_snapshot.json',
             out / 'metrics/baseline_selection.json', Path(__file__),
             ROOT / 'tools/distillation_core.py', ROOT / 'tools/training_safety.py',
             out / 'teacher_cache/bioclip25/metadata.json']
    save_json(run / 'config.json', config)
    return backup_before_training(run, [*files, run / 'config.json'], config)


def screen(out):
    torch.manual_seed(SEED)
    groups = read_json(out / 'classes.json')['groups']
    train = read_json(out / 'train_rows.json')
    planning = read_json(out / 'planning_rows.json')
    selection = read_json(out / 'selection_rows.json')
    x = torch.tensor(np.load(out / 'student_cache/train.npz')['features'], device='cuda')
    xv = torch.tensor(np.load(out / 'student_cache/planning.npz')['features'], device='cuda')
    xs = torch.tensor(np.load(out / 'student_cache/selection.npz')['features'], device='cuda')
    y = torch.tensor([r['target'] for r in train], dtype=torch.long, device='cuda')
    yp, ys = [r['target'] for r in planning], [r['target'] for r in selection]
    p = torch.tensor(teacher_prob(out, 'train'), dtype=torch.float32, device='cuda')
    w = torch.tensor(weights_for(train), device='cuda')
    initial = out / 'student_cache/initial_head.pt'
    state = torch.load(initial, weights_only=True)
    trials = [(0., 2.)] + [(lam, t) for t in [2., 4.] for lam in [.1, .3, .5]]
    results = []
    for lam, temp in trials:
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        run = out / f'runs/screen_{lam}_{temp}_{stamp}'
        config = {'stage': 'frozen_head_screen', 'lambda': lam, 'temperature': temp, 'seed': SEED,
                  'epochs': 40, 'lr': .001, 'backbone': 'frozen', 'dropout': .1}
        backup = backup_fit(out, run, [initial], config)
        torch.manual_seed(SEED)
        head = torch.nn.Linear(state['weight'].shape[1], state['weight'].shape[0]).cuda()
        head.load_state_dict(state)
        optimizer = torch.optim.AdamW(head.parameters(), lr=.001)
        best = -1
        for epoch in range(40):
            head.train(); optimizer.zero_grad(set_to_none=True)
            logits = collapse_logits(head(F.dropout(x, .1, training=True)), groups)
            ce = (F.cross_entropy(logits, y, reduction='none') * w).sum() / w.sum()
            loss = ce + lam * kd_loss(logits, p, temp, w)
            if not torch.isfinite(loss):
                raise RuntimeError('Head loss NaN')
            loss.backward(); optimizer.step()
            with torch.inference_mode():
                prob = collapse_logits(head(xv), groups).softmax(-1).cpu().numpy()
            score = metrics(yp, prob)['macro_f1']
            if score > best:
                best = score
                torch.save({'model_state': head.state_dict(), 'optimizer': optimizer.state_dict(),
                    'epoch': epoch, 'global_step': epoch+1, 'config': config, 'best_metric': best,
                    'class_hash': digest(read_json(out / 'classes.json'))}, run / 'best.pt')
        head.load_state_dict(torch.load(run / 'best.pt', weights_only=False)['model_state'])
        with torch.inference_mode():
            sp = collapse_logits(head(xs), groups).softmax(-1).cpu().numpy()
        m = metrics(ys, sp)
        save_json(run / 'selection_metrics.json', m)
        verify_backup(backup)
        result = {**config, 'run': str(run), 'selection_macro_f1': m['macro_f1'], 'selection_top1': m['top1']}
        results.append(result)
        save_json(out / 'screen_results.json', results)
        status(out, 'head_screen_completed_trial', result=result)
    winner = max((r for r in results if r['lambda'] > 0), key=lambda r: r['selection_macro_f1'])
    save_json(out / 'kd_choice.json', winner)
    del x, xv, xs, p, head, optimizer
    gc.collect(); torch.cuda.empty_cache()


def evaluate_model(model, rows, groups, pool):
    model.eval()
    result = []
    with torch.inference_mode():
        for start in range(0, len(rows), 128):
            output = model(load_batch(pool, rows[start:start+128]).cuda())
            raw = output[1] if isinstance(output, tuple) else output
            result.append(collapse_logits(raw, groups).softmax(-1).cpu().numpy())
    return np.concatenate(result)


def full_fit(out, lam, temp, smoke=False, resume_path=None):
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    run = out / f'runs/{"smoke" if smoke else "full"}_{"GT" if lam == 0 else "KD"}_{stamp}'
    config = {'stage': 'smoke' if smoke else 'full_backbone', 'lambda': lam, 'temperature': temp,
              'seed': SEED, 'epochs': 1 if smoke else 3, 'batch': 16 if smoke else 64, 'lr': .00001,
              'precision': 'bf16' if torch.cuda.is_bf16_supported() else 'fp16'}
    previous = None
    if resume_path is not None:
        resume_path = Path(resume_path).resolve()
        previous = torch.load(resume_path, map_location='cpu', weights_only=False)
        prior = previous['config']
        if prior['stage'] != 'full_backbone' or previous['epoch'] + 1 >= prior['epochs']:
            raise ValueError('Resume requires an incomplete full-backbone epoch checkpoint')
        if previous['class_manifest'] != read_json(out / 'classes.json'):
            raise ValueError('Resume class mapping changed')
        old_backup = verify_backup(resume_path.parent / 'pre_training_backup', verify_sources=False)
        for artifact in old_backup['files']:
            # A fix may change code, not original weights, labels or data snapshots.
            if Path(artifact['source']).suffix != '.py' and sha256(artifact['source']) != artifact['sha256']:
                raise ValueError('Non-code resume input changed: ' + artifact['source'])
        recorded_rows = [r for r in old_backup['files'] if Path(r['source']) == (out / 'rows.json')]
        if len(recorded_rows) != 1 or recorded_rows[0]['sha256'] != sha256(out / 'rows.json'):
            raise ValueError('Resume dataset snapshot mismatch')
        config = {**prior, 'resumed_from': str(resume_path)}
        lam, temp = config['lambda'], config['temperature']
        run = out / f'runs/resume_{"GT" if lam == 0 else "KD"}_{stamp}'
    resume_files = [resume_path, resume_path.parent / 'best.pt'] if resume_path else []
    backup = backup_fit(out, run, [INITIAL, *resume_files], config)
    classes = read_json(out / 'classes.json'); groups = classes['groups']
    train = read_json(out / 'train_rows.json')
    selection = read_json(out / 'selection_rows.json')
    pt = teacher_prob(out, 'train').astype(np.float32)
    if smoke:
        train = train[:32]; selection = selection[:32]; pt = pt[:32]
    weights = weights_for(train)
    torch.manual_seed(SEED); np.random.seed(SEED)
    model = YOLO(str(INITIAL)).model.float().cuda()
    model.requires_grad_(True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])
    use_bf16 = config.get('precision', 'fp16') == 'bf16'
    scaler = torch.amp.GradScaler('cuda', enabled=not use_bf16, init_scale=1024)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    best, step, history, first_epoch = -1, 0, [], 0
    if previous is not None:
        import shutil
        model.load_state_dict(previous['model_state'])
        optimizer.load_state_dict(previous['optimizer'])
        scheduler.load_state_dict(previous['scheduler'])
        scaler.load_state_dict(previous['scaler'])
        torch.set_rng_state(previous['torch_rng_state'])
        torch.cuda.set_rng_state_all(previous['cuda_rng_state'])
        best, step, first_epoch = previous['best_metric'], previous['global_step'], previous['epoch'] + 1
        history = read_json(resume_path.parent / 'history.json')[:first_epoch]
        for name in ['best.pt', 'selection_metrics.json', 'selection_probabilities.npy']:
            shutil.copy2(resume_path.parent / name, run / name)
    with ThreadPoolExecutor(max_workers=8) as pool:
        for epoch in range(first_epoch, config['epochs']):
            model.train()
            indices = np.random.default_rng(SEED + epoch).permutation(len(train))
            loss_sum = 0.
            for start in range(0, len(indices), config['batch']):
                ids = indices[start:start+config['batch']]
                x = load_batch(pool, [train[i] for i in ids]).cuda()
                y = torch.tensor([train[i]['target'] for i in ids], device='cuda', dtype=torch.long)
                p = torch.tensor(pt[ids], device='cuda')
                w = torch.tensor(weights[ids], device='cuda')
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.bfloat16 if use_bf16 else torch.float16):
                    logits = collapse_logits(model(x).float(), groups)
                    ce = (F.cross_entropy(logits, y, reduction='none') * w).mean()
                    # Global-normalized weights: matched GT/KD data weighting, no batch-ratio bias.
                    kd = kd_loss(logits, p, temp, w) * w.mean() if lam else logits.sum() * 0
                    loss = ce + lam * kd
                if not torch.isfinite(loss):
                    raise RuntimeError('Non-finite full-model loss')
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=use_bf16)
                scaler.step(optimizer); scaler.update()
                step += 1; loss_sum += loss.item()
                if start % (config['batch'] * 40) == 0:
                    status(out, 'training', run=str(run), epoch=epoch+1, completed=start, total=len(train), loss=loss.item())
            scheduler.step()
            valp = evaluate_model(model, selection, groups, pool)
            m = metrics([r['target'] for r in selection], valp)
            optimizer_steps = max((int(v['step'].item()) for v in optimizer.state.values() if 'step' in v), default=0)
            if optimizer_steps == 0:
                raise RuntimeError('No optimizer update occurred; smoke/training is not successful')
            history.append({'epoch': epoch+1, 'optimizer_steps': optimizer_steps,
                            'attempted_steps': step, 'amp_skipped_steps': step - optimizer_steps,
                            'mean_batch_loss': loss_sum / max(1, int(np.ceil(len(train)/config['batch']))),
                            'macro_f1': m['macro_f1'], 'top1': m['top1']})
            state = {'model_state': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(), 'epoch': epoch,
                'global_step': step, 'best_metric': max(best, m['macro_f1']), 'config': config,
                'torch_rng_state': torch.get_rng_state(), 'cuda_rng_state': torch.cuda.get_rng_state_all(),
                'class_manifest': classes, 'initial_sha256': sha256(INITIAL)}
            torch.save(state, run / 'last.pt')
            if m['macro_f1'] > best:
                best = m['macro_f1']
                torch.save(state, run / 'best.pt')
                save_json(run / 'selection_metrics.json', m)
                np.save(run / 'selection_probabilities.npy', valp)
            save_json(run / 'history.json', history)
            verify_backup(backup)
    # Verify real checkpoint load, not merely file existence.
    checkpoint = torch.load(run / 'best.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state'])
    if not all(torch.isfinite(v).all() for v in model.state_dict().values() if v.is_floating_point()):
        raise RuntimeError('Checkpoint failed finite check')
    del model, optimizer, state, checkpoint
    gc.collect(); torch.cuda.empty_cache()
    return {'run': str(run), 'lambda': lam, 'temperature': temp, 'selection_macro_f1': best}


def final_evaluate(out):
    lock = read_json(out / 'final_lock.json')
    student_features(out, ['test'])
    rows = read_json(out / 'test_rows.json')
    groups = read_json(out / 'classes.json')['groups']
    y = np.array([r['target'] for r in rows])
    basep = np.load(out / 'student_cache/test.npz')['probabilities']
    results = {'baseline': metrics(y, basep)}
    all_prob = {'baseline': basep}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for kind in ['gt_control', 'kd_candidate']:
            path = Path(lock[kind]['run']) / 'best.pt'
            model = YOLO(str(INITIAL)).model.float().cuda()
            model.load_state_dict(torch.load(path, map_location='cpu', weights_only=False)['model_state'])
            p = evaluate_model(model, rows, groups, pool)
            results[kind] = metrics(y, p)
            all_prob[kind] = p
            results[kind]['group_metrics'] = grouped_metrics(rows, p)
            np.save(out / f'{kind}_test_probabilities.npy', p)
            del model
            gc.collect(); torch.cuda.empty_cache()
    results['kd_vs_matched_gt_accuracy_ci'] = bootstrap_gain(y, all_prob['gt_control'], all_prob['kd_candidate'], [r['group_id'] for r in rows])
    save_json(out / 'final_test.json', results)
    labels = read_json(out / 'classes.json')['labels']
    write_csv(out / 'analysis/test_per_class_comparison.csv', [
        {'species': name, 'support': results['baseline']['support'][i],
         **{kind + '_f1': results[kind]['per_class_f1'][i] for kind in ['baseline', 'gt_control', 'kd_candidate']},
         'KD_minus_matched_GT_F1': results['kd_candidate']['per_class_f1'][i] - results['gt_control']['per_class_f1'][i]}
        for i, name in enumerate(labels)])
    write_csv(out / 'analysis/test_predictions.csv', [
        {'sample_id': r['sample_id'], 'source_group': r['group_id'], 'truth': labels[r['target']],
         **{kind + '_prediction': labels[int(p[i].argmax())] for kind, p in all_prob.items()},
         **{kind + '_confidence': float(p[i].max()) for kind, p in all_prob.items()}}
        for i, r in enumerate(rows)])
    # Figures are diagnostics of all classes, not cherry-picked improvements.
    import matplotlib.pyplot as plt
    figures = out / 'figures'; figures.mkdir(exist_ok=True)
    for kind in ['baseline', 'gt_control', 'kd_candidate']:
        fig, ax = plt.subplots(figsize=(9, 8))
        cm = np.asarray(results[kind]['confusion_matrix'])
        scaled = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        plot = ax.imshow(scaled, vmin=0, vmax=1, cmap='Blues')
        ax.set(title=kind + ' / historical regression', xlabel='Predicted class index', ylabel='True class index')
        fig.colorbar(plot, ax=ax); fig.tight_layout()
        fig.savefig(figures / f'{kind}_confusion_matrix.png', dpi=140); plt.close(fig)
    gains = np.array(results['kd_candidate']['per_class_f1']) - results['gt_control']['per_class_f1']
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.bar(np.arange(len(gains)), gains, color=np.where(gains >= 0, '#247c58', '#af4848'))
    ax.set(xlabel='Class index (see classes.json)', ylabel='F1 difference', title='KD minus matched GT continuation: all classes')
    fig.tight_layout(); fig.savefig(figures / 'per_class_gain.png', dpi=140); plt.close(fig)
    status(out, 'core_experiments_complete', results={k: v.get('macro_f1') for k, v in results.items()})


def report(out):
    summary = []
    def add(name, phase, m, state='COMPLETED'):
        summary.append({'experiment': name, 'split': phase, 'status': state,
            **{k: m.get(k, 'NA') for k in ['n', 'macro_f1', 'balanced_accuracy', 'top1', 'top5']}})
    for kind in ['baseline', 'teacher']:
        for phase in ['planning', 'selection', 'street']:
            p = out / f'metrics/{kind}_{phase}.json'
            if p.exists(): add(kind, phase, read_json(p))
    if (out / 'final_test.json').exists():
        for key, value in read_json(out / 'final_test.json').items():
            if 'macro_f1' in value: add(key, 'historical_regression_test', value)
    for name in ['multiview_teacher_E2', 'multiview_KD_E4']:
        add(name, 'NA', {}, 'BLOCKED_INSUFFICIENT_VERIFIED_TREE_IDS')
    add('DINO_feature_KD_E5', 'NA', {}, 'BLOCKED_BY_GATED_WEIGHTS' if
        (out / 'audit/DINO_STAGE_BLOCKED_BY_WEIGHTS.md').exists() else 'NOT_RUN')
    write_csv(out / 'experiment_summary.csv', summary)
    protected = read_json(out / 'protocol.json')['protected_sources'] if (out / 'protocol.json').exists() else {}
    unchanged = all(sha256(p) == expected for p, expected in protected.items())
    conclusion = 'NEED MORE EXPERIMENTS: core comparison incomplete.'
    if (out / 'final_test.json').exists():
        final = read_json(out / 'final_test.json')
        gain = final['kd_candidate']['macro_f1'] - final['gt_control']['macro_f1']
        old_gain = final['kd_candidate']['macro_f1'] - final['baseline']['macro_f1']
        conclusion = (f'KD test Macro-F1 minus matched GT continuation: {gain:+.6f}; '
                      f'minus original baseline: {old_gain:+.6f}. '
                      'Single-seed pilot only; retain production pending independent validation and user approval.')
    lines = ['# Distillation experiment report', '', '## Executive summary', '',
        f'- Current stage: {read_json(out / "status.json").get("stage") if (out / "status.json").exists() else "audit"}',
        f'- Production unchanged: {unchanged}. No automatic deployment.',
        '- Conclusion: ' + conclusion,
        '- This run preserves the existing 148-output YOLO architecture; Ficus supervision and metrics are merged into 147 labels.',
        '- Core supervised data: original web training images plus currently human-accepted herbarium images. No pending/rejected additions.',
        '- Existing web group IDs are not verified VMMS tree IDs. No claim of tree-level independent evaluation.',
        '- Small accepted VMMS holdout is a separate street diagnostic, never used for training or prompt selection.',
        '', '| Experiment | Split | Status | Macro-F1 | Top-1 |', '|---|---|---|---:|---:|']
    for r in summary:
        lines.append(f'| {r["experiment"]} | {r["split"]} | {r["status"]} | {r["macro_f1"]} | {r["top1"]} |')
    lines += ['', '## Reliability limits', '',
        '- Historical test is a regression set, not a new blind benchmark. Old model validation exposure remains.',
        '- BioCLIP internet pretraining overlap with the web benchmark is unknown; generalization claims require a clean field benchmark.',
        '- Multi-view spatial linkage is not verified identity. E2/E4 remain blocked rather than guessing Tree-ID.',
        '- Only one seed and three full-backbone epochs in this initial controlled pilot. A 0.5-point gain alone does not prove significance.',
        '- Macro-F1 averages classes with GT support; street diagnostic has few supported classes and must not be compared directly to full web Macro-F1.',
        '- Teacher confidence is candidate-set conditional, not calibrated open-world recognition probability.',
        '- Cached teachers and students use the same deterministic spatial crop; no claim of optimal augmentation.',
        '', '## Recovery and artifacts', '',
        '- Each fit has its own runs/*/pre_training_backup with physically copied weights and SHA256 verification.',
        '- Full fits save model/optimizer/scheduler/scaler/RNG/config in last.pt and best.pt.',
        '- Teacher chunks are resumable inference caches, not resumed training. Any training retry requires a new fit directory.',
        '- See protocol.json, audit/, analysis/, teacher_choice.json, screen_results.json, final_lock.json and final_test.json when present.',
        '- NOT_RUN/BLOCKED entries are not completed experiments. FAILURES.md records actual stage failures.',
        '', '## Sources', '',
        '- https://huggingface.co/imageomics/bioclip-2.5-vith14',
        '- https://github.com/facebookresearch/dinov3']
    (out / 'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    (out / 'NEXT_STEPS.md').write_text('# Next steps\n\n'
        '1. Finish any incomplete single-view matched GT/KD stages before interpreting gains.\n'
        '2. Verify independent VMMS tree identities and more held-out trees before E2/E4.\n'
        '3. Use real teacher/student complementarity and per-class regressions to decide further KD.\n'
        '4. Run DINO global feature ablation separately from view consistency; do not combine untested changes.\n'
        '5. Require user approval before any production model switch.\n', encoding='utf-8')
    save_json(out / 'production_verification.json', {'unchanged': unchanged, 'sources': protected})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--stage', choices=['audit', 'features', 'teacher', 'calibrate', 'screen', 'full', 'resume', 'final', 'report', 'pipeline'], required=True)
    parser.add_argument('--resume-checkpoint', type=Path)
    args = parser.parse_args(); out = args.out.resolve()
    try:
        if args.stage == 'audit': audit(out)
        elif args.stage == 'features': student_features(out, ['planning', 'selection', 'street', 'train'])
        elif args.stage == 'teacher': teacher(out, ['planning', 'selection', 'street', 'train'])
        elif args.stage == 'calibrate': calibrate(out)
        elif args.stage == 'screen': screen(out)
        elif args.stage == 'resume':
            if args.resume_checkpoint is None:
                raise ValueError('--resume-checkpoint is required')
            resumed = full_fit(out, 0, 2, resume_path=args.resume_checkpoint)
            save_json(out / ('resume_result_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.json'), resumed)
        elif args.stage == 'full':
            choice = read_json(out / 'kd_choice.json')
            full_fit(out, choice['lambda'], choice['temperature'], smoke=True)
            control = full_fit(out, 0, choice['temperature'])
            candidate = full_fit(out, choice['lambda'], choice['temperature'])
            save_json(out / 'final_lock.json', {'gt_control': control, 'kd_candidate': candidate,
                'selection_policy': 'Both preregistered matched arms, best epoch each on selection validation. No test-based reselection.'})
        elif args.stage == 'final': final_evaluate(out)
        elif args.stage == 'pipeline':
            for stage in ['features', 'teacher', 'calibrate', 'screen', 'full', 'final']:
                import subprocess
                result = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--out', str(out), '--stage', stage])
                if result.returncode:
                    raise RuntimeError(f'Dependent stage {stage} failed; no downstream fake result')
    except Exception:
        error = traceback.format_exc()
        out.mkdir(parents=True, exist_ok=True)
        with (out / 'FAILURES.md').open('a', encoding='utf-8') as f:
            f.write(f'\n## {datetime.now().isoformat()} {args.stage}\n\n```\n{error}\n```\n')
        status(out, 'FAILED_' + args.stage)
        raise
    finally:
        if (out / 'protocol.json').exists(): report(out)


if __name__ == '__main__':
    main()
