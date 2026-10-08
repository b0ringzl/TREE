"""Locked-checkpoint-only Aquilaria regression; no encoder, fitting or deployment.

Reuses the fixed historical human binary 92-photo Demo-letterbox x/z cache.
The challenge is already inspected regression evidence, never fresh test data
or a checkpoint-selection input. Negative source names are NOT species truth.
"""
import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ['HF_HUB_OFFLINE'] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

import numpy as np
import torch

import train_artocarpus_user_20260918 as experiment
from distillation_core import digest, read_json, save_json
from training_safety import sha256, verify_backup

RUN = ROOT / 'outputs/SSTv1_Artocarpus_User_20260918_v1'
CACHE = ROOT / 'outputs/SST_Conserve_Browser_20260917_v2/post_training_human_regression'
CHALLENGE = ROOT / 'outputs/SST_Conserve_v1_20260917_human_challenge2'
DESTINATION_NAME = 'post_training_aq92_regression'
AQ = 11
NATIVE_CLASSES = 148
ENCODER_SHA256 = 'ac2e37c2f89ef8e6b889176a9a3f418970ad9db15a218bd29e3321e95c46ae97'
ROWS_HASH = '048df701a3e861ce60a99b5c90604df90a70a5b5ee4ca8574ccbe2a82203a756'
FIXED_FILES = {
    CACHE / 'features.npz': '0198f7d4491f1f58f8c589759e94fcadfea8a5f70942990d517230f20a95a885',
    CACHE / 'challenge_rows.json': '612a5355d6c6498573a0dd5c878901f95848f5cd6c9019d7f18d3c8fefebe4dd',
    CACHE / 'protocol.json': '024e8f36f2453dde77f1b86d89a70abac461a3d7b519a8e13b51f09c5f2c45c1',
    CACHE.parent / 'classes.json': 'c327c18b1d94e3cd0d7eacc282bda5d4455f8d6b182535343d8a4cc27950e078',
    CHALLENGE / 'protocol.json': 'b82e82c800cf06ee3fe5a94cd2418fa553300dfedf617872fa8dbf2aa8b7daa6',
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def image_hash(row):
    value = row.get('sha256') or row.get('source_sha256')
    require(isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None,
            'Missing or malformed source image SHA-256')
    return value


def identity_tokens(row):
    """Normalize legacy observation spellings without inventing physical trees."""
    tokens = {'sha:' + image_hash(row)}
    groups = [row.get(key) for key in ('group_id', 'split_group', 'source_group')]
    groups += row.get('resolved_source_groups', [])
    for group in groups:
        if not group:
            continue
        text = str(group)
        tokens.add('group:' + text)
        for value in re.findall(r'(?:inat_observation_|observation[:_](?:inat[:_])?)(\d+)', text):
            tokens.add('observation:inat:' + value)
    for key in ('inat_observation_id', 'observation_id'):
        value = row.get(key)
        if value is not None and str(value).isdigit():
            tokens.add('observation:inat:' + str(value))
    for value in row.get('resolved_observation_ids', []):
        if str(value).startswith('inat:'):
            tokens.add('observation:' + str(value))
    return tokens


def verify_selection_lock(run):
    """This must execute before loading a checkpoint or constructing any head."""
    path = run / 'selection_lock.json'
    require(path.is_file(), 'No selection_lock.json: regression is forbidden before model selection')
    lock_sha = sha256(path)
    lock = read_json(path)
    require(lock.get('no_test_predictions_used') is True, 'Selection did not exclude test predictions')
    require(lock.get('shared_member_epoch') is True, 'Members must share the selected epoch')
    require(lock.get('deployed') is False, 'Expected the undeployed locked candidate')
    require(isinstance(lock.get('epoch'), int) and lock['epoch'] >= 0, 'Invalid selected epoch')
    selected = Path(lock['selected_checkpoint']).resolve()
    require(selected.is_relative_to(run.resolve()) and selected.is_file(),
            'Selected checkpoint is missing or outside the fixed run')
    require(sha256(selected) == lock['sha256'], 'Locked candidate checkpoint SHA-256 mismatch')
    require(sha256(path) == lock_sha, 'Selection lock changed while checking it')
    return lock, lock_sha, selected


def binary_metrics(truth, probabilities):
    """Only human AQ/non-AQ truth; full-vocabulary argmax, no tuned threshold."""
    y = np.asarray(truth)
    p = np.asarray(probabilities)
    require(y.ndim == 1 and len(y) > 0 and set(y.tolist()) <= {0, 1}, 'Invalid binary truth')
    require(p.shape == (len(y), NATIVE_CLASSES), 'Unexpected probability shape')
    require(np.isfinite(p).all() and np.all(p >= 0), 'Invalid probabilities')
    require(np.allclose(p.sum(1), 1., atol=1e-5), 'Unnormalized probabilities')
    positive = p.argmax(1) == AQ
    tp = int(((y == 1) & positive).sum())
    fn = int(((y == 1) & ~positive).sum())
    fp = int(((y == 0) & positive).sum())
    tn = int(((y == 0) & ~positive).sum())
    return dict(n=len(y), positive=tp + fn, negative=fp + tn,
                tp=tp, fn=fn, fp=fp, tn=tn,
                precision=tp / (tp + fp) if tp + fp else None,
                recall=tp / (tp + fn) if tp + fn else None,
                f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
                accuracy=(tp + tn) / len(y),
                false_positive_rate=fp / (fp + tn) if fp + tn else None)


def preflight(run):
    require(run.resolve() == RUN.resolve(), 'Only the fixed Artocarpus user run is authorized')
    lock, lock_sha, selected = verify_selection_lock(run)
    require(not (run / DESTINATION_NAME).exists(), 'Regression output already exists; refusing overwrite')
    protocol = read_json(run / 'protocol.json')
    require(protocol['encoder_sha256'] == ENCODER_SHA256, 'Training encoder differs from frozen cache')
    require(protocol['feature_batch_size'] == 1, 'Training feature batch contract changed')
    require(protocol['feature_x'] == 'normalized embedding e rounded float16 then float32 for MLP',
            'Training x rounding contract changed')
    require(protocol['feature_z'] == 'raw text logits computed from e.float BEFORE float16 rounding',
            'Training z contract changed; do not mix reconstructed and original raw z')
    print('PREFLIGHT locked candidate; verifying backup and protected deployment', flush=True)
    experiment.unchanged(protocol)
    experiment.check_prepared(run, protocol)
    verify_backup(lock['backup'])
    require(sha256(experiment.TEACHER) == ENCODER_SHA256, 'Actual frozen encoder changed')
    for path, expected in FIXED_FILES.items():
        require(sha256(path) == expected, 'Historical fixed artifact changed: ' + str(path))
    historical = read_json(CHALLENGE / 'protocol.json')
    historical_inputs = {}
    for filename, key in (
        ('challenge_rows.json', 'rows_sha256'),
        ('human_review.sqlite3', 'database_snapshot_sha256'),
        ('source_manifest.json', 'source_manifest_sha256'),
        ('source_group_audit.json', 'source_group_audit_sha256'),
    ):
        path = CHALLENGE / filename
        require(sha256(path) == historical[key], 'Human-review snapshot changed: ' + filename)
        historical_inputs[path] = historical[key]
    rows = read_json(CACHE / 'challenge_rows.json')
    require(rows == read_json(CHALLENGE / 'challenge_rows.json'), 'Cached rows differ from human snapshot')
    require(len(rows) == 92 and len({image_hash(r) for r in rows}) == 92, 'Fixed 92-image cohort changed')
    require(len({r['group_id'] for r in rows}) == 83, 'Fixed observation grouping changed')
    truth = np.array([r['binary_target'] for r in rows], dtype=np.int64)
    require(set(truth.tolist()) == {0, 1} and int(truth.sum()) == 33,
            'Expected 33 human positives and 59 human negatives')
    for row in rows:
        review = row['human_review']
        require(review['reviewer_identity'] == 'human', 'Non-human truth is forbidden')
        require(review['decision'] in ('aquilaria', 'not_aquilaria'), 'Uncertain human decision')
        require(row['binary_target'] == int(review['decision'] == 'aquilaria'), 'Human truth mismatch')
        require(review['sha256'] == image_hash(row), 'Human decision belongs to another image')
        require(sha256(row['source_image']) == image_hash(row), 'Human-reviewed source image changed')
    require(digest(rows) == ROWS_HASH, 'Historical row digest changed')
    cache_protocol = read_json(CACHE / 'protocol.json')
    require(cache_protocol['negative_source_species_used_as_classification_truth'] is False,
            'Unexpected historical truth contract')
    expected_prep = ('prepare_pixels: EXIF, RGB, gray letterbox <=1024, JPEG95, decode224; '
                     'batch1 frozen encoder; raw z from unrounded e, x=e.half().float.')
    require(cache_protocol['preprocessing'] == expected_prep, 'Historical preprocessing differs')
    with np.load(CACHE / 'features.npz', allow_pickle=False) as cache:
        require(set(cache.files) == {'x', 'z', 'rows_hash', 'encoder_sha256'}, 'Unexpected feature schema')
        require(str(cache['rows_hash']) == ROWS_HASH, 'Cached row ordering differs')
        require(str(cache['encoder_sha256']) == ENCODER_SHA256, 'Cached encoder differs')
        x, z = cache['x'].copy(), cache['z'].copy()
    require(x.shape == (92, 1024) and z.shape == (92, 148), 'Historical feature shape changed')
    require(x.dtype == np.float32 and z.dtype == np.float32, 'Historical feature dtype changed')
    require(np.isfinite(x).all() and np.isfinite(z).all(), 'Nonfinite historical features')
    train = read_json(run / 'train_rows.json')
    require(all(r['phase'] == 'train' for r in train), 'Training manifest has non-training rows')
    train_tokens = set().union(*(identity_tokens(r) for r in train))
    challenge_tokens = set().union(*(identity_tokens(r) for r in rows))
    overlap = sorted(train_tokens & challenge_tokens)
    require(not overlap, 'Training/challenge image or observation identity overlap: ' + str(overlap[:10]))
    labels, classes = read_json(run / 'labels.json'), read_json(run / 'classes.json')
    require(len(labels) == NATIVE_CLASSES and labels[AQ] == 'Aquilaria sinensis', 'AQ index/labels changed')
    require(classes == read_json(CACHE.parent / 'classes.json'), 'Raw text-logit class mapping changed')
    with np.load(run / 'train_features.npz', allow_pickle=False) as f:
        require(str(f['rows_hash']) == digest(train), 'Training feature row digest differs')
        require(str(f['encoder_sha256']) == ENCODER_SHA256, 'Training feature encoder differs')
        extra, scale = f['extra_bank'].copy(), float(f['scale'])
    require(extra.ndim == 2 and extra.shape[1] == 1024 and np.isfinite(extra).all(), 'Invalid extra bank')
    require(len(classes['groups']) + len(extra) == NATIVE_CLASSES, 'Output vocabulary changed')
    require(np.isfinite(scale) and scale > 0 and protocol['temperature'] > 0, 'Invalid scale/temperature')
    state = torch.load(selected, map_location='cpu', weights_only=False)
    require(state['epoch'] == lock['epoch'], 'Candidate epoch differs from lock')
    require(state['protocol'] == protocol and state['classes'] == classes, 'Candidate contract differs')
    model_state = state['model_state']
    require(isinstance(model_state, dict) and model_state, 'Missing ModuleList checkpoint weights')
    require({key.split('.')[0] for key in model_state} == {'0', '1', '2'}, 'Expected three jointly selected heads')
    require(all(torch.is_tensor(value) and bool(torch.isfinite(value).all()) for value in model_state.values()),
            'Invalid candidate tensor')
    require(len(protocol['deployment_manifest']['members']) == 3, 'Baseline member count differs')
    paths = [run / name for name in ('selection_lock.json', 'protocol.json', 'train_rows.json',
             'train_features.npz', 'classes.json', 'labels.json', 'contract.json', 'feature_contract.json')]
    paths += [selected, Path(__file__), Path(experiment.__file__), Path(lock['backup']) / 'backup_manifest.json']
    before = {str(p.resolve()): sha256(p) for p in paths}
    before.update({str(p.resolve()): value for p, value in {**FIXED_FILES, **historical_inputs}.items()})
    require(before[str((run / 'selection_lock.json').resolve())] == lock_sha, 'Selection changed during preflight')
    print('PREFLIGHT PASS 92 photos, 33 positive, 59 negative, 83 observation groups; no overlap', flush=True)
    return dict(lock=lock, lock_sha=lock_sha, selected=selected, protocol=protocol, rows=rows,
                truth=truth, x=x, z=z, extra=extra, scale=scale, classes=classes, labels=labels,
                model_state=model_state, before=before,
                overlap=dict(training_photos=len(train), challenge_photos=92,
                             observation_groups=83, identity_overlap=overlap,
                             caveat='Observation groups are not proven independent physical trees.'))


def assert_unchanged(context):
    for path, expected in context['before'].items():
        require(sha256(path) == expected, 'Source artifact changed during evaluation: ' + path)
    for row in context['rows']:
        require(sha256(row['source_image']) == image_hash(row), 'Challenge image changed during evaluation')
    experiment.unchanged(context['protocol'])
    verify_backup(context['lock']['backup'])


def evaluate(run):
    c = preflight(run.resolve())
    # No Engine.load(), encoding, head construction, or model prediction above this point.
    require(torch.cuda.is_available(), 'The root head functions require CUDA; no CPU fallback mutation')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    baseline = experiment.heads_from_deployment(c['protocol']).eval().requires_grad_(False)
    candidate = copy.deepcopy(baseline).eval().requires_grad_(False)
    candidate.load_state_dict(c['model_state'], strict=True)
    data = (c['rows'], torch.tensor(c['x'], device='cuda'), torch.tensor(c['z'], device='cuda'))
    params = dict(groups=c['classes']['groups'], temp=c['protocol']['temperature'],
                  extra=torch.tensor(c['extra'], device='cuda'), scale=c['scale'])
    with torch.inference_mode():
        probabilities = {'baseline': experiment.predict(baseline, data, params),
                         'candidate': experiment.predict(candidate, data, params)}
    results = {key: binary_metrics(c['truth'], p) for key, p in probabilities.items()}
    b, q = [probabilities[name].argmax(1) == AQ for name in ('baseline', 'candidate')]
    correct = c['truth'].astype(bool)
    records = []
    for i, row in enumerate(c['rows']):
        record = dict(id=row['id'], source_image=row['source_image'], sha256=image_hash(row),
                      group_id=row['group_id'], truth_is_aquilaria=int(c['truth'][i]),
                      human_decision=row['human_review']['decision'])
        for name, p in probabilities.items():
            top = int(p[i].argmax())
            record[name] = dict(top1_species=c['labels'][top], top1_score=float(p[i, top]),
                                aquilaria_score=float(p[i, AQ]), aquilaria_positive=top == AQ)
        records.append(record)
    assert_unchanged(c)
    destination = run / DESTINATION_NAME
    destination.mkdir(exist_ok=False)
    for source, name in ((run / 'selection_lock.json', 'selection_lock.json'),
                         (CACHE / 'challenge_rows.json', 'challenge_rows.json'),
                         (CHALLENGE / 'protocol.json', 'human_snapshot_protocol.json'),
                         (Path(__file__), 'evaluation_code.py')):
        shutil.copy2(source, destination / name)
        require(sha256(source) == sha256(destination / name), 'Snapshot copy failed')
    scope = 'Previously inspected historical Aquilaria human binary regression, NOT fresh independent evidence.'
    report = dict(scope=scope, selected_epoch=c['lock']['epoch'], selected_sha256=c['lock']['sha256'],
                  results=results, used_for_selection=False, threshold_tuned=False, trained=False,
                  encoder_inference=False, deployed=False, production_unchanged=True,
                  negative_source_species_used_as_truth=False,
                  binary_corrected_ids=[r['id'] for i, r in enumerate(c['rows']) if b[i] != correct[i] and q[i] == correct[i]],
                  binary_regressed_ids=[r['id'] for i, r in enumerate(c['rows']) if b[i] == correct[i] and q[i] != correct[i]])
    report['candidate_minus_baseline'] = {
        key: results['candidate'][key] - results['baseline'][key]
        if results['candidate'][key] is not None and results['baseline'][key] is not None else None
        for key in ('tp', 'fn', 'fp', 'tn', 'precision', 'recall', 'f1', 'accuracy')}
    save_json(destination / 'protocol.json', dict(
        created_at=datetime.now(timezone.utc).isoformat(), scope=scope, input_hashes=c['before'],
        selection_locked_before_inference=True, selected_checkpoint=str(c['selected']),
        selected_checkpoint_sha256=c['lock']['sha256'], selection_lock_sha256=c['lock_sha'],
        rule='Positive iff full 148-class top1 is index11 Aquilaria sinensis; no threshold tuning.',
        truth='Fixed human binary_target only: 33 Aquilaria / 59 non-Aquilaria.',
        cache=str(CACHE / 'features.npz'), rows_hash=ROWS_HASH, encoder_sha256=ENCODER_SHA256,
        features='Verified original Demo-pad x/z reused, not recomputed from rounded x.',
        negative_source_species_used_as_truth=False, no_training=True, no_deployment=True,
        prediction_implementation='train_artocarpus_user_20260918.heads_from_deployment + predict/logprob',
        limitations=['Historical reviewed regression, not new validation or test evidence.',
                     'The 83 observations are not established independent individual trees.',
                     'No threshold, recipe, epoch, deployment or future retuning decision is made here.']))
    save_json(destination / 'overlap_audit.json', c['overlap'])
    save_json(destination / 'predictions.json', records)
    np.savez_compressed(destination / 'predictions.npz', **probabilities,
                        binary_target=c['truth'], rows_hash=ROWS_HASH)
    save_json(destination / 'evaluation.json', report)
    def pct(value):
        return 'N/A (undefined)' if value is None else f'{value:.2%}'
    lines = ['# Aquilaria historical human regression', '', scope,
             '92 photos: 33 human-confirmed Aquilaria and 59 human-confirmed non-Aquilaria; 83 observation groups.',
             'Only full-vocabulary top1 at class 11 counts as positive. Negative source species are not exact-species truth.',
             'Locked candidate vs its unchanged deployed initialization; no fitting, selection or deployment.', '',
             '| Model | TP | FN | FP | TN | Precision | Recall | F1 | Accuracy |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name, result in results.items():
        lines.append(f"| {name} | {result['tp']} | {result['fn']} | {result['fp']} | {result['tn']} | "
                     + ' | '.join(pct(result[k]) for k in ('precision', 'recall', 'f1', 'accuracy')) + ' |')
    lines += ['', f"Corrected: {len(report['binary_corrected_ids'])}; regressed: {len(report['binary_regressed_ids'])}.",
              'Original cached x/z, source-image SHA, row order, human snapshot, model lock, training exclusions and backup verified.',
              'This result does not authorize changing the selected checkpoint, fitting again or deployment.']
    (destination / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    assert_unchanged(c)
    save_json(destination / 'COMPLETE.json', dict(completed_at=datetime.now(timezone.utc).isoformat(),
        evaluation_sha256=sha256(destination / 'evaluation.json'), selected_sha256=c['lock']['sha256'],
        selection_lock_sha256=c['lock_sha'], source_artifacts_unchanged=True,
        trained=False, encoded=False, used_for_selection=False, deployed=False))
    print('AQ92 REGRESSION COMPLETE', json.dumps(results, ensure_ascii=False), str(destination), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=RUN, help='Fixed run only; no alternate cohort or checkpoint')
    args = parser.parse_args()
    evaluate(args.run.resolve())


if __name__ == '__main__':
    main()
