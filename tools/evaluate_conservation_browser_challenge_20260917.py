"""Post-selection binary Aquilaria regression only; never fit or select a model.

The 92 historical human-reviewed photos have Aquilaria/non-Aquilaria truth,
not trusted 150-species labels. Features are re-extracted with the deployed
frozen encoder and the same letterbox preprocessing used by the browser run.
"""
import argparse
import gc
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ['HF_HUB_OFFLINE'] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

import train_conservation_browser_20260917 as experiment
from distillation_core import digest, read_json, save_json
from train_sstv1_incremental import prepare_pixels
from training_safety import sha256, verify_backup

DEFAULT_RUN = ROOT / 'outputs/SST_Conserve_Browser_20260917_v2'
CHALLENGE = ROOT / 'outputs/SST_Conserve_v1_20260917_human_challenge2'
AQ = 11


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def image_hash(row):
    result = row.get('sha256') or row.get('source_sha256')
    require(bool(result), 'An evaluation or fitting row has no image SHA-256')
    return result


def binary_argmax_metrics(truth, probabilities):
    """Only answer whether Aquilaria is the first-ranked species."""
    truth = np.asarray(truth, dtype=np.int64)
    require(set(truth.tolist()) <= {0, 1}, 'Non-binary human truth')
    p = np.asarray(probabilities)
    require(p.ndim == 2 and len(p) == len(truth), 'Probability row mismatch')
    require(np.isfinite(p).all() and np.all(p >= 0), 'Invalid probabilities')
    require(np.allclose(p.sum(1), 1, atol=1e-5), 'Probabilities not normalized')
    positive = p.argmax(1) == AQ
    tp = int(((truth == 1) & positive).sum())
    fn = int(((truth == 1) & ~positive).sum())
    fp = int(((truth == 0) & positive).sum())
    tn = int(((truth == 0) & ~positive).sum())
    return {
        'n': len(truth), 'positive': tp + fn, 'negative': fp + tn,
        'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn,
        'recall': tp / (tp + fn) if tp + fn else None,
        'fpr': fp / (fp + tn) if fp + tn else None,
        'precision': tp / (tp + fp) if tp + fp else None,
        'binary_accuracy': (tp + tn) / len(truth) if len(truth) else None,
        'rule': 'positive iff argmax over all branch candidates is Aquilaria sinensis',
    }


def preflight(run):
    require(run.is_dir(), 'Training run does not exist')
    lock_path = run / 'selection_lock.json'
    require(lock_path.is_file(), 'Select and lock a candidate before regression')
    lock = read_json(lock_path)
    require(lock.get('selection_uses_validation_only') is True,
            'Expected a validation-only model-selection lock')
    selected = Path(lock['selected']).resolve()
    require(selected.is_relative_to(run), 'Selected checkpoint is outside this run')
    require(selected.is_file() and sha256(selected) == lock['sha256'],
            'Locked candidate checkpoint hash mismatch')
    experiment.verify_contract(run)
    verify_backup(lock['backup'])
    cfg = read_json(run / 'config.json')
    labels = read_json(run / 'labels.json')
    require(len(labels) == 150 and labels[AQ] == 'Aquilaria sinensis',
            'Unexpected output label mapping')
    require([labels[i] for i in cfg['target_indices']] == experiment.TARGETS,
            'Target residual mapping changed')
    original_protocol = read_json(CHALLENGE / 'protocol.json')
    for filename, key in (
        ('challenge_rows.json', 'rows_sha256'),
        ('human_review.sqlite3', 'database_snapshot_sha256'),
        ('source_manifest.json', 'source_manifest_sha256'),
        ('source_group_audit.json', 'source_group_audit_sha256'),
    ):
        require(sha256(CHALLENGE / filename) == original_protocol[key],
                'Historical challenge snapshot changed: ' + filename)
    rows = read_json(CHALLENGE / 'challenge_rows.json')
    require(len(rows) == 92 and sum(r['binary_target'] for r in rows) == 33,
            'Expected the fixed 92-photo, 33-positive challenge2 snapshot')
    require(len({image_hash(r) for r in rows}) == len(rows), 'Challenge repeats an image')
    for row in rows:
        review = row['human_review']
        require(review['reviewer_identity'] == 'human', 'Truth is not human-reviewed')
        require(review['decision'] in ('aquilaria', 'not_aquilaria'), 'Uncertain human label')
        require(row['binary_target'] == int(review['decision'] == 'aquilaria'),
                'Binary truth does not match the immutable human decision')
        require(review['sha256'] == row['sha256'], 'Reviewed image hash mismatch')
    fit = read_json(run / 'accepted_replay_manifest.json') + [
        r for r in read_json(run / 'new_manifest.json') if r['phase'] == 'train'
    ]
    hash_overlap = sorted({image_hash(r) for r in fit} & {image_hash(r) for r in rows})
    group_overlap = sorted({r['group_id'] for r in fit} & {r['group_id'] for r in rows})
    require(not hash_overlap and not group_overlap, 'Training/challenge overlap detected')
    state = torch.load(selected, map_location='cpu', weights_only=False)
    require(state['labels'] == labels and state['config'] == cfg,
            'Locked residual config or labels differ from the run snapshot')
    require(int(state['epoch']) == int(lock['epoch']), 'Locked epoch mismatch')
    require(state['baseline_deployment']['version'] == cfg['base_version'],
            'Candidate was initialized on a different deployed baseline')
    audit = {
        'fitting_rows': len(fit), 'challenge_rows': len(rows),
        'image_sha_overlap': hash_overlap, 'source_group_overlap': group_overlap,
        'fitting_manifest_sha256': sha256(run / 'accepted_replay_manifest.json'),
        'new_manifest_sha256': sha256(run / 'new_manifest.json'),
        'source_group_caveat': 'Source groups are not proven independent physical trees.',
    }
    return lock, cfg, labels, rows, state, audit


def evaluate(run):
    run = run.resolve()
    # Fail before any encoding if selection, review, weights, or split checks fail.
    lock, cfg, labels, rows, state, overlap = preflight(run)
    destination = run / 'post_training_human_regression'
    destination.mkdir(exist_ok=False)
    protected_inputs = [
        run / 'selection_lock.json', Path(lock['selected']), run / 'config.json',
        run / 'new_features.npz', run / 'contract.json', run / 'labels.json',
        Path(experiment.__file__), Path(__file__),
        CHALLENGE / 'challenge_rows.json', CHALLENGE / 'protocol.json',
        CHALLENGE / 'human_review.sqlite3', CHALLENGE / 'source_manifest.json',
    ]
    before = {str(p.resolve()): sha256(p) for p in protected_inputs}
    for source, name in (
        (run / 'selection_lock.json', 'model_selection_lock.json'),
        (run / 'config.json', 'training_config.json'),
        (run / 'labels.json', 'labels.json'),
        (CHALLENGE / 'challenge_rows.json', 'challenge_rows.json'),
        (CHALLENGE / 'protocol.json', 'historical_protocol.json'),
        (Path(__file__), 'evaluation_code.py'),
    ):
        shutil.copy2(source, destination / name)
        require(sha256(source) == sha256(destination / name), 'Snapshot copy failed')
    save_json(destination / 'overlap_audit.json', overlap)
    protocol = {
        'created_at': datetime.now(timezone.utc).isoformat(),
        'training_run': str(run), 'challenge_source': str(CHALLENGE),
        'selected_checkpoint': lock['selected'], 'selected_sha256': lock['sha256'],
        'selected_epoch': lock['epoch'], 'input_hashes': before,
        'truth': 'Immutable human binary_target: Aquilaria or non-Aquilaria only.',
        'negative_source_species_used_as_classification_truth': False,
        'selection_locked_before_this_evaluation': True,
        'trained': False, 'retuned': False, 'deployed': False,
        'scope': 'Previously inspected historical regression, not a fresh blind test.',
        'preprocessing': 'prepare_pixels: EXIF, RGB, gray letterbox <=1024, JPEG95, decode224; batch1 frozen encoder; raw z from unrounded e, x=e.half().float.',
        'branches': ['native148', 'baseline150', 'candidate150'],
        'decision_rule': 'Aquilaria positive only when top1 across the entire candidate vocabulary is index11. No threshold tuning.',
        'extension': 'Use the same float32-renormalized new text bank as training models().',
        'limitations': ['Challenge1 is a subset and is not added again.',
                       'Human negative labels do not establish another exact species.',
                       'Observation groups are not verified unique trees.',
                       'This is a regression safety check, not validation or epoch selection.'],
    }
    save_json(destination / 'protocol.json', protocol)

    from tree_species_teacher_demo.engine import Engine
    engine = Engine()
    engine.load()
    require(engine.sstv1_state == 'ready' and engine.sstv1_version == cfg['base_version'],
            'Deployed baseline does not match the locked experiment')
    require(engine.encoder_sha256 == state['encoder_sha256'], 'Encoder hash mismatch')
    require(engine.labels == labels[:148], 'Native148 output ordering mismatch')
    require(engine.classes == read_json(run / 'classes.json'), 'Raw class grouping mismatch')
    require(engine.choice['calibration_temperature'] == cfg['temperature'], 'Temperature changed')
    xs, zs = [], []
    with torch.inference_mode():
        for i, row in enumerate(rows):
            require(sha256(row['source_image']) == row['sha256'], 'Human-reviewed image changed')
            pixels = prepare_pixels(row['source_image'])[None].cuda()
            e = engine.model.encode_image(((pixels - engine.mean) / engine.std).to(engine.dtype),
                                          normalize=True).float()
            z = e @ engine.bank.T * engine.model.logit_scale.exp().float()
            xs.append(e.half().float().cpu().numpy())
            zs.append(z.cpu().numpy())
            if i % 10 == 0:
                print('CHALLENGE EMBED', i + 1, len(rows), flush=True)
    x, z = np.concatenate(xs), np.concatenate(zs)
    require(x.shape == (92, 1024) and z.shape == (92, 148), 'Feature shape mismatch')
    require(np.isfinite(x).all() and np.isfinite(z).all(), 'Nonfinite features')
    np.savez_compressed(destination / 'features.npz', x=x, z=z, rows_hash=digest(rows),
                        encoder_sha256=engine.encoder_sha256)
    del engine
    gc.collect()
    torch.cuda.empty_cache()

    # tensor_data needs an integer target for its generic container, but scores()
    # never uses that field. -1 is explicitly unassigned, not a negative species.
    binary_rows = [dict(r, target=AQ if r['binary_target'] else -1,
                       classification_target_authority='human_Aquilaria_only_other_species_unassigned')
                   for r in rows]
    model = experiment.models(run)
    require(torch.allclose(model[2].norm(dim=1), torch.ones(2, device='cuda'), atol=1e-5),
            'Training models() must supply the normalized extension text bank')
    data = experiment.tensor_data(run, (binary_rows, x, z), model)
    zero = torch.nn.Linear(1024, 3).cuda().eval().requires_grad_(False)
    with torch.no_grad():
        zero.weight.zero_()
        zero.bias.zero_()
    candidate = torch.nn.Linear(1024, 3).cuda().eval().requires_grad_(False)
    candidate.load_state_dict(state['model_state'])
    probabilities = {
        'native148': experiment.scores(zero, data, native=True),
        'baseline150': experiment.scores(zero, data),
        'candidate150': experiment.scores(candidate, data),
    }
    truth = np.array([r['binary_target'] for r in rows], dtype=np.int64)
    for branch, p in probabilities.items():
        require(p.shape == (92, 148 if branch == 'native148' else 150), 'Output shape mismatch')
    report = {
        'scope': protocol['scope'], 'selected_epoch': lock['epoch'],
        'all_confirmed': {k: binary_argmax_metrics(truth, p) for k, p in probabilities.items()},
        'threshold_tuned': False, 'trained_on_challenge': False, 'used_for_selection': False,
        'deployed': False,
    }
    previous_cache = CHALLENGE / 'features.npz'
    with np.load(previous_cache, allow_pickle=False) as previous:
        require(str(previous['rows_hash']) == digest(rows), 'Historical cache row order changed')
        historical = previous['p']
    report['native_replay_numerical_check'] = {
        'historical_cache': str(previous_cache), 'sha256': sha256(previous_cache),
        'max_absolute_probability_difference': float(np.max(np.abs(historical - probabilities['native148']))),
        'top1_disagreements': int((historical.argmax(1) != probabilities['native148'].argmax(1)).sum()),
        'note': 'Reference only; current baseline and candidate both use the freshly extracted x/z.',
    }
    predictions = []
    for i, row in enumerate(rows):
        record = {'id': row['id'], 'sha256': row['sha256'], 'source_image': row['source_image'],
                  'group_id': row['group_id'], 'truth_is_aquilaria': int(truth[i]),
                  'human_decision': row['human_review']['decision']}
        for branch, p in probabilities.items():
            top = int(p[i].argmax())
            record[branch] = {'top1_species': labels[top], 'top1_score': float(p[i, top]),
                              'aquilaria_score': float(p[i, AQ]), 'aquilaria_positive': top == AQ}
        predictions.append(record)
    b = probabilities['baseline150'].argmax(1) == AQ
    c = probabilities['candidate150'].argmax(1) == AQ
    report['candidate_vs_baseline150'] = {
        'binary_corrected_ids': [r['id'] for i, r in enumerate(rows) if b[i] != bool(truth[i]) and c[i] == bool(truth[i])],
        'binary_regressed_ids': [r['id'] for i, r in enumerate(rows) if b[i] == bool(truth[i]) and c[i] != bool(truth[i])],
    }
    experiment.verify_contract(run)
    verify_backup(lock['backup'])
    for path, expected in before.items():
        require(sha256(path) == expected, 'Source artifact changed during evaluation: ' + path)
    report['source_artifacts_unchanged'] = True
    save_json(destination / 'predictions.json', predictions)
    np.savez_compressed(destination / 'predictions.npz', **probabilities, binary_target=truth,
                        rows_hash=digest(rows))
    save_json(destination / 'evaluation.json', report)
    lines = ['# Post-training Aquilaria human regression', '',
             'Historical 92-photo human binary challenge: 33 Aquilaria and 59 non-Aquilaria. Not a fresh blind test.',
             'No fitting, threshold tuning or checkpoint selection took place here. Negative source species are not classification truth.', '',
             '| Branch | TP | FN | FP | TN | Recall | FPR |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for branch, result in report['all_confirmed'].items():
        lines.append(f"| {branch} | {result['tp']} | {result['fn']} | {result['fp']} | {result['tn']} | {result['recall']:.2%} | {result['fpr']:.2%} |")
    lines += ['', 'Same-vocabulary comparison: baseline150 versus candidate150. Native148 is a deployed-model regression reference.',
              'All decisions are full-vocabulary top1, not the earlier binary pilot score thresholds.',
              'Production artifacts and training backups verified unchanged; no deployment.',
              'Full evidence: protocol.json, overlap_audit.json, features.npz, predictions.json, evaluation.json.']
    (destination / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    save_json(destination / 'COMPLETE.json', {
        'evaluated': True, 'trained': False, 'deployed': False,
        'selected_checkpoint_sha256': lock['sha256'], 'evaluation_sha256': sha256(destination / 'evaluation.json'),
        'production_unchanged': True,
    })
    print('HUMAN REGRESSION', report['all_confirmed'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    evaluate(args.run)
