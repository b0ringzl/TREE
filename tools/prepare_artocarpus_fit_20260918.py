"""Prepare a versioned Artocarpus fit; no encoder inference or fitting.

The sole model computation is CPU inference over existing TRAIN features for
training-only hard-negative ranking. Training parameters belong to root's
separate, pre-fit protocol. This module never alters old data or the Demo.
"""
import argparse
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'tools'))

import numpy as np
import torch

import train_conservation_browser_20260917 as prior
from bioclip_probe import ResidualHead, TEACHER
from distillation_core import collapse_logits, digest, read_json, save_json
from import_artocarpus_user_20260918 import current_human_review_exclusions
from training_safety import sha256, deployed_artifacts

PREVIOUS = ROOT/'outputs/SST_Conserve_Browser_20260917_v3'
AUDIT = ROOT/'outputs/artocarpus_preprocess_audit_20260917_v1'
AQ = ROOT/'outputs/SSTv1_Aquilaria_User_20260914_v1'
INCREMENTAL = ROOT/'outputs/SSTv1_incremental_20260914_v1'
DEPLOY = prior.DEPLOY
SEED = 20260918


def require(ok, message):
    if not ok: raise RuntimeError(message)


def normalize(row, phase, role, native148=True):
    r = dict(row)
    r['sample_id'] = str(row.get('sample_id') or row.get('id') or row.get('item_id'))
    r['source_image'] = str(Path(row['source_image']).resolve())
    r['sha256'] = row.get('sha256') or row.get('source_sha256') or row.get('audit_sha256')
    r['target'] = int(row['target']) if row.get('target') is not None else None
    # Keep the source catalog target for exact audit-cache alignment. Classes
    # 148/149 are development negatives only, never fitted by the148-class head.
    r['native148_supported'] = r['target'] is not None and 0 <= r['target'] < 148
    if native148 and r['target'] is not None and r['target'] >= 148:
        r['original_catalog150_target'] = r['target']
    r['group_id'] = str(row.get('split_group') or row.get('group_id') or row.get('source_group'))
    r['phase'] = phase; r['data_role'] = role
    require(r['sample_id'] not in ('None', '') and r['group_id'] not in ('None', ''), 'Missing ID/group')
    require(r['sha256'] and len(r['sha256']) == 64, 'Missing source hash')
    r['label_authority'] = row.get('label_authority') or row.get('review_authority') or (
        'existing_human_accepted_incremental_image' if row.get('human_decision', {}).get('status') == 'accepted'
        else 'existing_source_label_not_claimed_independent_human_certification')
    return r


def identities(row):
    """Conservative exact SHA/group/observation identity; no invented tree IDs."""
    result = set()
    h = row.get('sha256') or row.get('source_sha256') or row.get('audit_sha256')
    if h: result.add('sha:'+h)
    gs = [row.get(k) for k in ('group_id', 'source_group', 'split_group')]
    gs += row.get('resolved_source_groups', [])
    for group in gs:
        if not group: continue
        text = str(group); result.add('group:'+text)
        for match in re.findall(r'(?:inat_observation_|observation[:_](?:inat[:_])?)(\d+)', text):
            result.add('observation:inat:'+match)
    for key in ('inat_observation_id', 'observation_id'):
        value = row.get(key)
        if value is not None and str(value).isdigit(): result.add('observation:inat:'+str(value))
    for value in row.get('resolved_observation_ids', []):
        if str(value).startswith('inat:'): result.add('observation:'+str(value))
    return result


def row_ids(row):
    ids = {str(row[k]) for k in ('id', 'sample_id', 'item_id') if row.get(k)}
    ids |= {s.removeprefix('herbarium_').split('_brightness_')[0] for s in list(ids)}
    return ids


def heldout_metadata(planning, newdev):
    records = []; sources = []
    def add(rows, source, scope):
        sources.append({'path': str(source), 'sha256': sha256(source), 'scope': scope})
        for row in rows:
            h = row.get('sha256') or row.get('source_sha256') or row.get('audit_sha256')
            records.append({'sample_id': row.get('sample_id') or row.get('id'), 'sha256': h,
                'group_id': row.get('group_id') or row.get('source_group'),
                'phase': row.get('phase', scope), 'scope': scope, 'origin': str(source),
                'identity_tokens': sorted(identities(row))})
    add(planning, AUDIT/'rows.json', 'planning2640')
    add(newdev, AUDIT/'rows.json', 'previous_source_development15')
    for phase in ('selection', 'test', 'field', 'case'):
        path = AQ/f'old_{phase}_rows.json'
        add(read_json(path), path, 'historical_'+phase)
    path = INCREMENTAL/'accepted_new_manifest.json'
    add([r for r in read_json(path) if r['phase'] != 'train'], path, 'prior_incremental_heldout')
    path = AQ/'feature_rows.json'
    add([r for r in read_json(path) if r['phase'] != 'train'], path, 'prior_aquilaria_heldout')
    path = ROOT/'outputs/SST_Conserve_v1_20260917_human_challenge2/challenge_rows.json'
    add(read_json(path), path, 'historical_human_binary_challenge92')
    return records, sources


def round_robin(rows, rng, limit=24):
    chosen = []
    by_class = defaultdict(list)
    for i, row in enumerate(rows): by_class[row['target']].append(i)
    for label in sorted(by_class):
        groups = defaultdict(list)
        for i in by_class[label]: groups[rows[i]['group_id']].append(i)
        keys = sorted(groups); rng.shuffle(keys)
        for key in keys: rng.shuffle(groups[key])
        count = 0
        while count < limit:
            progressed = False
            for key in keys:
                if groups[key] and count < limit:
                    chosen.append(groups[key].pop()); count += 1; progressed = True
            if not progressed: break
    return chosen


def train_only_white_scores(expected_rows, dep, classes, constants):
    """No encoder. No validation or test probabilities enter negative mining."""
    original, x, z = prior.replay(PREVIOUS)
    require(digest(original) == digest(expected_rows), 'Replay feature/row order mismatch')
    heads = []
    for member in dep['members']:
        path = DEPLOY.parent/member['file']
        require(sha256(path) == member['sha256'], 'Deployed head changed')
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        require(ckpt['classes'] == classes and ckpt['encoder_sha256'] == dep['encoder_sha256'], 'Incompatible frozen head')
        head = ResidualHead(ckpt['config']['architecture']).cpu().eval().requires_grad_(False)
        head.load_state_dict(ckpt['model_state']); heads.append(head)
    extra = torch.tensor(constants['extra_bank'], dtype=torch.float32)
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(original), 512):
            a = torch.tensor(x[start:start+512], dtype=torch.float32)
            b = torch.tensor(z[start:start+512], dtype=torch.float32)
            ps = [(torch.cat([collapse_logits(h(a, b), classes['groups']), a@extra.T*constants['scale']], -1)
                   / dep['temperature']).softmax(-1)[:, 16] for h in heads]
            predictions.append(torch.stack(ps).mean(0).numpy())
    return np.concatenate(predictions)


def build(out, user_manifest):
    out = Path(out).resolve(); user_manifest = Path(user_manifest).resolve()
    require(out.is_relative_to(ROOT/'outputs') and not out.exists(), 'Preparation output must be new and under outputs')
    require(user_manifest.is_file(), 'Final user acceptance manifest is required')
    torch.set_num_threads(4)
    dep = read_json(DEPLOY); classes = read_json(AQ/'classes.json'); labels = read_json(AQ/'labels.json')
    require(len(labels) == 148 and labels[11] == 'Aquilaria sinensis' and labels[16] == 'Artocarpus hypargyreus'
            and labels[147] == 'Wodyetia bifurcata', 'Unexpected output mapping')
    require(dep == read_json(AUDIT/'deployment_snapshot.json'), 'Current weights differ from fixed pad validation cache')
    require(read_json(AUDIT/'COMPLETE.json')['production_and_source_hashes_unchanged'], 'Incomplete preprocessing audit')
    audit_protocol = read_json(AUDIT/'protocol.json')
    for path, expected in audit_protocol['protected_sha256_before'].items():
        # Review databases may legitimately advance; v1 does not pin those.
        require(sha256(path) == expected, 'Audit/cache source changed: '+path)
    review = current_human_review_exclusions()
    blocked_sha = set(review['blocked_sha256']); blocked_ids = set(review['blocked_ids'])
    old = read_json(PREVIOUS/'accepted_replay_manifest.json')
    require(len(old) == 23906, 'Expected the audited23906 replay source')
    imported = read_json(user_manifest)
    require(isinstance(imported, list), 'Final user manifest must be a list of records')
    accepted = []
    for row in imported:
        if row.get('final_lane') != 'accepted_new_training': continue
        require(row.get('training_eligible') is True and row.get('source_species') == labels[16], 'User row not explicitly eligible/exact species')
        require(not row.get('historical_heldout_content_records'), 'Accepted user row has historical heldout evidence')
        r = normalize(dict(row, target=16), 'train', 'new_user_accepted')
        require(r['sha256'] not in blocked_sha and not row_ids(r)&blocked_ids, 'User row conflicts with real human decision')
        accepted.append(r)
    require(bool(accepted), 'No explicitly accepted new user training rows')
    require(len({r['sha256'] for r in accepted}) == len(accepted), 'Duplicate accepted user SHA')
    audit_rows = read_json(AUDIT/'rows.json')
    planning_ids = [i for i, r in enumerate(audit_rows) if r['audit_cohort'] == 'old_planning']
    newdev_ids = [i for i, r in enumerate(audit_rows) if r['audit_cohort'] == 'new_source_development']
    require(len(planning_ids) == 2640 and len(newdev_ids) == 15, 'Fixed validation cohort changed')
    planning = [normalize(audit_rows[i], 'planning', 'old_planning2640') for i in planning_ids]
    newdev = [normalize(audit_rows[i], 'planning', 'old_source_development15') for i in newdev_ids]
    for r in newdev: r['binary_white_target'] = int(r['target'] == 16)
    held, held_sources = heldout_metadata(planning, newdev)
    held_tokens = {token for r in held for token in r['identity_tokens']}
    excluded = []; pool = []; pool_original_indices = []
    for i, original in enumerate(old):
        r = normalize(original, 'train', 'existing_replay')
        conflicts = sorted(identities(r)&held_tokens)
        reason = ('actual_human_reject_pending_uncertain' if r['sha256'] in blocked_sha or row_ids(r)&blocked_ids
                  else 'heldout_identity_overlap' if conflicts else None)
        if reason:
            excluded.append({'sample_id': r['sample_id'], 'sha256': r['sha256'], 'reason': reason,
                             'heldout_identity_tokens': conflicts, 'original_replay_index': i})
            continue
        require(r['target'] is not None and 0 <= r['target'] < 148, 'Bad replay label')
        pool.append(r); pool_original_indices.append(i)
    for r in accepted:
        require(not identities(r)&held_tokens, 'New user training overlaps heldout by SHA/group/observation')
        require(r['sha256'] not in {p['sha256'] for p in pool}, 'User row already exists in replay; do not double count')
    # Snapshots of immutable sources are checked before CPU training-only scoring.
    protected = [TEACHER, DEPLOY, *[DEPLOY.parent/m['file'] for m in dep['members']], *deployed_artifacts(),
        ROOT/'spacescan_trees_demo_v1/server.py', ROOT/'tree_species_teacher_demo/engine.py',
        ROOT/'spacescan_trees_demo_v1/point_recognition.py',
        ROOT/'spacescan_trees_demo_v1/models/ptv2_seven_20260913_v1/model.joblib',
        ROOT/'spacescan_trees_demo_v1/models/ptv2_seven_20260913_v1/deployment.json',
        Path(__file__).resolve(), ROOT/'tools/import_artocarpus_user_20260918.py',
        ROOT/'tools/train_conservation_browser_20260917.py', ROOT/'tools/train_sstv1_incremental.py',
        ROOT/'tools/bioclip_probe.py', ROOT/'tools/distillation_core.py', ROOT/'tools/training_safety.py',
        PREVIOUS/'accepted_replay_manifest.json', PREVIOUS/'replay_source_indices.json',
        PREVIOUS/'new_features.npz', user_manifest,
        AUDIT/'rows.json', AUDIT/'paired_features_predictions.npz', AUDIT/'protocol.json', AUDIT/'COMPLETE.json']
    protected += [Path(r['path']) for r in review['databases']]
    protected += [Path(r['source_manifest']) for r in review['current_live_queue_decisions'] if r.get('source_manifest')]
    protected += [Path(r['path']) for r in held_sources]
    protected = list(dict.fromkeys(p.resolve() for p in protected))
    before = {str(p): sha256(p) for p in protected}
    with np.load(PREVIOUS/'new_features.npz') as f:
        require(str(f['encoder_sha256']) == dep['encoder_sha256'], 'Frozen text-bank encoder mismatch')
        constants = {'extra_bank': f['extra_bank'].copy(), 'scale': float(f['logit_scale'])}
    scores = train_only_white_scores(old, dep, classes, constants)
    require(scores.shape == (23906,) and np.isfinite(scores).all(), 'Bad training-only mining scores')
    rng = np.random.default_rng(SEED); base_indices = round_robin(pool, rng, limit=24)
    focus_indices = [i for i, r in enumerate(pool) if r['target'] in (11, 16)]
    negative_ids = [i for i, r in enumerate(pool) if r['target'] != 16]
    negative_ids = sorted(negative_ids, key=lambda i: (-float(scores[pool_original_indices[i]]), pool_original_indices[i]))[:512]
    require(len(negative_ids) == 512, 'Not enough eligible training negatives')
    reasons = defaultdict(list)
    for name, ids in [('class_group_round_robin24', base_indices), ('retain_all_white_and_aquilaria', focus_indices), ('training_only_white_hard_negative512', negative_ids)]:
        for i in ids: reasons[i].append(name)
    selected = []; seen = set(); selected_indices = []
    for i in sorted(reasons, key=lambda j: pool_original_indices[j]):
        row = dict(pool[i]); h = row['sha256']
        if h in seen: continue
        seen.add(h); row['selection_reasons'] = reasons[i]
        row['original_replay_index'] = pool_original_indices[i]
        selected.append(row); selected_indices.append(pool_original_indices[i])
    selected.extend(accepted)
    require(len({r['sha256'] for r in selected}) == len(selected), 'Final train contains duplicate bytes')
    require(not {t for r in selected for t in identities(r)}&held_tokens, 'Final fitting/heldout identities overlap')
    for r in selected+planning+newdev:
        require(sha256(r['source_image']) == r['sha256'], 'Image changed: '+r['source_image'])
    # Full validation is retained even if a newer review raises a label concern;
    # such a case blocks preparation rather than removing a poor prediction.
    require(not any(r['sha256'] in blocked_sha or row_ids(r)&blocked_ids for r in planning+newdev),
            'Fixed development cohort now has a conflicting actual review; resolve a new protocol before fit')
    require({p: sha256(p) for p in before} == before, 'Preparation inputs changed')
    out.mkdir(parents=True, exist_ok=False)
    save_json(out/'train_rows.json', selected)
    save_json(out/'accepted_user_rows.json', accepted)
    save_json(out/'user_manifest_snapshot.json', imported)
    save_json(out/'planning_rows.json', planning)
    save_json(out/'newdev_rows.json', newdev)
    white_ids = [i for i, r in enumerate(newdev) if r['target'] == 16]
    require(len(white_ids) == 10, 'Expected10 white-guimu development images')
    save_json(out/'newdev_white_rows.json', [newdev[i] for i in white_ids])
    save_json(out/'heldout_metadata.json', {'records': held, 'sources': held_sources,
        'scope': 'Identity metadata only; no heldout predictions or features read for mining/selection.'})
    save_json(out/'human_review_snapshot.json', review)
    save_json(out/'excluded_replay.json', excluded)
    save_json(out/'classes.json', classes); save_json(out/'labels.json', labels)
    choice = read_json(AQ/'teacher_choice.json')
    require(choice['calibration_temperature'] == dep['temperature'], 'Temperature changed')
    save_json(out/'teacher_choice.json', choice); save_json(out/'deployment_snapshot.json', dep)
    save_json(out/'contract.json', {'artifacts': [{'path': p, 'sha256': h} for p, h in before.items()],
        'training_not_authorized_until_verified_physical_backup': True, 'production_switch_authorized': False})
    np.savez_compressed(out/'inference_constants.npz', extra_bank=constants['extra_bank'],
                        logit_scale=constants['scale'], encoder_sha256=dep['encoder_sha256'])
    # Root's extraction step writes the feature files with its shared schema.
    # Preserve the verified source and ordered indices without creating a
    # conflicting baseline_p/audit_baseline cache here.
    with np.load(AUDIT/'paired_features_predictions.npz') as cache:
        require(str(cache['rows_hash']) == digest(audit_rows), 'Validation cache rows mismatch')
    save_json(out/'validation_cache_source.json', {
        'path': str(AUDIT/'paired_features_predictions.npz'),
        'sha256': before[str((AUDIT/'paired_features_predictions.npz').resolve())],
        'source_rows_hash': digest(audit_rows),
        'planning_indices': planning_ids, 'newdev_indices': newdev_ids,
        'newdev_white_indices': [newdev_ids[i] for i in white_ids]})
    mining = [{'sample_id': pool[i]['sample_id'], 'sha256': pool[i]['sha256'], 'target': pool[i]['target'],
        'group_id': pool[i]['group_id'], 'original_replay_index': pool_original_indices[i],
        'baseline_white_probability': float(scores[pool_original_indices[i]])} for i in negative_ids]
    save_json(out/'training_only_hard_negatives.json', {'rows': mining,
        'scope': 'Scored existing training caches only. These old caches have mixed historical center/letterbox preprocessing; ranking only. Selected images must all be re-encoded through Demo prepare for the actual fit.',
        'encoder_ran': False, 'device': 'cpu', 'label_changes': False})
    save_json(out/'selection_manifest.json', {'seed': SEED, 'rule': 'Up to24/class source-group round robin + all white/AQ + top512 eligible training-only white false-positive scores + accepted new user originals; SHA deduplicate.',
        'round_robin_selected': len(base_indices), 'forced_focus': len(focus_indices),
        'original_replay_indices': selected_indices, 'new_user_ids': [r['sample_id'] for r in accepted],
        'no_new_institutional38': True, 'random_views_not_independent_trees': True})
    summary = {'created_at': datetime.now(timezone.utc).isoformat(), 'replay_source_count': len(old),
        'eligible_replay_count': len(pool), 'excluded_replay': len(excluded),
        'train_count': len(selected), 'old_replay_selected': len(selected)-len(accepted),
        'new_user_count': len(accepted), 'train_species_count': len({r['target'] for r in selected}),
        'white_train': sum(r['target'] == 16 for r in selected), 'aquilaria_train': sum(r['target'] == 11 for r in selected),
        'train_counts': dict(Counter(str(r['target']) for r in selected)),
        'label_authorities': dict(Counter(r['label_authority'] for r in selected)),
        'planning': len(planning), 'newdev': len(newdev), 'newdev_white': len(white_ids),
        'native148_newdev_outside_catalog': sum(not r['native148_supported'] for r in newdev),
        'fit_heldout_sha_group_observation_overlap': 0, 'train_preprocessing': 'MUST re-encode every selected original with exact Demo gray-letterbox/JPEG95/decode224; no old train embedding reuse for fitting.',
        'planning_preprocessing': 'Exact previously audited Demo pad input; xfloat16,zfloat32, frozen baseline148probabilities.',
        'encoder_inference_performed': False, 'cpu_head_mining_only': True, 'training_performed': False,
        'deployed': False, 'protected_sources_unchanged': True}
    save_json(out/'data_summary.json', summary)
    save_json(out/'PREPARED.json', {'completed_at': datetime.now(timezone.utc).isoformat(),
        'train_rows_hash': digest(selected), 'planning_rows_hash': digest(planning), 'newdev_rows_hash': digest(newdev),
        'manifest_hashes': {p.name: sha256(p) for p in out.glob('*.json')},
        'no_fit': True, 'no_encoder': True, 'production_unchanged': True})
    print('PREPARED', out, 'train', len(selected), 'new', len(accepted), 'white', summary['white_train'], 'AQ', summary['aquilaria_train'], flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--user-manifest', type=Path, required=True)
    args = parser.parse_args(); build(args.out, args.user_manifest)
