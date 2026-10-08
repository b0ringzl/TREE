"""Prepare four-class incremental experiments, without fitting or encoder use.

All x/z are immutable, source-aligned exact-Demo pixel caches. No old predicted
probabilities are reused, and no historical test predictions are opened.
"""
import argparse
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'tools'))
import numpy as np
import torch
import torch.nn.functional as F
from audit_artocarpus_preprocessing_20260917 import exact_demo_prepare
from bioclip_probe import TEACHER
from distillation_core import digest, read_json, save_json
from import_artocarpus_user_20260918 import current_human_review_exclusions
from prepare_artocarpus_fit_20260918 import identities, normalize, row_ids
from train_sstv1_incremental import prepare_pixels
from training_safety import sha256
from tree_species_teacher_demo.engine import SSTV1_MANIFEST, decode

BASE = ROOT/'outputs/SSTv1_Artocarpus_User_20260918_v1'
V3 = ROOT/'outputs/SST_Conserve_Browser_20260917_v3'
EXTRA = ROOT/'outputs/conservation_fourclass_extra_audit_20260918_v1'
DEFAULT = ROOT/'outputs/SST_Conservation_Iterative_20260918_v1'
FOCUS = [11, 16, 148, 149]
BASE_BEST_SHA = '9d18b096feb4ef7af027f4ecc921517503df50873f5e84eb0b3cc09ad0eb34a7'


def require(ok, message):
    if not ok: raise RuntimeError(message)


def verify_artifacts(rows):
    for r in rows:
        require(sha256(r['path']) == r['sha256'], 'Protected source changed: '+r['path'])


def feature_file(path, rows, encoder=None):
    with np.load(path, allow_pickle=False) as f:
        require(str(f['rows_hash']) == digest(rows), 'Feature/manifest mismatch: '+str(path))
        if encoder is not None and 'encoder_sha256' in f:
            require(str(f['encoder_sha256']) == encoder, 'Feature encoder mismatch')
        x, z = f['x'].copy(), f['z'].copy()
    require(x.shape == (len(rows), 1024) and z.shape == (len(rows), 148), 'Feature shape mismatch')
    require(x.dtype == np.float16 and z.dtype == np.float32, 'Unexpected precision contract')
    require(np.isfinite(x).all() and np.isfinite(z).all(), 'Nonfinite cache')
    return x, z


def build(out=DEFAULT):
    out = Path(out).resolve()
    require(out.is_relative_to(ROOT/'outputs') and not out.exists(), 'Output must be new and under outputs')
    base_contract = read_json(BASE/'contract.json')
    extra_contract = read_json(EXTRA/'input_contract.json')
    verify_artifacts(base_contract['artifacts']); verify_artifacts(extra_contract['artifacts'])
    lock = read_json(BASE/'selection_lock.json'); ckpt = Path(lock['selected_checkpoint'])
    require(lock['epoch'] == 58 and lock['sha256'] == BASE_BEST_SHA and sha256(ckpt) == BASE_BEST_SHA,
            'Initialization is not the verified best58 checkpoint')
    dep = read_json(SSTV1_MANIFEST); encoder = dep['encoder_sha256']
    require(dep == read_json(BASE/'deployment_snapshot.json'), 'Production initialization changed')
    require(sha256(TEACHER) == encoder, 'Encoder changed')
    classes = read_json(BASE/'classes.json'); labels = read_json(V3/'labels.json')
    require(classes == read_json(V3/'classes.json') and len(classes['groups']) == 147, 'Collapse mapping changed')
    require(len(labels) == 150 and labels[:148] == read_json(BASE/'labels.json'), '150-label extension mismatch')
    require([labels[k] for k in FOCUS] == ['Aquilaria sinensis', 'Artocarpus hypargyreus',
            'Rhodoleia championii', 'Keteleeria fortunei'], 'Focus labels differ')
    choice = read_json(BASE/'teacher_choice.json')
    require(choice['calibration_temperature'] == dep['temperature'], 'Calibration changed')
    review = current_human_review_exclusions()
    blocked_sha, blocked_ids = set(review['blocked_sha256']), set(review['blocked_ids'])
    old = read_json(BASE/'train_rows.json')
    require(len(old) == 4198, 'Previous accepted training cohort changed')
    extra = read_json(EXTRA/'extra_train_accepted.json')
    require(len(extra) == 38, 'Expected the separately audited38 source-labelled images')
    extra_norm = []
    for r in extra:
        require(r['training_eligible'] is True and r['phase'] == 'train' and
                r['final_lane'] == 'accepted_existing_source_verified_training', 'Extra row not explicitly eligible')
        q = normalize(r, 'train', 'existing_source_verified_extra', native148=False)
        require(q['target'] in (16, 148, 149) and labels[q['target']] == q['source_species'], 'Incorrect extra label')
        require(q.get('label_authority') == 'institutional_source_plus_agent_visual_review_not_human_certified',
                'Extra source authority changed')
        extra_norm.append(q)
    require(Counter(r['target'] for r in extra_norm) == {16: 8, 148: 14, 149: 16}, 'Extra class composition changed')
    train = old+extra_norm
    require(len({r['sha256'] for r in train}) == len(train), 'Duplicate bytes; resolve before freezing4236 cohort')
    require(len({r['sample_id'] for r in train}) == len(train), 'Duplicate sample identity')
    planning = read_json(BASE/'planning_rows.json'); newdev = read_json(BASE/'newdev_rows.json')
    require(len(planning) == 2640 and len(newdev) == 15, 'Fixed development cohort changed')
    held = read_json(BASE/'heldout_metadata.json')
    held_tokens = {t for r in held['records'] for t in r['identity_tokens']}
    require(not {t for r in train for t in identities(r)}&held_tokens, 'Training/heldout SHA/group overlap')
    for r in train+planning+newdev:
        require(r['sha256'] not in blocked_sha and not row_ids(r)&blocked_ids, 'Actual human rejection/pending conflict: '+r['sample_id'])
        require(sha256(r['source_image']) == r['sha256'], 'Image source changed: '+r['sample_id'])
    # Check the old actual-Demo cache's completed extraction contract. There is
    # no probability reuse: all baselines will be recomputed by the fit driver.
    fc = read_json(BASE/'feature_contract.json')
    require(fc['all_reencoded_pixels_equal_production'] and fc['feature_batch'] == 1 and
            fc['encoder_sha256'] == encoder and fc['train_rows_hash'] == digest(old), 'Old pixels not verified exact Demo')
    bx, bz = feature_file(BASE/'train_features.npz', old, encoder)
    v3rows = read_json(V3/'new_manifest.json')
    nx, nz = feature_file(V3/'new_features.npz', v3rows, encoder)
    v3cfg = read_json(V3/'config.json')
    require(v3cfg['text_prompts'] == ['a photo of Rhodoleia championii', 'a photo of Keteleeria fortunei'], 'Novel prompts changed')
    require(v3cfg['temperature'] == dep['temperature'], 'Old novel baseline temperature differs')
    # Immutable historical extraction-code and cache hashes recorded before its
    # fit establish the old cache provenance without opening its predictions.
    old_backup = V3/'fit_seed_20260917/pre_training_backup/backup_manifest.json'
    backup = read_json(old_backup)
    bp = {str(Path(r['source']).resolve()): r for r in backup['files']}
    provenance = [V3/'new_features.npz', ROOT/'tools/train_sstv1_incremental.py',
                  ROOT/'tools/train_conservation_browser_conservative_20260917.py']
    for path in provenance:
        r = bp[str(path.resolve())]
        require(sha256(path) == r['sha256'], 'Historical extraction provenance changed: '+str(path))
        require(sha256(old_backup.parent/r['backup']) == r['sha256'], 'Historical backup provenance corrupted')
    lookup = {(r['sample_id'], r['sha256']): i for i, r in enumerate(v3rows)}
    require(len(lookup) == len(v3rows), 'Duplicate old cache key')
    extra_indices = []
    production = exact_demo_prepare()
    for r in extra_norm:
        key = (r['sample_id'], r['sha256']); require(key in lookup, 'Extra lacks exact cached ID+SHA')
        index = lookup[key]; original = v3rows[index]
        require(original['target'] == r['target'] and original['group_id'] == r['group_id'] and
                original['phase'] == 'train', 'Extra feature target/group/phase changed')
        prepared, _, _ = production(Path(r['source_image']).read_bytes())
        require(torch.equal(decode(prepared)[0], prepare_pixels(Path(r['source_image']))),
                'Historical helper/actual Demo pixels differ')
        extra_indices.append(index)
    x = np.concatenate([bx, nx[extra_indices]]); z = np.concatenate([bz, nz[extra_indices]])
    with np.load(V3/'new_features.npz') as f, np.load(BASE/'train_features.npz') as b:
        extra_bank = f['extra_bank'].copy()
        raw_extension = f['extension_bank'].copy()
        extension_bank = F.normalize(torch.tensor(raw_extension, dtype=torch.float32), dim=-1).numpy()
        scale = float(f['logit_scale'])
        require(np.array_equal(extra_bank, b['extra_bank']) and scale == float(b['scale']), 'Old/new text constants differ')
    require(extra_bank.shape == (1, 1024) and extension_bank.shape == (2, 1024), 'Bad appended bank shapes')
    require(np.allclose(np.linalg.norm(extension_bank, axis=1), 1., atol=1e-6), 'Novel text bank not unit normalized')
    dev_features = {}
    for phase, rows in [('planning', planning), ('newdev', newdev)]:
        dev_features[phase] = feature_file(BASE/(phase+'_features.npz'), rows)
    source_paths = [Path(a['path']) for a in base_contract['artifacts']+extra_contract['artifacts']]
    source_paths += [BASE/n for n in ['train_rows.json', 'planning_rows.json', 'newdev_rows.json',
        'train_features.npz', 'planning_features.npz', 'newdev_features.npz', 'feature_contract.json',
        'labels.json', 'classes.json', 'teacher_choice.json', 'contract.json', 'selection_lock.json',
        'heldout_metadata.json', 'deployment_snapshot.json']]
    source_paths += [EXTRA/n for n in ['input_contract.json', 'extra_train_accepted.json', 'SUMMARY.json',
        'eligibility_audit.json', 'current_human_review_snapshot.json']]
    source_paths += [V3/n for n in ['new_features.npz', 'new_manifest.json', 'labels.json', 'classes.json', 'config.json']]
    source_paths += provenance
    source_paths += [ckpt, old_backup, Path(__file__), ROOT/'tools/prepare_artocarpus_fit_20260918.py',
        ROOT/'tools/import_artocarpus_user_20260918.py', ROOT/'tools/audit_artocarpus_preprocessing_20260917.py',
        ROOT/'spacescan_trees_demo_v1/server.py', ROOT/'tree_species_teacher_demo/engine.py']
    source_paths += [Path(d['path']) for d in review['databases']]
    source_paths = list(dict.fromkeys(p.resolve() for p in source_paths))
    source_hashes = {str(p): sha256(p) for p in source_paths}
    verify_artifacts(base_contract['artifacts']); verify_artifacts(extra_contract['artifacts'])
    out.mkdir(parents=True, exist_ok=False)
    for name, value in [('train_rows', train), ('planning_rows', planning), ('newdev_rows', newdev),
                        ('classes', classes), ('labels', labels), ('teacher_choice', choice),
                        ('human_review_snapshot', review), ('heldout_metadata', held),
                        ('extra_train_rows', extra_norm), ('deployment_snapshot', dep)]:
        save_json(out/(name+'.json'), value)
    save_json(out/'init.json', {'checkpoint': str(ckpt), 'sha256': BASE_BEST_SHA, 'epoch': 58,
        'encoder_sha256': encoder, 'native_labels_count': 148, 'candidate_labels_count': 150,
        'new_residual': 'Each member retains its existing raw148 MLP and appends a zero-initialized1024-to2 branch; do not change the original147 collapse groups.',
        'production_is_not_initialization': True, 'deployed': False})
    np.savez_compressed(out/'train_features.npz', x=x, z=z, extra_bank=extra_bank,
        extension_bank=extension_bank, scale=scale, encoder_sha256=encoder, rows_hash=digest(train))
    for phase, rows in [('planning', planning), ('newdev', newdev)]:
        dx, dz = dev_features[phase]
        np.savez_compressed(out/(phase+'_features.npz'), x=dx, z=dz, rows_hash=digest(rows), encoder_sha256=encoder)
    save_json(out/'feature_contract.json', {'encoder_sha256': encoder,
        'preprocessing': 'Exact actual Demo gray-pad/JPEG95/224 decode; normalized encoder e, x=e.float16 and raw z=e.float32@old148text.T*scale',
        'old4198_cache': str(BASE/'train_features.npz'), 'old4198_rows_hash': digest(old),
        'extra38_cache': str(V3/'new_features.npz'), 'extra38_source_indices': extra_indices,
        'extra38_id_sha_target_group_phase_and_source_verified': True,
        'extra38_current_pixels_equal_historical_helper': True,
        'old4198_pixels_equal_demo_contract': str(BASE/'feature_contract.json'),
        'validation_reused_from_verified_demo_cache': True, 'probabilities_reused': False,
        'extension_prompts': v3cfg['text_prompts'], 'raw_extension_norms': np.linalg.norm(raw_extension, axis=1).tolist(),
        'extension_normalization': 'F.normalize(float32 cached vectors, dim=-1); no new text fit',
        'new_encoder_inference': False, 'source_feature_hashes': {str(p): source_hashes[str(p.resolve())]
            for p in [BASE/'train_features.npz', BASE/'planning_features.npz', BASE/'newdev_features.npz', V3/'new_features.npz']}})
    save_json(out/'selection_manifest.json', {'base_train': len(old), 'extra_train': len(extra_norm),
        'extra_ids': [r['sample_id'] for r in extra_norm], 'extra_indices_in_v3': extra_indices,
        'train_all_sha_unique': True, 'fit_heldout_sha_group_observation_overlap': 0,
        'augmentation_images_are_not_new_individuals': True})
    save_json(out/'contract.json', {'artifacts': [{'path': p, 'sha256': h} for p, h in source_hashes.items()],
        'training_requires_new_per_attempt_verified_physical_backup': True, 'production_switch_authorized': False})
    summary = {'created_at': datetime.now(timezone.utc).isoformat(), 'train_count': len(train),
        'prior_training_count': len(old), 'additional_source_verified_count': len(extra_norm),
        'target_indices': FOCUS, 'target_names': [labels[k] for k in FOCUS],
        'focus_train_counts': {labels[k]: sum(r['target'] == k for r in train) for k in FOCUS},
        'focus_train_group_counts': {labels[k]: len({r['group_id'] for r in train if r['target'] == k}) for k in FOCUS},
        'planning_count': len(planning), 'newdev_count': len(newdev),
        'newdev_support': dict(Counter(labels[r['target']] for r in newdev)),
        'authority_counts': dict(Counter(r['label_authority'] for r in train)),
        'taxonomy': '150 output labels; unchanged147 collapsed groups plus fixed foxtail and two novel species',
        'heldout_overlap': 0, 'independent_new_test_for_red_or_keteleeria': False,
        'new_species_training_one_gallery_group_each': True,
        'source_rights_summary': read_json(EXTRA/'SUMMARY.json')['rights_scope'],
        'scope': 'Local research only. Source attribution and watermarked originals preserved; no deployment or publication.',
        'fitted': False, 'encoder_ran': False, 'test_predictions_opened': False, 'production_unchanged': True}
    require(summary['train_count'] == 4236, 'Prepared training count changed')
    save_json(out/'data_summary.json', summary)
    require({p: sha256(p) for p in source_hashes} == source_hashes, 'Source changed during preparation')
    save_json(out/'PREPARED.json', {'completed_at': datetime.now(timezone.utc).isoformat(),
        'train_rows_hash': digest(train), 'planning_rows_hash': digest(planning), 'newdev_rows_hash': digest(newdev),
        'prepared_file_sha256': {p.name: sha256(p) for p in out.iterdir() if p.is_file()},
        'training_performed': False, 'encoder_used': False, 'no_test_predictions': True})
    print('PREPARED', str(out), 'train', len(train), 'focus', summary['focus_train_counts'], flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--out', type=Path, default=DEFAULT)
    args = parser.parse_args(); torch.set_num_threads(4); build(args.out)
