"""Locked TRAIN-only deterministic views for the four-species conservation run.

prepare: metadata, human-review and original-file checks only; no model/GPU.
extract: separately authorized frozen-encoder inference; no fitting/deployment.
Augmentations retain their original parent and observation group. A trainer must
average the original plus three views within each parent before weighting it.
"""
import argparse
import hashlib
import io
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

os.environ['HF_HUB_OFFLINE'] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'tools'))

import numpy as np
import torch
from PIL import Image, ImageOps

from audit_artocarpus_preprocessing_20260917 import exact_demo_prepare
from distillation_core import digest, read_json, save_json
from import_artocarpus_user_20260918 import current_human_review_exclusions
from prepare_artocarpus_fit_20260918 import identities, row_ids
from train_sstv1_incremental import prepare_pixels
from training_safety import sha256
from tree_species_teacher_demo.engine import Engine, SSTV1_MANIFEST, decode
from bioclip_probe import TEACHER

BASE_RUN = ROOT / 'outputs/SSTv1_Artocarpus_User_20260918_v1'
DEFAULT_OUT = ROOT / 'outputs/SST_Conservation_Iterative_20260918_v1/augmentation'
TARGETS = {11: 'Aquilaria sinensis', 16: 'Artocarpus hypargyreus',
           148: 'Rhodoleia championii', 149: 'Keteleeria fortunei'}
VIEWS = (
    {'name': 'horizontal_flip', 'gamma': None, 'operation': 'horizontal flip of exact Demo 224 tensor'},
    {'name': 'gamma_0.85', 'gamma': 0.85, 'operation': 'EXIF/RGB source gamma then actual Demo prepare/decode'},
    {'name': 'gamma_1.15', 'gamma': 1.15, 'operation': 'EXIF/RGB source gamma then actual Demo prepare/decode'},
)
SCHEMA = 'conservation_training_augmentation_v1'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def rows_from(path):
    value = read_json(path)
    if isinstance(value, dict):
        value = value.get('rows', value.get('items', value.get('records')))
    require(isinstance(value, list), 'Expected a row list: ' + str(path))
    return value


def validate_training_rows(rows):
    require(rows, 'Empty training manifest')
    for row in rows:
        require(row.get('phase', 'train') == 'train', 'Non-training row is forbidden')
        require(row.get('data_role') and row.get('sample_id') and row.get('group_id'),
                'Training rows need data_role, sample_id and group_id')
        require(not any(word in str(row['data_role']).lower() for word in ('heldout', 'test', 'validation', 'planning', 'development')),
                'Heldout role is forbidden in augmentation input')
        require(isinstance(row.get('sha256'), str) and len(row['sha256']) == 64,
                'Training row needs original SHA-256')
        require(isinstance(row.get('target'), int) and 0 <= row['target'] < 150, 'Invalid 150-label target')
        require(row.get('source_image'), 'Missing original image path')
        if row.get('training_eligible') is False:
            # Some legacy download rows retain the original pre-review flag.
            # Their frozen acceptance is necessary but not sufficient: the
            # corresponding CURRENT SQLite decision is checked in safety_check.
            decision = row.get('human_decision', {})
            require(decision.get('status') == 'accepted' and decision.get('sha256') == row['sha256'],
                    'Ineligible row lacks a same-image human acceptance snapshot')
        require(not row.get('is_augmentation'), 'Input must contain originals, not old synthetic views')
        if row['target'] in TARGETS and row.get('source_species'):
            require(row['source_species'] == TARGETS[row['target']], 'Target/source species mismatch')
    require(len({r['sha256'] for r in rows}) == len(rows), 'Input repeats a parent SHA')
    require(len({r['sample_id'] for r in rows}) == len(rows), 'Input repeats a parent sample ID')


def select_parents(train, hard):
    validate_training_rows(train)
    require(len(hard) == 512 and len({r['sha256'] for r in hard}) == 512,
            'Expected the immutable previous 512 training-only hard-negative SHAs')
    by_hash = {r['sha256']: r for r in train}
    for row in hard:
        require(row['sha256'] in by_hash, 'A fixed hard negative is absent from the authorized training manifest')
        parent = by_hash[row['sha256']]
        require(parent['target'] == row['target'] and parent['group_id'] == row['group_id'],
                'Hard-negative parent target/group changed')
    hard_sha = {r['sha256'] for r in hard}
    parents = [r for r in train if r['target'] in TARGETS or r['sha256'] in hard_sha]
    require(all(any(r['target'] == target for r in parents) for target in TARGETS),
            'The four-species training manifest is not ready; no partial preparation')
    return parents, hard_sha


def build_aug_rows(parents, hard_sha):
    rows = []
    for index, parent in enumerate(parents):
        reasons = []
        if parent['target'] in TARGETS:
            reasons.append('four_species_training_focus')
        if parent['sha256'] in hard_sha:
            reasons.append('fixed_previous_training_only_hard_negative512')
        for view in VIEWS:
            row = dict(parent)
            row.update(sample_id=parent['sample_id'] + '__aug_' + view['name'],
                parent_sample_id=parent['sample_id'], parent_sha256=parent['sha256'],
                parent_source_image=parent['source_image'], parent_group_id=parent['group_id'],
                parent_index=index, is_augmentation=True, augmentation=view.copy(),
                phase='train', data_role='training_only_deterministic_view',
                selection_reasons=reasons, independent_tree=False, parent_total_views=4,
                view_weight_with_original=0.25,
                loss_aggregation='Mean original + three views within parent BEFORE class/source-group weighting')
            # sha256/source_image/group_id remain original values, never fake view identities.
            rows.append(row)
    require(len(rows) == len(parents) * 3, 'Augmentation accounting failed')
    require(len({r['sample_id'] for r in rows}) == len(rows), 'Augmentation sample IDs collide')
    return rows


def safety_check(parents, heldout, review, verify_images):
    require(isinstance(heldout, dict) and isinstance(heldout.get('records'), list),
            'Need metadata-only heldout records with identity_tokens')
    held_tokens = set()
    for row in heldout['records']:
        held_tokens.update(row.get('identity_tokens', []))
        held_tokens.update(identities(row))
    require(held_tokens, 'Empty heldout exclusions are not allowed')
    blocked_sha, blocked_ids = set(review['blocked_sha256']), set(review['blocked_ids'])
    conflicts, legacy_flags = [], []
    live_accepts = {(str(r['id']), r.get('sha256')) for r in review['current_live_queue_decisions']
                    if r.get('status') == 'accepted' and r.get('valid_for_current_bytes')}
    for row in parents:
        tokens = sorted(identities(row) & held_tokens)
        blocked = row['sha256'] in blocked_sha or bool(row_ids(row) & blocked_ids)
        if row.get('training_eligible') is False:
            require(any((rid, row['sha256']) in live_accepts for rid in row_ids(row)),
                    'Legacy false flag has no actual current same-image human acceptance')
            legacy_flags.append(dict(sample_id=row['sample_id'], sha256=row['sha256'],
                note='Imported pre-review false flag retained; current same-byte human SQLite acceptance verified.'))
        if tokens or blocked:
            conflicts.append(dict(sample_id=row['sample_id'], heldout_tokens=tokens, human_blocked=blocked))
        if verify_images:
            require(sha256(row['source_image']) == row['sha256'], 'Original training image changed: ' + row['source_image'])
    require(not conflicts, 'Training/heldout or real human-review conflict: ' + str(conflicts[:8]))
    return dict(parent_count=len(parents), unique_sha=len({r['sha256'] for r in parents}),
                source_groups=len({r['group_id'] for r in parents}),
                blocked_sha_count=len(blocked_sha), heldout_identity_count=len(held_tokens),
                conflicts=[], original_image_sha_checked=verify_images,
                imported_false_flag_with_current_human_acceptance=legacy_flags,
                heldout_image_or_prediction_reads=False, review_db_mutation=False)


def assert_hashes(mapping):
    for path, expected in mapping.items():
        require(sha256(path) == expected, 'Pinned input changed: ' + path)


def prepare(train_manifest, heldout_metadata, hard_negatives, out, base_run):
    train_manifest, heldout_metadata, hard_negatives = [Path(p).resolve() for p in (train_manifest, heldout_metadata, hard_negatives)]
    out, base_run = out.resolve(), base_run.resolve()
    require(out.is_relative_to(ROOT / 'outputs') and not out.exists(), 'Output must be a NEW directory under outputs')
    require(out.parent.is_dir(), 'The root-prepared study must exist before its new augmentation subdirectory')
    protocol = read_json(base_run / 'protocol.json')
    require(protocol['feature_batch_size'] == 1 and protocol['feature_z'] == 'raw text logits computed from e.float BEFORE float16 rounding',
            'Unexpected previous feature contract')
    base_rows = read_json(base_run / 'train_rows.json')
    with np.load(base_run / 'train_features.npz', allow_pickle=False) as cache:
        require(str(cache['rows_hash']) == digest(base_rows), 'Previous training feature ordering mismatch')
        require(str(cache['encoder_sha256']) == protocol['encoder_sha256'], 'Previous feature encoder mismatch')
        require(cache['x'].shape == (len(base_rows), 1024) and cache['z'].shape == (len(base_rows), 148),
                'Previous current-pad feature dimensions changed')
    dep = read_json(SSTV1_MANIFEST)
    require(dep == protocol['deployment_manifest'], 'Current deployment differs from the initialization contract')
    require(sha256(TEACHER) == dep['encoder_sha256'], 'Actual frozen encoder changed')
    train, hard = rows_from(train_manifest), rows_from(hard_negatives)
    require(hard == rows_from(base_run / 'training_only_hard_negatives.json'), 'Hard-negative selection changed')
    parents, hard_sha = select_parents(train, hard)
    heldout = read_json(heldout_metadata)
    review = current_human_review_exclusions()
    audit = safety_check(parents, heldout, review, verify_images=True)
    aug = build_aug_rows(parents, hard_sha)
    protected = [train_manifest, heldout_metadata, hard_negatives, TEACHER, SSTV1_MANIFEST,
                 base_run / 'train_rows.json', base_run / 'train_features.npz', base_run / 'protocol.json',
                 base_run / 'training_only_hard_negatives.json', Path(__file__).resolve(),
                 ROOT / 'tools/import_artocarpus_user_20260918.py',
                 ROOT / 'tools/prepare_artocarpus_fit_20260918.py',
                 ROOT / 'tools/train_sstv1_incremental.py', ROOT / 'tools/distillation_core.py',
                 ROOT / 'tools/bioclip_probe.py', ROOT / 'tools/training_safety.py',
                 ROOT / 'tools/audit_artocarpus_preprocessing_20260917.py',
                 ROOT / 'spacescan_trees_demo_v1/server.py', ROOT / 'tree_species_teacher_demo/engine.py']
    protected += [SSTV1_MANIFEST.parent / m['file'] for m in dep['members']]
    protected += [Path(r['path']) for r in review['databases']]
    protected += [Path(r['source_manifest']) for r in review['current_live_queue_decisions'] if r.get('source_manifest')]
    before = {str(p.resolve()): sha256(p) for p in dict.fromkeys(protected)}
    # No validation/test image opens and no inference happen in prepare.
    require(digest(current_human_review_exclusions()) == digest(review), 'Human review changed during preparation')
    assert_hashes(before)
    out.mkdir(parents=True, exist_ok=False)
    save_json(out / 'train_aug_parents.json', parents)
    save_json(out / 'train_aug_rows.json', aug)
    save_json(out / 'augmentation_human_review_snapshot.json', review)
    save_json(out / 'augmentation_safety_audit.json', audit)
    summary = dict(original_training_rows=len(train), selected_parent_photos=len(parents),
        selected_source_groups=audit['source_groups'], augmentation_rows=len(aug),
        fixed_hard_negative_photos=512,
        focus_parent_counts={str(k): sum(r['target'] == k for r in parents) for k in TARGETS},
        focus_and_hard_overlap=sum(r['target'] in TARGETS and r['sha256'] in hard_sha for r in parents),
        no_new_independent_trees=True, original_images_unchanged=True, encoder_ran=False,
        training_performed=False, deployed=False)
    save_json(out / 'augmentation_summary.json', summary)
    artifacts = {name: sha256(out / name) for name in (
        'train_aug_parents.json', 'train_aug_rows.json', 'augmentation_human_review_snapshot.json',
        'augmentation_safety_audit.json', 'augmentation_summary.json')}
    save_json(out / 'augmentation_lock.json', dict(schema=SCHEMA, created_at=datetime.now(timezone.utc).isoformat(),
        encoder_sha256=dep['encoder_sha256'], deployed_version=dep['version'],
        train_manifest=str(train_manifest), heldout_metadata=str(heldout_metadata), base_run=str(base_run),
        hard_negatives=str(hard_negatives), target_labels=TARGETS, views=list(VIEWS),
        parents_hash=digest(parents), rows_hash=digest(aug), artifacts=artifacts, protected_sha256=before,
        source_images='Only original training photos; never modified or stripped of watermarks.',
        preprocessing='Flip exact Demo224 tensor; gamma on EXIF-corrected RGB followed by actual Demo gray-letterbox/JPEG95/decode224.',
        gamma_definition='uint8 round(255 * (channel/255)**gamma), common lookup table over RGB',
        pixel_check='Every view is checked against an independent reference padding/JPEG/decode implementation.',
        feature_batch_size=1, feature_x='e.half() stored float16',
        feature_z='raw148 logits from unrounded normalized e.float(), before float16 x rounding',
        parent_loss='Average original+3 views per parent BEFORE class/source-group loss weighting',
        raw_z_names_unchanged=True, new_class_text_logits='Handled separately by the fixed four-class fit driver',
        no_fit=True, no_deployment=True, gpu_extraction_requires_explicit_command=True))
    print('AUGMENTATION MANIFEST LOCKED', summary, str(out), flush=True)
    return summary


def reference_pixels(rgb):
    edge = min(1024, max(rgb.size))
    padded = ImageOps.pad(rgb, (edge, edge), method=Image.Resampling.BICUBIC, color=(128, 128, 128))
    stream = io.BytesIO(); padded.save(stream, format='JPEG', quality=95)
    return decode(stream.getvalue())[0]


def gamma_rgb(rgb, gamma):
    require(gamma in (0.85, 1.15), 'Only the two frozen mild exposure transforms are allowed')
    table = np.rint(255. * (np.arange(256, dtype=np.float64) / 255.) ** gamma).astype(np.uint8).tolist()
    return rgb.point(table * 3)


def parent_pixels(parent, production):
    blob = Path(parent['source_image']).read_bytes()
    require(hashlib.sha256(blob).hexdigest() == parent['sha256'], 'Training parent bytes changed')
    prepared, _, _ = production(blob)
    original = decode(prepared)[0]
    require(torch.equal(original, prepare_pixels(parent['source_image'])), 'Original Demo/helper pixels differ')
    with Image.open(io.BytesIO(blob)) as image:
        rgb = ImageOps.exif_transpose(image).convert('RGB')
    reference = reference_pixels(rgb)
    require(torch.equal(original, reference), 'Reference Demo pixels differ')
    views = [torch.flip(original, dims=(-1,))]
    require(torch.equal(views[0], torch.flip(reference, dims=(-1,))), 'Horizontal flip mismatch')
    for gamma in (0.85, 1.15):
        exposed = gamma_rgb(rgb, gamma)
        stream = io.BytesIO(); exposed.save(stream, format='PNG')
        # The lossless in-memory PNG is a synthetic training input, not a new source file.
        prepared, _, _ = production(stream.getvalue())
        pixel = decode(prepared)[0]
        require(torch.equal(pixel, reference_pixels(exposed)), 'Gamma Demo/helper pixels differ')
        views.append(pixel)
    require(all(tuple(p.shape) == (3, 224, 224) and bool(torch.isfinite(p).all()) for p in views), 'Bad view tensor')
    return views


def verify_lock(out):
    path = out / 'augmentation_lock.json'
    require(path.is_file(), 'prepare must create the lock before extraction')
    lock = read_json(path)
    require(lock['schema'] == SCHEMA and lock['views'] == list(VIEWS), 'Augmentation recipe changed')
    require(lock['target_labels'] == {str(k): v for k, v in TARGETS.items()}, 'Target indices changed')
    for name, expected in lock['artifacts'].items():
        require(sha256(out / name) == expected, 'Prepared augmentation artifact changed: ' + name)
    assert_hashes(lock['protected_sha256'])
    parents, rows = read_json(out / 'train_aug_parents.json'), read_json(out / 'train_aug_rows.json')
    require(digest(parents) == lock['parents_hash'] and digest(rows) == lock['rows_hash'], 'Manifest row order changed')
    train, hard = rows_from(lock['train_manifest']), rows_from(lock['hard_negatives'])
    selected, hard_sha = select_parents(train, hard)
    require(selected == parents and build_aug_rows(parents, hard_sha) == rows, 'Locked views differ from training parents')
    review = current_human_review_exclusions()
    require(digest(review) == digest(read_json(out / 'augmentation_human_review_snapshot.json')), 'Current human decisions changed')
    safety_check(parents, read_json(lock['heldout_metadata']), review, verify_images=True)
    require(read_json(SSTV1_MANIFEST)['encoder_sha256'] == lock['encoder_sha256'], 'Encoder contract changed')
    return lock, parents, rows


def extract(out, authorize_gpu):
    require(authorize_gpu, 'GPU extraction is not authorized; pass --authorize-gpu-extract only after root approval')
    out = out.resolve()
    require(out.is_relative_to(ROOT / 'outputs'), 'Output outside project outputs')
    feature_path = out / 'train_aug_features.npz'
    chunk_dir = out / 'augmentation_feature_chunks'
    require(not feature_path.exists() and not chunk_dir.exists(), 'Extraction already started; never overwrite or resume')
    lock, parents, rows = verify_lock(out)
    lock_sha = sha256(out / 'augmentation_lock.json')
    require(torch.cuda.is_available(), 'CUDA is required for exact batch1 frozen-encoder extraction')
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
    chunk_dir.mkdir(exist_ok=False)
    engine = Engine(); engine.load()
    require(engine.encoder_sha256 == lock['encoder_sha256'], 'Loaded encoder differs')
    require(engine.sstv1_state == 'ready' and engine.sstv1_version == lock['deployed_version'], 'Loaded deployment changed')
    production = exact_demo_prepare()
    xbuf, zbuf, chunk_paths, audits = [], [], [], []
    start = 0
    def flush():
        nonlocal start
        if not xbuf:
            return
        path = chunk_dir / f'{start:06d}.npz'
        require(not path.exists(), 'Chunk overwrite forbidden')
        stop = start + len(xbuf)
        np.savez_compressed(path, x=np.concatenate(xbuf), z=np.concatenate(zbuf),
                            rows_hash=digest(rows[start:stop]), encoder_sha256=lock['encoder_sha256'])
        chunk_paths.append(path); xbuf.clear(); zbuf.clear(); start = stop
        print('AUGMENTATION FEATURES', start, '/', len(rows), flush=True)
    with torch.inference_mode():
        for parent_index, parent in enumerate(parents):
            views = parent_pixels(parent, production)
            for view_index, pixel in enumerate(views):
                row = rows[parent_index * 3 + view_index]
                e = engine.model.encode_image(((pixel[None].cuda() - engine.mean) / engine.std).to(engine.dtype), normalize=True).float()
                z = e @ engine.bank.T * engine.model.logit_scale.exp().float()
                require(tuple(e.shape) == (1, 1024) and tuple(z.shape) == (1, 148), 'Encoder/text bank dimension changed')
                xbuf.append(e.half().cpu().numpy()); zbuf.append(z.cpu().numpy())
                audits.append(dict(sample_id=row['sample_id'], parent_sha256=parent['sha256'],
                    group_id=parent['group_id'], pixel_sha256=hashlib.sha256(pixel.contiguous().numpy().tobytes()).hexdigest(),
                    pixel_shape=[3, 224, 224], actual_demo_equals_reference=True))
                if len(xbuf) == 64:
                    flush()
        flush()
    xx, zz, offset = [], [], 0
    for path in chunk_paths:
        with np.load(path, allow_pickle=False) as cache:
            n = len(cache['x'])
            require(str(cache['rows_hash']) == digest(rows[offset:offset+n]), 'Chunk order changed')
            xx.append(cache['x']); zz.append(cache['z']); offset += n
    x, z = np.concatenate(xx), np.concatenate(zz)
    require(x.shape == (len(rows), 1024) and z.shape == (len(rows), 148), 'Final feature dimensions changed')
    require(np.isfinite(x).all() and np.isfinite(z).all(), 'Nonfinite features')
    verify_lock(out)
    require(sha256(out / 'augmentation_lock.json') == lock_sha, 'Augmentation lock changed during extraction')
    require(not feature_path.exists(), 'Feature output already exists')
    np.savez_compressed(feature_path, x=x, z=z, rows_hash=lock['rows_hash'], encoder_sha256=lock['encoder_sha256'],
                        augmentation_lock_sha256=lock_sha)
    save_json(out / 'augmentation_pixel_audit.json', audits)
    save_json(out / 'AUGMENTATION_COMPLETE.json', dict(completed_at=datetime.now(timezone.utc).isoformat(),
        augmentation_rows=len(rows), parent_photos=len(parents), source_groups=len({r['group_id'] for r in parents}),
        rows_hash=lock['rows_hash'], features_sha256=sha256(feature_path), augmentation_lock_sha256=lock_sha,
        original_files_unchanged=True, production_unchanged=True, human_review_unchanged=True,
        all_views_actual_demo_equals_reference=True, raw_z_from_unrounded_embedding=True,
        fitted=False, deployed=False, independent_tree_count_not_increased=True))
    print('AUGMENTATION COMPLETE', len(rows), 'views from', len(parents), 'original photos', str(feature_path), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'extract'))
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--train-manifest', type=Path)
    parser.add_argument('--heldout-metadata', type=Path)
    parser.add_argument('--hard-negatives', type=Path, default=BASE_RUN / 'training_only_hard_negatives.json')
    parser.add_argument('--base-run', type=Path, default=BASE_RUN)
    parser.add_argument('--authorize-gpu-extract', action='store_true')
    args = parser.parse_args()
    if args.stage == 'prepare':
        require(args.train_manifest and args.heldout_metadata, 'prepare requires --train-manifest and --heldout-metadata')
        prepare(args.train_manifest, args.heldout_metadata, args.hard_negatives, args.out, args.base_run)
    else:
        extract(args.out, args.authorize_gpu_extract)


if __name__ == '__main__':
    main()
