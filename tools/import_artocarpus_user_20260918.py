"""Versioned user-image import and historical review firewall. Never trains.

Copies original bytes using copy2, verifies SHA-256, and generates derivative QA
contact sheets. Does not mutate originals, historical manifests, review DBs or models.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps, ExifTags
from scipy.fft import dctn

from artocarpus_fresh_split_20260917 import (
    ROOT, HELD, BLOCKED_REVIEW, ACCEPTED_REVIEW, record_identity, merge_records,
    lookup_maps, historical_matches, queue_config, list_rows, load, digest, save_new,
)

SOURCE = Path('C:/Users/25041725G/Desktop/Artocarpus hypargyreus')
OUT = ROOT / 'external_datasets/artocarpus_user_20260918_v1'
REGISTRY = ROOT / 'outputs/artocarpus_fresh_split_20260917_v1/refined_v2/history_registry.json'


def perceptual_hash(im):
    coeff = dctn(np.asarray(im.convert('L').resize((32, 32)), dtype=float), norm='ortho')[:8, :8].flatten()
    return f"{int(''.join('1' if x else '0' for x in coeff > np.median(coeff[1:])), 2):016x}"


def json_safe(value):
    if isinstance(value, (int, float, str, bool)) or value is None: return value
    if isinstance(value, bytes):
        for encoding in ('utf-8', 'utf-16-le', 'latin1'):
            try:
                s = value.decode(encoding).strip('\x00')
                if s: return s[:3000]
            except UnicodeDecodeError: continue
        return {'bytes': len(value), 'sha256': hashlib.sha256(value).hexdigest()}
    if isinstance(value, (list, tuple)): return [json_safe(v) for v in value[:40]]
    if isinstance(value, dict): return {str(k): json_safe(v) for k, v in value.items()}
    return str(value)[:3000]


def font(size):
    p = Path('C:/Windows/Fonts/arial.ttf')
    return ImageFont.truetype(str(p), size) if p.exists() else ImageFont.load_default()


def contact_sheets(rows, out):
    paths = []
    for start in range(0, len(rows), 6):
        sheet = Image.new('RGB', (1920, 1440), '#f1f3f0'); draw = ImageDraw.Draw(sheet)
        for j, row in enumerate(rows[start:start + 6]):
            x, y = (j % 3) * 640, (j // 3) * 720
            draw.rounded_rectangle((x + 10, y + 10, x + 630, y + 708), radius=14, fill='white')
            with Image.open(row['source_image']) as im:
                image = ImageOps.exif_transpose(im).convert('RGB'); image.thumbnail((600, 590))
            sheet.paste(image, (x + 20 + (600 - image.width) // 2, y + 20 + (590 - image.height) // 2))
            draw.text((x + 24, y + 623), f"{start+j+1:02d}  {row['original_filename']}", font=font(26), fill='#18352a')
            draw.text((x + 24, y + 659), f"{row['dimensions'][0]} x {row['dimensions'][1]}   user-supplied label", font=font(19), fill='#46584f')
            draw.text((x + 24, y + 684), 'Artocarpus hypargyreus | audit pending', font=font(17), fill='#6f4824')
        path = out / 'contact_sheets' / f'contact_{start//6+1:02d}.jpg'
        sheet.save(path, quality=96, subsampling=0); paths.append(str(path.resolve()))
    return paths


def import_originals(source, out):
    source = source.resolve(strict=True); out = out.resolve()
    files = sorted(p for p in source.iterdir() if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg'})
    if len(files) != 18: raise ValueError(f'Expected the supplied 18 photographs; found {len(files)}')
    out.mkdir(parents=True, exist_ok=False); (out / 'images').mkdir(); (out / 'contact_sheets').mkdir()
    rows = []
    for i, path in enumerate(files, 1):
        before = digest(path); dest = out / 'images' / path.name
        shutil.copy2(path, dest)
        assert digest(dest) == before == digest(path), f'Copy verification failed: {path}'
        with Image.open(dest) as im:
            raw_exif = {str(ExifTags.TAGS.get(k, k)): json_safe(v) for k, v in im.getexif().items()}
            info = {k: json_safe(v) for k, v in im.info.items() if k in {'comment', 'description', 'xmp', 'XML:com.adobe.xmp'}}
            oriented = ImageOps.exif_transpose(im).convert('RGB')
            ph = perceptual_hash(oriented); width, height = oriented.size
            gray = np.asarray(oriented.convert('L'))
        meta_text = json.dumps({'exif': raw_exif, 'info': info}, ensure_ascii=False)
        urls = sorted(set(re.findall(r'https?://[^\s<>"\\]+', meta_text)))
        series = 'filename_series_35' if re.fullmatch(r'35[a-z]?', path.stem, re.I) else 'unresolved_large_filename_collection'
        rows.append({'id': f'artouser_{i:02d}_{before[:12]}', 'source_species': 'Artocarpus hypargyreus',
            'source_species_zh_hant': '白桂木', 'label_authority': 'user_supplied_target_species_for_training',
            'human_expert_identification_claimed': False, 'source': 'user_supplied_folder',
            'original_source_image': str(path), 'original_filename': path.name, 'source_image': str(dest),
            'file': dest.relative_to(out).as_posix(), 'sha256': before, 'copy_sha256_verified': True,
            'phash': ph, 'dimensions': [width, height], 'bytes': dest.stat().st_size,
            'metadata': {'exif': raw_exif, 'embedded_info': info, 'embedded_urls': urls},
            'quality': {'dark_fraction': float((gray < 25).mean()), 'bright_fraction': float((gray > 245).mean())},
            'provisional_series': series, 'group_id': 'user_artocarpus_20260918:' + series,
            'individual_tree_identity': 'unknown; filenames do not establish separate individuals',
            'observer_identity': 'unknown_pending_provenance_matching', 'rights_status': 'user_supplied_rights_not_independently_verified',
            'training_authority': 'User requested local additional Artocarpus training; does not override prior human exclusions or heldout isolation',
            'training_eligible': False, 'human_review_status': 'not_asserted_by_importer',
            'audit_lane': 'pending_history_and_content_audit', 'split': 'quarantine_unsplit'})
    save_new(out / 'import_manifest.json', rows)
    sheets = contact_sheets(rows, out)
    save_new(out / 'IMPORT_VERIFIED.json', {'created_at': datetime.now(timezone.utc).isoformat(), 'source_folder': str(source),
        'original_count': len(rows), 'all_copied_bytes_verified': True, 'source_images_unchanged': True,
        'label_authority': 'user_supplied_target_species_for_training', 'contact_sheets': sheets,
        'importer_sha256': digest(__file__), 'training_performed': False, 'historical_files_changed': False})
    print(json.dumps({'imported': len(rows), 'contact_sheets': sheets, 'manifest': str(out / 'import_manifest.json')}, ensure_ascii=False), flush=True)


def read_decisions(path):
    before = digest(path)
    with sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True) as db:
        db.execute('PRAGMA query_only=ON'); db.row_factory = sqlite3.Row
        rows = [dict(r) for r in db.execute('SELECT * FROM decisions')]
    assert digest(path) == before, f'Review DB changed during read-only snapshot: {path}'
    return rows, before


def current_human_review_exclusions():
    """Read-only callable for prepare code. Lists are JSON-serializable.

    Only ACTIVE source queues create pending exclusions. Copied challenge DBs
    are fingerprinted/read for provenance, not treated as live pending queues.
    This function neither approves new user images nor changes a single decision.
    """
    queues = [(ROOT / m, ROOT / d, palm) for m, d, palm in queue_config()]
    queues.append((ROOT / 'outputs/artocarpus_fresh_review_20260917_v1/curated_manifest.json',
                   ROOT / 'outputs/artocarpus_fresh_review_20260917_v1/human_review.sqlite3', False))
    blocked_sha = set(); blocked_ids = set(); details = []; databases = []
    all_current = []
    for mp, dp, palm in queues:
        rows = list_rows(load(mp)); decisions, dbsha = read_decisions(dp)
        by_id = {str(d.get('id') or d.get('item_id')): d for d in decisions}
        represented = set()
        for row in rows:
            if palm and row.get('exact_duplicate_matches'): continue
            rid = str(row.get('id') or row.get('sample_id') or row.get('item_id'))
            sha = row.get('sha256') or row.get('source_sha256') or row.get('image_sha256')
            d = by_id.get(rid)
            valid = d and (not d.get('sha256') or d['sha256'] == sha) and (not d.get('source_species') or d['source_species'] == row.get('source_species'))
            state = str(d.get('decision') or d.get('status')) if valid else 'pending'
            if d: represented.add(rid)
            entry = {'id': rid, 'sha256': sha, 'status': state, 'database': str(dp),
                'source_manifest': str(mp), 'decision': d, 'valid_for_current_bytes': bool(valid),
                'authority': 'actual_live_queue_and_current_sqlite_decision'}
            all_current.append(entry)
            if state in BLOCKED_REVIEW:
                if sha: blocked_sha.add(sha)
                blocked_ids.add(rid); details.append(entry)
        for d in decisions:
            rid = str(d.get('id') or d.get('item_id')); state = str(d.get('decision') or d.get('status'))
            # Preserve a real rejected/stale SHA even when current queue metadata
            # no longer references those old bytes; no importer may erase it.
            if rid not in represented or (d.get('sha256') and any(x['id'] == rid and not x['valid_for_current_bytes'] for x in all_current if x['database'] == str(dp))):
                entry = {'id': rid, 'sha256': d.get('sha256'), 'status': state, 'database': str(dp), 'decision': d,
                    'authority': 'real_current_sqlite_decision_not_resolved_to_current_queue'}
                all_current.append(entry)
                if state in BLOCKED_REVIEW:
                    if d.get('sha256'): blocked_sha.add(d['sha256'])
                    blocked_ids.add(rid); details.append(entry)
        databases.append({'path': str(dp), 'sha256': dbsha, 'role': 'active_source_review', 'decision_count': len(decisions)})
    for base in sorted((ROOT / 'outputs').glob('SST_Conserve*human_challenge*')):
        dp = base / 'human_review.sqlite3'
        if dp.is_file():
            ds, sh = read_decisions(dp)
            databases.append({'path': str(dp), 'sha256': sh, 'role': 'historical_evaluation_snapshot_not_live_pending_queue', 'decision_count': len(ds)})
    for d in databases: assert digest(d['path']) == d['sha256'], d['path']
    return {'schema_version': 1, 'blocked_sha256': sorted(blocked_sha), 'blocked_ids': sorted(blocked_ids),
        'blocked_details': details, 'current_live_queue_decisions': all_current, 'databases': databases,
        'authority_note': 'Current active SQLite decisions and actual queue membership only. Historical challenge copies do not create new pending statuses or override newer live decisions.',
        'db_mutation': False}


def audit(out, registry_path):
    imported = load(out / 'import_manifest.json')
    reg = load(registry_path)
    # Historical decisions are snapshots. Clear them and reconstruct current real
    # queues and DB decisions; never treat importer placeholder fields as decisions.
    for r in reg['records']: r['review_statuses'] = []
    extra_inputs = [ROOT / 'external_datasets/artocarpus_whole_source_20260917_v1/manifest.json',
        ROOT / 'external_datasets/artocarpus_fresh_inat_20260917_v1/manifest.json']
    extra_records = []; snapshots = []; db_snapshots = []
    for path in extra_inputs:
        before = digest(path)
        extra_records.extend(record_identity(r, path) for r in list_rows(load(path)))
        assert digest(path) == before
        snapshots.append({'path': str(path), 'sha256': before})
    queues = [(ROOT / m, ROOT / d, palm) for m, d, palm in queue_config()]
    queues.append((ROOT / 'outputs/artocarpus_fresh_review_20260917_v1/curated_manifest.json',
                   ROOT / 'outputs/artocarpus_fresh_review_20260917_v1/human_review.sqlite3', False))
    for base in sorted((ROOT / 'outputs').glob('SST_Conserve*human_challenge*')):
        if (base / 'human_review.sqlite3').is_file():
            queues.append((base / 'source_manifest.json', base / 'human_review.sqlite3', False))
    decision_details = []
    for mp, dp, palm in queues:
        before_manifest = digest(mp); rows = list_rows(load(mp)); ds, db_hash = read_decisions(dp)
        by_id = {str(d.get('id') or d.get('item_id')): d for d in ds}
        by_sha = collections.defaultdict(list)
        for d in ds:
            if d.get('sha256'): by_sha[d['sha256']].append(d)
        counts = collections.Counter(); represented = set()
        archived_snapshot = 'human_challenge' in dp.parent.name
        for row in rows:
            if palm and row.get('exact_duplicate_matches'): continue
            r = record_identity(row, mp)
            d = by_id.get(r['id'])
            if not d and len(r['tokens']['sha256']) == 1:
                choices = by_sha.get(r['tokens']['sha256'][0], [])
                d = choices[0] if len(choices) == 1 else None
            valid = d and (not d.get('sha256') or d['sha256'] in r['tokens']['sha256']) and (not d.get('source_species') or d['source_species'] == r['species'])
            state = str(d.get('decision') or d.get('status')) if valid else 'pending'
            r['review_status'] = None if archived_snapshot else state; r['origin'] = str(dp)
            if archived_snapshot: r['phase'] = 'challenge'
            counts[state] += 1; extra_records.append(r)
            if d:
                represented.add(str(d.get('id') or d.get('item_id')))
                decision_details.append({'database': str(dp), 'manifest': str(mp), 'current_decision': d,
                    'matched_queue_id': r['id'], 'valid_for_current_bytes': bool(valid)})
        # Preserve real decisions whose rows no longer appear in a current queue.
        # A removed queue row is not permission to revive its rejected SHA.
        for d in ds:
            key = str(d.get('id') or d.get('item_id'))
            if key in represented: continue
            r = record_identity(d, dp)
            r['review_status'] = None if archived_snapshot else str(d.get('decision') or d.get('status'))
            if r['tokens']['sha256']: extra_records.append(r)
            decision_details.append({'database': str(dp), 'manifest': str(mp), 'current_decision': d,
                'matched_queue_id': None, 'valid_for_current_bytes': None, 'meaning': 'Current DB decision retained by SHA even if absent from queue'})
        assert digest(mp) == before_manifest and digest(dp) == db_hash
        snapshots.append({'path': str(mp), 'sha256': before_manifest})
        db_snapshots.append({'path': str(dp), 'sha256': db_hash, 'decision_count': len(ds),
            'authority_role': 'historical_evaluation_snapshot_not_live_pending_queue' if archived_snapshot else 'active_source_review',
            'actual_queue_status_counts': dict(counts), 'read_only': True})
    reg['records'].extend(merge_records(extra_records))
    maps, visual = lookup_maps(reg)
    output = []; hit_records = set()
    for row in imported:
        assert digest(row['original_source_image']) == row['sha256'] == digest(row['source_image'])
        r = record_identity(row, out / 'import_manifest.json')
        hits, ids, cached = historical_matches(r, reg, maps, visual, 6)
        # Direct image identity is SHA/photo/sample/path/URL; observations and
        # source galleries join only when metadata actually establishes them.
        content = set(ids) - set(hits.get('observer', []))
        content |= {i for k, values_ in hits.items() if k != 'observer' for i in values_}
        blocked = sorted(i for i in content if BLOCKED_REVIEW & set(reg['records'][i]['review_statuses']))
        held = sorted(i for i in content if HELD & set(reg['records'][i]['phases']))
        train = sorted(i for i in content if 'train' in reg['records'][i]['phases'])
        source_species = {s for i in content for s in reg['records'][i]['species']}
        conflict = sorted(source_species - {'Artocarpus hypargyreus'})
        observers = {o for i in content for o in reg['records'][i]['tokens']['observer']}
        observer_hits = {i for o in observers for i in maps['observer'].get(o, ())}
        observer_held = sorted(i for i in observer_hits if HELD & set(reg['records'][i]['phases']))
        groups = sorted({g for i in content for g in reg['records'][i]['tokens']['group']})
        photo_ids = sorted({v for i in content for v in reg['records'][i]['tokens']['photo']})
        observations = sorted({v for i in content for v in reg['records'][i]['tokens']['observation']})
        near_details = []
        for i in sorted(content):
            rec = reg['records'][i]
            distance = min(((int(row['phash'], 16) ^ int(p, 16)).bit_count() for p in rec['phashes']), default=None)
            near_details.append({'record_index': i, 'ids': rec['ids'], 'sha256': rec['tokens']['sha256'],
                'phash_distance': distance, 'phases': rec['phases'], 'review_statuses': rec['review_statuses'],
                'source_species': rec['species'], 'source_paths': rec['paths'], 'origins': rec['origins']})
        weak = []
        for h, recids in visual.items():
            distance = (int(row['phash'], 16) ^ h).bit_count()
            if 7 <= distance <= 12:
                weak.extend({'record_index': i, 'distance': distance, 'phases': reg['records'][i]['phases'],
                    'review_statuses': reg['records'][i]['review_statuses']} for i in recids)
        if blocked: lane = 'actual_rejected_pending_uncertain'
        elif held: lane = 'heldout_content_overlap'
        elif conflict: lane = 'historical_label_conflict'
        elif train: lane = 'duplicate_existing_train'
        elif observer_held: lane = 'observer_holdout_overlap'
        elif content or cached: lane = 'unresolved_historical_overlap'
        else: lane = 'new_candidate'
        split_group = groups[0] if len(groups) == 1 else row['group_id']
        authority = [x for x in decision_details if x['current_decision'].get('sha256') in {s for i in content for s in reg['records'][i]['tokens']['sha256']}]
        output.append(dict(row, audit_lane=lane, historical_matches=hits, cached_matches=cached,
            historical_content_evidence=near_details, weak_phash_7_to12_review=weak,
            actual_current_human_decisions=authority, historical_reject_pending_uncertain_records=blocked,
            historical_heldout_content_records=held, historical_training_records=train,
            inherited_observer_tokens=sorted(observers), observer_heldout_records=observer_held,
            resolved_photo_ids=photo_ids, resolved_observation_ids=observations, resolved_source_groups=groups,
            identity_mapping_authority='metadata/SHA/pHash; perceptual-only matches require visual confirmation, never model relabelling',
            split_group=split_group, fresh_holdout_eligible=False,
            new_independent_tree_claim=False, eligible_new_training_count=0,
            existing_training_reuse_candidate=lane == 'duplicate_existing_train',
            training_eligible=False, submit_as_new_review_item=not bool(blocked),
            remaining_gates=['root visual content/target review', 'rights/local-use scope check',
                'preserve current human decisions and frozen group/split assignments',
                'pre-training weight backup and run-specific accepted manifest']) )
        hit_records |= content | set(observer_held)
    # Images within filename series remain explicitly linked, but do not infer
    # that arbitrary "large" names refer to distinct trees or a single exact tree.
    pairs = []
    for i, a in enumerate(output):
        for b in output[i+1:]:
            distance = (int(a['phash'], 16) ^ int(b['phash'], 16)).bit_count()
            if a['sha256'] == b['sha256'] or distance <= 12:
                pairs.append({'a': a['id'], 'b': b['id'], 'a_file': a['original_filename'],
                    'b_file': b['original_filename'], 'same_sha256': a['sha256'] == b['sha256'],
                    'phash_distance': distance, 'action': 'Bind if duplicate; visual review for non-exact perceptual matches'})
    for x in snapshots + db_snapshots: assert digest(x['path']) == x['sha256'], x['path']
    for row in imported: assert digest(row['original_source_image']) == row['sha256']
    save_new(out / 'manifest.json', output)
    save_new(out / 'historical_review_snapshot.json', {'databases': db_snapshots, 'decision_rows': decision_details,
        'all_unchanged_at_end': True, 'current_review_overrides_importer_placeholder': True})
    save_new(out / 'historical_evidence_records.json', [{'record_index': i, **reg['records'][i]} for i in sorted(hit_records)])
    save_new(out / 'within_batch_near_duplicates.json', pairs)
    summary = {'count': len(output), 'lane_counts': dict(collections.Counter(r['audit_lane'] for r in output)),
        'source_files_unchanged': True, 'copied_original_sha256_verified': True,
        'registry': {'path': str(registry_path), 'sha256': digest(registry_path)}, 'refreshed_source_files': snapshots,
        'all_real_review_databases': db_snapshots, 'script_sha256': digest(__file__),
        'fresh_holdout_eligible': 0, 'independent_trees_proven': 0, 'training_eligible': 0,
        'train_reuse_not_new_images': [r['original_filename'] for r in output if r['audit_lane'] == 'duplicate_existing_train'],
        'new_candidates_pending_final_gate': [r['original_filename'] for r in output if r['audit_lane'] == 'new_candidate'],
        'authority': 'user_supplied_target_species_for_training, NOT invented expert certification',
        'source_unknown_rule': '35* bound as a filename series; arbitrary large filenames provide no independent-tree evidence. No fresh split from this folder.',
        'training_performed': False, 'gpu_used': False, 'historical_files_changed': False}
    save_new(out / 'AUDIT_SUMMARY.json', summary)
    gate = {'meaning': 'Handoff audit only; root must explicitly snapshot accepted manifests before any backed-up fit.',
        'current_new_candidates': [{'id': r['id'], 'file': r['original_filename'], 'sha256': r['sha256'], 'split_group': r['split_group'],
            'weak_near_match_count': len(r['weak_phash_7_to12_review'])} for r in output if r['audit_lane'] == 'new_candidate'],
        'prior_training_reuse': [{'id': r['id'], 'file': r['original_filename'], 'resolved_photo_ids': r['resolved_photo_ids'],
            'observations': r['resolved_observation_ids'], 'split_groups': r['resolved_source_groups'],
            'observer_overlap_warning': bool(r['observer_heldout_records']), 'new_independent_sample': False} for r in output if r['audit_lane'] == 'duplicate_existing_train'],
        'quarantine': [{'id': r['id'], 'file': r['original_filename'], 'lane': r['audit_lane']} for r in output if r['audit_lane'] not in {'new_candidate', 'duplicate_existing_train'}],
        'never_allowed': ['Use folder name to override an actual old reject/pending/uncertain decision', 'Move old heldout content into training',
            'Count resized duplicates or unidentified same-series views as independent trees', 'Create training/validation splits by randomly scattering these 18 files'],
        'all_training_eligible_remain_false': True}
    save_new(out / 'ACCEPTANCE_GATE.json', gate)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def finalize_after_root_visual_review(out):
    """Encode the root's explicitly supplied visual decision, never model labels.

    Root inspected all three contact sheets and instructed: exclude only 35.jpg
    montage, approve remaining nine new candidates under the user's folder label,
    retain eight historical training duplicates as ineligible/non-new records.
    """
    rows = load(out / 'manifest.json')
    assert len(rows) == 18
    decisions = current_human_review_exclusions()
    blocked = set(decisions['blocked_sha256'])
    new_files = {'35a.jpg', '35c.jpg', '35d.jpg', '35f.jpg', '35s.jpg',
                 'large (1).jpeg', 'large (2).jpeg', 'large 11.jpg', 'large.jpeg'}
    result = []
    for r in rows:
        assert digest(r['original_source_image']) == r['sha256'] == digest(r['source_image'])
        filename = r['original_filename']
        if filename == '35.jpg':
            lane = 'excluded_montage_near_duplicate_lower_resolution'
            note = 'Root visual review: 400-pixel low-resolution montage; primary content similar to 35c.jpg. Original retained, not deleted.'
            eligible = False
        elif filename in new_files:
            assert r['audit_lane'] == 'new_candidate', filename
            assert not r['historical_reject_pending_uncertain_records'] and not r['historical_heldout_content_records'], filename
            assert r['sha256'] not in blocked, filename
            lane = 'accepted_new_training'; eligible = True
            note = 'Root inspected the contact sheets: no obvious unreadable/non-plant content. Species authority remains the user-supplied training folder, not invented expert or per-image human certification.'
        else:
            assert r['audit_lane'] == 'duplicate_existing_train', filename
            lane = 'duplicate_existing_train'; eligible = False
            note = 'Historical training near-duplicate/resolution variant. Retain evidence but do not add a second training item or count as new.'
        group_note = ('35* is one source/visual series; 35a/35c appear to show the same fruit cluster. Bind all to one split; no independent-tree claim.'
                      if filename.startswith('35') else 'Arbitrary large filenames do not imply separate trees. Unresolved new files stay in one conservative group; prior training duplicates retain their resolved history.')
        result.append(dict(r, final_lane=lane, training_eligible=eligible, root_visual_review_note=note,
            review_authority='root_agent_contact_sheet_inspection_plus_explicit_user_training_label',
            label_authority='user_supplied_target_species_for_training', human_expert_identification_claimed=False,
            grouping_note=group_note, data_use_scope='local_user_requested_research_training_only_no_image_or_weight_publication_authorized',
            independent_tree_identity_verified=False, fresh_holdout_eligible=False))
    assert sum(r['training_eligible'] for r in result) == 9
    assert {r['original_filename'] for r in result if r['training_eligible']} == new_files
    assert len({r['split_group'] for r in result if r['training_eligible'] and r['original_filename'].startswith('35')}) == 1
    save_new(out / 'final_training_manifest.json', result)
    save_new(out / 'FINAL_GATE_VERIFIED.json', {
        'accepted_new_training': 9, 'retained_prior_training_duplicates_not_added': 8, 'excluded_montage': 1,
        'originals_deleted': False, 'source_bytes_unchanged': True,
        'accepted_rows_clear_of_current_live_blocked_sha': True,
        'accepted_rows_have_no_historical_heldout_content_matches': True,
        'label_authority': 'user_supplied_target_species_for_training',
        'not_expert_human_certified': True, 'new_independent_tree_count': None, 'fresh_test_count': 0,
        'accepted_training_group_count': len({r['split_group'] for r in result if r['training_eligible']}),
        'root_visual_instruction': 'Exclude 35.jpg low-resolution montage; accept remaining9 new candidates; retain8 duplicates ineligible. Bind35series and unknown large-series conservatively.',
        'source_manifest_sha256': digest(out / 'manifest.json'),
        'final_manifest_sha256': digest(out / 'final_training_manifest.json'),
        'live_review_db_fingerprints': decisions['databases'], 'current_blocked_sha_count': len(blocked),
        'historical_review_snapshot_interpretation': 'The two human_challenge DBs are evaluation snapshots, not current pending queues. LIVE_REVIEW_EXCLUSIONS and this final gate use active-source current DB authority; prior audit snapshot pending counts must not be used to block replay.',
        'script_sha256': digest(__file__), 'training_performed': False, 'gpu_used': False,
        'weight_backup_required_before_any_fit': True})
    print(json.dumps({'final_manifest': str(out / 'final_training_manifest.json'), 'accepted_new_training': 9,
                      'duplicates_not_added': 8, 'excluded': 1, 'groups': len({r['split_group'] for r in result if r['training_eligible']})}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('mode', choices=('import', 'audit', 'exclusions', 'finalize'))
    p.add_argument('--source', type=Path, default=SOURCE); p.add_argument('--out', type=Path, default=OUT)
    p.add_argument('--registry', type=Path, default=REGISTRY); a = p.parse_args()
    if a.mode == 'import': import_originals(a.source, a.out)
    elif a.mode == 'audit': audit(a.out.resolve(), a.registry.resolve())
    elif a.mode == 'finalize': finalize_after_root_visual_review(a.out.resolve())
    else:
        result = current_human_review_exclusions()
        save_new(a.out / 'LIVE_REVIEW_EXCLUSIONS.json', result)
        print(json.dumps({'blocked_sha256': len(result['blocked_sha256']), 'blocked_ids': len(result['blocked_ids']), 'databases': len(result['databases'])}))


if __name__ == '__main__': main()
