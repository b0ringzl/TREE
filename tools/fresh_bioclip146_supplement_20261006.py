"""Coverage-only older-photo supplement, fixed before any benchmark inference.

Copies the locked first cohort into a new version; never mutates it, trains,
accepts human labels, changes production, or selects by model performance.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import random
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageOps

from fresh_bioclip146_benchmark_20261006 import (
    Api, CAP, Denied, LICENSES, RECENT_DATE, SEED, UA, identifiers, load,
    nearest_hash, phash, prior_index, resolve_taxon, save, sha256, sheets,
    verify_backup,
)

DATES = ('2025-01-01', '2026-05-24')


def collect(out, first):
    verify_backup(out / 'pre_training_backup')
    verify_backup(first / 'pre_training_backup')
    lock = load(first / 'collection_lock.json')
    if sha256(first / 'manifest.json') != lock['manifest_sha256']:
        raise RuntimeError('Locked primary cohort changed')
    if (first / 'predictions.jsonl').exists():
        raise RuntimeError('Coverage supplement must be fixed BEFORE any model scores')
    if lock['source_stopped']:
        raise RuntimeError('Do not continue after a source denial or quota stop')
    if (out / 'PLAN.json').exists():
        raise FileExistsError('Never overwrite/resume a collection version')
    for name in ('images', 'source_responses', 'contact_sheets'):
        (out / name).mkdir()
    categories = load(first / 'catalog146.json')
    save(out / 'catalog146.json', categories)
    plan = load(first / 'PLAN.json')
    plan.update({
        'primary_cohort': str(first), 'primary_manifest_sha256': lock['manifest_sha256'],
        'primary_observation_dates': plan['observation_dates'],
        'supplement_observation_dates': DATES,
        'supplement_rule': 'Only fill classes with fewer than five primary observations. Same gates and one observer/class across both cohorts. No inference occurred before this rule.',
        'observation_dates': [DATES[0], plan['observation_dates'][1]],
        'reporting': 'Primary recent-photo cohort and older coverage supplement reported separately, plus combined screening. Not all images were newly photographed.',
    })
    save(out / 'PLAN.json', plan)
    rows = load(first / 'manifest.json')
    for row in rows:
        source = first / row['file']; target = out / row['file']
        shutil.copy2(source, target)
        if sha256(target) != row['sha256']:
            raise RuntimeError('Primary image physical copy mismatch')
        row['cohort'] = 'primary_20260525_20261006'
    save(out / 'manifest.json', rows)
    old_obs, old_photos, old_hashes, old_visual = prior_index(out)
    api = Api()
    place = load(first / 'source_responses/hong_kong_place.json')
    hk_id = next(p['id'] for p in place['results'] if p['name'] == 'Hong Kong')
    rng = random.Random(SEED + 1)
    summaries, errors, exclusions = [], [], []
    new_hashes, new_visual, used_obs = set(), {}, {r['observation_id'] for r in rows}
    total_bytes = sum(r['bytes'] for r in rows)
    stopped = False
    for category in categories:
        label = category['label']
        seeds = [r for r in rows if r['label'] == label]
        if len(seeds) >= CAP:
            continue
        print('SUPPLEMENT', category['index'] + 1, '/146', label, 'starting', len(seeds), flush=True)
        count = len(seeds); observers = {r['observer_id'] for r in seeds}; reasons = Counter()
        try:
            resolved, taxon_ids = [], set()
            for term in category['query_terms']:
                term_hash = hashlib.sha1(term.encode()).hexdigest()[:8]
                cached = list((first / 'source_responses').glob(f"{category['index']:03d}_*_taxon_{term_hash}.json"))
                data = load(cached[0]) if len(cached) == 1 else api.get('taxa/autocomplete', {'q': term, 'per_page': 10})
                taxon, mapping = resolve_taxon(data, term)
                if taxon is None:
                    errors.append({'label': label, 'term': term, 'reason': mapping}); continue
                if taxon['id'] in taxon_ids:
                    continue
                taxon_ids.add(taxon['id']); resolved.append((term, taxon, mapping))
            for region in ('hong_kong', 'global'):
                candidates = []
                for term, taxon, mapping in resolved:
                    params = dict(taxon_id=taxon['id'], photos='true', quality_grade='research',
                                  photo_license=','.join(sorted(LICENSES)), d1=DATES[0], d2=DATES[1],
                                  per_page=100, order_by='id', order='desc')
                    if region == 'hong_kong':
                        params['place_id'] = hk_id
                    data = api.get('observations', params)
                    save(out / 'source_responses' / f"{category['index']:03d}_{taxon['id']}_{region}.json", data)
                    for obs in data.get('results', []):
                        evidence = identifiers(obs, taxon['id']); why = []
                        if (obs.get('taxon') or {}).get('id') != taxon['id']: why.append('not_exact_taxon')
                        if obs.get('flags') or evidence['tier'] != 'multiple_independent_exact_ids': why.append('insufficient_source_evidence')
                        if str(obs['id']) in old_obs: why.append('historical_observation')
                        if obs['id'] in used_obs: why.append('already_selected_observation')
                        if obs['user']['id'] in observers: why.append('same_observer_for_class')
                        if not DATES[0] <= (obs.get('observed_on') or '') <= DATES[1]: why.append('date_out_of_scope')
                        photos = [p for p in obs.get('photos', []) if p.get('license_code') in LICENSES
                                  and str(p['id']) not in old_photos and not p.get('flags') and not p.get('hidden')]
                        if not photos: why.append('no_new_licensed_photo')
                        if why:
                            reasons.update(why); exclusions.append({'label': label, 'observation_id': obs['id'], 'reasons': why}); continue
                        candidates.append((obs, photos[0], taxon, term, mapping, evidence))
                rng.shuffle(candidates)
                for obs, photo, taxon, term, mapping, evidence in candidates:
                    if count >= CAP: break
                    if obs['id'] in used_obs or obs['user']['id'] in observers: continue
                    try:
                        url = photo['url'].replace('/square.', '/large.')
                        response = requests.get(url, headers=UA, timeout=30)
                        if response.status_code in (401, 403, 429, 503): raise Denied(f'Media source stopped on {response.status_code}')
                        response.raise_for_status(); raw = response.content
                        if len(raw) > 20_000_000: raise ValueError('too_large')
                        with Image.open(io.BytesIO(raw)) as src:
                            if src.width * src.height > 40_000_000: raise ValueError('too_many_pixels')
                            im = ImageOps.exif_transpose(src).convert('RGB')
                        digest = hashlib.sha256(raw).hexdigest(); visual = phash(im)
                        gray = np.asarray(im.convert('L'))
                        near_old, near_new = nearest_hash(visual, old_visual), nearest_hash(visual, new_visual)
                        why = []
                        if digest in old_hashes or digest in new_hashes: why.append('exact_duplicate')
                        if near_old: why.append('near_historical_image')
                        if near_new: why.append('near_new_image')
                        if min(im.size) < 224: why.append('low_resolution')
                        if float((gray < 15).mean()) > .85 or float((gray > 245).mean()) > .85: why.append('extreme_lighting')
                        if why:
                            reasons.update(why); exclusions.append({'label': label, 'observation_id': obs['id'],
                                'photo_id': photo['id'], 'reasons': why, 'near_old': near_old, 'near_new': near_new}); continue
                        total_bytes += len(raw)
                        if total_bytes > 1_000_000_000: raise Denied('Bounded media budget reached')
                        path = out / 'images' / f"inat_{photo['id']}.jpg"
                        if path.exists(): raise RuntimeError('Original image cannot be overwritten')
                        path.write_bytes(raw)
                        rows.append({'id': f"inat_{photo['id']}", 'label': label, 'source_species': taxon['name'],
                            'query_term': term, 'taxonomy_mapping': mapping, 'taxon_id': taxon['id'],
                            'observation_id': obs['id'], 'photo_id': photo['id'], 'observer_id': obs['user']['id'],
                            'observed_on': obs['observed_on'], 'source_created_at': obs.get('created_at'),
                            'source_page': obs['uri'], 'photo_url': url, 'source_evidence': evidence,
                            'license': photo['license_code'], 'attribution': photo.get('attribution'),
                            'region_query': region, 'hong_kong': hk_id in obs.get('place_ids', []),
                            'recent_after_base_project': obs['observed_on'] >= RECENT_DATE,
                            'cohort': 'older_20250101_20260524', 'file': path.relative_to(out).as_posix(),
                            'sha256': digest, 'phash': f'{visual:016x}', 'dimensions': list(im.size), 'bytes': len(raw),
                            'training_eligible': False, 'human_review_status': 'not_expert_confirmed',
                            'split': 'fresh_source_labelled_evaluation'})
                        new_hashes.add(digest); new_visual[visual] = str(path)
                        used_obs.add(obs['id']); observers.add(obs['user']['id']); count += 1
                        save(out / 'manifest.json', rows)
                    except Denied: raise
                    except Exception as exc:
                        exclusions.append({'label': label, 'observation_id': obs['id'], 'reasons': ['download_decode_error'], 'error': str(exc)})
                if count >= CAP: break
        except Denied as exc:
            errors.append({'label': label, 'error': str(exc)}); stopped = True
        except Exception as exc:
            errors.append({'label': label, 'error': str(exc)})
        summaries.append({'label': label, 'primary': len(seeds), 'supplemented': count-len(seeds), 'combined': count, 'exclusions': dict(reasons)})
        save(out / 'collection_summary.json', {'categories': summaries, 'errors': errors, 'source_stopped': stopped,
             'api_requests': api.calls, 'downloaded': len(rows), 'primary': lock['rows'], 'supplement': len(rows)-lock['rows'], 'bytes': total_bytes})
        save(out / 'exclusions.json', exclusions)
        print('PROGRESS', len(rows), 'combined images;', count-len(seeds), 'added this category', flush=True)
        if stopped: break
    save(out / 'manifest.json', rows); save(out / 'collection_errors.json', errors)
    sheets(out, rows)
    save(out / 'collection_lock.json', {'manifest_sha256': sha256(out / 'manifest.json'), 'plan_sha256': sha256(out / 'PLAN.json'),
         'catalog_sha256': sha256(out / 'catalog146.json'), 'rows': len(rows), 'source_stopped': stopped,
         'locked_before_inference': True, 'training_eligible': False})
    print('SUPPLEMENT COMPLETE', len(rows), 'source stopped:', stopped, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--first', type=Path, required=True)
    args = parser.parse_args()
    collect(args.out.resolve(), args.first.resolve())
