"""Prospective, source-labelled, licensed evaluation; NEVER fit or deploy.

Collection is independent of every model output. Raw labels come from multiple
current community identifiers, not predictions and not human_review acceptance.
Novel to project != proven absent from BioCLIP pretraining. Save all exclusions,
source attribution, missing classes and date/geographic strata.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import re
import sqlite3
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
from tools.training_safety import sha256, verify_backup
from tools.collect_sstv1_incremental import phash
from tools.collect_conservation_sources_20260917 import identifiers
from spacescan_trees_demo_v1.presentation_groups import BAUHINIA_GROUP, BAUHINIA_MEMBERS, grouped_distribution

SEED = 20261006
LICENSES = {'cc0', 'cc-by', 'cc-by-sa'}
START_DATE, END_DATE = '2026-05-25', '2026-10-06'
RECENT_DATE = '2026-09-08'
CAP = 5
HEAD = ROOT / 'tree_species_teacher_demo/models/SSTv1_Aquilaria_UserCrops_20260919_H120'
CLASSES = ROOT / 'outputs/distillation/bioclip_probe_20260908_0227/classes.json'
ALIASES = ROOT / 'outputs/distillation/20260908_0045/aliases.json'
LOOKUP = np.array([int(i).bit_count() for i in range(256)], dtype=np.uint8)
UA = {'User-Agent': 'SST-local-146-validation/1.0 (bounded licensed botanical research)'}


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def catalog():
    classes, aliases = load(CLASSES), load(ALIASES)
    raw_labels = classes['labels'] + ['Wodyetia bifurcata']
    terms = {label: [term for index in group for term in aliases[index]['prompt_terms']]
             for label, group in zip(classes['labels'], classes['groups'])}
    terms['Wodyetia bifurcata'] = ['Wodyetia bifurcata']
    names = list(grouped_distribution(dict.fromkeys(raw_labels, 1 / 148)))
    terms[BAUHINIA_GROUP] = [term for name in BAUHINIA_MEMBERS for term in terms[name]]
    assert len(names) == 146 and len(set(names)) == 146
    return [{'index': i, 'label': name, 'query_terms': sorted(set(terms[name]))}
            for i, name in enumerate(names)]


def walk(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            if isinstance(child, (dict, list)):
                yield from walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk(child)


def prior_index(out):
    args = ['rg', '--files', 'external_datasets', 'outputs']
    for pattern in ('*manifest*.json', '*rows.json', '*metadata*.jsonl', '*catalog*.csv',
                    '*human_review*.json', '*source_audit*.json', '*review*.sqlite3',
                    '*existing_image_index*.jsonl'):
        args += ['-g', pattern]
    files = subprocess.check_output(args, cwd=ROOT, text=True, encoding='utf-8').splitlines()
    obs, photos, hashes, visuals, examined, errors = set(), set(), set(), {}, [], []

    def record(row):
        for key in ('inat_observation_id', 'observation_id'):
            value = str(row.get(key, ''))
            if value.isdigit(): obs.add(value)
        for key in ('photo_id', 'inat_photo_id'):
            value = str(row.get(key, ''))
            if value.isdigit(): photos.add(value)
        for key in ('sha256', 'source_sha256', 'file_sha256', 'image_sha256'):
            value = row.get(key)
            if isinstance(value, str) and re.fullmatch('[a-fA-F0-9]{64}', value): hashes.add(value.lower())
        value = row.get('phash')
        if isinstance(value, str) and re.fullmatch('[a-fA-F0-9]{16}', value):
            visuals[int(value, 16)] = str(row.get('path') or row.get('source_image') or row.get('file') or '')
        for key in ('source_page', 'source_page_url', 'source_url', 'url', 'group_id',
                    'source_group', 'path', 'file', 'source_image', 'absolute_path'):
            value = row.get(key)
            if not isinstance(value, str): continue
            obs.update(re.findall(r'/observations/(\d+)', value))
            obs.update(re.findall(r'observation[_:](\d+)', value))
            photos.update(re.findall(r'inat_photo_(\d+)', value))
            photos.update(re.findall(r'/photos/(\d+)', value))

    for name in sorted(set(files)):
        path = (ROOT / name).resolve()
        if out == path or out in path.parents: continue
        try:
            if path.suffix == '.csv':
                with path.open(encoding='utf-8-sig', newline='') as stream:
                    for row in csv.DictReader(stream): record(row)
            elif path.suffix == '.jsonl':
                with path.open(encoding='utf-8-sig') as stream:
                    for line in stream:
                        if line.strip():
                            for row in walk(json.loads(line)): record(row)
            elif path.suffix == '.sqlite3':
                with sqlite3.connect(path.as_uri() + '?mode=ro', uri=True) as db:
                    db.row_factory = sqlite3.Row
                    for name_row in db.execute("SELECT name FROM sqlite_master WHERE type='table'"):
                        if name_row[0] in ('decisions', 'images'):
                            for row in db.execute('SELECT * FROM ' + name_row[0]): record(dict(row))
            else:
                for row in walk(load(path)): record(row)
            examined.append({'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha256(path)})
        except Exception as exc:
            errors.append({'path': str(path), 'error': str(exc)})
    save(out / 'prior_index_audit.json', {'files': examined, 'errors': errors,
         'observations': len(obs), 'photos': len(photos), 'hashes': len(hashes), 'phashes': len(visuals),
         'scope': 'All rg-listed source/training/evaluation/review manifests including rejected and pending images',
         'limitation': 'Unrecorded images and very different photographs of the same tree cannot be ruled out.'})
    if errors:
        raise RuntimeError('Historical index incomplete; inspect prior_index_audit.json before collection')
    return obs, photos, hashes, visuals


class Denied(RuntimeError): pass


class Api:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(UA)
        self.last = 0
        self.calls = 0

    def get(self, endpoint, params):
        time.sleep(max(0, 1.2 - (time.monotonic() - self.last)))
        self.last = time.monotonic()
        response = self.session.get('https://api.inaturalist.org/v1/' + endpoint, params=params, timeout=40)
        self.calls += 1
        if response.status_code in (401, 403, 429, 503):
            raise Denied(f'Source stopped on {response.status_code}; no bypass/retry')
        response.raise_for_status()
        return response.json()


def resolve_taxon(data, term):
    def norm(value): return re.sub(r'\s+', ' ', value.replace('×', 'x')).strip().lower()
    exact = [row for row in data.get('results', []) if row.get('is_active') and
             row.get('rank') in ('species', 'hybrid') and norm(row.get('name', '')) == norm(term)]
    if len(exact) == 1: return exact[0], 'exact_active_scientific_name'
    synonym = [row for row in data.get('results', []) if row.get('is_active') and
               row.get('rank') in ('species', 'hybrid') and norm(row.get('matched_term', '')) == norm(term)]
    if len(synonym) == 1: return synonym[0], 'API_exact_scientific_synonym_match'
    return None, 'no_unique_exact_scientific_match'


def nearest_hash(value, old):
    if not old: return None
    keys = np.array(list(old), dtype=np.uint64)
    distances = LOOKUP[np.bitwise_xor(keys, np.uint64(value)).view(np.uint8).reshape(-1, 8)].sum(axis=1)
    index = int(distances.argmin())
    return {'distance': int(distances[index]), 'path': old[int(keys[index])]} if distances[index] <= 4 else None


def sheets(out, rows):
    for start in range(0, len(rows), 20):
        sheet = Image.new('RGB', (1500, 1320), 'white'); draw = ImageDraw.Draw(sheet)
        for index, row in enumerate(rows[start:start + 20]):
            with Image.open(out / row['file']) as src: im = ImageOps.exif_transpose(src).convert('RGB')
            im.thumbnail((285, 255))
            x, y = index % 5 * 300, index // 5 * 330
            sheet.paste(im, (x + (285 - im.width) // 2, y))
            draw.text((x + 3, y + 258), f"{start+index:03d} {row['source_species'][:34]}", fill='black')
            draw.text((x + 3, y + 277), f"obs {row['observation_id']}", fill='black')
            draw.text((x + 3, y + 296), f"{row['observed_on']} / {row['region_query']}", fill='black')
        sheet.save(out / 'contact_sheets' / f'batch_{start//20:03d}.jpg', quality=92)


def collect(out):
    verify_backup(out / 'pre_training_backup')
    if (out / 'PLAN.json').exists(): raise FileExistsError('Collection cannot overwrite/resume a prior run')
    for name in ('images', 'source_responses', 'contact_sheets'): (out / name).mkdir()
    categories = catalog(); save(out / 'catalog146.json', categories)
    save(out / 'PLAN.json', {'date': END_DATE, 'seed': SEED, 'category_count': 146, 'cap_per_category': CAP,
         'observation_dates': [START_DATE, END_DATE], 'recent_stratum_from': RECENT_DATE,
         'source': 'iNaturalist research-grade, >=2 current exact identifiers, including a non-observer, no species conflicts',
         'selection': 'Hong Kong first, then global; random seed within fixed query pool; one first licensed photo per independent observation/observer per class. No model-based sample removal.',
         'query_bounds': 'One page <=100 observations per resolved taxon and region, serial >=1.2s API spacing. Stop on 401/403/429/503.',
         'licenses': sorted(LICENSES), 'media_budget_bytes': 1_000_000_000,
         'filter': 'Known observation/photo IDs, exact SHA256, pHash Hamming<=4, minimum native side224, extreme lighting. No prediction/confidence filtering.',
         'baseline': 'Project original BioCLIP2.5 zero-shot with frozen project prompts; not BioCLIP1',
         'comparisons': ['teacher_center vs sstv1_center: head-only', 'teacher_context vs sstv1_context: same views', 'teacher_center vs sstv1_context: total recipe'],
         'grouping': 'Same 146 display labels for all modes, including coarse Ficus/Araucaria/Bauhinia. Not 146 strictly resolved biological species.',
         'human_review_status': 'Source community labels, NOT expert-confirmed; not added to human_review.sqlite3 or training',
         'training_performed': False, 'deployment_changed': False,
         'limitations': ['Worldwide botanical photos are not Hong Kong street-view accuracy.', 'Cannot prove absence from foundation-model pretraining.', 'Five samples/class are a screening audit, not precise per-species estimates.']})
    print('Indexing all prior provenance before any download...', flush=True)
    old_obs, old_photos, old_hashes, old_visual = prior_index(out)
    api = Api()
    places = api.get('places/autocomplete', {'q': 'Hong Kong', 'per_page': 10})
    save(out / 'source_responses/hong_kong_place.json', places)
    hk_places = [r for r in places.get('results', []) if r.get('name') == 'Hong Kong']
    if len(hk_places) != 1: raise RuntimeError('Hong Kong place not uniquely resolved')
    hk_id = hk_places[0]['id']
    rng = random.Random(SEED)
    rows, exclusions, errors, summaries = [], [], [], []
    new_hashes, new_visual, used_obs = set(), {}, set()
    total_bytes = 0; stopped = False
    for category in categories:
        print('COLLECT', category['index'] + 1, '/146', category['label'], flush=True)
        count = 0; observers = set(); taxon_ids = set(); reasons = Counter()
        try:
            resolved = []
            for term in category['query_terms']:
                taxa = api.get('taxa/autocomplete', {'q': term, 'per_page': 10})
                save(out / 'source_responses' / f"{category['index']:03d}_{len(resolved)}_taxon_{hashlib.sha1(term.encode()).hexdigest()[:8]}.json", taxa)
                taxon, mapping = resolve_taxon(taxa, term)
                if taxon is None:
                    errors.append({'label': category['label'], 'term': term, 'reason': mapping}); continue
                if taxon['id'] in taxon_ids: continue
                taxon_ids.add(taxon['id']); resolved.append((term, taxon, mapping))
            for region in ('hong_kong', 'global'):
                candidates = []
                for term, taxon, mapping in resolved:
                    params = dict(taxon_id=taxon['id'], photos='true', quality_grade='research',
                                  photo_license=','.join(sorted(LICENSES)), d1=START_DATE, d2=END_DATE,
                                  per_page=100, order_by='id', order='desc')
                    if region == 'hong_kong': params['place_id'] = hk_id
                    data = api.get('observations', params)
                    save(out / 'source_responses' / f"{category['index']:03d}_{taxon['id']}_{region}.json", data)
                    for obs in data.get('results', []):
                        evidence = identifiers(obs, taxon['id']); why = []
                        if (obs.get('taxon') or {}).get('id') != taxon['id']: why.append('not_exact_taxon')
                        if obs.get('flags') or evidence['tier'] != 'multiple_independent_exact_ids': why.append('insufficient_source_evidence')
                        if str(obs['id']) in old_obs: why.append('historical_observation')
                        if obs['id'] in used_obs: why.append('already_selected_observation')
                        if obs['user']['id'] in observers: why.append('same_observer_for_class')
                        if not START_DATE <= (obs.get('observed_on') or '') <= END_DATE: why.append('date_out_of_scope')
                        photos = [p for p in obs.get('photos', []) if p.get('license_code') in LICENSES
                                  and str(p['id']) not in old_photos and not p.get('flags') and not p.get('hidden')]
                        if not photos: why.append('no_new_licensed_photo')
                        if why:
                            reasons.update(why); exclusions.append({'label': category['label'], 'observation_id': obs['id'], 'reasons': why}); continue
                        candidates.append((obs, photos[0], taxon, term, mapping, evidence))
                rng.shuffle(candidates)
                # Source pool order is frozen before scores; take one per observer.
                for obs, photo, taxon, term, mapping, evidence in candidates:
                    if count >= CAP: break
                    if obs['id'] in used_obs or obs['user']['id'] in observers: continue
                    url = photo['url'].replace('/square.', '/large.')
                    try:
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
                        quality_why = []
                        if digest in old_hashes or digest in new_hashes: quality_why.append('exact_duplicate')
                        if near_old: quality_why.append('near_historical_image')
                        if near_new: quality_why.append('near_new_image')
                        if min(im.size) < 224: quality_why.append('low_resolution')
                        if float((gray < 15).mean()) > .85 or float((gray > 245).mean()) > .85: quality_why.append('extreme_lighting')
                        if quality_why:
                            reasons.update(quality_why); exclusions.append({'label': category['label'], 'observation_id': obs['id'],
                                'photo_id': photo['id'], 'reasons': quality_why, 'near_old': near_old, 'near_new': near_new}); continue
                        total_bytes += len(raw)
                        if total_bytes > 1_000_000_000: raise Denied('Bounded media budget reached')
                        path = out / 'images' / f"inat_{photo['id']}.jpg"
                        if path.exists(): raise RuntimeError('Original image cannot be overwritten')
                        path.write_bytes(raw)
                        row = {'id': f"inat_{photo['id']}", 'label': category['label'], 'source_species': taxon['name'],
                            'query_term': term, 'taxonomy_mapping': mapping, 'taxon_id': taxon['id'],
                            'observation_id': obs['id'], 'photo_id': photo['id'], 'observer_id': obs['user']['id'],
                            'observed_on': obs.get('observed_on'), 'source_created_at': obs.get('created_at'),
                            'source_page': obs['uri'], 'photo_url': url, 'source_evidence': evidence,
                            'license': photo['license_code'], 'attribution': photo.get('attribution'),
                            'region_query': region, 'hong_kong': hk_id in obs.get('place_ids', []),
                            'recent_after_base_project': obs.get('observed_on', '') >= RECENT_DATE,
                            'file': path.relative_to(out).as_posix(), 'sha256': digest, 'phash': f'{visual:016x}',
                            'dimensions': list(im.size), 'bytes': len(raw), 'training_eligible': False,
                            'human_review_status': 'not_expert_confirmed', 'split': 'fresh_source_labelled_evaluation'}
                        rows.append(row); new_hashes.add(digest); new_visual[visual] = str(path)
                        used_obs.add(obs['id']); observers.add(obs['user']['id']); count += 1
                        save(out / 'manifest.json', rows)
                    except Denied: raise
                    except Exception as exc:
                        exclusions.append({'label': category['label'], 'observation_id': obs['id'], 'reasons': ['download_decode_error'], 'error': str(exc)})
                if count >= CAP: break
        except Denied as exc:
            errors.append({'label': category['label'], 'error': str(exc)}); stopped = True
        except Exception as exc:
            errors.append({'label': category['label'], 'error': str(exc)})
        summaries.append({'label': category['label'], 'downloaded': count, 'exclusions': dict(reasons), 'taxon_ids': sorted(taxon_ids)})
        save(out / 'collection_summary.json', {'categories': summaries, 'errors': errors, 'source_stopped': stopped,
             'api_requests': api.calls, 'downloaded': len(rows), 'bytes': total_bytes})
        save(out / 'exclusions.json', exclusions)
        print('PROGRESS', len(rows), 'images,', count, 'this category', flush=True)
        if stopped: break
    save(out / 'manifest.json', rows); save(out / 'collection_errors.json', errors)
    sheets(out, rows)
    save(out / 'collection_lock.json', {'manifest_sha256': sha256(out / 'manifest.json'), 'plan_sha256': sha256(out / 'PLAN.json'),
         'catalog_sha256': sha256(out / 'catalog146.json'), 'rows': len(rows), 'source_stopped': stopped,
         'locked_before_inference': True, 'training_eligible': False})
    print('COLLECTION COMPLETE', len(rows), 'source stopped:', stopped, flush=True)


def rank(distribution):
    return [label for label, _ in sorted(grouped_distribution(distribution).items(), key=lambda row: (-row[1], row[0]))]


def evaluate(out):
    verify_backup(out / 'pre_training_backup')
    lock = load(out / 'collection_lock.json')
    if sha256(out / 'manifest.json') != lock['manifest_sha256']: raise RuntimeError('Frozen manifest changed')
    if (out / 'predictions.jsonl').exists(): raise FileExistsError('Never overwrite/resume evaluation; use new version')
    rows = load(out / 'manifest.json'); release = load(ROOT / 'spacescan_trees_demo_v1/release.json')
    health = requests.get('http://127.0.0.1:8039/api/health', timeout=10).json()
    if health['sstv1_version'] != release['classifier_version'] or health['status'] != 'ready': raise RuntimeError('Live model mismatch')
    # Exact production padding function, imported but no app lifespan or new GPU model.
    from spacescan_trees_demo_v1.server import prepare, infer
    started = time.monotonic()
    for index, row in enumerate(rows):
        data = (out / row['file']).read_bytes()
        if hashlib.sha256(data).hexdigest() != row['sha256']: raise RuntimeError('Frozen image changed')
        whole, _, _ = prepare(data)
        distributions = {}
        evidence = {}
        for view, raw in (('center', data), ('whole', whole)):
            response = requests.post('http://127.0.0.1:8039/api/predict', files={'image': ('image.jpg', raw, 'image/jpeg')}, data={'mode': 'sstv1'}, timeout=120)
            response.raise_for_status(); result = response.json()
            if result['deployment'] != release['classifier_version']: raise RuntimeError('Live deployment switched')
            if result['encoder_sha256'] != load(HEAD / 'deployment.json')['encoder_sha256']: raise RuntimeError('Encoder switched')
            for mode in ('teacher', 'sstv1'):
                distributions[mode + '_' + view] = result[mode + '_distribution']
            evidence[view] = {'inference_seconds': result['inference_seconds'], 'quality': result['quality']}
        for mode in ('teacher', 'sstv1'):
            distributions[mode + '_context'] = {label: .5 * (distributions[mode + '_whole'][label] + distributions[mode + '_center'][label])
                                                for label in distributions[mode + '_center']}
        if index in (0, len(rows)//2, len(rows)-1):
            actual = infer(data, 'street', 'sstv1', 'context')
            expected = grouped_distribution(distributions['sstv1_context'])
            error = max(abs(expected[name] - actual['image_display_distribution'][name]) for name in expected)
            if error > 1e-6: raise RuntimeError('Actual production inference path mismatch')
            evidence['production_parity_max_error'] = error
        ranked = {name: rank(dist)[:3] for name, dist in distributions.items()}
        record = {'id': row['id'], 'label': row['label'], 'distributions': distributions, 'top3': ranked, 'evidence': evidence}
        with (out / 'predictions.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
        if index % 20 == 0 or index == len(rows)-1:
            save(out / 'evaluation_status.json', {'processed': index+1, 'total': len(rows), 'elapsed_seconds': round(time.monotonic()-started, 1)})
            print('EVAL', index+1, '/', len(rows), flush=True)
    summarize(out)
    verify_backup(out / 'pre_training_backup')
    print('EVALUATION COMPLETE; all protected sources/weights unchanged', flush=True)


def metrics(rows, predictions, labels):
    if not rows: return {'n': 0, 'covered_classes': 0}
    modes = tuple(predictions[0]['top3'])
    result = {'n': len(rows), 'covered_classes': len(set(row['label'] for row in rows)), 'modes': {}, 'per_class': []}
    for label in labels:
        indices = [i for i, row in enumerate(rows) if row['label'] == label]
        result['per_class'].append({'label': label, 'n': len(indices), 'modes': {
            mode: {'correct': sum(predictions[i]['top3'][mode][0] == label for i in indices),
                   'top3_correct': sum(label in predictions[i]['top3'][mode] for i in indices)} for mode in modes}})
    for mode in modes:
        correct = np.array([p['top3'][mode][0] == row['label'] for p, row in zip(predictions, rows)])
        tp, fp, fn = [], [], []
        for label in labels:
            true = np.array([row['label'] == label for row in rows]); pred = np.array([p['top3'][mode][0] == label for p in predictions])
            tp.append(int((true & pred).sum())); fp.append(int((~true & pred).sum())); fn.append(int((true & ~pred).sum()))
        covered = [i for i, label in enumerate(labels) if any(row['label'] == label for row in rows)]
        result['modes'][mode] = {'correct': int(correct.sum()), 'top1': float(correct.mean()),
            'top3': sum(row['label'] in p['top3'][mode] for row, p in zip(rows, predictions)) / len(rows),
            'macro_recall_covered': float(np.mean([tp[i] / (tp[i]+fn[i]) for i in covered])),
            'macro_f1_covered': float(np.mean([2*tp[i] / max(1, 2*tp[i]+fp[i]+fn[i]) for i in covered]))}
    rng = np.random.default_rng(SEED)
    result['paired_comparisons'] = {}
    for before, after in (('teacher_context', 'sstv1_context'), ('teacher_center', 'sstv1_center'), ('teacher_center', 'sstv1_context')):
        a = np.array([p['top3'][before][0] == row['label'] for row, p in zip(rows, predictions)])
        b = np.array([p['top3'][after][0] == row['label'] for row, p in zip(rows, predictions)])
        d = b.astype(float) - a.astype(float)
        samples = d[rng.integers(0, len(d), size=(10000, len(d)))].mean(axis=1)
        fixed, broken = int((~a & b).sum()), int((a & ~b).sum())
        from scipy.stats import binomtest
        result['paired_comparisons'][before + '_vs_' + after] = {'gain_percentage_points': float(d.mean()*100),
            'paired_bootstrap95_percentage_points': (np.quantile(samples, [.025, .975])*100).tolist(),
            'corrected': fixed, 'regressed': broken,
            'mcnemar_exact_p': float(binomtest(fixed, fixed+broken, .5).pvalue) if fixed+broken else 1.0}
    return result


def summarize(out):
    rows = load(out / 'manifest.json')
    predictions = [json.loads(line) for line in (out / 'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
    if len(rows) != len(predictions) or any(row['id'] != pred['id'] for row, pred in zip(rows, predictions)): raise RuntimeError('Row alignment mismatch')
    labels = [r['label'] for r in load(out / 'catalog146.json')]
    result = {'date': END_DATE, 'source_labelled_not_expert_ground_truth': True, 'training_performed': False,
        'catalog_categories': 146, 'all': metrics(rows, predictions, labels), 'subsets': {},
        'source_species_counts': dict(Counter(r['source_species'] for r in rows)),
        'manifest_sha256': sha256(out / 'manifest.json'), 'predictions_sha256': sha256(out / 'predictions.jsonl')}
    for name, condition in (('hong_kong', lambda r:r['hong_kong']), ('outside_hong_kong', lambda r:not r['hong_kong']),
                             ('recent_after_base_project', lambda r:r['recent_after_base_project'])):
        indices = [i for i, row in enumerate(rows) if condition(row)]
        result['subsets'][name] = metrics([rows[i] for i in indices], [predictions[i] for i in indices], labels)
    result['missing_classes'] = [c['label'] for c in result['all']['per_class'] if not c['n']]
    save(out / 'RESULTS.json', result)
    lines = ['# BioCLIP 2.5 vs current SSTv1: fresh 146-category screening', '',
        'This is a new source-labelled community-identification audit, NOT expert-confirmed Hong Kong street-view accuracy.', '',
        f"Images: {len(rows)}; covered classes: {result['all']['covered_classes']}/146; Hong Kong images: {result['subsets']['hong_kong']['n']}.", '',
        '| Mode | Correct | Top-1 | Top-3 | Macro recall (covered only) |', '| --- | ---: | ---: | ---: | ---: |']
    for name, value in result['all']['modes'].items():
        lines.append(f"| {name} | {value['correct']}/{len(rows)} | {value['top1']:.2%} | {value['top3']:.2%} | {value['macro_recall_covered']:.2%} |")
    lines += ['', 'Paired gains (percentage points; bootstrap95):']
    for name, value in result['all']['paired_comparisons'].items():
        lines.append(f"- {name}: {value['gain_percentage_points']:+.2f} pp, CI {value['paired_bootstrap95_percentage_points']}; corrected {value['corrected']}, regressed {value['regressed']}.")
    lines += ['', '## Per category', '', '| Category | N | BioCLIP context correct | SST context correct | Change |', '| --- | ---: | ---: | ---: | ---: |']
    for value in result['all']['per_class']:
        a, b = value['modes']['teacher_context']['correct'], value['modes']['sstv1_context']['correct']
        lines.append(f"| {value['label']} | {value['n']} | {a} | {b} | {b-a:+d} |")
    lines += ['', 'Missing classes are not silently scored as zero or averaged away in a claimed full146 accuracy.',
        'Coarse Ficus/Araucaria/Bauhinia categories do not measure within-group species discrimination.',
        'Same images and 146-label probability aggregation used for all models; no sample removal based on errors or confidence.',
        'Current context exactly reuses production padding + original center, equal-probability averaging; head-only results isolate training from preprocessing.',
        'Sampling dates/regions/quality gates fixed before inference. Five/class is not a precise per-species estimate.',
        'No assertion of absolute absence from BioCLIP pretraining; known project observation/photo/hash/near-duplicate matches excluded.',
        'Licensed originals and attribution retained for private local evaluation, not public redistribution.',
        'No training, calibration change, model selection or production deployment. Backup: pre_training_backup/backup_manifest.json.']
    (out / 'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('collect', 'evaluate', 'summarize'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    {'collect': collect, 'evaluate': evaluate, 'summarize': summarize}[args.phase](args.out.resolve())
