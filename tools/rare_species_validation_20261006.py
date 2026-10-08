"""Prospective rare-species source audit. Evaluation only; never fit or deploy."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
import fresh_bioclip146_benchmark_20261006 as base
from training_safety import backup_before_training, sha256, verify_backup

TARGETS = ('Aquilaria sinensis', 'Artocarpus hypargyreus')
SEED = 2026100607
END_DATE = '2026-10-06'


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    plan = {
        'targets': TARGETS, 'seed': SEED, 'date': END_DATE,
        'purpose': 'Prospective comparison of original BioCLIP2.5 and deployed SSTv1-H120; no training or model selection',
        'source_plan': ['iNaturalist licensed exact-species observations, all dates through 2026-10-06',
                        'Open-access author-labelled natural photographs from papers',
                        'Explicitly licensed botanical photographs, original provenance preserved'],
        'sampling': 'Up to 100 iNaturalist observations per species, up to 3 original photos per observation; seeded order before inference',
        'labels': 'Current multiple independent species identifiers or explicit author photograph caption; no model relabelling',
        'dedup': 'Historical observation/photo IDs, SHA256 and pHash Hamming<=4; paper panels additionally checked against historical images',
        'quality': 'Native minimum side224, <=20MB/40MP, extreme-dark/bright fraction<=0.85; no score-based exclusions',
        'paper_rules': 'Record complete original figure, panel rectangle and caption. Never count multiple panels as independent trees. Charts, microscope tissue, agarwood products and mixed/unlabelled targets excluded.',
        'review': 'Agent quality review only, not human-reviewed botanical truth; excluded/pending training decisions remain untouched',
        'evaluation': 'Same full 146-label candidate catalog and same production whole+center processing for baseline and current; center-only diagnostic also retained',
        'uncertainty': 'Paired source-group clustered bootstrap; image-level counts and equal-source-group recall both reported',
        'rights': 'Private local scientific evaluation only; preserve original watermarks/attribution; no redistributable image or weight package',
        'limitations': ['Novel to project is not proof absent from BioCLIP pretraining',
                        'Two species and botanical photos are not all rare-plant or Hong Kong street-view accuracy',
                        'Older public figures are domain-shift challenge evidence, not proven foundation-model-unseen data'],
        'training_performed': False, 'deployment_changed': False,
    }
    base.save(out / 'PLAN.json', plan)
    base.save(out / 'catalog146.json', base.catalog())
    prior = base.load(ROOT / 'outputs/bioclip_sstv1_fresh146_20261006_v2/pre_training_backup/backup_manifest.json')
    inputs = [Path(r['source']) for r in prior['files']]
    inputs += [Path(__file__), ROOT / 'tools/training_safety.py', out / 'PLAN.json', out / 'catalog146.json',
               ROOT / 'outputs/bioclip_sstv1_fresh146_20261006_v2/FINAL_RESULTS.json']
    dest = backup_before_training(out, inputs, {'purpose': 'Evaluation-only pre-plan rollback snapshot, no fitting/deployment'})
    print('BACKUP VERIFIED', dest, flush=True)


def collect(out):
    verify_backup(out / 'pre_training_backup')
    if (out / 'inat_manifest.json').exists():
        raise FileExistsError('Do not silently redo a completed collection')
    for directory in ('images', 'source_responses', 'contact_sheets', 'paper_originals'):
        (out / directory).mkdir(exist_ok=True)
    old_obs, old_photos, old_hashes, old_visual = base.prior_index(out)
    api = base.Api()
    rows, exclusions, summary = [], [], []
    new_hashes, new_visual = set(), {}
    rng = random.Random(SEED)
    for label in TARGETS:
        raw = api.get('taxa/autocomplete', {'q': label, 'per_page': 10})
        base.save(out / 'source_responses' / (label.replace(' ', '_') + '_taxon.json'), raw)
        taxon, mapping = base.resolve_taxon(raw, label)
        if taxon is None:
            raise RuntimeError('Exact taxon unavailable: ' + label)
        observations = []
        for page in range(1, 11):
            response = api.get('observations', {'taxon_id': taxon['id'], 'photos': 'true',
                'quality_grade': 'research', 'photo_license': ','.join(sorted(base.LICENSES)),
                'd2': END_DATE, 'per_page': 100, 'page': page, 'order_by': 'id', 'order': 'desc'})
            base.save(out / 'source_responses' / f"{taxon['id']}_page{page}.json", response)
            observations += response.get('results', [])
            if len(observations) >= response.get('total_results', 0) or not response.get('results'):
                break
        candidates = []
        reasons = Counter()
        for obs in observations:
            evidence = base.identifiers(obs, taxon['id'])
            why = []
            if (obs.get('taxon') or {}).get('id') != taxon['id']: why.append('not_exact_taxon')
            if obs.get('flags') or evidence['tier'] != 'multiple_independent_exact_ids': why.append('insufficient_independent_source_label')
            if str(obs['id']) in old_obs: why.append('historical_observation')
            photos = [p for p in obs.get('photos', []) if p.get('license_code') in base.LICENSES
                      and str(p['id']) not in old_photos and not p.get('flags') and not p.get('hidden')]
            if not photos: why.append('no_new_licensed_photo')
            if why:
                exclusions.append({'label': label, 'observation_id': obs['id'], 'reasons': why})
                reasons.update(why)
            else:
                candidates.append((obs, photos, evidence))
        rng.shuffle(candidates)
        accepted_obs = 0
        for obs, photos, evidence in candidates:
            if accepted_obs >= 100: break
            accepted_photo_count = 0
            for photo in photos:
                if accepted_photo_count >= 3: break
                url = photo['url'].replace('/square.', '/large.')
                response = requests.get(url, headers=base.UA, timeout=35)
                if response.status_code in (401, 403, 429, 503):
                    raise base.Denied('Media source denial; collection stopped')
                response.raise_for_status()
                raw = response.content
                with Image.open(io.BytesIO(raw)) as src:
                    im = ImageOps.exif_transpose(src).convert('RGB')
                digest = hashlib.sha256(raw).hexdigest()
                visual = base.phash(im)
                near_old = base.nearest_hash(visual, old_visual)
                near_new = base.nearest_hash(visual, new_visual)
                gray = np.asarray(im.convert('L'))
                why = []
                if digest in old_hashes or digest in new_hashes: why.append('exact_duplicate')
                if near_old: why.append('near_historical_image')
                if near_new: why.append('near_new_image')
                if min(im.size) < 224: why.append('low_resolution')
                if len(raw) > 20_000_000 or im.width * im.height > 40_000_000: why.append('oversized')
                if (gray < 15).mean() > .85 or (gray > 245).mean() > .85: why.append('extreme_lighting')
                if why:
                    reasons.update(why)
                    exclusions.append({'label': label, 'observation_id': obs['id'], 'photo_id': photo['id'],
                                       'reasons': why, 'near_old': near_old, 'near_new': near_new})
                    continue
                path = out / 'images' / f"inat_{photo['id']}.jpg"
                if path.exists(): raise FileExistsError(path)
                path.write_bytes(raw)
                rows.append({'id': f"inat_{photo['id']}", 'label': label, 'source_species': label,
                    'query_term': label, 'taxonomy_mapping': mapping, 'taxon_id': taxon['id'],
                    'observation_id': obs['id'], 'photo_id': photo['id'], 'observer_id': obs['user']['id'],
                    'source_group': f"inat_observation_{obs['id']}", 'source_kind': 'community_observation',
                    'observed_on': obs.get('observed_on'), 'source_created_at': obs.get('created_at'),
                    'source_page': obs['uri'], 'photo_url': url, 'source_evidence': evidence,
                    'license': photo['license_code'], 'attribution': photo.get('attribution'),
                    'region_query': 'global', 'hong_kong': 7613 in obs.get('place_ids', []),
                    'recent_after_base_project': (obs.get('observed_on') or '') >= '2026-09-08',
                    'file': path.relative_to(out).as_posix(), 'sha256': digest, 'phash': f'{visual:016x}',
                    'dimensions': list(im.size), 'bytes': len(raw), 'training_eligible': False,
                    'human_review_status': 'source_labelled_not_expert_confirmed', 'split': 'prospective_evaluation'})
                new_hashes.add(digest); new_visual[visual] = str(path)
                accepted_photo_count += 1
                base.save(out / 'inat_manifest_partial.json', rows)
                print('PHOTO', label, len(rows), obs['id'], photo['id'], flush=True)
                time.sleep(.3)
            accepted_obs += int(accepted_photo_count > 0)
        summary.append({'label': label, 'queried_observations': len(observations),
            'candidate_observations': len(candidates), 'accepted_observations': accepted_obs,
            'photos': sum(r['label'] == label for r in rows), 'exclusions': dict(reasons)})
        base.save(out / 'inat_collection_summary.json', summary)
        base.save(out / 'inat_exclusions.json', exclusions)
    base.save(out / 'inat_manifest.json', rows)
    base.sheets(out, rows)
    print('INAT COMPLETE', summary, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('prepare', 'collect'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    {'prepare': prepare, 'collect': collect}[args.phase](args.out.resolve())
