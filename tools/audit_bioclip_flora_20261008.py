"""Fixed, attribution-preserving scope pilot. No fitting, threshold search or publication."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spacescan_trees_demo_v1.server import prepare
from tools.training_safety import sha256, verify_backup

RUN = ROOT / 'outputs/bioclip_flora_integration_20261008_v1'
SPECIES = ['Helianthus annuus', 'Solanum lycopersicum', 'Monstera deliciosa', 'Quercus robur',
           'Picea abies', 'Betula pendula', 'Pinus sylvestris', 'Eucalyptus globulus']


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def get(url, **params):
    response = requests.get(url, params=params, timeout=45)
    response.raise_for_status()
    return response.json()


def collect():
    if (RUN / 'audit_protocol.json').exists():
        raise RuntimeError('Frozen pilot already exists; no replacement')
    verify_backup(RUN / 'pre_training_backup', verify_sources=False)
    manifest = json.loads((ROOT / 'outputs/plantnet_demo_comparison_20260929_v2/manifest.json').read_text(encoding='utf-8'))
    hk = [{**row, 'source_label': row['species'], 'scope_truth': 'hong_kong',
           'label_authority': 'iNaturalist research-grade source; not independent expert verification'}
          for row in manifest['samples']]
    frozen = {'router_policy': 'flora-advisory-v1-margin2-same-genus-no-auto-switch',
              'thresholds_selected_before_this_audit': True, 'threshold_tuning_on_audit': False,
              'hk_regression_count': len(hk), 'outside_requested_species': SPECIES,
              'outside_images_per_species': 2, 'outside_selection': 'First distinct public CC0 photo per observation, id ascending',
              'acceptance_gate': {'sst_same_top1_all': True, 'sst_max_probability_drift': 0.000005,
                                  'strong_wrong_scope_advice_max_fraction_hk': 0.05},
              'limitations': ['Previously inspected HK cohort is regression, not fresh blind evidence',
                             'Public iNaturalist images may overlap foundation pretraining',
                             'No claim of complete open-set rejection or per-species broad accuracy'],
              'manual_broad_results_are_never_mixed_with_sst': True}
    write(RUN / 'audit_protocol.json', frozen)
    images = RUN / 'ood_images'
    images.mkdir(exist_ok=False)
    outside = []
    for species in SPECIES:
        taxa = get('https://api.inaturalist.org/v1/taxa', q=species, per_page=10)['results']
        matches = [r for r in taxa if r['name'].casefold() == species.casefold() and r.get('rank') == 'species']
        if not matches:
            print('No exact source taxon:', species, flush=True)
            continue
        taxon = matches[0]
        observations = get('https://api.inaturalist.org/v1/observations', taxon_id=taxon['id'],
                           quality_grade='research', photo_license='cc0', photos=True,
                           order_by='id', order='asc', per_page=20)['results']
        count = 0
        for observation in observations:
            if observation.get('taxon', {}).get('name', '').casefold() != species.casefold():
                continue
            photo = next((p for p in observation.get('photos', []) if p.get('license_code') == 'cc0'), None)
            if not photo:
                continue
            url = photo['url'].replace('/square.', '/large.')
            response = requests.get(url, timeout=45)
            response.raise_for_status()
            data = response.content
            _, width, height = prepare(data)
            path = images / (species.replace(' ', '_') + '_' + str(photo['id']) + '.jpg')
            path.write_bytes(data)
            outside.append({'image_path': str(path.resolve()), 'source_label': species,
                            'scope_truth': 'outside_hk', 'species': species,
                            'sha256': sha256(path), 'photo_id': photo['id'], 'width': width, 'height': height,
                            'license': 'CC0', 'source_url': f"https://www.inaturalist.org/observations/{observation['id']}",
                            'image_url': url, 'taxon_id': taxon['id'], 'label_authority': 'iNaturalist research-grade source'})
            count += 1
            if count == 2:
                break
        print('Frozen source sample:', species, count, flush=True)
    # Fail if any supposed outside taxon is actually in the closed-set map.
    classes = json.loads((ROOT / 'outputs/distillation/bioclip_probe_20260908_0227/classes.json').read_text(encoding='utf-8'))
    if any(r['source_label'] in classes['original_names'] for r in outside):
        raise RuntimeError('An outside pilot taxon is in the HK model')
    write(RUN / 'audit_samples.json', hk + outside)


def evaluate():
    output = RUN / 'audit_results.json'
    if output.exists():
        raise RuntimeError('Audit already completed; never overwrite')
    rows = json.loads((RUN / 'audit_samples.json').read_text(encoding='utf-8'))
    results = []
    for i, row in enumerate(rows):
        path = Path(row['image_path'])
        if sha256(path) != row['sha256']:
            raise RuntimeError('Frozen image changed')
        data = path.read_bytes()
        pixels, _, _ = prepare(data)
        baseline = requests.post('http://127.0.0.1:8039/api/predict-context',
                                 files={'image': ('whole.jpg', pixels, 'image/jpeg'), 'detail': ('detail', data)}, timeout=120)
        baseline.raise_for_status()
        old = baseline.json()['sstv1_distribution']
        response = requests.post('http://127.0.0.1:8057/api/predict',
                                 files={'image': ('image.jpg', data, 'image/jpeg')},
                                 data={'projection': 'photo', 'recognition_scope': 'hong_kong'}, timeout=120)
        response.raise_for_status()
        candidate = response.json()
        broad = candidate.get('flora_check', {})
        if not broad.get('available'):
            raise RuntimeError('Scope check unavailable: inspect service log')
        new = candidate['image_distribution']
        drift = max(abs(old[k] - new[k]) for k in old)
        record = {**row, 'sst_top3': [{k: r[k] for k in ['species', 'score']} for r in candidate['top3']],
                  'baseline_raw_top1': max(old, key=old.get), 'candidate_raw_top1': max(new, key=new.get),
                  'max_sst_probability_drift': drift, 'scope_advisory': candidate['scope_advisory'],
                  'flora_check': broad, 'quality_warnings': candidate['warnings']}
        results.append(record)
        print(f"{i + 1}/{len(rows)} {row['source_label']}: SST={candidate['top3'][0]['species']}; broad={broad['top3'][0]['species']}; action={candidate['scope_advisory']['action']}", flush=True)
    hk = [r for r in results if r['scope_truth'] == 'hong_kong']
    outside = [r for r in results if r['scope_truth'] == 'outside_hk']
    wrong_advice = sum(r['scope_advisory']['action'] == 'suggest_broad' for r in hk)
    summary = {'hk_n': len(hk), 'outside_n': len(outside),
               'all_sst_same_raw_top1': all(r['baseline_raw_top1'] == r['candidate_raw_top1'] for r in results),
               'max_sst_probability_drift': max(r['max_sst_probability_drift'] for r in results),
               'hk_strong_wrong_scope_advice': wrong_advice,
               'outside_strong_scope_advice': sum(r['scope_advisory']['action'] == 'suggest_broad' for r in outside),
               'outside_global_exact_top1': sum(r['flora_check']['top3'][0]['species'] == r['source_label'] for r in outside),
               'hk_global_exact_top1': sum(r['flora_check']['top3'][0]['species'] == r['source_label'] for r in hk)}
    summary['pilot_acceptance_pass'] = (summary['all_sst_same_raw_top1'] and summary['max_sst_probability_drift'] <= .000005
                                       and wrong_advice / len(hk) <= .05)
    write(output, {'summary': summary, 'results': results})
    print(summary, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['collect', 'evaluate'])
    args = parser.parse_args()
    collect() if args.phase == 'collect' else evaluate()
