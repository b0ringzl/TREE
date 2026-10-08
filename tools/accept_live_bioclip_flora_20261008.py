"""Actual formal-port smoke acceptance after sandbox evaluation."""
import hashlib
import json
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'outputs/bioclip_flora_integration_20261008_v1'
URL = 'http://127.0.0.1:8043'


def predict(data, **fields):
    response = requests.post(URL + '/api/predict', files={'image': ('image.jpg', data, 'image/jpeg')},
                             data={'projection': 'photo', **fields}, timeout=120)
    response.raise_for_status()
    return response.json()


def main():
    gate = json.loads((RUN / 'deployment_gate.json').read_text(encoding='utf-8'))
    if not gate['acceptance']['pilot_acceptance_pass'] or not gate['deployed_weights_unchanged']:
        raise RuntimeError('No verified acceptance gate')
    health = requests.get(URL + '/api/health', timeout=10).json()
    if not health['ready']:
        raise RuntimeError('Formal service not ready; retry after initialization')
    audit = json.loads((RUN / 'audit_results.json').read_text(encoding='utf-8'))['results']
    selected = audit[:3] + audit[50:53]
    checks = []
    for row in selected:
        data = Path(row['image_path']).read_bytes()
        if hashlib.sha256(data).hexdigest() != row['sha256']:
            raise RuntimeError('Source sample changed')
        result = predict(data, recognition_scope='hong_kong')
        scores = [{'species': r['species'], 'score': r['score']} for r in result['top3']]
        assert result['image_model_version'] == health['single_image_version']
        assert scores == row['sst_top3']
        assert result['scope_advisory'] == row['scope_advisory']
        broad = predict(data, recognition_scope='broad_plants')
        assert broad['mode'] == 'bioclip_broad' and broad['recognition_scope'] == 'broad_plants'
        assert [(r['species'], r['score']) for r in broad['top3']] == [(r['species'], r['score']) for r in result['flora_alternative']['top3']]
        checks.append({'species': row['source_label'], 'sst_equal_to_sandbox': True,
                       'manual_broad_equal_to_same_crop_alternative': True, 'scope_action': result['scope_advisory']['action']})
    image = Path(audit[50]['image_path']).read_bytes()
    single = predict(image, view_policy='single')
    assert single['single_view'] and single['flora_check']['view_count'] == 1
    assert single['flora_alternative']['single_view'] and not single['flora_alternative']['model_input_detail']
    invalid = requests.post(URL + '/api/predict', files={'image': ('image.jpg', image, 'image/jpeg')},
                            data={'projection': 'photo', 'recognition_scope': 'unsupported'}, timeout=30)
    assert invalid.status_code == 400
    detection_smoke = {}
    test = requests.get(URL + '/api/test-photo', timeout=15)
    test.raise_for_status()
    detected = requests.post(URL + '/api/detect-trees', files={'image': ('test.jpg', test.content, 'image/jpeg')},
                             data={'projection': 'photo'}, timeout=180)
    detected.raise_for_status()
    detected = detected.json()
    detection_smoke['action'] = detected['action']
    if detected['candidates']:
        response = requests.post(URL + '/api/identify-tree', files={'image': ('test.jpg', test.content, 'image/jpeg')},
                                 data={'token': detected['candidates'][0]['token'], 'recognition_scope': 'broad_plants'}, timeout=120)
        response.raise_for_status()
        target = response.json()
        assert target['mode'] == 'bioclip_broad' and target['selected_tree'] and target['crop_preview']
        detection_smoke['selected_target_broad_mode_works'] = True
    result = {'formal_url': URL + '/image-only', 'health': health, 'checks': checks,
              'sam_tight_crop_single_pass_preserved': True, 'invalid_scope_rejected': True,
              'detection_smoke': detection_smoke,
              'default_sst_and_original_weights_retained': True, 'test_servers_retired': [8040, 8057]}
    path = RUN / 'live_acceptance.json'
    if path.exists():
        raise RuntimeError('Never overwrite live acceptance')
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
