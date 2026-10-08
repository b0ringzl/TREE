"""Read-only model audit and real local inference; creates a new report only."""
import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import requests

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from spacescan_trees_demo_v1 import official_release
from spacescan_trees_demo_v1.species_names import bilingual_names


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    # Prior checks/reports are never overwritten.
    args.output.mkdir(parents=True, exist_ok=False)
    expected = official_release.RELEASE
    delivery = PROJECT / expected['android_delivery']
    deployed = read(PROJECT / 'tree_species_teacher_demo/models/SSTv1_Artocarpus_User_20260918_v1/deployment.json')
    mobile = read(delivery / 'model_manifest.json')
    results = {'created_at': datetime.now().astimezone().isoformat(), 'checks': [], 'training_performed': False}

    def check(name, ok, **evidence):
        results['checks'].append({'name': name, 'passed': bool(ok), **evidence})
        print(name, bool(ok), flush=True)

    check('expected_deployment', deployed['version'] == expected['classifier_version'])
    check('same_android_encoder_heads_temperature_labels', mobile['source_deployment'] == deployed)
    labels = read(PROJECT / 'outputs/distillation/bioclip_probe_20260908_0227/classes.json')['labels'] + ['Wodyetia bifurcata']
    species = read(delivery / 'species.json')
    check('same_148_class_order', [row['label'] for row in species] == labels and len(labels) == 148)
    check('same_bilingual_species_names', all(row['scientific_name'] == bilingual_names(row['label'])['scientific_name'] and row['chinese_name'] == bilingual_names(row['label'])['chinese_name'] for row in species))
    for key in ('apk', 'model'):
        path = official_release.download_path(key)
        sha = digest(path)
        check('android_' + key + '_sha256', sha == expected['android_' + key + '_sha256'], path=str(path), sha256=sha, bytes=path.stat().st_size)

    backup_path = PROJECT / 'outputs/SpaceScan_Android_SSTv1_20260919_v1/pre_training_backup/backup_manifest.json'
    for row in read(backup_path)['files']:
        path = Path(row['source'])
        # Compare existing encoder/heads/point weights and mappings to verified pre-port snapshot.
        if path.suffix in ('.pt', '.safetensors', '.onnx', '.joblib', '.json'):
            sha = digest(path)
            check('unchanged_' + row['backup'], sha == row['sha256'], path=str(path), sha256=sha)
    results['pre_port_backup_manifest'] = str(backup_path)

    historical = read(delivery / 'paired_regression.json')
    check('prior_conversion_regression_preserved', len(historical) == 21 and all(r['same_top1'] for r in historical))
    results['historical_conversion_regression'] = {'n': len(historical), 'same_top1': sum(r['same_top1'] for r in historical), 'max_abs_score_drift': max(r['max_abs'] for r in historical), 'interpretation': 'Previously inspected fixed-RGB conversion regression, NOT a new species-accuracy evaluation or full Android pipeline parity.'}
    live = requests.get('http://127.0.0.1:8043/api/release', timeout=10)
    live.raise_for_status()
    results['live_release'] = live.json()
    check('official_web_ready_and_version_aligned', results['live_release']['ready'] and results['live_release']['web_version_state'] == 'aligned')
    for key in ('apk', 'model'):
        response = requests.get('http://127.0.0.1:8043/api/official-download/' + key, headers={'Range': 'bytes=0-7'}, timeout=15)
        check('live_download_' + key, response.status_code == 206 and response.content[:2] == b'PK' and len(response.content) == 8)

    example = Path('C:/Users/25041725G/Desktop/验证/微信图片_20260916084639_130_2.jpg')
    with example.open('rb') as stream:
        response = requests.post('http://127.0.0.1:8043/api/predict', files={'image': (example.name, stream, 'image/jpeg')}, data={'projection': 'photo'}, timeout=120)
    response.raise_for_status()
    prediction = response.json()
    results['whole_image_api_regression'] = {key: prediction.get(key) for key in ('image_model_version', 'release_provenance', 'width', 'height', 'tree_detection')}
    results['whole_image_api_regression']['top3'] = [{k: row[k] for k in ('species', 'score')} for row in prediction['top3']]
    check('terminalia_whole_image_still_top1', prediction['top3'][0]['scientific_name'] == 'Terminalia mantaly')
    check('actual_prediction_version_matches', prediction['image_model_version'] == expected['classifier_version'])
    check('input_provenance_present', prediction['release_provenance']['target_bytes_sha256'] == digest(example))
    results['all_checks_passed'] = all(row['passed'] for row in results['checks'])
    (args.output / 'verification.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    if not results['all_checks_passed']:
        raise SystemExit('Release verification failed; see report. No weights were changed.')
    print(str(args.output / 'verification.json'))


if __name__ == '__main__':
    main()
