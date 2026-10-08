"""Explicitly authorized H120 deployment: stage/verify only, never fit or auto-switch."""
import argparse
import copy
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
from training_safety import backup_before_training, verify_backup, sha256
from bioclip_probe import ResidualHead

RUN = ROOT / 'outputs/SSTv1_Aquilaria_UserCrops_20260919_v2'
OUT = ROOT / 'outputs/SSTv1_Aquilaria_H120_deployment_20260919_v1'
OLD = ROOT / 'tree_species_teacher_demo/models/SSTv1_Artocarpus_User_20260918_v1'
NEW = ROOT / 'tree_species_teacher_demo/models/SSTv1_Aquilaria_UserCrops_20260919_H120'
CHECKPOINT = RUN / 'H_crops_targeted_002/step_120.pt'
VERSION = 'sstv1-aquilaria-user-crops-20260919-h120'
WEB = 'http://127.0.0.1:8043'
TEACHER = 'http://127.0.0.1:8039'
PAIR = ROOT / 'outputs/SpaceScan_Trees_Demo_Files_20260913_56b49420/02_cloud_assist_gain/01_TreeID_13_view0'
EDITABLE = [ROOT / p for p in (
    'tree_species_teacher_demo/engine.py', 'spacescan_trees_demo_v1/release.json',
    'spacescan_trees_demo_v1/official_release.py', 'spacescan_trees_demo_v1/templates/index.html',
    'spacescan_trees_demo_v1/static/apps.html', 'spacescan_trees_demo_v1/static/apps.js')]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def get(url):
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    return r.json()


def post(url, files, fields):
    r = requests.post(url, files=files, data=fields, timeout=180)
    r.raise_for_status()
    result = r.json()
    result.pop('model_input', None)
    result.pop('crop_preview', None)
    return result


def pair():
    cloud, = PAIR.glob('cloud_for_replay.*')
    return post(WEB + '/api/predict-with-cloud', {
        'image': ('image.jpg', (PAIR / 'image.jpg').read_bytes()),
        'cloud': (cloud.name, cloud.read_bytes())},
        {'projection': 'photo', 'paired': 'true', 'units': 'metres', 'z_up': 'true', 'scenario': 'street'})


def capture():
    images = []
    for row in read(RUN / 'H_crops_targeted_002/crop_replay_single.json')['images']:
        path = Path(row['source_image'])
        result = post(WEB + '/api/predict', {'image': (path.name, path.read_bytes())},
                      {'projection': 'photo', 'source': 'image'})
        images.append(dict(filename=path.name, sha256=sha256(path), result=result))
        print('REPLAY', path.name, result['top3'][0]['species'], flush=True)
    # Original teacher and adapted branches must remain numerically unchanged.
    from spacescan_trees_demo_v1.server import prepare
    data, _, _ = prepare(Path(read(RUN / 'H_crops_targeted_002/crop_replay_single.json')['images'][0]['source_image']).read_bytes())
    upstream = post(TEACHER + '/api/predict', {'image': ('prepared.jpg', data)}, {'mode': 'sstv1'})
    paired = pair()
    return dict(health=get(WEB + '/api/health'), release=get(WEB + '/api/release'),
                images=images, upstream=upstream, pair=paired)


def stage():
    require(not OUT.exists() and not NEW.exists(), 'New deployment directory required')
    old = read(OLD / 'deployment.json')
    require(get(WEB + '/api/health')['single_image_version'] == old['version'], 'Unexpected deployed version')
    audit = read(RUN / 'false_positive_audit_v1/details.json')
    require(sha256(CHECKPOINT) == audit['checkpoint_sha256'], 'Not the audited H120 checkpoint')
    state = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    require(state['step'] == 120 and state['recipe']['name'] == 'H_crops_targeted_002', 'Wrong checkpoint')
    classes = read(RUN / 'classes.json')
    require(read(RUN / 'labels.json') == classes['labels'] + ['Wodyetia bifurcata'], 'Label mapping changed')
    experiment = ROOT / 'outputs/distillation/bioclip_probe_20260908_0227'
    require(classes == read(experiment / 'classes.json'), 'Original classes changed')
    point_path = ROOT / 'spacescan_trees_demo_v1/models/ptv2_seven_20260913_v1/deployment.json'
    point = read(point_path)
    immutable = [OLD / 'deployment.json', *[OLD / m['file'] for m in old['members']],
        CHECKPOINT, *[RUN / p for p in ('protocol.json', 'selection_lock.json', 'crop_manifest.json',
        'train_rows.json', 'labels.json', 'classes.json', 'teacher_choice.json', 'REPORT.md',
        'last_steps_single_dev.json', 'false_positive_audit_v1/details.json')],
        point_path, ROOT / point['model'], ROOT / point['encoder'],
        *[ROOT / p for p in ('spacescan_trees_demo_v1/server.py', 'spacescan_trees_demo_v1/point_recognition.py',
        'spacescan_trees_demo_v1/tree_detection.py', 'spacescan_trees_demo_v1/species_names.json',
        'spacescan_trees_demo_v1/static/polygon-crop.js', 'tree_species_teacher_demo/server.py')],
        *[experiment / p for p in ('classes.json', 'teacher_choice.json', 'selection_lock.json', 'protocol.json')]]
    immutable += [Path(read(experiment / 'protocol.json')['teacher_weight'])]
    immutable += [Path(r['run']) / 'best.pt' for r in read(experiment / 'selection_lock.json')['selected_runs']]
    immutable = list(dict.fromkeys(p.resolve() for p in immutable))
    OUT.mkdir()
    backup = backup_before_training(OUT, immutable + EDITABLE, {
        'purpose': 'Pre-deployment physical rollback backup, NO TRAINING',
        'user_authorization': '没关系，先替换正式demo',
        'prior_version': old['version'], 'new_version': VERSION,
        'validation_gate_passed': False,
        'policy': 'Explicit user override after disclosure of false positives 7 to 10; historical selection lock unchanged'})
    info = verify_backup(backup)
    for entry in info['files']:
        target = backup / entry['backup']
        require(target.stat().st_size == entry['bytes'] and not target.samefile(entry['source']), 'Invalid physical copy')
    save(OUT / 'protected.json', {str(p): sha256(p) for p in immutable})
    save(OUT / 'BEFORE.json', capture())
    NEW.mkdir()
    members = []
    for i, member in enumerate(old['members']):
        require(sha256(OLD / member['file']) == member['sha256'], 'Old weight changed')
        payload = torch.load(OLD / member['file'], map_location='cpu', weights_only=False)
        require(payload['classes'] == classes and payload['encoder_sha256'] == old['encoder_sha256'], 'Incompatible member')
        prefix = str(i) + '.'
        weights = {k[len(prefix):]: v for k, v in state['model_state'].items() if k.startswith(prefix)}
        head = ResidualHead('mlp').eval()
        head.load_state_dict(weights, strict=True)
        payload = dict(model_state=weights, config={'architecture': 'mlp'}, classes=classes,
                       encoder_sha256=old['encoder_sha256'], step=120,
                       provenance={'checkpoint': str(CHECKPOINT), 'sha256': audit['checkpoint_sha256'], 'member_index': i})
        target = NEW / f'member_{i+1}.pt'
        torch.save(payload, target)
        reloaded = torch.load(target, map_location='cpu', weights_only=False)['model_state']
        require(weights.keys() == reloaded.keys() and all(torch.equal(v, reloaded[k]) for k, v in weights.items()), 'Tensor export mismatch')
        members.append(dict(file=target.name, sha256=sha256(target), step=120, source=str(CHECKPOINT)))
    require(sum(len(torch.load(NEW/m['file'],map_location='cpu',weights_only=False)['model_state']) for m in members) == len(state['model_state']), 'Unexported joint keys')
    manifest = {k: copy.deepcopy(old[k]) for k in ('model','output_count','encoder_sha256','temperature','ensemble')}
    manifest.update(version=VERSION, display_version='SSTv1-H120', release_date='2026-09-19', members=members,
        source_run=str(RUN), source_checkpoint=str(CHECKPOINT), source_checkpoint_sha256=audit['checkpoint_sha256'],
        labels_sha256=sha256(RUN/'labels.json'), pre_deployment_backup=str(backup),
        user_authorized=True, authorization='没关系，先替换正式demo', training_performed_during_deployment=False,
        selection='Explicit user-authorized H120 override, NOT the validation-selected checkpoint', validation_gate_passed=False,
        preserved_selection_lock_sha256=sha256(RUN/'selection_lock.json'),
        known_regressions={'development_Aquilaria_FP':[7,10], 'development_Artocarpus_TP':[10,9],
                          'development_overall_correct':[2304,2302], 'development_overall_n':2650},
        scope='Formal Web/Windows single-image branch only; unchanged image preprocessing, original teacher/PTv2 fusion and Android package',
        rights='Private local evaluation only; not a public redistribution grant')
    save(NEW / 'deployment.json', manifest)
    shutil.copy2(RUN / 'labels.json', NEW / 'labels.json')
    save(OUT / 'STAGED.json', dict(version=VERSION, manifest=str(NEW/'deployment.json'), backup=str(backup),
        backup_files=len(info['files']), backup_bytes=sum(x['bytes'] for x in info['files']),
        tensor_parity=True, pointer_changed=False, training=False, created_at=datetime.now(timezone.utc).isoformat()))
    print(json.dumps(read(OUT/'STAGED.json'), ensure_ascii=False), flush=True)


def verify():
    before = read(OUT/'BEFORE.json')
    # Preserve the first post-switch capture, including any failed assertion evidence.
    if (OUT/'AFTER.json').exists():
        after = read(OUT/'AFTER.json')
    else:
        after = capture()
        save(OUT/'AFTER.json', after)
    require(get(WEB+'/api/health')['single_image_version'] == VERSION, 'Current service changed since capture')
    require(after['health']['ready'] and after['health']['single_image_version'] == VERSION, 'Live version mismatch')
    require(after['release']['web_version_state'] == 'aligned', 'Release mismatch')
    deltas = []
    expected = read(RUN/'H_crops_targeted_002/crop_replay_single.json')['images']
    for actual, row in zip(after['images'], expected):
        result = actual['result']
        require(result['image_model_version'] == VERSION, 'Wrong image model')
        require(result['top3'][0]['species'] == row['candidate']['top1'], 'Unexpected candidate')
        delta = abs(result['image_distribution']['Aquilaria sinensis'] - row['candidate']['true_score'])
        require(delta < 1e-4, 'Cached vs live H120 mismatch')
        require(len(result['image_distribution']) == 148, 'Label count mismatch')
        deltas.append(delta)
    for mode in ('teacher_distribution', 'adapted_distribution'):
        delta = max(abs(v-after['upstream'][mode][s]) for s,v in before['upstream'][mode].items())
        require(delta < 1e-5, 'Original teacher/adapted branch changed')
    # GPU point reduction order has sub-microprobability rounding differences.
    # Use the project's existing unchanged-branch tolerance (1e-5), exclude latency.
    pair_delta = 0.
    for branch in ('image', 'point', 'fusion'):
        a, b = before['pair']['comparison'][branch], after['pair']['comparison'][branch]
        require([r['species'] for r in a] == [r['species'] for r in b], 'Paired ranking differs: '+branch)
        delta = max(abs(x['score']-y['score']) for x,y in zip(a,b))
        require(delta < 1e-5, 'Paired scores differ: '+branch)
        pair_delta = max(pair_delta, delta)
    a, b = before['pair']['point_model'], after['pair']['point_model']
    require({k:v for k,v in a.items() if k not in ('scores','inference_seconds')} ==
            {k:v for k,v in b.items() if k not in ('scores','inference_seconds')}, 'Point contract changed')
    np.testing.assert_allclose(a['scores'], b['scores'], atol=1e-5, rtol=0)
    for p, expected_hash in read(OUT/'protected.json').items():
        require(sha256(p) == expected_hash, 'Protected file changed: '+p)
    backup = Path(read(OUT/'STAGED.json')['backup'])
    verify_backup(backup, verify_sources=False)
    for route in ('/image-only', '/multimodal'):
        html = requests.get(WEB+route,timeout=15).text
        require('SSTv1-H120 · 2026-09-19' in html, 'UI version not updated')
    require(after['release']['android_classifier_version'] == before['release']['classifier_version'], 'Android version mislabeled')
    require(after['release']['android_classifier_alignment'] == 'mismatch', 'Android divergence hidden')
    summary = dict(version=VERSION, crop_replay_correct=sum(x['result']['top3'][0]['species']=='Aquilaria sinensis' for x in after['images']),
        crop_replay_n=8, evidence='Training-image replay, NOT held-out accuracy', max_live_vs_cached_delta=max(deltas),
        original_teacher_adapted_unchanged=True, paired_branch_unchanged=True,
        paired_probability_max_abs=pair_delta, paired_comparison_atol=1e-5,
        initial_verification_note='Exact float/duration equality was replaced with the pre-existing 1e-5 unchanged-branch tolerance; first AFTER evidence preserved.',
        protected_hashes_unchanged=True, backup_verified=True, android_package_unchanged=True, training=False)
    save(OUT/'VERIFIED.json', summary)
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    torch.set_num_threads(4)
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['stage','verify'])
    args=parser.parse_args()
    stage() if args.action=='stage' else verify()
