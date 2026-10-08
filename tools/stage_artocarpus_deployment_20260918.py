"""Package the validation-selected 148-class ensemble, without fitting or switching."""
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from bioclip_probe import ResidualHead
from distillation_core import collapse_logits, digest
from training_safety import backup_before_training, sha256, verify_backup

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'outputs/SSTv1_Artocarpus_User_20260918_v1'
OUT = ROOT / 'outputs/SSTv1_Artocarpus_deployment_20260918_v1'
OLD = ROOT / 'tree_species_teacher_demo/models/SSTv1_Aquilaria_User_20260915_v1'
NEW = ROOT / 'tree_species_teacher_demo/models/SSTv1_Artocarpus_User_20260918_v1'
VERSION = 'sstv1-artocarpus-user-20260918-v1'


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def save(p, data):
    Path(p).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def main():
    torch.set_num_threads(4)
    require(not OUT.exists() and not NEW.exists(), 'Use a new output version; never overwrite a deployment')
    lock = read(RUN / 'selection_lock.json')
    protocol = read(RUN / 'protocol.json')
    old = read(OLD / 'deployment.json')
    classes, labels = read(RUN / 'classes.json'), read(RUN / 'labels.json')
    require(lock['validation_recommends_candidate'] and lock['epoch'] == 58 and lock['no_test_predictions_used'], 'Candidate not validation locked')
    require(sha256(lock['selected_checkpoint']) == lock['sha256'], 'Locked checkpoint changed')
    require(len(labels) == 148 and labels == classes['labels'] + ['Wodyetia bifurcata'], 'Label order changed')
    require(old['temperature'] == protocol['temperature'] and old['encoder_sha256'] == protocol['encoder_sha256'], 'Inference contract changed')
    engine_source = ROOT / 'tree_species_teacher_demo/engine.py'
    require("SSTV1_MANIFEST = ROOT / 'tree_species_teacher_demo/models/SSTv1_Aquilaria_User_20260915_v1/deployment.json'" in engine_source.read_text(encoding='utf-8'), 'Unexpected active manifest pointer')
    for path, expected in protocol['protected_sha256'].items():
        require(sha256(path) == expected, 'Protected pre-deployment file changed: ' + path)
    for name, expected in protocol['prepared_files'].items():
        require(sha256(RUN / name) == expected, 'Training snapshot changed: ' + name)
    checkpoints = [RUN / f'selected_members/member_{i}.pt' for i in (1, 2, 3)]
    joint = torch.load(lock['selected_checkpoint'], map_location='cpu', weights_only=False)
    heads = []
    for i, path in enumerate(checkpoints):
        state = torch.load(path, map_location='cpu', weights_only=False)
        require(state['epoch'] == lock['epoch'] and state['config']['architecture'] == 'mlp', 'Wrong member epoch/type')
        require(state['classes'] == classes and state['encoder_sha256'] == old['encoder_sha256'], 'Wrong encoder/labels')
        expected_state = {k[len(str(i)) + 1:]: v for k, v in joint['model_state'].items() if k.startswith(str(i) + '.')}
        require(expected_state.keys() == state['model_state'].keys(), 'Member state keys differ')
        require(all(torch.equal(v, expected_state[k]) for k, v in state['model_state'].items()), 'Export differs from locked checkpoint')
        head = ResidualHead('mlp').eval().requires_grad_(False)
        head.load_state_dict(state['model_state'])
        heads.append(head)
    # Probability parity on the existing development cache; no optimization.
    rows = read(RUN / 'planning_rows.json')
    with np.load(RUN / 'planning_features.npz', allow_pickle=False) as cache:
        require(str(cache['rows_hash']) == digest(rows), 'Cached row order changed')
        x = torch.from_numpy(cache['x'].astype(np.float32))
        z = torch.from_numpy(cache['z'].astype(np.float32))
        baseline = cache['audit_baseline'].copy()
    with np.load(RUN / 'inference_constants.npz', allow_pickle=False) as constants:
        extra = torch.from_numpy(constants['extra_bank'].astype(np.float32))
        scale = float(constants['logit_scale'])
    with torch.inference_mode():
        result = []
        for start in range(0, len(rows), 256):
            features, raw = x[start:start + 256], z[start:start + 256]
            tail = features @ extra.T * scale
            result.append(torch.stack([
                (torch.cat([collapse_logits(h(features, raw), classes['groups']), tail], -1) / old['temperature']).softmax(-1)
                for h in heads]).mean(0).numpy())
        actual = np.concatenate(result)
    with np.load(RUN / 'fit/selected_predictions.npz', allow_pickle=False) as predictions:
        expected = predictions['planning'].copy()
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-5)
    require(np.array_equal(actual.argmax(1), expected.argmax(1)), 'Packaged heads change validation predictions')
    OUT.mkdir()
    inputs = [*[Path(p) for p in protocol['protected_sha256']], *checkpoints,
              Path(lock['selected_checkpoint']), RUN / 'selection_lock.json', RUN / 'protocol.json',
              *[RUN / name for name in protocol['prepared_files']],
              ROOT / 'spacescan_trees_demo_v1/static/index.html', ROOT / 'tree_species_teacher_demo/server.py',
              ROOT / 'spacescan_trees_demo_v1/start-live.ps1', RUN / 'fit/selected_evaluation.json', RUN / 'fit/baseline_evaluation.json']
    backup = backup_before_training(OUT, inputs, {'purpose': 'User-authorized deployment rollback; NO training',
        'user_request': '先把现在最好的模型接入demo吧', 'prior_version': old['version'],
        'new_version': VERSION, 'selection': 'Fixed development gates and locked epoch 58; not test selection'})
    info = verify_backup(backup)
    for row in info['files']:
        copy = backup / row['backup']
        require(copy.stat().st_size == row['bytes'] and not copy.samefile(row['source']), 'Backup is not a verified physical copy')
    NEW.mkdir()
    manifest = dict(old)
    manifest.update(version=VERSION, source_run=str(RUN), members=[], pre_deployment_backup=str(backup),
                    user_authorized=True, training_performed_during_deployment=False,
                    selection_lock_sha256=sha256(RUN / 'selection_lock.json'),
                    labels_sha256=sha256(RUN / 'labels.json'), selected_epoch=lock['epoch'],
                    scope='148-class image-only update; unchanged teacher/PTv2 point fusion; no 150-class claim')
    for i, path in enumerate(checkpoints, 1):
        target = NEW / f'member_{i}.pt'
        shutil.copy2(path, target)
        require(sha256(target) == sha256(path) and not target.samefile(path), 'Staged weight copy verification failed')
        manifest['members'].append({'file': target.name, 'sha256': sha256(target), 'source': str(path), 'epoch': lock['epoch']})
    save(NEW / 'deployment.json', manifest)
    shutil.copy2(RUN / 'labels.json', NEW / 'labels.json')
    smoke = []
    for target in (0, 11, 16):
        index = next(i for i, row in enumerate(rows) if row['target'] == target)
        row = rows[index]
        image = Path(row['source_image'])
        require(sha256(image) == row['source_sha256'], 'Smoke image changed')
        smoke.append({'image': str(image), 'sha256': sha256(image), 'row_index': index,
                      'labels': labels, 'expected_candidate_distribution': expected[index].tolist(),
                      'baseline_cached_distribution': baseline[index].tolist(), 'target': target})
    save(OUT / 'expected.json', {'smoke': smoke, 'selection': 'First planning row for class indices 0, 11, 16, not cherry-picked by output'})
    save(OUT / 'STAGED.json', {'created_at': datetime.now(timezone.utc).isoformat(), 'manifest': str(NEW / 'deployment.json'),
         'version': VERSION, 'old_manifest': str(OLD / 'deployment.json'), 'backup': str(backup),
         'backup_files': len(info['files']), 'backup_bytes': sum(r['bytes'] for r in info['files']),
         'export_tensor_parity': True, 'planning_samples': len(rows), 'probability_parity_max_abs': float(abs(actual - expected).max()),
         'top1_parity': True, 'training_performed': False, 'active_pointer_changed': False})
    print(json.dumps(read(OUT / 'STAGED.json'), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
