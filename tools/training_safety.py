"""Fail-closed, run-specific physical copies of pre-training model artifacts."""
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def deployed_artifacts():
    return [ROOT / p for p in [
        'full_image_species_baseline/experiments/full_web_yolo11s_cls_20260906_seed1/pretrain/weights/selected_by_val_group_macro_f1.pt',
        'experiments/whu18_yolo11_ptv2_fusion_20260826/yolo11s_cls/run/weights/selected_by_val_macro_f1.pt',
        'experiments/whu18_yolo11_ptv2_fusion_20260826/ptv2/run/best.pt',
        'experiments/whu18_yolo11_ptv2_fusion_20260826/fusion/run/best.pt',
        'hk-tree-vmms/derived/hk_dualsource_three_species_20260907/experiment/model.joblib',
        'hk-tree-vmms/derived/reviewed_five_species_pilot_v1_20260907/model.joblib',
        'tree_species_demo/catalog.json', 'tree_species_demo/engine.py',
    ]]


def verify_backup(directory, verify_sources=True):
    directory = Path(directory)
    info = json.loads((directory / 'backup_manifest.json').read_text(encoding='utf-8'))
    for r in info['files']:
        p = directory / r['backup']
        if not p.is_file() or sha256(p) != r['sha256']:
            raise RuntimeError('Backup missing or corrupted: ' + str(p))
        if verify_sources and sha256(r['source']) != r['sha256']:
            raise RuntimeError('Original artifact changed: ' + r['source'])
    return info


def backup_before_training(run_dir, inputs, metadata=None):
    dest = Path(run_dir) / 'pre_training_backup'
    # An existing directory cannot be silently reused for another fit/resume.
    dest.mkdir(parents=True, exist_ok=False)
    files = []
    for i, source in enumerate(dict.fromkeys(Path(p).resolve() for p in inputs)):
        if not source.is_file():
            raise FileNotFoundError('Training stopped: missing backup input ' + str(source))
        before = sha256(source)
        target = dest / f'{i:02d}_{source.name}'
        shutil.copy2(source, target)
        if sha256(target) != before or sha256(source) != before:
            raise RuntimeError('Training stopped: backup verification failed ' + str(source))
        files.append({'source': str(source), 'backup': target.name, 'sha256': before, 'bytes': target.stat().st_size})
    manifest = {'created_at': datetime.now(timezone.utc).isoformat(), 'files': files, 'metadata': metadata or {},
                'rollback_policy': 'Production stays unchanged. To switch later, verify this manifest and restore only the explicitly selected original paths under separate user authorization.'}
    (dest / 'backup_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    verify_backup(dest)
    return dest
