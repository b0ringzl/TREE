"""Read-only inventory of assets needed by the current desktop inference paths."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / 'outputs/distillation/bioclip_probe_20260908_0227'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    check = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            check.update(block)
    return check.hexdigest()


assets = []


def add(name, path, expected=None, hash_now=True):
    path = Path(path)
    exists = path.is_file()
    result = {'name': name, 'path': str(path), 'inside_project': path.is_relative_to(ROOT),
              'exists': exists, 'bytes': path.stat().st_size if exists else None,
              'expected_sha256': expected}
    if exists and hash_now:
        result['sha256'] = digest(path)
        result['verified'] = expected is None or result['sha256'] == expected
    assets.append(result)


protocol = read(PROBE / 'protocol.json')
selection = read(PROBE / 'selection_lock.json')
deployment_path = ROOT / 'tree_species_teacher_demo/models/SSTv1_Aquilaria_UserCrops_20260919_H120/deployment.json'
deployment = read(deployment_path)
flora_path = ROOT / 'external_datasets/bioclip25_flora_20261008_v1/manifest.json'
flora = read(flora_path)
point_path = ROOT / 'spacescan_trees_demo_v1/models/ptv2_seven_20260913_v1/deployment.json'
point = read(point_path)

add('teacher_encoder', protocol['teacher_weight'], protocol['teacher_sha256'])
for index, row in enumerate(selection['selected_runs'], 1):
    add(f'adapted_head_{index}', Path(row['run']) / 'best.pt', selection['weights_sha256'][row['run']])
for index, row in enumerate(deployment['members'], 1):
    add(f'sst_head_{index}', deployment_path.parent / row['file'], row['sha256'])
add('sst_labels', deployment_path.parent / 'labels.json', deployment['labels_sha256'])
add('class_aliases', ROOT / 'outputs/distillation/20260908_0045/aliases.json')
add('tree_detector', ROOT / 'hk-tree-vmms/derived/training_runs/combined_tree_segmentation/yolo11m_seg_1024_20260901_104757/weights/best.pt')
add('clip_detector', ROOT / 'outputs/tree_detector_benchmark_20260909_v1/assets/clip/ViT-B-32.pt')
add('grounding_dino', ROOT / 'outputs/tree_detector_benchmark_20260909_v1/assets/grounding_dino_base/model.safetensors')
add('sam', ROOT / 'LabelPaw-web-images/weights/sam_weights/sam2.1_hiera_tiny.pt')
add('ptv2_model', ROOT / point['model'], point['model_sha256'])
add('ptv2_encoder', ROOT / point['encoder'], point['encoder_sha256'])
for row in flora['derived_files']:
    add('flora_' + row['file'], flora_path.parent / row['file'], row['sha256'])

print(json.dumps({
    'project': str(ROOT), 'asset_count': len(assets),
    'missing': [row['name'] for row in assets if not row['exists']],
    'hash_mismatches': [row['name'] for row in assets if row.get('verified') is False],
    'total_asset_bytes': sum(row['bytes'] or 0 for row in assets),
    'absolute_config_paths': ['protocol.teacher_weight', 'selection_lock.selected_runs[*].run'],
    'assets': assets,
}, ensure_ascii=False, indent=2))
