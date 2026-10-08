"""Read-only candidate screening. No fitting, fusion claims or production edits."""
import hashlib
import html
import json
import shutil
from pathlib import Path
from collections import Counter

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/point_assist_case_selection_20260913_v1'

def read(p):
    return json.loads((ROOT / p).read_text(encoding='utf8'))

def save(p, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf8')

def quality(path):
    with Image.open(path) as im:
        w, h = im.size
        im.thumbnail((512, 512))
        gray = np.asarray(im.convert('L'), dtype=float)
    return dict(width=w, height=h, dark_fraction=float((gray < 40).mean()),
                contrast=float(gray.std()), note='Whole crop quality proxy; not a tree-mask measurement')

def main():
    OUT.mkdir(exist_ok=False)
    catalog = read('outputs/hk_point_assist_20260911_v1/catalog.json')
    geometry = read('outputs/hk_point_assist_20260911_v1/geometry_reviews.json')
    training = {r['key']: r for r in read('outputs/hk_point_training_20260912_v1/manifest.json')}
    folds = read('outputs/bauhinia_ptv3_fewshot_20260912_v1/ptv2_results.json')
    rows = []
    for r in catalog:
        if r['species'] not in ['Livistona chinensis', 'Wodyetia bifurcata']:
            continue
        t = training.get(r['key'], {})
        g = geometry.get(r['key'], {})
        exact = g.get('asset_hashes') == r.get('sha256')
        complete = exact and all(g.get(k) is True for k in ['same_tree', 'isolated_tree', 'base_visible', 'crown_complete'])
        measurement = r.get('historical_measurement', {})
        rows.append(dict(id=r['id'], key=r['key'], species=r['species'], case='palm',
            group=t.get('group'), split=t.get('split', 'unassigned'), label=r['review'],
            geometry_review=g, geometry_review_matches_current_assets=exact,
            height_candidate=bool(complete and measurement.get('height_feature_eligible')),
            estimated_height_m=measurement.get('estimated_tree_height_m'),
            height_is_surveyed_truth=False, original_measurement=measurement,
            cloud=r['asset_paths']['cleaned'], preview=t.get('views'),
            views=[dict(image=r['asset_paths']['image'], overlay=r['asset_paths']['projection'],
                        frame_key=r['id'], index=0)],
            screening='geometry_candidate' if complete else 'needs_geometry_recheck'))
    priorities = read('outputs/single_tree_panorama_selection_v1/priority_pairs.json')
    for r in priorities:
        if r['id'] not in ['01_TreeID_13', '01_TreeID_14', '01_TreeID_15']:
            continue
        rows.append(dict(id=r['id'], key=r['id'], species='Bauhinia x blakeana', case='bauhinia',
            group=r['id'], split='leave_one_tree_out', label='Explicit user confirmation of first three trees',
            height_candidate=False, estimated_height_m=None, height_is_surveyed_truth=False,
            cloud=r['path'], preview=r['geometry_preview'], views=r['views'],
            calibration_status=r.get('calibration_status'), screening='needs_pair_and_quality_confirmation'))
    assets = {}
    for row in rows:
        row['point_results'] = []
        for fold in folds:
            for split in ['validation', 'test_regression']:
                for pred in fold[split]['predictions']:
                    if pred['id'] == row['id']:
                        row['point_results'].append(dict(fold=fold['fold'], split=split,
                            model=fold['run'], **pred))
        for view in row['views']:
            view['quality'] = quality(view['image'])
        # Quality-only rank, before any new image predictions. Keep ALL views.
        row['views'].sort(key=lambda v: -v['quality']['dark_fraction'])
        for path in [row['cloud'], row['preview']] + [v[k] for v in row['views'] for k in ['image', 'overlay']]:
            if path and Path(path).is_file():
                assets[str(path)] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    rows.sort(key=lambda r: (r['case'] != 'bauhinia', not r['height_candidate'], r['species'], r['id']))
    save(OUT/'manifest.json', rows)
    save(OUT/'source_hashes.json', assets)
    save(OUT/'geometry_reviews_snapshot.json', geometry)
    save(OUT/'ptv2_results_snapshot.json', folds)
    summary = dict(screened_accepted_pair_catalog=len(catalog),
        screened_segmented_priority_catalog=len(priorities), selected_instances=len(rows),
        selected_views=sum(len(r['views']) for r in rows), species_counts=dict(Counter(r['species'] for r in rows)),
        palm_height_candidates=[r['id'] for r in rows if r['height_candidate']],
        scope='All target palms in existing accepted paired catalog, plus three confirmed segmented Bauhinia; NOT exhaustive raw-scene species discovery.',
        image_inference_completed=False, fusion_validated=False, training_performed=False, deployment_changed=False)
    save(OUT/'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False))

if __name__ == '__main__':
    main()
