"""Non-destructive metric-height audit and fixed-split morphology ablation.

Ground planes are local candidates, NOT surveyed root collars. Original point
clouds, models, manifests and human decisions are never overwritten.
"""
import hashlib
import json
from collections import Counter
from pathlib import Path
import joblib
import numpy as np
from scipy.spatial import cKDTree
from sklearn.linear_model import LinearRegression, LogisticRegression, RANSACRegressor
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, normalize

DATA = Path(__file__).resolve().parents[1] / 'derived/hk_dualsource_three_species_20260907'
OUT = DATA / 'metric_height_v1'

def read(p):
    return json.loads(p.read_text(encoding='utf-8'))

def write(p, value):
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def ground_candidate(local, axis):
    """Fit a low-envelope plane around the stem; reject poor spatial support."""
    xy = local[:, :2] - axis
    radius = np.linalg.norm(xy, axis=1)
    p = local[(radius > 1.2) & (radius < 6)]
    if len(p) < 100:
        return {'status': 'insufficient_ground_context'}
    cells = np.floor((p[:, :2]-axis) / .5).astype(int)
    _, inverse = np.unique(cells, axis=0, return_inverse=True)
    order = np.argsort(inverse)
    groups = np.split(p[order], np.flatnonzero(np.diff(inverse[order]))+1)
    low = np.array([np.r_[np.median(g[:, :2], axis=0), np.quantile(g[:, 2], .1)] for g in groups])
    low = low[low[:, 2] <= np.quantile(low[:, 2], .4) + .3]
    if len(low) < 20:
        return {'status': 'insufficient_low_cells'}
    model = RANSACRegressor(LinearRegression(), min_samples=8,
                           residual_threshold=.15, max_trials=150, random_state=7)
    model.fit(low[:, :2]-axis, low[:, 2])
    inlier = low[model.inlier_mask_]
    sectors = np.unique(np.floor((np.arctan2(inlier[:, 1]-axis[1], inlier[:, 0]-axis[0])+np.pi) / (np.pi/4)))
    residual = float(np.median(np.abs(model.predict(inlier[:, :2]-axis)-inlier[:, 2])))
    slope = float(np.linalg.norm(model.estimator_.coef_))
    supported = len(inlier) >= 20 and len(sectors) >= 5 and slope < .5 and residual < .12
    return {'status': 'local_plane_candidate' if supported else 'unreliable_ground_plane',
            'ground_z_m': float(model.predict(np.zeros((1, 2)))[0]),
            'plane_xy_slope': model.estimator_.coef_.tolist(),
            'inlier_cells': len(inlier), 'angular_sectors': len(sectors),
            'median_residual_m': residual, 'slope': slope,
            'root_collar_verified': False}

def measure(cleaned, record, local=None):
    q = record['quality']; axis = np.array(q['axis_local_xy'])
    top = float(np.quantile(cleaned[:, 2], .995))
    bottom = float(cleaned[:, 2].min())
    result = {'units': 'metres', 'metric_source': 'original_VMMS_local_XYZ',
              'observed_vertical_span_m': float(np.ptp(cleaned[:, 2])),
              'robust_observed_height_m': top-bottom, 'canopy_top_q995_z_m': top,
              'observed_bottom_z_m': bottom, 'total_height_verified': False,
              'estimated_tree_height_m': None, 'height_feature_eligible': False,
              'flags': []}
    if local is None:
        result.update(ground={'status': 'dense_ground_context_not_audited'})
        result['flags'].append('ground_unverified')
        return result
    ground = ground_candidate(local, axis); result['ground'] = ground
    if ground['status'] != 'local_plane_candidate':
        result['flags'].append('ground_unverified')
        return result
    z0 = ground['ground_z_m']; gap = bottom-z0
    # This tests the entire ground-to-crown interval, unlike the old crop-base test.
    floor = float(q['crown_floor_z'])
    edges = np.arange(z0+.2, floor+.4, .4)
    stem = cleaned[(cleaned[:, 2] < floor) & (np.linalg.norm(cleaned[:, :2]-axis, axis=1) < 2.5)]
    hist = np.histogram(stem[:, 2], edges)[0] if len(edges) >= 2 else np.array([])
    occupancy = float((hist >= 3).mean()) if len(hist) else 0.
    result.update(estimated_tree_height_m=top-z0, bottom_ground_gap_m=gap,
                  ground_to_crown_stem_occupancy=occupancy,
                  height_feature_eligible=bool(-.25 <= gap <= .6 and occupancy >= .8 and top-z0 > 1))
    if gap > .6: result['flags'].append('missing_basal_stem_or_wrong_ground')
    if gap < -.25: result['flags'].append('points_below_estimated_ground')
    if occupancy < .8: result['flags'].append('ground_to_crown_stem_gaps')
    result['flags'].append('ground_and_crown_require_human_confirmation')
    return result

def audit():
    OUT.mkdir(exist_ok=True)
    rows = [r for r in read(DATA/'manifest.json') if r.get('point') and Path(r['point']).exists()]
    results = {}; loaded_path = None; cloud = tree = None
    for i, r in enumerate(sorted(rows, key=lambda r: r.get('dense_cache', ''))):
        path = r.get('dense_cache'); local = None
        if path and Path(path).exists():
            if path != loaded_path:
                cloud = np.load(path)['xyz']; tree = cKDTree(cloud[:, :2]); loaded_path = path
            axis = np.array(r['quality']['axis_local_xy'])
            local = cloud[tree.query_ball_point(axis, 6)]
        cleaned = np.load(Path(r['point']).with_name('cleaned_local.npz'))['xyz']
        results[r['id']] = measure(cleaned, r, local)
        if (i+1) % 30 == 0: print('height audit', i+1, '/', len(rows), flush=True)
    write(OUT/'measurements.json', results)
    reviews = read(DATA/'pair_reviews.json') if (DATA/'pair_reviews.json').exists() else {}
    write(OUT/'review_snapshot.json', reviews)
    summary = {'pairs': len(results), 'eligible_provisional_heights': sum(x['height_feature_eligible'] for x in results.values()),
               'verified_total_heights': 0, 'flags': dict(Counter(f for x in results.values() for f in x['flags'])),
               'human_review_counts': dict(Counter(v['decision'] for v in reviews.values()))}
    write(OUT/'summary.json', summary)
    return results

def ablation(measurements):
    split_path = DATA/'experiment/split.json'; split = read(split_path)
    reviews = read(OUT/'review_snapshot.json')
    # Freeze original membership, exclude rejected pairs AND their conservative
    # groups symmetrically in ALL variants. No heldout-driven split reselection.
    excluded_groups = {r['tree_group'] for r in split if reviews.get(r['id'], {}).get('decision') == 'reject'}
    selected = [r for r in split if r['split'] in ('train', 'heldout') and r['tree_group'] not in excluded_groups]
    names = ['榕树', 'Livistona chinensis 蒲葵', 'Wodyetia bifurcata 狐尾椰子']
    xi = []; xp = []; physical = []; labels = []
    for r in selected:
        fingerprint = hashlib.sha256(Path(r['point']).read_bytes()+Path(r['image']).read_bytes()).hexdigest()[:16]
        with np.load(DATA/'experiment/features'/(r['id']+'_'+fingerprint+'.npz')) as f:
            xi.append(f['image']); xp.append(f['point'])
        m = measurements[r['id']]; eligible = m['height_feature_eligible']
        # Unknown height contributes neutral height + explicit missingness.
        physical.append([m['estimated_tree_height_m'] if eligible else np.nan, float(eligible)])
        labels.append(names.index(reviews.get(r['id'], {}).get('species', r['species'])))
    xi = normalize(xi); xp = normalize(xp); y = np.array(labels); physical = np.array(physical)
    train = np.array([r['split'] == 'train' for r in selected]); test = ~train
    if not np.isfinite(physical[train, 0]).any(): raise ValueError('No training height available')
    median = float(np.nanmedian(physical[train, 0]))
    physical[:, 0] = np.nan_to_num(physical[:, 0], nan=median)
    scaler = StandardScaler().fit(physical[train]); h = scaler.transform(physical)
    # Fixed branch weight; no hyperparameter search on heldout labels.
    weight = .25
    variants = {'image': xi, 'point': xp, 'fusion': normalize(np.c_[xi, xp]),
                'height_only': h, 'point_height': np.c_[xp, h*weight],
                'fusion_height': np.c_[normalize(np.c_[xi, xp]), h*weight],
                'point_availability_only': np.c_[xp, h[:, 1:]*weight],
                'fusion_availability_only': np.c_[normalize(np.c_[xi, xp]), h[:, 1:]*weight]}
    counts = Counter(r['tree_group'] for r, t in zip(selected, train) if t)
    weights = np.array([1/counts[r['tree_group']] for r, t in zip(selected, train) if t])
    heads = {}; metrics = {}
    for name, x in variants.items():
        clf = LogisticRegression(C=5, class_weight='balanced', max_iter=3000, random_state=7)
        clf.fit(x[train], y[train], sample_weight=weights); p = clf.predict(x[test]); heads[name] = clf
        metrics[name] = {'accuracy': float(accuracy_score(y[test], p)),
                         'balanced_accuracy': float(balanced_accuracy_score(y[test], p)),
                         'confusion_matrix': confusion_matrix(y[test], p, labels=[0, 1, 2]).tolist()}
    report = {'status': 'research_only_not_deployed', 'split_sha256': hashlib.sha256(split_path.read_bytes()).hexdigest(),
              'membership_policy': 'frozen original split, reject groups removed identically; review overrides applied identically',
              'excluded_rejected_groups': sorted(excluded_groups), 'train_pairs': int(train.sum()), 'heldout_pairs': int(test.sum()),
              'height_branch_weight': weight, 'train_median_imputation_m': median,
              'height_available_train': int(physical[train, 1].sum()), 'height_available_heldout': int(physical[test, 1].sum()),
              'scope': 'weak image-transfer label agreement, not accepted classification accuracy; local ground and crown not verified',
              'class_order': names, 'metrics': metrics}
    joblib.dump({'heads': heads, 'scaler': scaler, 'median_height_m': median, 'height_branch_weight': weight,
                 'experimental': True, 'promoted': False}, OUT/'ablation_model.joblib')
    write(OUT/'ablation_report.json', report); write(OUT/'ablation_split.json', selected)
    print(json.dumps(report, ensure_ascii=True, indent=2))

if __name__ == '__main__':
    ablation(audit())
