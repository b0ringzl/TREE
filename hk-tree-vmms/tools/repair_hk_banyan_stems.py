"""Versioned, review-only recovery of explicitly rejected banyan stems."""
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
from build_hk_dense_dataset import coordinates, polygon_inside, coverage_sample
from build_hk_lidar_pairs import project, voxel
from audit_hk_metric_height import DATA, OUT, read, write, measure

def balanced_ids(xyz, limit=8000):
    """Show scarce lower stems instead of letting dense crowns dominate display."""
    if len(xyz) <= limit: return np.arange(len(xyz))
    labels = np.minimum(39, ((xyz[:, 2]-xyz[:, 2].min())/max(np.ptp(xyz[:, 2]), 1e-6)*40).astype(int))
    return np.concatenate([idx[np.linspace(0, len(idx)-1, min(len(idx), limit//40), dtype=int)]
                           for i in range(40) if len(idx := np.flatnonzero(labels == i))])

def views(xyz, path, floor=None):
    sheet = Image.new('RGB', (1200, 460), '#eff4ed'); draw = ImageDraw.Draw(sheet)
    center = (xyz.min(0)+xyz.max(0))/2; p = xyz-center
    scale = 350/max(np.ptp(xyz, axis=0).max(), 1)
    for panel, (a, b, title) in enumerate([(0, 2, 'XZ'), (1, 2, 'YZ'), (0, 1, 'XY')]):
        draw.text((panel*400+12, 12), title+' | metres, height-stratified preview', fill='#193d2b')
        for i in balanced_ids(xyz):
            x = panel*400+200+p[i, a]*scale; y = 230-p[i, b]*scale
            color = '#b45d18' if floor is not None and xyz[i, 2] < floor else '#26744d'
            draw.ellipse((x, y, x+2, y+2), fill=color)
    sheet.save(path, quality=94)

def main():
    rows = coordinates(); manifest = read(DATA/'manifest.json'); reviews = read(DATA/'pair_reviews.json')
    metrics = read(OUT/'measurements.json'); result = []
    preview_dir = OUT/'previews'; preview_dir.mkdir(exist_ok=True)
    for r in manifest:
        if r['id'] not in metrics: continue
        cleaned = np.load(Path(r['point']).with_name('cleaned_local.npz'))['xyz']
        views(cleaned, preview_dir/(r['id']+'.jpg'), r['quality']['crown_floor_z'])
    for r in manifest:
        review = reviews.get(r['id'], {})
        if r['species'] != '榕树' or review.get('decision') != 'reject' or '干' not in review.get('note', ''): continue
        folder = OUT/'banyan_repairs'/r['id']; folder.mkdir(parents=True, exist_ok=True)
        old = np.load(Path(r['point']).with_name('cleaned_local.npz'))['xyz']
        cloud = np.load(r['dense_cache'])['xyz']; axis = np.array(r['quality']['axis_local_xy'])
        local = cloud[np.linalg.norm(cloud[:, :2]-axis, axis=1) < 6]
        m = metrics[r['id']]; ground = m['ground']
        if ground['status'] != 'local_plane_candidate':
            result.append({'id': r['id'], 'status': 'blocked_unreliable_ground'}); continue
        row = rows[r['frame_key']]; uv, _ = project(local, row)
        floor = r['quality']['crown_floor_z']; z0 = ground['ground_z_m']
        # Broad lower-stem ROI supports buttresses/aerial roots, while image
        # polygon and 3D connected components constrain neighboring objects.
        candidate = local[polygon_inside(uv, np.array(r['polygon']), dilation=3)
                          & (np.linalg.norm(local[:, :2]-axis, axis=1) < 4.5)
                          & (local[:, 2] > z0+.35) & (local[:, 2] < floor+.8)]
        labels = DBSCAN(eps=.38, min_samples=4).fit_predict(candidate)
        components = []
        for label in set(labels)-{-1}:
            s = candidate[labels == label]
            if len(s) < 100 or np.ptp(s[:, 2]) < 1.5: continue
            # Must connect both low stem and crown; proximity alone is insufficient.
            if s[:, 2].min() > z0+1.5 or s[:, 2].max() < floor-.5: continue
            components.append(s)
        if not components:
            result.append({'id': r['id'], 'status': 'no_ground_to_crown_component'}); continue
        stem = max(components, key=len)
        # Preserve the original crown, replace only lower-stem extraction.
        recovered = np.r_[old[old[:, 2] >= floor], stem]
        recovered = recovered[voxel(recovered, .10)]
        ti = np.flatnonzero(recovered[:, 2] < floor); ci = np.flatnonzero(recovered[:, 2] >= floor)
        nt = min(768, len(ti)); nc = min(2048-nt, len(ci))
        ids = np.r_[ti[coverage_sample(recovered[ti], nt)], ci[coverage_sample(recovered[ci], nc)]]
        center = recovered.mean(0); scale = np.linalg.norm(recovered-center, axis=1).max()
        np.savez_compressed(folder/'cleaned_local.npz', xyz=recovered)
        np.savez_compressed(folder/'point.npz', points_xyz=((recovered[ids]-center)/scale).astype(np.float32),
                            centroid_xyz=center, scale=scale, metric_units='m')
        views(old, folder/'before.jpg', floor); views(recovered, folder/'after.jpg', floor)
        with Image.open(r['source_panorama']) as src:
            im = src.convert('RGB'); im.thumbnail((1280, 640))
        draw = ImageDraw.Draw(im); puv = project(stem, row)[0]
        for u, v in puv[balanced_ids(stem, 5000)]:
            x, y = u*im.width, v*im.height; draw.ellipse((x-1, y-1, x+1, y+1), fill='#ff942e')
        im.save(folder/'projection.jpg', quality=94)
        report = {'id': r['id'], 'status': 'repair_candidate_requires_review', 'original_review_preserved': review,
                  'old_trunk_points': r['quality']['trunk_points'], 'new_trunk_points': len(ti),
                  'sample_unique_points': len(np.unique(recovered[ids], axis=0)),
                  'ground': ground, 'new_height_audit': measure(recovered, r, local),
                  'caveat': 'Planter walls and intertwined aerial roots need visual review; crown unchanged, not certified complete.',
                  'source': r['dense_cache'], 'production_data_replaced': False}
        write(folder/'report.json', report); result.append(report)
        print(r['id'], report['old_trunk_points'], '->', len(ti), flush=True)
    write(OUT/'banyan_repairs.json', result)

if __name__ == '__main__': main()
