"""Recover measured dense neighbours of image-supported geometric proposals.

No invented points, labels, training or claim of complete individual trees.
"""
import sys,time,argparse
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
from hk_fullscene_extract_20260912 import OUT,read,write,header,chunks,voxel
from hk_fullscene_refine_20260912 import DEST
from rebuild_banyan_coverage_preview import sample_ids,orthographic

def main(map_id):
    source=next(s for s in read(OUT/'sources.json') if s['id']==map_id)
    rows=read(DEST/'maps'/map_id/'candidates.json');chosen=[]
    for r in rows:
        p=DEST/'image_evidence'/r['id']/'record.json'
        if p.exists() and read(p)['status'] in ['one_view_tree_support','two_view_tree_support']:chosen.append(r)
    if not chosen:raise ValueError('No image-supported proposals available')
    dest=DEST/'dense_recovery'/map_id;dest.mkdir(parents=True,exist_ok=False)
    source_path=Path(source['dense_source']);before=source_path.stat();h=header(source_path)
    entries=[]
    for r in chosen:
        q=np.load(r['cloud'])['xyz'];entries.append((r,q.min(0)-.6,q.max(0)+.6,cKDTree(q),[]))
    seen=0;t=time.time()
    for x in chunks(source_path,h):
        seen+=len(x);x=x[np.isfinite(x).all(1)]
        for r,lo,hi,tree,parts in entries:
            a=x[((x>=lo)&(x<=hi)).all(1)]
            if len(a):
                a=a[tree.query(a,workers=2,distance_upper_bound=.6)[0]<=.6]
                if len(a):parts.append(voxel(a,.08))
                if len(parts)>10:parts[:]=[voxel(np.concatenate(parts),.08)]
        if seen%10000000<len(x):print(map_id,'dense scanned',seen,'/',h['POINTS'][0],flush=True)
    after=source_path.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
    result=[]
    for r,lo,hi,tree,parts in entries:
        if not parts:result.append(dict(id=r['id'],status='no_dense_overlap'));continue
        q=voxel(np.concatenate(parts),.08);d=dest/r['id'];d.mkdir();ids=sample_ids(q,2048);center=(q.max(0)+q.min(0))/2;scale=max(np.linalg.norm(q-center,axis=1).max(),1e-8)
        np.savez_compressed(d/'cloud.npz',xyz=q);np.savez_compressed(d/'point.npz',metric_xyz=q[ids],points_xyz=((q[ids]-center)/scale).astype(np.float32),centroid_xyz=center,scale=scale)
        orthographic(q,sample_ids(q,6000),d/'views.jpg',float(np.quantile(q[:,2],.25)))
        result.append(dict(id=r['id'],map_id=map_id,coarse_points=r['coarse_points'],dense_retained_points=len(q),sample_points=len(ids),cloud=str(d/'cloud.npz'),point=str(d/'point.npz'),views=str(d/'views.jpg'),source=str(source_path),extent_m=np.ptp(q,axis=0).tolist(),status='measured_dense_neighbours_not_whole_tree_certification',species=None,training_eligible=False))
    write(dest/'manifest.json',result);write(dest/'audit.json',dict(source=str(source_path),scanned_points=seen,source_points=int(h['POINTS'][0]),size=before.st_size,mtime_ns=before.st_mtime_ns,source_unchanged=True,seconds=time.time()-t,policy='Within 0.6 m of screened measured coarse support; voxel 0.08 m; independent 2048 point sample. Does not reconstruct missing trunk/crown.',selected=len(chosen)))
    print('DENSE DONE',map_id,[(r['id'],r.get('dense_retained_points')) for r in result],flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--map',required=True);main(p.parse_args().map)
