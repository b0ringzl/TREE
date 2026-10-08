"""Second-pass geometry QC: suppress dense stem seeds and planar/fragment artifacts."""
import sys,argparse
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
from hk_fullscene_extract_20260912 import OUT,read,write
from rebuild_banyan_coverage_preview import sample_ids,orthographic
DEST=ROOT/'outputs/hk_fullscene_refined_20260912_v2'
def shape(q):
    ev=np.maximum(np.linalg.eigvalsh(np.cov(q.T)),1e-9);xy=np.maximum(np.linalg.eigvalsh(np.cov(q[:,:2].T)),1e-9);extent=np.ptp(q,axis=0);upper=q[q[:,2]>=np.quantile(q[:,2],.55)];crown=np.ptp(upper[:,:2],axis=0);ratio=float(xy[0]/xy[1]);volume=float(ev[0]/ev.sum())
    return dict(extent_m=extent.tolist(),xy_minor_major_ratio=ratio,volumetric_ratio=volume,crown_width_m=crown.tolist())
def refine(map_id):
    source=OUT/'maps'/map_id
    if not (source/'candidates.json').exists():return
    folder=DEST/'maps'/map_id
    if (folder/'candidates.json').exists():return
    folder.mkdir(parents=True,exist_ok=False);rows=read(source/'candidates.json');x=np.load(source/'coarse.npy');cloudids={}
    for r in rows:
        with np.load(r['cloud']) as f:cloudids[r['id']]=f['coarse_source_indices']
    # Non-transitive NMS avoids joining an entire row of adjacent trees/facades.
    remaining=set(r['id'] for r in rows);ordered=sorted(rows,key=lambda r:-r['coarse_points']);groups=[]
    for a in ordered:
        if a['id'] not in remaining:continue
        near=[r for r in rows if r['id'] in remaining and np.linalg.norm(np.array(r['axis_xy'])-a['axis_xy'])<3.]
        for r in near:remaining.remove(r['id'])
        groups.append((a,near))
    kept=[];rejected=[]
    for a,near in groups:
        ids=np.unique(np.concatenate([cloudids[r['id']] for r in near]));q=x[ids];stats=shape(q);ext=np.array(stats['extent_m']);reasons=[]
        if len(q)<180:reasons.append('sparse_fragment')
        if min(stats['crown_width_m'])<1.3:reasons.append('insufficient_two_dimensional_crown_spread')
        if stats['xy_minor_major_ratio']<.12:reasons.append('line_or_facade_like_xy_shape')
        if stats['volumetric_ratio']<.018:reasons.append('near_planar_object')
        if ext[2]<2.5:reasons.append('low_vegetation_or_fragment')
        if ext[:2].max()>18 or ext[2]>35:reasons.append('oversized_mixed_object')
        # A dominant fitted plane is recorded as a warning; facade-shaped planes reject.
        pool=q[sample_ids(q,1200)];rng=np.random.default_rng(20260912);best=(0.,0.)
        for _ in range(40):
            tri=pool[rng.choice(len(pool),3,replace=False)];normal=np.cross(tri[1]-tri[0],tri[2]-tri[0]);norm=np.linalg.norm(normal)
            if norm<1e-8:continue
            normal/=norm;frac=float((abs((pool-tri[0])@normal)<.15).mean())
            if frac>best[0]:best=(frac,float(abs(normal[2])))
        stats['dominant_plane_fraction']=best[0];stats['dominant_plane_abs_z_normal']=best[1]
        if best[0]>.6:reasons.append('dominant_plane_contamination')
        record=dict(a,merged_v1_ids=[r['id'] for r in near],quality=stats,rejection_reasons=reasons)
        if reasons:rejected.append(record);continue
        key=map_id+'_r'+str(len(kept)+1).zfill(4);d=folder/key;d.mkdir();sid=sample_ids(q,2048);center=(q.min(0)+q.max(0))/2;scale=max(np.linalg.norm(q-center,axis=1).max(),1e-8)
        np.savez_compressed(d/'cloud.npz',xyz=q,coarse_source_indices=ids);np.savez_compressed(d/'point.npz',points_xyz=((q[sid]-center)/scale).astype(np.float32),metric_xyz=q[sid],centroid_xyz=center,scale=scale);orthographic(q,sample_ids(q,6000),d/'views.jpg',float(np.quantile(q[:,2],.25)))
        record.update(id=key,point=str(d/'point.npz'),cloud=str(d/'cloud.npz'),views=str(d/'views.jpg'),coarse_points=len(q),sample_points=len(sid),extent_m=stats['extent_m'],observed_z_span_m=float(ext[2]),status='geometry_screened_candidate_not_confirmed_tree',training_eligible=False,species=None,whole_tree_verified=False)
        kept.append(record)
    write(folder/'candidates.json',kept);write(folder/'rejected.json',rejected);write(folder/'summary.json',dict(input_candidates=len(rows),after_seed_merge=len(groups),retained_geometry_candidates=len(kept),rejected_after_merge=len(rejected),reasons=dict(Counter(v for r in rejected for v in r['rejection_reasons']))));print(map_id,'v1',len(rows),'merged',len(groups),'QC candidates',len(kept),flush=True)
def summarize():
    sources=read(OUT/'sources.json');rows=[];coverage=[]
    for s in sources:
        folder=DEST/'maps'/s['id'];exists=(folder/'candidates.json').exists();items=read(folder/'candidates.json') if exists else [];rows+=items;coverage.append(dict(id=s['id'],complete=exists,candidates=len(items)))
    parent=list(range(len(rows)))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    links=[]
    for survey in sorted({r['survey'] for r in rows}):
        ids=[i for i,r in enumerate(rows) if r['survey']==survey];tree=cKDTree([rows[i]['axis_xy'] for i in ids])
        for a,b in tree.query_pairs(3):
            i,j=ids[a],ids[b]
            if rows[i]['map_id']==rows[j]['map_id']:continue
            qa=np.load(rows[i]['cloud'])['xyz'];qb=np.load(rows[j]['cloud'])['xyz'];sa=qa[sample_ids(qa,1000)];sb=qb[sample_ids(qb,1000)];ab=float((cKDTree(qb).query(sa)[0]<.65).mean());ba=float((cKDTree(qa).query(sb)[0]<.65).mean())
            if max(ab,ba)>=.3:parent[find(i)]=find(j);links.append(dict(a=rows[i]['id'],b=rows[j]['id'],overlap_ab=ab,overlap_ba=ba))
    groups={}
    for i,r in enumerate(rows):r['dedup_group']='provisional_'+str(find(i));groups.setdefault(r['dedup_group'],[]).append(r)
    for group,rs in groups.items():
        best=max(rs,key=lambda r:(r['base_support_points']>=5,r['coarse_points']))
        for r in rs:r['representative']=r is best
    summary=dict(maps_complete=sum(r['complete'] for r in coverage),maps_planned=len(sources),candidate_files=len(rows),provisional_groups=len(groups),verified_trees=None,training_eligible=0,by_survey=dict(Counter(r['survey'] for r in rows if r['representative'])),coverage=coverage,limitations=['Geometry screening not tree existence truth','Cannot merge new capture sessions without verified registration','Ground elevation approximate; no complete tree/root certification','Dense recovery and image verification tracked separately'])
    write(DEST/'candidates.json',rows);write(DEST/'summary.json',summary);write(DEST/'dedup_links.json',links);print(summary,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--ids',nargs='*');p.add_argument('--summarize',action='store_true');a=p.parse_args();DEST.mkdir(exist_ok=True)
    if not (DEST/'config.json').exists():write(DEST/'config.json',dict(source=str(OUT),method='3 m nontransitive seed suppression and merged source point membership; sparse/XY-spread/planarity/oversize QC',species_labels_used=False,source_mutations=False))
    if a.summarize:summarize()
    else:
        for s in read(OUT/'sources.json'):
            if not a.ids or s['id'] in a.ids:refine(s['id'])
