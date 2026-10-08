"""Exclusive, measured-point recovery of provisional 991 geometry candidates."""
import sys,json,time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
from hk_fullscene_extract_20260912 import header,chunks,gridkeys
from rebuild_banyan_coverage_preview import sample_ids,orthographic
from training_safety import sha256
OUT=ROOT/'outputs/tree991_single_tree_review_20260913_v1'
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8')
def main():
    OUT.mkdir(exist_ok=False)
    source=ROOT/'lidar data/hk/tree9/tree991/map_dense.pcd'
    proposal=ROOT/'outputs/hk_fullscene_refined_20260912_v2/maps/recent_tree991/candidates.json'
    coarse_path=ROOT/'outputs/hk_fullscene_extract_20260912_v1/maps/recent_tree991/coarse.npy'
    rows=json.loads(proposal.read_text(encoding='utf8'));coarse=np.load(coarse_path)
    original_hash=sha256(source);membership=np.zeros(len(coarse),np.int32);owner=np.full(len(coarse),-1,np.int16);distance=np.full(len(coarse),np.inf)
    for i,r in enumerate(rows):
        ids=np.load(r['cloud'])['coarse_source_indices'];membership[ids]+=1
        d=np.linalg.norm(coarse[ids,:2]-r['axis_xy'],axis=1);better=d<distance[ids]
        owner[ids[better]]=i;distance[ids[better]]=d[better]
    support_ids=np.flatnonzero(owner>=0);tree=cKDTree(coarse[support_ids]);pieces=[[] for r in rows];rawids=[[] for r in rows];offset=0
    for chunk in chunks(source,header(source)):
        finite=np.flatnonzero(np.isfinite(chunk).all(1));q=chunk[finite]
        d,ix=tree.query(q,workers=4,distance_upper_bound=.4);valid=np.flatnonzero(d<=.4)
        labels=owner[support_ids[ix[valid]]]
        for i in np.unique(labels):
            idx=valid[labels==i];x=q[idx];keep=np.unique(gridkeys(x,.08),return_index=True)[1]
            pieces[i].append(x[keep]);rawids[i].append((offset+finite[idx[keep]]).astype(np.int64))
        offset+=len(chunk);print('scanned',offset,flush=True)
    manifest=[]
    for i,r in enumerate(rows):
        if not pieces[i]:continue
        q=np.concatenate(pieces[i]);ids=np.concatenate(rawids[i]);keep=np.unique(gridkeys(q,.08),return_index=True)[1];q=q[keep];ids=ids[keep]
        folder=OUT/r['id'];folder.mkdir();np.savez_compressed(folder/'cloud.npz',xyz=q,source_indices=ids)
        sampled=sample_ids(q,2048);np.savez_compressed(folder/'point.npz',metric_xyz=q[sampled],source_indices=ids[sampled])
        orthographic(q,sample_ids(q,6000),folder/'views.jpg',float(np.quantile(q[:,2],.25)))
        save(folder/'display.json',np.round(q[sample_ids(q,7000)],3).tolist())
        overlap_ids=np.load(r['cloud'])['coarse_source_indices']
        flags=[]
        if r['base_support_points']<10:flags.append('weak_base_support')
        if r['observed_z_span_m']>20:flags.append('tall_or_mixed_check_building')
        if (membership[overlap_ids]>1).mean()>.2:flags.append('overlapping_proposals_boundary_uncertain')
        manifest.append(dict(id=r['id'],number=i+1,axis_xy=r['axis_xy'],point_count=len(q),sample_count=len(sampled),
            extent_m=np.ptp(q,axis=0).tolist(),review_flags=flags,status='provisional_geometry_instance_not_confirmed_tree',
            species=None,cloud_to_google_transform=None,projection_status='blocked_missing_verified_panorama_and_registration',
            source=str(source),cloud=str(folder/'cloud.npz'),point=str(folder/'point.npz'),preview=str(folder/'views.jpg')))
    save(OUT/'manifest.json',manifest)
    save(OUT/'scene.json',dict(points=np.round(coarse[sample_ids(coarse,20000)],3).tolist(),candidates=manifest))
    fig,ax=plt.subplots(figsize=(12,10));q=coarse[sample_ids(coarse,50000)];ax.scatter(q[:,0],q[:,1],s=.15,c=q[:,2],cmap='Greys',alpha=.5)
    for r in manifest:ax.text(*r['axis_xy'],str(r['number']),fontsize=7,color='#aa3300',bbox=dict(facecolor='white',alpha=.7,pad=.6,edgecolor='none'))
    ax.scatter([0],[0],marker='+',s=150,c='blue',label='Local origin (not Google camera)');ax.set_aspect('equal');ax.set_xlabel('Local X / metres');ax.set_ylabel('Local Y / metres');ax.set_title('991 provisional candidates — NOT geographic registration');ax.legend();fig.tight_layout();fig.savefig(OUT/'overview.png',dpi=160);plt.close(fig)
    assert sha256(source)==original_hash
    allids=np.concatenate([np.load(r['cloud'])['source_indices'] for r in manifest]);assert len(np.unique(allids))==len(allids)
    save(OUT/'audit.json',dict(source=str(source),source_sha256=original_hash,source_unchanged=True,scanned_points=offset,candidates=len(manifest),retained_points=len(allids),
        source_index_overlap=0,method='Existing stem/canopy geometric proposals; resolve shared coarse points by nearest seed in XY; assign each raw point to closest labelled support within 0.4m; 0.08m measured-point sampling',
        limitations=['Not a trained instance segmentation model','Nearest support ownership is approximate and can retain buildings or merge trees','Rejected earlier proposals not recovered; not exhaustive tree count','No species labels assigned','No Google panorama, camera pose, or projection verified'],
        input_proposals=str(proposal),input_proposals_sha256=sha256(proposal),training=False))
    print('DONE',len(manifest),len(allids))
if __name__=='__main__':main()
