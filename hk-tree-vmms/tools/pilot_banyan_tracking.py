"""Ten-tree, non-destructive lower-stem tracking pilot; no synthetic points.

Uses overlapping existing map caches from the SAME survey. Missing original
coverage remains missing. Neighbour images are evidence, not inferred labels.
"""
from collections import Counter
from pathlib import Path
import hashlib
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
from build_hk_dense_dataset import coordinates, polygon_inside, coverage_sample
from build_hk_lidar_pairs import project, voxel
from audit_hk_metric_height import DATA, read, write, ground_candidate
from repair_hk_banyan_stems import views, balanced_ids

OUT = DATA/'banyan_tracking_pilot_v1'
REQUIRED = ['jianshazui_pano_2__002821__65726f344e',
            'jianshazui_pano_2__002734__154764da16',
            'jianshazui_pano_2__002794__c55635bd6b']

def select(records):
    candidates = [r for r in records if r['species']=='榕树'
                  and r['status']=='geometry_candidate_pending_review'
                  and r.get('dense_cache') and Path(r['dense_cache']).exists()]
    chosen = [next(r for r in candidates if r['id']==i) for i in REQUIRED]
    # Spread routes and stem sparsity; never select by recovery/model success.
    bins = {}
    for r in candidates:
        bins.setdefault(r['survey'], []).append(r)
    for key in bins:
        bins[key].sort(key=lambda r:(r['quality']['trunk_points']/r['quality']['clean_points'], r['id']))
    while len(chosen)<10:
        added = False
        for survey in sorted(bins):
            for r in bins[survey]:
                axis = np.array(r['quality']['axis_local_xy'])
                if any(s['survey']==survey and np.linalg.norm(axis-s['quality']['axis_local_xy'])<12 for s in chosen): continue
                chosen.append(r); added=True; break
            if len(chosen)==10: break
        if not added: break
    return chosen

def track(candidate, axis, z0, floor):
    """Layer graph: allow drifting, splitting stems but reject broad wall slices."""
    nodes=[]; layers={}; step=.4
    for layer,z in enumerate(np.arange(z0+.25, floor+.8, step)):
        ids=np.flatnonzero((candidate[:,2]>=z)&(candidate[:,2]<z+step))
        if len(ids)<8: continue
        labels=DBSCAN(eps=.32,min_samples=4).fit_predict(candidate[ids,:2])
        for label in set(labels)-{-1}:
            idx=ids[labels==label]; p=candidate[idx]; span=np.ptp(p[:,:2],axis=0)
            if len(idx)<8 or span.max()>4.5: continue
            eig=np.linalg.eigvalsh(np.cov(p[:,:2].T))
            if span.max()>1.5 and eig[0]/max(eig.sum(),1e-9)<.007: continue
            n=len(nodes); nodes.append({'ids':idx,'layer':layer,'xy':p[:,:2], 'links':set()});layers.setdefault(layer,[]).append(n)
    for layer, ids in layers.items():
        for a in ids:
            tree=cKDTree(nodes[a]['xy'])
            for gap in (1,2):
                for b in layers.get(layer+gap,[]):
                    # Permit a single missing 40cm slice, but record actual gaps.
                    if tree.query(nodes[b]['xy'])[0].min()<(.45 if gap==1 else .55):
                        nodes[a]['links'].add(b);nodes[b]['links'].add(a)
    seen=set(); options=[]
    for n in range(len(nodes)):
        if n in seen: continue
        todo=[n]; component=[]; seen.add(n)
        while todo:
            j=todo.pop();component.append(j)
            for k in nodes[j]['links']-seen:seen.add(k);todo.append(k)
        idx=np.unique(np.concatenate([nodes[j]['ids'] for j in component])); p=candidate[idx]
        if len(p)<100 or np.ptp(p[:,2])<2 or p[:,2].min()>z0+1.5 or p[:,2].max()<floor-.5: continue
        low=p[p[:,2]<p[:,2].min()+1]
        distance=np.linalg.norm(np.median(low[:,:2],axis=0)-axis)
        if distance>4.5:continue
        bins=np.histogram(p[:,2], np.arange(z0+.25,floor+.4,.4))[0]
        coverage=float((bins>=3).mean()) if len(bins) else 0
        score=coverage*np.log1p(len(p))/(1+.3*distance)
        options.append((score,idx,coverage))
    if not options:return None, {'reason':'no_supported_low_stem_to_crown_path','slice_nodes':len(nodes)}
    _,idx,coverage=max(options,key=lambda x:x[0])
    return candidate[idx], {'slice_nodes':len(nodes),'observed_stem_bin_coverage':coverage,
                            'tracking_policy':'40cm layers, one-gap maximum, branching allowed'}

def main():
    OUT.mkdir(exist_ok=True); records=read(DATA/'manifest.json'); chosen=select(records); rows=coordinates()
    write(OUT/'selection.json', [{'id':r['id'],'survey':r['survey'],'tree_group':r['tree_group']} for r in chosen])
    review_path=DATA/'pair_reviews.json'; review_hash=hashlib.sha256(review_path.read_bytes()).hexdigest()
    # Collect nearby real points from every available same-survey cache once.
    local_parts={r['id']:[] for r in chosen}; source_paths={r['id']:[] for r in chosen}
    for survey in sorted({r['survey'] for r in chosen}):
        paths=sorted({r.get('dense_cache') for r in records if r.get('survey')==survey and r.get('dense_cache') and Path(r['dense_cache']).exists()})
        subset=[r for r in chosen if r['survey']==survey]
        for path in paths:
            cloud=np.load(path)['xyz']; tree=cKDTree(cloud[:,:2])
            for r in subset:
                ids=tree.query_ball_point(r['quality']['axis_local_xy'],8)
                if ids:local_parts[r['id']].append(cloud[ids]);source_paths[r['id']].append(path)
            print('map cache',Path(path).name,flush=True)
    results=[]
    for r in chosen:
        ident=r['id']; folder=OUT/ident; folder.mkdir(exist_ok=True)
        old=np.load(Path(r['point']).with_name('cleaned_local.npz'))['xyz']; q=r['quality']; floor=q['crown_floor_z'];axis=np.array(q['axis_local_xy'])
        local=np.concatenate(local_parts[ident]);local=local[voxel(local,.10)]
        ground=ground_candidate(local,axis)
        # A failed ground plane must NOT prevent exploring a lower stem. A low
        # elevation bound only defines an ROI; it is never exported as tree height.
        verified_plane=ground['status']=='local_plane_candidate'
        z0=ground['ground_z_m'] if verified_plane else float(np.quantile(local[:,2],.01))
        uv,_=project(local,rows[r['frame_key']])
        mask=polygon_inside(uv,np.array(r['polygon']),dilation=7)
        candidate=local[mask&(local[:,2]>z0+.25)&(local[:,2]<floor+.8)&(np.linalg.norm(local[:,:2]-axis,axis=1)<6)]
        stem,stats=track(candidate,axis,z0,floor)
        result={'id':ident,'survey':r['survey'],'old_trunk_points':q['trunk_points'],
                'source_cache_paths':source_paths[ident], 'source_map_count':len(source_paths[ident]),
                'raw_lower_roi_points':len(candidate),'ground':ground,'roi_bottom_z_m':z0,
                'total_height_verified':False, 'status':'blocked','tracking':stats,
                'original_review':read(review_path).get(ident), 'original_points_unchanged':True}
        # Nearby views are provided for human same-tree cross-check, not votes
        # from frames whose independent alignment has not been verified.
        nearby=sorted([s for s in records if s.get('survey')==r['survey'] and s.get('point') and s['id']!=ident
                       and s.get('quality') and np.linalg.norm(np.array(s['quality']['axis_local_xy'])-axis)<2],key=lambda s:s['id'])[:3]
        result['nearby_view_ids']=[s['id'] for s in nearby]
        views(old,folder/'before.jpg',floor)
        with Image.open(r['source_panorama']) as src:im=src.convert('RGB');im.thumbnail((1280,640))
        im.save(folder/'panorama.jpg',quality=94)
        if stem is not None:
            recovered=np.r_[old[old[:,2]>=floor],stem];recovered=recovered[voxel(recovered,.10)]
            ti=np.flatnonzero(recovered[:,2]<floor);ci=np.flatnonzero(recovered[:,2]>=floor)
            # Three vertical stem bands prevent the upper stem taking every slot.
            bands=np.array_split(ti[np.argsort(recovered[ti,2])],3);sample=[]
            for band in bands:
                sample.extend(band[coverage_sample(recovered[band],min(256,len(band)))].tolist())
            sample.extend(ci[coverage_sample(recovered[ci],min(2048-len(sample),len(ci)))].tolist());ids=np.array(sample)
            center=recovered.mean(0);scale=np.linalg.norm(recovered-center,axis=1).max()
            np.savez_compressed(folder/'cleaned_local.npz',xyz=recovered)
            np.savez_compressed(folder/'point.npz',points_xyz=((recovered[ids]-center)/scale).astype(np.float32),centroid_xyz=center,scale=scale,metric_units='m')
            views(recovered,folder/'after.jpg',floor)
            puv=project(stem,rows[r['frame_key']])[0];draw=ImageDraw.Draw(im)
            for u,v in puv[balanced_ids(stem,5000)]:
                x,y=u*im.width,v*im.height;draw.ellipse((x-1,y-1,x+1,y+1),fill='#ff942e')
            im.save(folder/'projection.jpg',quality=94)
            # Every added point must be traceable to a real local cache point.
            error=float(cKDTree(local).query(stem)[0].max()); assert error<1e-5
            assert len(ids)==len(np.unique(recovered[ids],axis=0))
            result.update(status='candidate_requires_review',new_trunk_points=len(ti),sample_unique_points=len(ids),
                          source_membership_error_m=error,base_gap_to_roi_m=float(stem[:,2].min()-z0),
                          warning='未认证完整；花坛、背景、邻树和冠部归属待核查。无可靠地面时不提供整树高度。')
        write(folder/'report.json',result);results.append(result)
        print(ident, result['status'],q['trunk_points'],'->',result.get('new_trunk_points'),flush=True)
    write(OUT/'results.json',results)
    write(OUT/'verification.json',{'selected':len(chosen),'statuses':dict(Counter(r['status'] for r in results)),
                                 'source_reviews_unchanged':review_hash==hashlib.sha256(review_path.read_bytes()).hexdigest(),
                                 'source_review_hash_before':review_hash,'no_model_retraining':True})

if __name__=='__main__':main()
