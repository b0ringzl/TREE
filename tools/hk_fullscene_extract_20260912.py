"""Label-independent, full-scene geometric tree candidate extraction.

Outputs are NOT confirmed species/instances. Never fits or deploys model weights.
Every spatially retained point is an original measured point; no completion.
"""
import argparse,csv,hashlib,json,sys,time
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.spatial import cKDTree
from scipy import ndimage
ROOT=Path(__file__).resolve().parents[1];VMMS=ROOT.parent/'vmms';OUT=ROOT/'outputs/hk_fullscene_extract_20260912_v1'
sys.path.insert(0,str(ROOT/'tools'))
from rebuild_banyan_coverage_preview import sample_ids,orthographic
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,obj):Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def gridkeys(x,size):
    q=np.floor(x/size).astype(np.int64)+(1<<20)
    if q.min()<0 or q.max()>=(1<<21):raise ValueError('Coordinate range exceeds grid index')
    return (q[:,0]<<42)|(q[:,1]<<21)|q[:,2]
def voxel(x,size):return x[np.unique(gridkeys(x,size),return_index=True)[1]]
def header(path):
    meta={}
    with Path(path).open('rb') as f:
        lines=[]
        for _ in range(100):
            line=f.readline();lines.append(line);s=line.decode('ascii').strip()
            if s and not s.startswith('#'):parts=s.split();meta[parts[0]]=parts[1:]
            if s.startswith('DATA '):break
        else:raise ValueError('PCD header missing DATA')
        meta['offset']=f.tell();meta['raw_header']=b''.join(lines)
    return meta
def chunks(path,h):
    count=int(h['POINTS'][0]);seen=0
    if h['DATA']==['binary']:
        fields=[]
        for name,size,kind,n in zip(h['FIELDS'],h['SIZE'],h['TYPE'],h.get('COUNT',['1']*len(h['FIELDS']))):
            dt={'F':'f','I':'i','U':'u'}[kind]+size;fields.append((name,'<'+dt) if int(n)==1 else (name,'<'+dt,(int(n),)))
        dtype=np.dtype(fields)
        if Path(path).stat().st_size!=h['offset']+count*dtype.itemsize:raise ValueError('Truncated binary PCD')
        mm=np.memmap(path,dtype=dtype,mode='r',offset=h['offset'],shape=(count,))
        for start in range(0,count,1000000):
            a=mm[start:start+1000000];x=np.column_stack([a[k] for k in ['x','y','z']]);seen+=len(x);yield x
    elif h['DATA']==['ascii']:
        width=sum(map(int,h.get('COUNT',['1']*len(h['FIELDS']))));indices=[h['FIELDS'].index(k) for k in ['x','y','z']]
        with Path(path).open('rb') as f:
            f.seek(h['offset']);tail=b''
            while True:
                b=f.read(32*1024*1024)
                if not b:
                    if tail.strip():
                        a=np.fromstring(tail.decode('ascii'),sep=' ',dtype=np.float32).reshape(-1,width);seen+=len(a);yield a[:,indices]
                    break
                b=tail+b;last=b.rfind(b'\n');tail=b[last+1:];a=np.fromstring(b[:last].decode('ascii'),sep=' ',dtype=np.float32)
                if a.size%width:raise ValueError('Malformed ASCII PCD row')
                a=a.reshape(-1,width);seen+=len(a);yield a[:,indices]
    else:raise ValueError('Unsupported PCD encoding')
    if seen!=count:raise ValueError(f'Point count mismatch {seen} != {count}')
def init():
    OUT.mkdir(exist_ok=False);sources=[]
    for p in sorted((ROOT/'lidar data/hk').rglob('map_dense.pcd')):
        sources.append(dict(id='recent_'+p.parent.name,survey='recent_'+p.parent.name,coarse_source=str(p),dense_source=str(p),role='full_dense_map',registered_with_other_surveys=False))
    assets=list(csv.DictReader((ROOT/'hk-tree-vmms/outputs/coordinates/pointcloud_assets.csv').open(encoding='utf-8-sig')))
    for r in assets:
        p=VMMS/r['source_relpath'];name=p.name
        if name.startswith('Opt_DenseMap_Riegl_C_'):
            sources.append(dict(id='hewentian_'+p.stem.split('_')[-1],survey='2023-10-27_hewentian',coarse_source=str(p),dense_source=str(p),role='full_optimized_dense_map',registered_with_other_surveys=True))
        elif name.startswith('pointcloud_DS_'):
            route=p.parts[len(VMMS.parts)];prefix='jianshazui' if route.startswith('20250210') else 'stubbs';dense=p.with_name(name.replace('_DS_','_'))
            sources.append(dict(id=prefix+'_'+p.stem.split('_')[-1],survey=route,coarse_source=str(p),dense_source=str(dense),role='full_scene_existing_downsampled_map_then_dense_refinement',registered_with_other_surveys=True))
    for r in sources:
        p=Path(r['coarse_source']);h=header(p);r.update(declared_points=int(h['POINTS'][0]),source_bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns)
    write(OUT/'sources.json',sources);write(OUT/'config.json',dict(version=1,voxel_m=.2,ground_grid_m=1.,ground_cell_quantile=.1,ground_opening_m=7,vegetation_geometry='local 24-neighbour PCA; scattering > .035, planarity < .8; not a trained semantic model',candidate_policy='trunk-supported seeds plus canopy height local maxima; keep unconfirmed objects, do not assign species',inputs_use_human_labels=False,minimum_candidate_points=60,production_unchanged=True,scope='7 new dense maps; 5 optimized HoManTin dense maps; all 12 TST and 4 Stubbs DS maps for scene-wide discovery; dense ROI refinement recorded separately',limitations=['Geometric proposals can include shrubs, poles or buildings; not verified tree count','No unverified cross-session registration; dedup only within the same known coordinate frame','No trained ground/vegetation checkpoint available at the recorded F: path','No independent instance segmentation accuracy claim']))
    print('initialized',len(sources),'maps',sum(r['declared_points'] for r in sources),'input points',flush=True)
def preprocess(source):
    folder=OUT/'maps'/source['id'];folder.mkdir(parents=True,exist_ok=True)
    if (folder/'coarse.json').exists():return
    if (folder/'coarse.npy').exists():raise ValueError('Uncommitted coarse output exists; inspect before retry')
    p=Path(source['coarse_source']);h=header(p);parts=[];seen=0;finite=0;t=time.time()
    for x in chunks(p,h):
        seen+=len(x);x=x[np.isfinite(x).all(1)];finite+=len(x)
        if len(x):parts.append(voxel(x,.2))
        if len(parts)>=12:parts=[voxel(np.concatenate(parts),.2)]
        if seen%5000000<len(x):print(source['id'],'scan',seen,'/',source['declared_points'],'seconds',round(time.time()-t),flush=True)
    x=voxel(np.concatenate(parts),.2);np.save(folder/'coarse.npy',x.astype(np.float32));st=p.stat();assert st.st_size==source['source_bytes'] and st.st_mtime_ns==source['mtime_ns']
    meta=dict(source=source,scanned_points=seen,finite_points=finite,voxel_points=len(x),bounds=[x.min(0).tolist(),x.max(0).tolist()],seconds=time.time()-t,species_labels_used=False,source_size_and_mtime_unchanged=True)
    write(folder/'coarse.json',meta);print(source['id'],'COARSE DONE',len(x),flush=True)
def ground_and_pca(x):
    xy=np.floor(x[:,:2]).astype(int);lo=xy.min(0);ij=xy-lo;shape=ij.max(0)+1
    if np.prod(shape)>16000000:raise ValueError('Ground raster too large; tile before processing')
    key=ij[:,0]*shape[1]+ij[:,1];order=np.lexsort((x[:,2],key));u,start,count=np.unique(key[order],return_index=True,return_counts=True);z=x[order[start+np.floor((count-1)*.1).astype(int)],2]
    grid=np.full(shape,np.nan,np.float32);grid.flat[u]=z;missing=~np.isfinite(grid);nearest=ndimage.distance_transform_edt(missing,return_distances=False,return_indices=True);filled=grid[tuple(nearest)];surface=ndimage.grey_opening(filled,size=(7,7));surface=ndimage.median_filter(surface,size=3);gz=surface[ij[:,0],ij[:,1]];height=x[:,2]-gz
    support=ndimage.distance_transform_edt(missing)[ij[:,0],ij[:,1]]
    valid=np.flatnonzero((height>.45)&(height<45));q=x[valid];scatter=np.zeros(len(x),np.float32);planar=np.ones(len(x),np.float32);vertical=np.zeros(len(x),np.float32);linear=np.zeros(len(x),np.float32)
    tree=cKDTree(x)
    for start in range(0,len(q),40000):
        ids=valid[start:start+40000];dist,near=tree.query(x[ids],k=24,workers=4);a=x[near];a-=a.mean(1,keepdims=True);cov=np.einsum('nki,nkj->nij',a,a)/24;ev,vec=np.linalg.eigh(cov);ev=np.maximum(ev,1e-10);sc=ev[:,0]/ev[:,2];pl=(ev[:,1]-ev[:,0])/ev[:,2];ln=(ev[:,2]-ev[:,1])/ev[:,2];ok=dist[:,-1]<3
        scatter[ids]=np.where(ok,sc,0);planar[ids]=pl;linear[ids]=ln;vertical[ids]=abs(vec[:,2,2])
    return gz,height,scatter,planar,linear,vertical,support
def extract(source):
    from sklearn.cluster import DBSCAN
    folder=OUT/'maps'/source['id'];result=folder/'candidates.json'
    if result.exists():return
    if (folder/'instances').exists():raise ValueError('Uncommitted instances; do not overwrite')
    x=np.load(folder/'coarse.npy');gz,h,sc,pl,ln,vert,support=ground_and_pca(x);veg=(h>2)&(h<40)&(sc>.035)&(pl<.8);trunk=(h>.5)&(h<4.5)&(vert>.65)&(ln>.5)
    seeds=[];tids=np.flatnonzero(trunk);vegetation=x[veg];vtree=cKDTree(vegetation[:,:2]) if len(vegetation) else None
    if len(tids):
        # Bound redundant XY points before clustering vertical traces.
        labs=DBSCAN(eps=.45,min_samples=5,n_jobs=4).fit_predict(x[tids,:2])
        for lab in sorted(set(labs)-{-1}):
            ids=tids[labs==lab];q=x[ids];axis=np.median(q[:,:2],axis=0);span=np.ptp(q,axis=0)
            if span[2]<.8 or span[:2].max()>2.5:continue
            vi=vtree.query_ball_point(axis,4) if vtree is not None else []
            if len(vi)<25:continue
            seeds.append(dict(xy=axis.tolist(),kind='vertical_trace_with_crown',stem_points=len(ids),estimated_ground=float(np.median(gz[ids]))))
    # Independent crown-first discovery covers trees whose trunks were not observed.
    if len(vegetation):
        vxy=np.floor(vegetation[:,:2]/.75).astype(int);vmin=vxy.min(0);ij=vxy-vmin;shape=ij.max(0)+1;chm=np.zeros(shape,np.float32);density=np.zeros(shape,np.int32);np.maximum.at(chm,(ij[:,0],ij[:,1]),h[veg]);np.add.at(density,(ij[:,0],ij[:,1]),1)
        smooth=ndimage.gaussian_filter(chm,1);local=ndimage.maximum_filter(smooth,size=7);peaks=np.argwhere((smooth==local)&(smooth>2.5)&(ndimage.maximum_filter(density,size=3)>=3));peaks=sorted(peaks,key=lambda k:-smooth[tuple(k)])
        for ijp in peaks:
            axis=(ijp+vmin+.5)*.75
            if seeds and np.min(np.linalg.norm(np.array([s['xy'] for s in seeds])-axis,axis=1))<3:continue
            vi=vtree.query_ball_point(axis,3)
            if len(vi)<25:continue
            seeds.append(dict(xy=axis.tolist(),kind='crown_only_proposal',stem_points=0,estimated_ground=float(np.median(vegetation[vi,2])-np.median(h[veg][vi]))))
    seeds.sort(key=lambda s:tuple(s['xy']));out=[];(folder/'instances').mkdir();proposal_mask=veg|((h>.45)&(h<40)&(pl<.9)&((sc>.015)|(vert>.6)))
    if seeds:
        seedxy=np.array([s['xy'] for s in seeds]);distance,assignment=cKDTree(seedxy).query(x[:,:2],workers=4)
        for i,seed in enumerate(seeds):
            ids=np.flatnonzero((assignment==i)&(distance<8)&proposal_mask);q=x[ids]
            if len(q)<60:continue
            extent=np.ptp(q,axis=0);scattering=float(np.mean(veg[ids]));nearbase=(h[ids]<2.5)&(distance[ids]<1.)
            if extent[2]<2 or extent[:2].max()<1 or scattering<.12:continue
            # Candidate, not confirmed tree: retain quality evidence rather than certify geometry.
            key=source['id']+'_'+str(len(out)+1).zfill(4);d=folder/'instances'/key;d.mkdir();sid=sample_ids(q,2048);center=(q.min(0)+q.max(0))/2;scale=max(np.linalg.norm(q-center,axis=1).max(),1e-8)
            np.savez_compressed(d/'cloud.npz',xyz=q,coarse_source_indices=ids);np.savez_compressed(d/'point.npz',points_xyz=((q[sid]-center)/scale).astype(np.float32),metric_xyz=q[sid],centroid_xyz=center,scale=scale,source_indices=ids[sid]);orthographic(q,sample_ids(q,6000),d/'views.jpg',float(np.quantile(q[:,2],.25)))
            row=dict(id=key,survey=source['survey'],map_id=source['id'],axis_xy=seed['xy'],seed_kind=seed['kind'],coarse_points=len(q),sample_points=len(sid),extent_m=extent.tolist(),observed_z_span_m=float(extent[2]),estimated_ground_z=seed['estimated_ground'],base_support_points=int(nearbase.sum()),vegetation_shape_fraction=scattering,status='unverified_geometric_tree_candidate',species=None,training_eligible=False,whole_tree_verified=False,geometry_verified=False,point=str(d/'point.npz'),cloud=str(d/'cloud.npz'),views=str(d/'views.jpg'),dense_refinement_required=source['coarse_source']!=source['dense_source'])
            out.append(row)
    write(result,out);write(folder/'segmentation_summary.json',dict(map_id=source['id'],input_points=len(x),ground_band_points=int((h<=.45).sum()),vegetation_shape_points=int(veg.sum()),seed_count=len(seeds),candidate_count=len(out),algorithm='geometry-only proposals, not learned semantic segmentation',training_eligible=0));print(source['id'],'EXTRACT DONE',len(out),'unverified candidates',flush=True)
def run(ids=None,stage='all'):
    for source in read(OUT/'sources.json'):
        if ids and source['id'] not in ids:continue
        try:
            if stage in ['all','coarse']:preprocess(source)
            if stage in ['all','extract']:extract(source)
        except Exception as e:
            folder=OUT/'maps'/source['id'];folder.mkdir(parents=True,exist_ok=True);write(folder/'error.json',dict(error=str(e),stage=stage));print(source['id'],'ERROR',repr(e),flush=True)
def summarize():
    sources=read(OUT/'sources.json');allrows=[];coverage=[]
    for s in sources:
        d=OUT/'maps'/s['id'];done=(d/'candidates.json').exists();rows=read(d/'candidates.json') if done else [];allrows+=rows;coverage.append(dict(id=s['id'],role=s['role'],coarse_complete=(d/'coarse.json').exists(),extraction_complete=done,candidates=len(rows),error=read(d/'error.json') if (d/'error.json').exists() else None))
    # Conservative cross-map proximity blocks only in a common survey coordinate system.
    parent=list(range(len(allrows)))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for survey in sorted({r['survey'] for r in allrows}):
        ids=[i for i,r in enumerate(allrows) if r['survey']==survey];tree=cKDTree(np.array([allrows[i]['axis_xy'] for i in ids]))
        for a,b in tree.query_pairs(2.):
            i,j=ids[a],ids[b]
            if allrows[i]['map_id']!=allrows[j]['map_id']:parent[find(i)]=find(j)
    groups={}
    for i,r in enumerate(allrows):r['dedup_group']='candidate_group_'+str(find(i));groups.setdefault(r['dedup_group'],[]).append(r)
    for rs in groups.values():
        best=max(rs,key=lambda r:(r['base_support_points']>=5,r['coarse_points']))
        for r in rs:r['representative']=r is best
    summary=dict(maps_planned=len(sources),maps_coarse_complete=sum(c['coarse_complete'] for c in coverage),maps_extraction_complete=sum(c['extraction_complete'] for c in coverage),raw_candidate_files=len(allrows),provisional_spatial_groups=len(groups),verified_independent_trees=None,training_eligible=0,by_survey=dict(Counter(r['survey'] for r in allrows if r['representative'])),coverage=coverage,warning='Geometric candidates and proximity groups, NOT confirmed tree count, tree species labels or extraction accuracy. No model fitting in this run.')
    write(OUT/'candidates.json',allrows);write(OUT/'summary.json',summary);print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['init','all','coarse','extract','summarize']);p.add_argument('--ids',nargs='*');a=p.parse_args()
    if a.stage=='init':init()
    elif a.stage=='summarize':summarize()
    else:run(a.ids,a.stage)
