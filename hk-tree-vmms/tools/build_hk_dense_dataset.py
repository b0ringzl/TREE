"""Auditable three-class VMMS pairing: image seed -> original map -> single tree.

Derived outputs only. Image labels supervise candidate pairs, never certify 3D
alignment. Review and training eligibility are separate from geometric checks.
"""
import argparse,csv,hashlib,json,re,time
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
from build_hk_lidar_pairs import PROJECT,VMMS,TARGETS,Maps,manual_records,species,read,write,voxel,project,render_preview
from recover_palm_scene import farthest_sample

OUT=PROJECT/'derived/hk_dualsource_three_species_20260907'
CACHE=PROJECT/'derived/hk_lidar_pairs_pilot_20260907'

def coverage_sample(points,count):
    # Bound FPS work for dense crowns. Points are spatially voxel-ordered; retain
    # axis extremes plus an evenly distributed deterministic pool of real points.
    if len(points)<=6000:return farthest_sample(points,count)
    pool=np.unique(np.r_[np.linspace(0,len(points)-1,6000,dtype=int),np.argmin(points,axis=0),np.argmax(points,axis=0)])
    return pool[farthest_sample(points[pool],count)]

def coordinates():
    return {r['stream_id']+'__'+r['frame_id']:r for r in csv.DictReader((PROJECT/'outputs/coordinates/frame_coordinates.csv').open(encoding='utf-8-sig'))}

def polygon_inside(uv,poly,dilation=7):
    mask=np.zeros((1024,2048),np.uint8);cv2.fillPoly(mask,[np.rint(poly*[2047,1023]).astype(np.int32)],1)
    if dilation:mask=cv2.dilate(mask,np.ones((dilation,dilation),np.uint8))
    return mask[np.clip((uv[:,1]*1023).astype(int),0,1023),np.clip((uv[:,0]*2047).astype(int),0,2047)]>0

def crown_seed(local,row,poly,palm):
    uv,dist=project(local,row)
    inside=polygon_inside(uv,poly)
    top=poly[:,1].min()+(.42 if palm else .7)*np.ptp(poly[:,1])
    p=local[inside&(uv[:,1]<top)&(dist>2)&(dist<40)]
    if len(p)<25:return None,'too_few_crown_projection_points'
    p=p[voxel(p,.2)]
    labs=DBSCAN(eps=.85,min_samples=4).fit_predict(p)
    pos=np.array([float(row['local_'+k]) for k in 'xyz']);options=[]
    for lab in set(labs):
        if lab<0:continue
        q=p[labs==lab]
        if len(q)<20:continue
        span=np.ptp(q,axis=0);eig=np.linalg.eigvalsh(np.cov(q.T));fraction=eig[0]/max(eig.sum(),1e-9)
        if fraction<.018 or span[2]<.8 or span[:2].max()>(12 if palm else 20):continue
        options.append((float(np.linalg.norm(np.median(q,axis=0)-pos)),q))
    if not options:return None,'no_nonplanar_foreground_crown'
    return min(options,key=lambda x:x[0])[1],None

def discover(out):
    rows=coordinates();records=manual_records();maps=Maps(CACHE);items=[];last=None
    for key,r in sorted(records.items()):
        if key not in rows:continue
        labels=[l for l in r.get('labels',[]) if species(l.get('species','')) in TARGETS and len(l.get('points',[]))>=3]
        if not labels:continue
        row=rows[key]
        try:local,sources=maps.nearby(row)
        except Exception as e:
            local=None;error=str(e)
        for label in labels:
            ident=key+'__'+hashlib.sha256(str(label.get('label_id')).encode()).hexdigest()[:10]
            item={'id':ident,'frame_key':key,'route':r['route'],'species':species(label['species']),'label_id':label.get('label_id'),'polygon':label['points'],'annotation_provenance':r['annotation_provenance'],'status':'rejected_seed','training_eligible':False}
            if local is None:item['reason']=error
            else:
                q,reason=crown_seed(local,row,np.array(label['points']),item['species']!=TARGETS[0])
                if reason:item['reason']=reason
                else:
                    item.update(status='seed_found',seed_center=np.median(q,axis=0).tolist(),seed_points=q.tolist(),seed_source_lidar=sources,survey=Path(row['source_image_relpath']).parts[0])
            items.append(item)
        if len(items)//50!= (len(items)-len(labels))//50:
            print('seeds',len(items),dict(Counter(x['status'] for x in items)),flush=True)
            write(out/'seeds.progress.json',items)
    write(out/'seeds.json',items)
    print('seed discovery',len(items),dict(Counter(x['status'] for x in items)),flush=True)
    return items

def choose_maps(items,out):
    # Rank maps by actual local downsampled support, not broad overlapping bounds.
    inventory=list(csv.DictReader((PROJECT/'outputs/coordinates/pointcloud_assets.csv').open(encoding='utf-8-sig')))
    for survey in sorted(set(x['survey'] for x in items if x['status']=='seed_found')):
        selected=[x for x in items if x.get('survey')==survey and x['status']=='seed_found']
        if survey.startswith('2023-10-27'):
            for x in selected:x['dense_source']='frame_color_clouds'
            continue
        ds=[r for r in inventory if r['source_relpath'].startswith(survey+'/') and '/pointcloud_DS_' in r['source_relpath']]
        centers=np.array([x['seed_center'][:2] for x in selected]);scores=[];names=[]
        from build_hk_lidar_pairs import load_ascii
        for r in ds:
            p=VMMS/r['source_relpath'];cloud=load_ascii(p);cloud=cloud[np.isfinite(cloud).all(1)]
            counts=cKDTree(cloud[:,:2]).query_ball_point(centers,5,return_length=True)
            scores.append(counts);names.append(str(p.with_name(p.name.replace('pointcloud_DS_','pointcloud_'))))
        for x,k in zip(selected,np.argmax(np.array(scores),axis=0)):x['dense_source']=names[int(k)]
    write(out/'seeds.json',items)
    print('dense map assignments',dict(Counter(Path(x.get('dense_source','')).name for x in items if x['status']=='seed_found')),flush=True)

def pcd_chunks(path):
    with path.open('rb') as f:
        header=[]
        while True:
            line=f.readline()
            if not line:raise ValueError('incomplete PCD header')
            header.append(line)
            if line.startswith(b'DATA '):break
        offset=f.tell()
    h=b''.join(header).decode('ascii');n=int(next(l.split()[1] for l in h.splitlines() if l.startswith('POINTS ')))
    if 'DATA binary\n' in h or 'DATA binary\r\n' in h:
        fields=next(l.split()[1:] for l in h.splitlines() if l.startswith('FIELDS '))
        sizes=next(l.split()[1:] for l in h.splitlines() if l.startswith('SIZE '))
        if len(fields)!=4 or sizes!=['4']*4:raise ValueError('unsupported binary layout')
        if path.stat().st_size!=offset+n*16:raise ValueError('truncated binary source')
        mm=np.memmap(path,dtype='<f4',mode='r',offset=offset,shape=(n,4))
        for start in range(0,n,1000000):yield np.asarray(mm[start:start+1000000,:3])
    elif 'DATA ascii' in h:
        import pandas as pd
        seen=0
        with path.open('rb') as f:
            f.seek(offset)
            for chunk in pd.read_csv(f,sep=r'\s+',header=None,usecols=[0,1,2],dtype=np.float32,chunksize=1000000):
                seen+=len(chunk);yield chunk.to_numpy()
        if seen!=n:raise ValueError('ASCII point count mismatch')
    else:raise ValueError('unsupported PCD encoding')

def build_dense_cache(items,out):
    cache=out/'dense_cache';cache.mkdir(exist_ok=True)
    sources=sorted(set(x.get('dense_source') for x in items if x.get('dense_source')!='frame_color_clouds' and x.get('dense_source')))
    for source in sources:
        selected=[x for x in items if x.get('dense_source')==source];ident=hashlib.sha256(source.encode()).hexdigest()[:12];path=cache/(ident+'.npz')
        for x in selected:x['dense_cache']=str(path)
        if path.exists():continue
        centers=np.array([x['seed_center'][:2] for x in selected]);tree=cKDTree(centers)
        # Grid filter avoids expensive tree searches for distant source regions.
        cells=np.floor(centers/10).astype(np.int64);allowed=set()
        for a,b in cells:
            for dx in range(-2,3):
                for dy in range(-2,3):allowed.add((a+dx)*100000+(b+dy))
        allowed=np.array(sorted(allowed));parts=[];scanned=0
        for xyz in pcd_chunks(Path(source)):
            scanned+=len(xyz);xyz=xyz[np.isfinite(xyz).all(1)]
            cells=np.floor(xyz[:,:2]/10).astype(np.int64);xyz=xyz[np.isin(cells[:,0]*100000+cells[:,1],allowed)]
            if len(xyz):
                d,ids=tree.query(xyz[:,:2]);keep=(d<13)&(xyz[:,2]<np.array([x['seed_center'][2] for x in selected])[ids]+12)&(xyz[:,2]>np.array([x['seed_center'][2] for x in selected])[ids]-35)
                xyz=xyz[keep];parts.append(xyz[voxel(xyz,.10)])
            if scanned%20000000==0:print(Path(source).parent.parent.parent.name,Path(source).name,scanned,'points scanned',flush=True)
        xyz=np.concatenate(parts) if parts else np.empty((0,3),np.float32);xyz=xyz[voxel(xyz,.10)]
        np.savez_compressed(path,xyz=xyz);write(path.with_suffix('.json'),{'source':source,'scanned_points':scanned,'unique_10cm_points':len(xyz),'seed_ids':[x['id'] for x in selected]})
        print('cached',Path(source).name,len(xyz),flush=True)
        write(out/'seeds.json',items)
    write(out/'seeds.json',items)

def extract_one(local,row,item):
    palm=item.get('morphology', 'broadleaf' if item['species']==TARGETS[0] else 'palm')=='palm';poly=np.array(item['polygon']);seed=np.array(item['seed_points'])
    # Rediscover the crown from dense observations rather than copying the coarse seed.
    uv,dist=project(local,row);inside=polygon_inside(uv,poly)
    near=cKDTree(seed).query(local)[0]<(.85 if palm else 1.1)
    q=local[near&inside];q=q[voxel(q,.12)]
    if len(q)<50:raise ValueError('dense_crown_not_supported')
    center=np.median(q[:,:2],axis=0);dxy=np.linalg.norm(local[:,:2]-center,axis=1)
    column=local[dxy<(2 if palm else 3)]
    # Data-dependent ground/clear-stem slices (no hardcoded coordinates or elevations).
    ground=float(np.quantile(column[:,2],.03));axis_options=[]
    for height in (1.5,2.5,3.5):
        sl=column[(column[:,2]>ground+height)&(column[:,2]<ground+height+.5)]
        if len(sl)<15:continue
        labs=DBSCAN(eps=.25,min_samples=5).fit_predict(sl[:,:2])
        for lab in set(labs):
            if lab<0:continue
            s=sl[labs==lab];span=np.ptp(s[:,:2],axis=0)
            if len(s)<10 or span.max()>(.9 if palm else 2.0):continue
            axis=np.median(s[:,:2],axis=0);axis_options.append((np.linalg.norm(axis-center),axis,span.max()))
    if not axis_options:raise ValueError('no_supported_stem_axis')
    _,axis,width=min(axis_options,key=lambda t:t[0]);dxy=np.linalg.norm(local[:,:2]-axis,axis=1)
    trunk_radius=max(.35,min(1.2,width*.65))
    trunk_col=local[dxy<trunk_radius]
    base=float(np.quantile(trunk_col[:,2],.01))+.1
    crown_floor=float(np.quantile(q[:,2],.02))-.35
    if crown_floor<base+1:raise ValueError('no_separated_trunk_and_crown')
    crown=(cKDTree(q).query(local)[0]<(.55 if palm else .8))&(local[:,2]>crown_floor)&(dxy<(4 if palm else 11))
    trunk=(dxy<trunk_radius)&(local[:,2]>base)&(local[:,2]<np.quantile(q[:,2],.8))
    cleaned=local[crown|trunk];cleaned=cleaned[voxel(cleaned,.10)]
    if len(cleaned)<512:raise ValueError('fewer_than_512_independent_points')
    knn=cKDTree(cleaned).query(cleaned,k=8)[0][:,-1];cleaned=cleaned[knn<.65]
    span=np.ptp(cleaned,axis=0);eig=np.linalg.eigvalsh(np.cov(cleaned.T))
    if eig[0]/eig.sum()<.012:raise ValueError('planar_or_linear_not_tree_crown')
    tids=np.where(cleaned[:,2]<crown_floor)[0];cids=np.where(cleaned[:,2]>=crown_floor)[0]
    edges=np.arange(base,crown_floor+.4,.4);hist=np.histogram(cleaned[tids,2],edges)[0]
    occupancy=float((hist>0).mean())
    if len(tids)<60 or len(cids)<200 or occupancy<(.95 if palm else .8):raise ValueError('incomplete_stem_or_crown')
    # FPS quotas reallocate deficits; never duplicate points.
    count=min(2048,len(cleaned));nt=min(len(tids),count//4);nc=min(len(cids),count-nt);nt=min(len(tids),count-nc)
    ids=np.r_[tids[coverage_sample(cleaned[tids],nt)],cids[coverage_sample(cleaned[cids],nc)]]
    stats={'axis_local_xy':axis.tolist(),'base_z':base,'crown_floor_z':crown_floor,'clean_points':len(cleaned),'trunk_points':len(tids),'crown_points':len(cids),'trunk_bin_occupancy':occupancy,'sample_unique_points':len(ids),'extent_m':span.tolist(),'shape_eigen_min_fraction':float(eig[0]/eig.sum())}
    return cleaned,ids,stats

def extract(items,out,available_only=False):
    rows=coordinates();maps=Maps(CACHE);loaded={};results=[];(out/'instances').mkdir(exist_ok=True);last_frame=None
    items=sorted(items,key=lambda r:(r.get('dense_source',''),r['frame_key'],r['id']))
    for i,item in enumerate(items):
        rec={k:v for k,v in item.items() if k!='seed_points'}
        if item['status']!='seed_found':results.append(rec);continue
        folder=out/'instances'/item['id'];folder.mkdir(exist_ok=True);row=rows[item['frame_key']]
        if (folder/'record.json').exists():
            previous=read(folder/'record.json')
            if previous.get('status') in ('geometry_candidate_pending_review','rejected_geometry','needs_geometry_refinement'):
                results.append(previous);continue
        if item['dense_source']!='frame_color_clouds':
            path=out/'dense_cache'/(hashlib.sha256(item['dense_source'].encode()).hexdigest()[:12]+'.npz')
            if available_only and not path.with_suffix('.json').exists():
                rec['status']='awaiting_dense_cache';results.append(rec);continue
            item['dense_cache']=str(path)
        try:
            if item['dense_source']=='frame_color_clouds':
                if last_frame!=item['frame_key']:local,sources=maps.nearby(row);last_frame=item['frame_key']
                rec['dense_source_paths']=sources
            else:
                path=item['dense_cache']
                if path not in loaded:
                    loaded.clear();cloud=np.load(path)['xyz'];loaded[path]=(cloud,cKDTree(cloud[:,:2]))
                cloud,tree=loaded[path];local=cloud[tree.query_ball_point(item['seed_center'][:2],13)]
            cleaned,ids,stats=extract_one(local,row,item)
            with Image.open(VMMS/row['source_image_relpath']) as src:im=src.convert('RGB');im.thumbnail((4096,2048))
            poly=np.array(item['polygon']);left,top=poly.min(0);right,bottom=poly.max(0)
            im.crop((int(left*im.width),int(top*im.height),int(right*im.width),int(bottom*im.height))).save(folder/'image.jpg',quality=94)
            center=cleaned.mean(0);scale=np.linalg.norm(cleaned-center,axis=1).max()
            np.savez_compressed(folder/'point.npz',points_xyz=((cleaned[ids]-center)/scale).astype(np.float32),centroid_xyz=center,scale=scale,source_point_count=len(cleaned),sampled_unique_point_count=len(ids))
            np.savez_compressed(folder/'cleaned_local.npz',xyz=cleaned)
            render_preview(im,poly,project(cleaned,row)[0],cleaned,folder,item['species'])
            rec.update(status='geometry_candidate_pending_review',quality=stats,point=str(folder/'point.npz'),image=str(folder/'image.jpg'),source_panorama=str(VMMS/row['source_image_relpath']),projection_status='yaw180_empirical; fine_extrinsics_not_verified',training_eligible=False)
        except Exception as e:rec.update(status='rejected_geometry',reason=str(e))
        results.append(rec);write(folder/'record.json',rec)
        if (i+1)%25==0:
            write(out/'manifest.progress.json',results);print('extraction',i+1,'/',len(items),dict(Counter(x['status'] for x in results)),flush=True)
    # Group by stem, rather than image frame, before any training/test split.
    for survey in {r.get('survey') for r in results}:
        valid=[r for r in results if r.get('survey')==survey and r['status']=='geometry_candidate_pending_review']
        if not valid:continue
        groups=DBSCAN(eps=2,min_samples=1).fit_predict(np.array([r['quality']['axis_local_xy'] for r in valid]))
        for r,g in zip(valid,groups):r['tree_group']=survey+'__'+str(g)
        for g in set(groups):
            members=[r for r in valid if r['tree_group']==survey+'__'+str(g)]
            for r in members:r['group_label_conflict']=len({x['species'] for x in members})>1
    write(out/'manifest.json',results)
    for r in results:
        if (out/'instances'/r['id']).exists():write(out/'instances'/r['id']/'record.json',r)
    valid=[r for r in results if r['status']=='geometry_candidate_pending_review']
    summary={'input_annotations':len(results),'processing':any(r['status']=='awaiting_dense_cache' for r in results),'status':dict(Counter(r['status'] for r in results)),'candidate_species':dict(Counter(r['species'] for r in valid)),'candidate_tree_groups':len({r['tree_group'] for r in valid}),'rejection_reasons':dict(Counter(r.get('reason') for r in results if r['status'].startswith('rejected'))),'training_policy':'Human image label transfer is weak 3D supervision, not accepted alignment. Experimental fits may use nonconflicting geometry candidates, never claim accepted-test accuracy. Deployment requires separate validation.'}
    write(out/'summary.json',summary);print(json.dumps(summary,ensure_ascii=False,indent=2))

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['seeds','cache','extract','all'],default='all');p.add_argument('--out',type=Path,default=OUT);p.add_argument('--available-only',action='store_true');a=p.parse_args();a.out.mkdir(exist_ok=True)
    items=discover(a.out) if a.stage in ('seeds','all') else read(a.out/'seeds.json')
    if a.stage in ('seeds','all'):choose_maps(items,a.out)
    if a.stage in ('cache','all'):build_dense_cache(items,a.out)
    if a.stage in ('extract','all'):extract(items,a.out,a.available_only)

if __name__=='__main__':main()
