"""Dense, read-only extraction of the three manually labelled palms in frame 1474."""
from pathlib import Path
import argparse
import csv
import json
import zipfile
from collections import Counter
import numpy as np
import cv2
from PIL import Image,ImageDraw
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
from build_hk_lidar_pairs import PROJECT, VMMS, read, write, project, voxel, render_preview

KEY='jianshazui_pano_2__001474'
OUT=PROJECT/'derived/foxtail_complete_scene_20260907'
SOURCE=VMMS/'20250210-jianshazui/pointcloud/mapping/pointcloud_5.pcd'

def load_scene(row):
    cache=OUT/'dense_scene.npz'
    if cache.exists():
        with np.load(cache) as f:return f['xyz'],f['rgb']
    with SOURCE.open('rb') as f:
        header=[]
        while True:
            line=f.readline();header.append(line)
            if not line:raise ValueError('Incomplete header')
            if line.startswith(b'DATA '):break
        offset=f.tell()
    h=b''.join(header).decode('ascii');assert 'FIELDS x y z rgb' in h and 'DATA binary' in h
    count=int(next(l.split()[1] for l in h.splitlines() if l.startswith('POINTS ')))
    assert SOURCE.stat().st_size==offset+count*16,'Truncated source'
    data=np.memmap(SOURCE,dtype=[('x','<f4'),('y','<f4'),('z','<f4'),('rgb','<u4')],mode='r',offset=offset,shape=(count,))
    pos=np.array([float(row['local_'+k]) for k in 'xyz']); parts=[];colors=[]
    for start in range(0,count,2000000):
        chunk=data[start:start+2000000]
        keep=((chunk['x']-pos[0])**2+(chunk['y']-pos[1])**2<35**2)&(chunk['z']>-5)&(chunk['z']<27)
        p=np.column_stack([chunk[k][keep] for k in 'xyz']);rgb=chunk['rgb'][keep]
        ids=voxel(p,.04);parts.append(p[ids]);colors.append(rgb[ids])
        if start%20000000==0:print('dense source scanned',start,'/',count,flush=True)
    xyz=np.concatenate(parts);rgb=np.concatenate(colors);ids=voxel(xyz,.04)
    xyz=xyz[ids];rgb=rgb[ids];np.savez_compressed(cache,xyz=xyz,rgb=rgb)
    write(OUT/'source.json',{'source':str(SOURCE),'source_points':count,'integrity':'exact binary size verified','scene_unique_voxels_4cm':len(xyz),'source_read_only':True})
    return xyz,rgb

def farthest_sample(points,n):
    """Keep spatial extremes rather than duplicating points to inflate density."""
    n=min(n,len(points));chosen=np.empty(n,int);dist=np.full(len(points),np.inf)
    current=int(np.argmax(points[:,2]))
    for i in range(n):
        chosen[i]=current;dist=np.minimum(dist,np.sum((points-points[current])**2,axis=1));current=int(np.argmax(dist))
    return chosen

def export_clean(folder,raw,seed,all_axes,axis_index,im,poly,row,info):
    center=all_axes[axis_index]
    # Trunk center is estimated in a clear horizontal slice, not from a biased crown.
    dxy=np.linalg.norm(raw[:,:2]-center,axis=1)
    stem_slice=raw[(dxy<.7)&(raw[:,2]>1.5)&(raw[:,2]<2)]
    if len(stem_slice)<20:raise ValueError('No supported trunk slice')
    axis=np.median(stem_slice[:,:2],axis=0);dxy=np.linalg.norm(raw[:,:2]-axis,axis=1)
    near=raw[dxy<.3];base=float(np.quantile(near[:,2],.005))+.05
    crown_floor=float(np.quantile(seed[:,2],.015))-.25
    own=cKDTree(all_axes).query(raw[:,:2])[1]==axis_index
    crown=(dxy<3.6)&(raw[:,2]>crown_floor)&own
    # Grow from image-supported foliage; independent fine leaves need not form
    # one dense DBSCAN cluster. This preserves real leaf returns, not synthetic fill.
    seed_distance=cKDTree(seed).query(raw,k=1)[0]
    crown &= seed_distance<.65
    trunk=(dxy<.35)&(raw[:,2]>=base)&(raw[:,2]<=np.quantile(seed[:,2],.80))
    points=raw[crown|trunk];points=points[voxel(points,.06)]
    distances=cKDTree(points).query(points,k=9)[0][:,-1]
    points=points[distances<.45]
    trunk_ids=np.where(points[:,2]<crown_floor)[0];crown_ids=np.where(points[:,2]>=crown_floor)[0]
    if min(len(trunk_ids),len(crown_ids))<128:raise ValueError('Insufficient independent trunk/crown points')
    ids=np.r_[trunk_ids[farthest_sample(points[trunk_ids],512)],crown_ids[farthest_sample(points[crown_ids],1536)]]
    sampled=points[ids];centroid=points.mean(0);scale=np.linalg.norm(points-centroid,axis=1).max()
    np.savez_compressed(folder/'cleaned_local.npz',xyz=points)
    np.savez_compressed(folder/'point.npz',points_xyz=((sampled-centroid)/scale).astype(np.float32),centroid_xyz=centroid,scale=scale,source_point_count=len(points),sampled_unique_point_count=len(ids))
    np.savetxt(folder/'point_local.xyz',sampled,fmt='%.5f')
    # Keep georeferenced and unit-sphere versions unambiguous.
    header='ply\nformat ascii 1.0\nelement vertex '+str(len(points))+'\nproperty float x\nproperty float y\nproperty float z\nend_header\n'
    with (folder/'cleaned_local.ply').open('w') as f:
        f.write(header);np.savetxt(f,points,fmt='%.5f')
    render_preview(im,poly,project(points,row)[0],points,folder,info['species'])
    left,top=poly.min(0);right,bottom=poly.max(0)
    im.crop((int(left*im.width),int(top*im.height),int(right*im.width),int(bottom*im.height))).save(folder/'image.jpg',quality=94)
    span=np.ptp(points,axis=0);edges=np.arange(base,crown_floor+.25,.25)
    hist=np.histogram(points[points[:,2]<crown_floor,2],edges)[0]
    sampled_hist=np.histogram(sampled[sampled[:,2]<crown_floor,2],edges)[0]
    info.update(status='recovered_dense_candidate_needs_review',axis_local_xy=axis.tolist(),base_local_z=base,crown_floor_local_z=crown_floor,clean_points=len(points),trunk_points=len(trunk_ids),crown_points=len(crown_ids),sampled_unique_points=len(ids),extent_m=span.tolist(),trunk_vertical_bin_occupancy=float((hist>0).mean()),sampled_trunk_vertical_bin_occupancy=float((sampled_hist>0).mean()),sampling='6cm instance voxel; farthest-point sampling: 512 trunk + 1536 crown; no duplicated or synthesized points',source_lidar=str(SOURCE),source_frame=KEY,completeness_caution='Continuous stem and crown recovered; not a claim of full 360-degree surface coverage. Crown tips/occlusion and fine extrinsics remain subject to visual review.',training_eligible=False)
    write(folder/'record.json',info)
    return points,sampled

def main():
    OUT.mkdir(exist_ok=True)
    row=next(r for r in csv.DictReader((PROJECT/'outputs/coordinates/frame_coordinates.csv').open(encoding='utf-8-sig')) if r['stream_id']+'__'+r['frame_id']==KEY)
    record=read(PROJECT/'derived/jianshazui_frame_labeler/training_exports/effective_yolo_20260901_001944/records/train'/f'{KEY}.json')
    xyz,rgb=load_scene(row);uv,dist=project(xyz,row)
    with Image.open(VMMS/row['source_image_relpath']) as src:im=src.convert('RGB');im.thumbnail((4096,2048))
    results=[]
    for i,label in enumerate(record['labels']):
        folder=OUT/f'palm_{i+1:02d}';folder.mkdir(exist_ok=True)
        poly=np.array(label['points']);mask=np.zeros((2048,4096),np.uint8);cv2.fillPoly(mask,[np.rint(poly*[4095,2047]).astype(np.int32)],1)
        # Small projection tolerance for seed discovery only; not whole-scene inclusion.
        mask=cv2.dilate(mask,np.ones((9,9),np.uint8))
        inside=mask[np.clip((uv[:,1]*2047).astype(int),0,2047),np.clip((uv[:,0]*4095).astype(int),0,4095)]>0
        crown_y=poly[:,1].min()+.30*np.ptp(poly[:,1]);seeds=inside&(uv[:,1]<crown_y)&(dist<24)&(dist>2)
        crown=xyz[seeds];crown=crown[voxel(crown,.12)]
        labs=DBSCAN(eps=.5,min_samples=5).fit_predict(crown);options=[]
        for lab in set(labs):
            if lab<0:continue
            p=crown[labs==lab]
            if len(p)<30:continue
            eig=np.linalg.eigvalsh(np.cov(p.T));fraction=eig[0]/max(eig.sum(),1e-9)
            if fraction<.015:continue
            options.append((len(p),p))
        if not options:results.append({'id':folder.name,'reason':'no_foreground_crown'});continue
        seed=max(options,key=lambda x:x[0])[1];center=np.median(seed[:,:2],axis=0)
        # Inspect a generously bounded raw column before deciding trunk/crown cleaning.
        region=(np.linalg.norm(xyz[:,:2]-center,axis=1)<5)&(xyz[:,2]<seed[:,2].max()+1)
        raw=xyz[region];col=rgb[region];np.savez_compressed(folder/'raw_column.npz',xyz=raw,rgb=col)
        render_preview(im,poly,project(raw,row)[0],raw,folder,label['species'])
        np.savez_compressed(folder/'crown_seed.npz',xyz=seed)
        info={'id':folder.name,'source_label_id':label['label_id'],'species':label['species'],'seed_points':len(seed),'seed_center_xy':center.tolist(),'seed_extent':np.ptp(seed,axis=0).tolist(),'seed_z_range':[float(seed[:,2].min()),float(seed[:,2].max())],'raw_column_points':len(raw),'status':'raw_column_for_geometric_diagnosis'}
        results.append(info);write(folder/'record.json',info)
    # Include other observed stem centers in the row when assigning crown ownership.
    slice_points=xyz[(xyz[:,0]>-883)&(xyz[:,0]<-880)&(xyz[:,2]>1.5)&(xyz[:,2]<2)]
    labs=DBSCAN(eps=.18,min_samples=15).fit_predict(slice_points[:,:2]);axes=[]
    for lab in set(labs):
        if lab<0:continue
        p=slice_points[labs==lab]
        if np.ptp(p[:,:2],axis=0).max()<.7 and len(p)>40:axes.append(np.median(p[:,:2],axis=0))
    axes=np.array(axes);panels=[];recovered=[]
    for info,label in zip(results,record['labels']):
        if 'seed_center_xy' not in info:continue
        folder=OUT/info['id'];raw=np.load(folder/'raw_column.npz')['xyz'];seed=np.load(folder/'crown_seed.npz')['xyz']
        axis_index=int(np.argmin(np.linalg.norm(axes-info['seed_center_xy'],axis=1)))
        clean,sampled=export_clean(folder,raw,seed,axes,axis_index,im,np.array(label['points']),row,info)
        recovered.append((info,clean,sampled));panels.append(Image.open(folder/'point_views.jpg').copy())
    sheet=Image.new('RGB',(960,430*len(panels)),'white');draw=ImageDraw.Draw(sheet)
    for i,(panel,(info,clean,sampled)) in enumerate(zip(panels,recovered)):
        draw.text((15,i*430+8),f"{info['id']} | {len(clean)} real points | {len(sampled)} unique sampled | height {np.ptp(clean[:,2]):.2f}m",fill='black');sheet.paste(panel,(0,i*430+35))
    sheet.save(OUT/'three_palms_comparison.jpg',quality=95)
    overlay=im.copy();draw=ImageDraw.Draw(overlay)
    for color,(info,clean,sampled) in zip(['#ff6633','#33ffff','#cc55ff'],recovered):
        for u,v in project(sampled,row)[0]:
            x,y=int(u*im.width),int(v*im.height);draw.ellipse((x-2,y-2,x+2,y+2),fill=color)
    overlay.thumbnail((1600,800));overlay.save(OUT/'three_palms_projection.jpg',quality=95)
    write(OUT/'manifest.json',results)
    checks=[];assigned=[]
    for info,clean,sampled in recovered:
        checks.append({'id':info['id'],'finite':bool(np.isfinite(sampled).all()),'sample_count':len(sampled),'unique_sample_count':len(np.unique(sampled,axis=0)),'trunk_bins_occupied':info['sampled_trunk_vertical_bin_occupancy']})
        assigned.append(set(map(tuple,clean)))
    overlaps=[len(assigned[i]&assigned[j]) for i in range(len(assigned)) for j in range(i+1,len(assigned))]
    assert all(c['finite'] and c['unique_sample_count']==2048 and c['trunk_bins_occupied']==1 for c in checks)
    assert not any(overlaps)
    write(OUT/'verification.json',{'samples':checks,'shared_points_between_trees':overlaps,'verdict':'structural checks passed; not human acceptance or classification accuracy'})
    # Retire only our known-bad derived pilot, preserving an audit copy and files.
    pilot=PROJECT/'derived/hk_lidar_pairs_pilot_20260907';old_id=KEY+'__79dca14b84'
    for name in ('manifest.json','instances/'+old_id+'/record.json'):
        path=pilot/name
        if not path.exists():continue
        backup=path.with_name(path.stem+'.before_dense_recovery.json')
        old=read(path)
        if not backup.exists():write(backup,old)
        entries=old if isinstance(old,list) else [old]
        for entry in entries:
            if entry.get('id')==old_id:entry.update(status='superseded_bad_background_cluster',geometry_pass=False,training_eligible=False,replaced_by=str(OUT/'palm_03/record.json'),rejection_reason='Largest cluster selected background surface rather than foreground palm; dense source recovery replaces this candidate.')
        write(path,old)
    summary_path=pilot/'summary.json'
    if summary_path.exists():
        summary=read(summary_path);entries=read(pilot/'manifest.json')
        backup=pilot/'summary.before_dense_recovery.json'
        if not backup.exists():write(backup,summary)
        summary.update(status_counts=dict(Counter(e['status'] for e in entries)),geometry_pass=sum(bool(e.get('geometry_pass')) for e in entries),dense_recovery_notice='The JTS frame1474 background candidate was invalidated; see foxtail_complete_scene_20260907 for three recovered dense palms.')
        write(summary_path,summary)
    package_files=['manifest.json','source.json','verification.json','三棵狐尾椰子点云修复说明.md','three_palms_comparison.jpg','three_palms_projection.jpg']
    with zipfile.ZipFile(OUT/'狐尾椰子_三棵双源样本.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in package_files:
            if (OUT/name).exists():z.write(OUT/name,name)
        for info,_,_ in recovered:
            for name in ('image.jpg','point.npz','point_local.xyz','cleaned_local.npz','cleaned_local.ply','record.json','projection.jpg','point_views.jpg'):
                p=OUT/info['id']/name;z.write(p,str(p.relative_to(OUT)))
    print(json.dumps({'output':str(OUT),'recovered':[{k:r[k] for k in ('id','clean_points','sampled_unique_points','extent_m')} for r in results],'verification':checks},indent=2,ensure_ascii=False))

if __name__=='__main__':main()
