"""Transfer manual panorama polygons into mapped LiDAR; export auditable candidates.

No source or annotation file is modified. Fine extrinsics are unavailable: every
pair is explicitly a projection candidate, not a human-validated 3D annotation.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter,defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from diagnose_lidar_camera_axes import world_from_flu
from render_pointcloud_panorama_preview import read_binary_xyzi_rgb
VMMS=PROJECT.parent.parent/'vmms'
OUTPUT=PROJECT/'derived/hk_lidar_pairs_20260907'
TARGETS=['榕树','Livistona chinensis 蒲葵','Wodyetia bifurcata 狐尾椰子']
EXPORTS={'hewentian':'hewentian_frame_labeler/training_exports/effective_yolo_20260901_003718','jianshazui':'jianshazui_frame_labeler/training_exports/effective_yolo_20260901_001944','stubbs_road':'stubbs_road_frame_labeler/training_exports/effective_yolo_20260901_020046'}

def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,value): Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
def species(name):
    if str(name).startswith(('Ficus microcarpa','Ficus benjamina')) or name in ('榕树','榕樹','细叶榕','垂叶榕'):return TARGETS[0]
    if str(name).startswith('Livistona chinensis'): return TARGETS[1]
    if str(name).startswith(('Wodyetia bifurcata','Archontophoenix alexandrae')):return TARGETS[2] if str(name).startswith('Wodyetia') else str(name)
    return str(name)

def manual_records():
    records={}
    for route,relative in EXPORTS.items():
        for path in sorted((PROJECT/'derived'/relative/'records').glob('*/*.json')):
            r=read(path); r['route']=route;r['annotation_provenance']=str(path);records[r['frame_key']]=r
    runtime=PROJECT/'derived/exhaustive_species_frame_labeler/runtime'
    reviews=read(runtime/'frame_review_state.json');drafts=read(runtime/'frame_drafts.json')
    # Later formal decisions override older export labels, including deletions.
    for key,r in reviews.items():
        route='hewentian' if key.startswith('hewentian') else 'jianshazui' if key.startswith('jianshazui') else 'stubbs_road'
        if r.get('frame_status')!='annotated': records.pop(key,None);continue
        records[key]={**r,'frame_key':key,'route':route,'annotation_provenance':str(runtime/'frame_review_state.json')}
    # Only explicit draft edits captured in the feedback audit are supervision.
    feedback=PROJECT/'derived/unsaved_review_feedback_iteration2_20260906/species_feedback.csv'
    for row in csv.DictReader(feedback.open(encoding='utf-8-sig')):
        key=row['frame_key']; draft=drafts.get(key,{})
        if key in reviews:continue
        label=next((x for x in draft.get('labels',[]) if x.get('label_id')==row['label_id']),None)
        if label is None:continue
        route='hewentian' if key.startswith('hewentian') else 'jianshazui' if key.startswith('jianshazui') else 'stubbs_road'
        rec=records.setdefault(key,{'frame_key':key,'route':route,'labels':[],'annotation_provenance':str(feedback)})
        rec['labels']=[x for x in rec['labels'] if x.get('label_id')!=label['label_id']]+[label]
    return records

def voxel(xyz,size=.15):
    _,ids=np.unique(np.floor(xyz/size).astype(np.int64),axis=0,return_index=True)
    return np.sort(ids)

def load_ascii(path):
    with path.open('rb') as f:
        while True:
            line=f.readline()
            if not line:raise ValueError('incomplete PCD header '+str(path))
            if line.startswith(b'DATA '):
                if line.strip()!=b'DATA ascii':raise ValueError('expected ASCII '+str(path))
                break
        return np.loadtxt(f,dtype=np.float32,usecols=(0,1,2))

class Maps:
    def __init__(self,out):self.out=out;self.loaded={};self.color_files=None
    def nearby(self,row):
        survey=Path(row['source_image_relpath']).parts[0];position=np.array([float(row['local_'+k]) for k in 'xyz'])
        if survey=='2023-10-27_hewentian':
            if self.color_files is None:
                self.color_files=[]
                for p in (VMMS/survey/'pointcloud/ColorCloudPoint').glob('*.pcd'):
                    m=re.search(r'_img(\d+)_',p.name)
                    if m:self.color_files.append((int(m.group(1)),p))
            nearest=sorted(self.color_files,key=lambda v:abs(v[0]-int(row['frame_id'])))[:3]
            arrays=[];sources=[]
            for _,p in nearest:
                r=read_binary_xyzi_rgb(p);arrays.append(np.column_stack([r[k] for k in 'xyz']));sources.append(str(p))
            xyz=np.concatenate(arrays);ids=np.isfinite(xyz).all(1)&(np.linalg.norm(xyz[:,:2]-position[:2],axis=1)<45)
            xyz=xyz[ids];xyz=xyz[voxel(xyz)]
            return xyz,sources
        if survey not in self.loaded:
            cache=self.out/'map_cache';cache.mkdir(exist_ok=True);path=cache/(survey+'.npy')
            sources=sorted((VMMS/survey/'pointcloud/mapping').glob('pointcloud_DS_*.pcd'))
            if not sources:raise ValueError('no downsampled maps '+survey)
            if path.exists(): xyz=np.load(path,allow_pickle=False)
            else:
                arrays=[]
                for p in sources:
                    print('loading map '+p.name,flush=True)
                    arr=load_ascii(p);arr=arr[np.isfinite(arr).all(1)];arrays.append(arr[voxel(arr)])
                xyz=np.concatenate(arrays);xyz=xyz[voxel(xyz)];np.save(path,xyz)
            self.loaded[survey]=(xyz,cKDTree(xyz[:,:2]),[str(x) for x in sources])
        xyz,tree,sources=self.loaded[survey];return xyz[tree.query_ball_point(position[:2],45)],sources

def project(xyz,row,yaw=180):
    pos=np.array([float(row['local_'+k]) for k in 'xyz']);rot=world_from_flu(*[float(row[k+'_rad']) for k in ('roll','pitch','heading')])
    angle=np.deg2rad(yaw);offset=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
    cam=(xyz-pos)@(rot@offset);dist=np.linalg.norm(cam,axis=1)
    uv=np.column_stack(((.5-np.arctan2(cam[:,1],cam[:,0])/(2*np.pi))%1,.5-np.arcsin(np.clip(cam[:,2]/np.maximum(dist,1e-6),-1,1))/np.pi))
    return uv,dist

def clean_points(xyz,full,camera_position=None):
    counts={'projected_polygon_points':len(xyz)}
    if len(xyz)<40:return xyz,counts,'too_few_projected_points'
    # Local XY cell minimum approximates terrain; avoid a global horizontal floor.
    grid=np.floor(full[:,:2]/1.5).astype(int);cells,inverse=np.unique(grid,axis=0,return_inverse=True)
    floors=np.full(len(cells),np.inf);np.minimum.at(floors,inverse,full[:,2])
    _,nearest=cKDTree(cells*1.5+.75).query(xyz[:,:2]);keep=xyz[:,2]>floors[nearest]+.35
    xyz=xyz[keep];counts['above_local_ground']=len(xyz)
    if len(xyz)<40:return xyz,counts,'too_few_above_ground'
    xyz=xyz[voxel(xyz,.18)]
    if len(xyz)>25000:xyz=xyz[np.random.default_rng(7).choice(len(xyz),25000,replace=False)]
    clusters=DBSCAN(eps=.65,min_samples=5,n_jobs=2).fit_predict(xyz)
    options=[];surface_rejections=0
    for lab,n in Counter(clusters).items():
        if lab<0 or n<35:continue
        group=xyz[clusters==lab];span=np.ptp(group,axis=0)
        if span[2]<1.3 or max(span[:2])>22:continue
        eig=np.linalg.eigvalsh(np.cov(group.T));fraction=float(eig[0]/max(eig.sum(),1e-12))
        if fraction<.01:
            surface_rejections+=1;continue
        distance=float(np.median(np.linalg.norm(group-camera_position,axis=1))) if camera_position is not None else 1.
        options.append((n,lab,distance,fraction))
    counts['planar_or_linear_clusters_rejected']=surface_rejections
    if not options:return xyz[:0],counts,'no_tree_sized_cluster'
    # A background facade must not win merely because it is denser than foliage.
    options.sort(key=lambda v:(v[2],-v[0]));xyz=xyz[clusters==options[0][1]];counts['cluster_count']=len(options)
    counts['largest_cluster_fraction']=options[0][0]/max(1,sum(x[0] for x in options))
    counts['selection_rule']='nearest nonplanar tree-sized cluster; not a completeness guarantee'
    counts['selected_eigen_min_fraction']=options[0][3]
    dist=cKDTree(xyz).query(xyz,k=min(12,len(xyz)))[0][:,1:].mean(1)
    xyz=xyz[dist<=np.median(dist)+3*max(np.median(abs(dist-np.median(dist))),.03)]
    counts['clean_points']=len(xyz)
    return xyz,counts,None

def render_preview(image,points,uv,xyz,path,label):
    canvas=image.copy();canvas.thumbnail((1280,640));draw=ImageDraw.Draw(canvas)
    polygon=[(int(x*canvas.width),int(y*canvas.height)) for x,y in points]
    draw.line(polygon+polygon[:1],fill='#ffcc33',width=3)
    for u,v in uv[::max(1,len(uv)//3500)]:
        x,y=int(u*canvas.width),int(v*canvas.height);draw.ellipse((x-1,y-1,x+1,y+1),fill='#19e9ae')
    canvas.save(path/'projection.jpg',quality=90)
    # Three orthographic views retain real aspect ratio and reveal mixed structures.
    sheet=Image.new('RGB',(960,390),'#eff4ed');d=ImageDraw.Draw(sheet)
    if len(xyz):
        center=(xyz.min(0)+xyz.max(0))/2;p=xyz-center;scale=300/max(np.ptp(p,axis=0).max(),1)
        for panel,(a,b,title) in enumerate([(0,2,'XZ'),(1,2,'YZ'),(0,1,'XY')]):
            d.text((panel*320+12,12),title,fill='#193d2b')
            for point in p[::max(1,len(p)//4000)]:
                x=panel*320+160+point[a]*scale;y=200-point[b]*scale;d.ellipse((x,y,x+2,y+2),fill='#26744d')
    sheet.save(path/'point_views.jpg',quality=90)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--limit',type=int);parser.add_argument('--all-species',action='store_true');parser.add_argument('--out',type=Path,default=OUTPUT);cfg=parser.parse_args()
    out=cfg.out;out.mkdir(parents=True,exist_ok=True);(out/'instances').mkdir(exist_ok=True)
    rows={r['stream_id']+'__'+r['frame_id']:r for r in csv.DictReader((PROJECT/'outputs/coordinates/frame_coordinates.csv').open(encoding='utf-8-sig'))}
    records=manual_records();annotations=[]
    for key,r in records.items():
        if key not in rows:continue
        for label in r.get('labels',[]):
            name=species(label.get('species',''));poly=label.get('points',[])
            if len(poly)<3 or '待定' in name or name=='Unknown':continue
            if not cfg.all_species and name not in TARGETS:continue
            annotations.append({'frame_key':key,'label':label,'species':name,'route':r['route'],'annotation_provenance':r['annotation_provenance']})
    annotations.sort(key=lambda x:(x['route'],x['frame_key'],x['label'].get('label_id','')))
    if cfg.limit:
        # Balanced pilot across species and routes.
        bins=defaultdict(list)
        for a in annotations:bins[(a['species'],a['route'])].append(a)
        annotations=[]
        while bins and len(annotations)<cfg.limit:
            for k in list(bins):
                annotations.append(bins[k].pop(len(bins[k])//2))
                if not bins[k]:del bins[k]
                if len(annotations)>=cfg.limit:break
    write(out/'annotation_snapshot.json',annotations)
    maps=Maps(out);results=[];last_key=None
    for idx,a in enumerate(annotations):
        key=a['frame_key'];row=rows[key];poly=np.array(a['label']['points'],dtype=float)
        ident=key+'__'+hashlib.sha256(str(a['label'].get('label_id',idx)).encode()).hexdigest()[:10]
        folder=out/'instances'/ident;folder.mkdir(exist_ok=True)
        record={k:v for k,v in a.items() if k!='label'};record.update(id=ident,points_per_sample=2048,source_panorama=str(VMMS/row['source_image_relpath']),position_hk80=[float(row[k]) for k in ('hk80_easting','hk80_northing','hk80_z')],projection={'yaw_deg':180,'fine_extrinsics':'unavailable','calibration_status':'empirical_HWT_yaw; other_routes_unverified','lever_arm_m':[0,0,0]},status='rejected',source_label_id=a['label'].get('label_id'))
        try:
            if key!=last_key:
                local,sources=maps.nearby(row);uv,dist=project(local,row)
                with Image.open(VMMS/row['source_image_relpath']) as source:im=source.convert('RGB');im.thumbnail((4096,2048))
                last_key=key
            record['source_lidar']=sources;record['nearby_map_points']=len(local)
            mask=np.zeros((1024,2048),dtype=np.uint8);cv2.fillPoly(mask,[np.rint(poly*[2047,1023]).astype(np.int32)],1)
            inside=mask[np.clip((uv[:,1]*1023).astype(int),0,1023),np.clip((uv[:,0]*2047).astype(int),0,2047)]>0
            selected=local[inside&(dist>2)&(dist<45)]
            cleaned,stats,reject=clean_points(selected,local,np.array([float(row['local_'+k]) for k in 'xyz']));record['cleaning']=stats
            left,top=np.clip(poly.min(0),0,1);right,bottom=np.clip(poly.max(0),0,1)
            crop=im.crop((int(left*im.width),int(top*im.height),max(int(left*im.width)+1,int(right*im.width)),max(int(top*im.height)+1,int(bottom*im.height))))
            crop.save(folder/'image.jpg',quality=94)
            if reject:record['reason']=reject
            elif len(cleaned)<128:record['reason']='less_than_128_clean_points'
            else:
                span=np.ptp(cleaned,axis=0);center=cleaned.mean(0);scale=np.linalg.norm(cleaned-center,axis=1).max()
                rng=np.random.default_rng(int(hashlib.sha256(ident.encode()).hexdigest()[:8],16));ids=rng.choice(len(cleaned),2048,replace=len(cleaned)<2048)
                np.savez_compressed(folder/'point.npz',points_xyz=((cleaned[ids]-center)/scale).astype(np.float32),centroid_xyz=center,scale=scale,source_point_count=len(cleaned),sampled_unique_point_count=len(set(ids)))
                np.savez_compressed(folder/'cleaned_local.npz',xyz=cleaned.astype(np.float32))
                origin=record['position_hk80']-np.array([float(row['local_'+k]) for k in 'xyz'])
                record.update(status='candidate_needs_alignment_review',center_local=center.tolist(),center_hk80=(center+origin).tolist(),extent_m=span.tolist(),unique_clean_points=len(cleaned),unique_sampled_points=len(set(ids)),point=str(folder/'point.npz'),image=str(folder/'image.jpg'),geometry_pass=len(cleaned)>=256 and span[2]>=2 and stats.get('largest_cluster_fraction',0)>=.55)
            clean_uv,_=project(cleaned,row);render_preview(im,poly,clean_uv,cleaned,folder,a['species'])
        except Exception as e:record['reason']=type(e).__name__+': '+str(e)
        write(folder/'record.json',record);results.append(record)
        if (idx+1)%10==0 or idx==len(annotations)-1:print(f"{idx+1}/{len(annotations)} {dict(Counter(x['status'] for x in results))}",flush=True)
    # Conservative grouping: never split near-identical centers across folds.
    for route in {r['route'] for r in results}:
        valid=[r for r in results if r.get('center_hk80')]
        valid=[r for r in valid if r['route']==route]
        if not valid:continue
        centers=np.array([r['center_hk80'][:2] for r in valid]);clusters=DBSCAN(eps=4,min_samples=1).fit_predict(centers)
        for r,c in zip(valid,clusters):r['tree_group']=route+'_'+str(c)
        groups=defaultdict(list)
        for r in valid:groups[r['tree_group']].append(r)
        for group,items in groups.items():
            conflict=len(set(r['species'] for r in items))>1
            for r in items:r['group_label_conflict']=conflict
    write(out/'manifest.json',results)
    for r in results:write(out/'instances'/r['id']/'record.json',r)
    summary={'scope':'all human named species' if cfg.all_species else 'three priority species','input_annotations':len(annotations),'species':dict(Counter(x['species'] for x in results)),'status_counts':dict(Counter(x['status'] for x in results)),'geometry_pass':sum(bool(x.get('geometry_pass')) for x in results),'rejections':dict(Counter(x.get('reason') for x in results if x['status']=='rejected')),'training_policy':'All point labels are provisional polygon-transfer candidates. Projection review is mandatory before claiming trusted paired training data. No HK weights are promoted on candidate metrics alone.','sampling':'0.15m map voxel / 0.18m instance voxel, local ground removal + DBSCAN + statistical outlier filtering; 2048 XYZ unit sphere; replacement only when fewer points, explicitly counted.','source_data_read_only':True}
    write(out/'summary.json',summary);print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
