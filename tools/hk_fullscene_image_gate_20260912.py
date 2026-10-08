"""Map-first proposals -> local panorama crops -> independent tree/background checks.

Image support does not certify species, point-cloud purity or whole-tree completeness.
"""
import sys,csv,io,hashlib,time,argparse
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'));sys.path.insert(0,str(ROOT/'hk-tree-vmms/tools'))
from hk_fullscene_refine_20260912 import DEST
from hk_fullscene_extract_20260912 import read,write,VMMS
from build_hk_lidar_pairs import project
from spacescan_trees_demo_v1 import panorama
from spacescan_trees_demo_v1.tree_detection import Engine
OUT=DEST/'image_evidence'
def main(ids=None,per_map=None):
    OUT.mkdir(exist_ok=True);records=[]
    for p in sorted((DEST/'maps').glob('*/candidates.json')):
        if ids and p.parent.name not in ids:continue
        items=read(p)
        if per_map and len(items)>per_map:items=[items[i] for i in np.linspace(0,len(items)-1,per_map,dtype=int)]
        records+=items
    coords=list(csv.DictReader((ROOT/'hk-tree-vmms/outputs/coordinates/frame_coordinates.csv').open(encoding='utf-8-sig')));surveys={}
    for c in coords:surveys.setdefault(Path(c['source_image_relpath']).parts[0],[]).append(c)
    indices={s:cKDTree([[float(r['local_x']),float(r['local_y'])] for r in rs]) for s,rs in surveys.items()};engine=Engine();write(OUT/'method.json',dict(weights=engine.weights,gate='CLIP tree-vs-background padded and central crops; both margins < -0.015, plus YOLO tree box >=0.4 covering central target region',projection='local XYZ -> empirical VMMS panorama transform -> pinhole crop',species_labels_used=False,training_eligible=False,warning='Independent image corroboration only; cannot prove point-cloud single-tree purity or taxonomic identity'))
    image_cache={};done=0
    for r in records:
        d=OUT/r['id']
        if (d/'record.json').exists():continue
        d.mkdir(exist_ok=True);result=dict(id=r['id'],map_id=r['map_id'],survey=r['survey'],status='no_registered_panorama',views=[],training_eligible=False,species=None)
        if r['survey'] not in indices:write(d/'record.json',result);continue
        ds,ix=indices[r['survey']].query(r['axis_xy'],k=min(500,len(surveys[r['survey']])));choices=[]
        for distance,index in zip(np.atleast_1d(ds),np.atleast_1d(ix)):
            if not 7<=distance<=45:continue
            row=surveys[r['survey']][int(index)];pos=np.array([float(row['local_x']),float(row['local_y'])]);bearing=np.arctan2(*(np.array(r['axis_xy'])-pos)[::-1]);rel=Path(row['source_image_relpath'])
            if not (VMMS/rel).is_file():continue
            if choices and min(abs(np.arctan2(np.sin(bearing-b),np.cos(bearing-b))) for _,b in choices)<np.deg2rad(10):continue
            choices.append((row,bearing))
            if len(choices)>=2:break
        q=np.load(r['cloud'])['xyz'];images=[];viewrows=[]
        for vi,(row,bearing) in enumerate(choices):
            try:
                uv,_=project(q,row);u=uv[:,0];center=np.arctan2(np.sin(u*2*np.pi).mean(),np.cos(u*2*np.pi).mean())/(2*np.pi);u=center+(u-center+.5)%1-.5;a,c=np.quantile(u,[.02,.98]);b,dv=np.quantile(uv[:,1],[.02,.98]);padx=.07*(c-a);pady=.07*(dv-b);region=[float(a-padx),float(max(0,b-pady)),float(c+padx),float(min(1,dv+pady))]
                path=VMMS/row['source_image_relpath']
                if str(path) not in image_cache:
                    if len(image_cache)>=6:image_cache.pop(next(iter(image_cache)))
                    with Image.open(path) as im:image_cache[str(path)]=im.convert('RGB')
                crop,spec=panorama.perspective_for_region(image_cache[str(path)],region,max_edge=1280)
                if min(crop.size)<48:raise ValueError('too_small_native_crop')
                file=d/f'view_{vi}.jpg';crop.save(file,quality=94);images.append(crop);viewrows.append(dict(frame_key=row['stream_id']+'__'+row['frame_id'],panorama=str(path),wgs84_platform=[float(row['wgs84_latitude']),float(row['wgs84_longitude'])],crop=str(file),perspective=spec,region=region,camera_distance_xy=float(np.linalg.norm(np.array([float(row['local_x']),float(row['local_y'])])-r['axis_xy']))))
            except Exception as e:result['views'].append(dict(error=str(e),frame_key=row['stream_id']+'__'+row['frame_id']))
        if images:
            checks=engine.background_checks(images)
            for image,v,check in zip(images,viewrows,checks):
                boxes=engine.boxes(image);center_boxes=[b for b in boxes if b['box'][0]<=.5<=b['box'][2] and b['box'][1]<=.5<=b['box'][3] and (b['box'][2]-b['box'][0])*(b['box'][3]-b['box'][1])>.12]
                supported=all(m<-.015 for m in check['background_minus_tree']) and bool(center_boxes)
                v.update(background_check=check,detector_boxes=boxes,image_support=supported);result['views'].append(v)
            n=sum(v.get('image_support',False) for v in result['views']);result['status']='two_view_tree_support' if n>=2 else 'one_view_tree_support' if n else 'image_not_correlated_or_background_veto'
        elif choices:result['status']='projection_or_crop_unusable'
        write(d/'record.json',result);done+=1
        if done%25==0:print('image evidence',done,'/',len(records),r['id'],result['status'],flush=True)
    allresults=[read(p) for p in OUT.glob('*/record.json')];from collections import Counter
    write(OUT/'manifest.json',allresults);write(OUT/'summary.json',dict(records=len(allresults),status_counts=dict(Counter(r['status'] for r in allresults)),training_eligible=0));print('image audit done',len(allresults),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--ids',nargs='*');p.add_argument('--per-map',type=int);a=p.parse_args();main(a.ids,a.per_map)
