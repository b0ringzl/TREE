"""First expansion: Terminalia catappa and Celtis sinensis, review-only pairs.

Selection uses existing validation + effective image labels, not test metrics.
No deployed classes/weights or human decisions are changed.
"""
import hashlib
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from scipy.spatial import cKDTree
from ultralytics import YOLO
from build_hk_lidar_pairs import manual_records,project,voxel,VMMS,Maps
from build_hk_dense_dataset import coordinates,crown_seed,polygon_inside,coverage_sample
from pilot_banyan_tracking import track
from repair_hk_banyan_stems import views,balanced_ids
from audit_hk_metric_height import DATA,read,write,ground_candidate

WORK=DATA.parents[2];OUT=DATA.parent/'species_expansion_round1_20260907'
NAMES={'Terminalia catappa':'榄仁树','Celtis sinensis':'朴树'}
MODEL=WORK/'full_image_species_baseline/experiments/full_web_yolo11s_cls_20260906_seed1/pretrain/weights/selected_by_val_group_macro_f1.pt'
USE_FRAME_SOURCES=False
LIMIT_PER_CLASS=None
LABEL_ALIASES={}

def main():
    OUT.mkdir(exist_ok=True);rows=coordinates();records=manual_records()
    model_report=read(MODEL.parent.parent/'final_metrics.json')
    pc=model_report['metrics']['val_group']['per_class'];canonical_names=model_report['class_names']
    labels=[];counts=Counter();routes={name:Counter() for name in NAMES}
    for key,r in records.items():
        for label in r.get('labels',[]):
            raw=label.get('species','')
            original_species=raw
            raw=next((v for k,v in LABEL_ALIASES.items() if raw.startswith(k)),raw)
            if raw in ('银合欢','銀合歡'):raw='Leucaena leucocephala'
            name=next((n for n in NAMES if raw.startswith(n)),None)
            if not name:continue
            counts[name]+=1;routes[name][r['route']]+=1
            if key in rows and (USE_FRAME_SOURCES or key.startswith('jianshazui')) and len(label.get('points',[]))>=3:
                labels.append({'id':key+'__'+hashlib.sha256(str(label.get('label_id')).encode()).hexdigest()[:10],
                               'frame_key':key,'species':name,'chinese_name':NAMES[name],'polygon':label['points'],
                               'original_species':original_species,
                               'annotation_provenance':r['annotation_provenance']})
    if LIMIT_PER_CLASS:
        chosen=[]
        for n in NAMES:
            group=sorted([v for v in labels if v['species']==n],key=lambda v:-np.prod(np.ptp(np.array(v['polygon']),axis=0)))
            kept=[]
            for v in group:
                row=rows[v['frame_key']];pos=np.array([float(row['local_'+k]) for k in 'xy'])
                if any(Path(row['source_image_relpath']).parts[0]==s and np.linalg.norm(pos-p)<15 for s,p in kept):continue
                chosen.append(v);kept.append((Path(row['source_image_relpath']).parts[0],pos))
                if len(kept)>=LIMIT_PER_CLASS:break
        labels=chosen
    write(OUT/'selection.json',{'classes':[{'species':n,'chinese_name':NAMES[n],'effective_image_labels':counts[n],
          'label_routes':dict(routes[n]),'validation':next((p for p in pc if p['scientific_name']==n),None)} for n in NAMES],
          'policy':'Effective image labels; source-mode recorded; subset prioritizes larger crops and 15m camera spacing, not classifier success. Not new trained dual-source species.',
          'source_mode':'original_frame_clouds_or_DS_maps' if USE_FRAME_SOURCES else 'existing_JTS_dense_ROI_caches',
          'attempts':len(labels)})
    # Cache coverage only; do not pretend these are exhaustive raw-map scans.
    maps=Maps(DATA.parent/'hk_lidar_pairs_pilot_20260907') if USE_FRAME_SOURCES else None
    if not USE_FRAME_SOURCES:
        clouds=[]
        for path in (DATA/'dense_cache').glob('*.npz'):
            meta=read(path.with_suffix('.json'))
            if '20250210-jianshazui' in meta['source']:clouds.append(np.load(path)['xyz'])
        cloud=np.concatenate(clouds);cloud=cloud[voxel(cloud,.10)];tree=cKDTree(cloud[:,:2])
    model=YOLO(str(MODEL));results=[];seen={n:[] for n in NAMES}
    for item in sorted(labels,key=lambda x:x['id']):
        folder=OUT/item['id'];folder.mkdir(exist_ok=True);row=rows[item['frame_key']];poly=np.array(item['polygon'])
        r={**item,'status':'needs_rework','training_eligible':False,'projection_verified':False}
        with Image.open(VMMS/row['source_image_relpath']) as src:im=src.convert('RGB');im.thumbnail((4096,2048))
        lo=poly.min(0);hi=poly.max(0);crop=im.crop((int(lo[0]*im.width),int(lo[1]*im.height),int(hi[0]*im.width),int(hi[1]*im.height)));crop.save(folder/'image.jpg',quality=94)
        pred=model.predict(crop,verbose=False,device=0)[0];probs=pred.probs.data.cpu().numpy();rank=np.argsort(-probs)
        top_index=int(rank[0]);raw_name=pred.names[top_index]
        assert int(raw_name.split('_',1)[0])==top_index, 'Checkpoint class ordering mismatch'
        r['image_evidence']={'top1':canonical_names[top_index],'raw_model_class':raw_name,'score':float(probs[rank[0]]),'margin':float(probs[rank[0]]-probs[rank[1]]),
                             'label_agreement':canonical_names[top_index]==item['species'],'model_has_target_class':item['species'] in canonical_names,'calibrated':False}
        pos=np.array([float(row['local_'+k]) for k in 'xyz'])
        try:
            if USE_FRAME_SOURCES:
                local,sources=maps.nearby(row);r['point_sources']=sources;r['source_sampling_m']=getattr(maps,'sampling_m',.15)
            else:local=cloud[tree.query_ball_point(pos[:2],40)]
            seed,reason=crown_seed(local[voxel(local,.2)],row,poly,False)
            if seed is None:raise ValueError(reason)
            axis=np.median(seed[:,:2],axis=0)
            local=local[np.linalg.norm(local[:,:2]-axis,axis=1)<10] if USE_FRAME_SOURCES else cloud[tree.query_ball_point(axis,10)]
            uv,_=project(local,row);inside=polygon_inside(uv,poly,dilation=5)
            crown=local[inside&(cKDTree(seed).query(local)[0]<.7)]
            if len(crown)<200:raise ValueError('insufficient_crown_support_in_existing_cache')
            floor=float(np.quantile(crown[:,2],.02)-.35);ground=ground_candidate(local,axis)
            z0=ground['ground_z_m'] if ground['status']=='local_plane_candidate' else float(np.quantile(local[:,2],.01))
            lower=local[inside&(local[:,2]>z0+.25)&(local[:,2]<floor+.8)&(np.linalg.norm(local[:,:2]-axis,axis=1)<6)]
            stem,stats=track(lower,axis,z0,floor)
            r.update(ground=ground,tracking=stats,axis_local_xy=axis.tolist(),crown_floor_z_m=floor,
                     roi_bottom_z_m=float(z0),lower_roi_points=len(lower),seed_points=len(seed),
                     crown_support_points=len(crown),total_height_verified=False)
            if stem is None:raise ValueError(stats['reason'])
            p=np.r_[crown[crown[:,2]>=floor],stem];p=p[voxel(p,.1)]
            ti=np.flatnonzero(p[:,2]<floor);ci=np.flatnonzero(p[:,2]>=floor);nt=min(768,len(ti));nc=min(2048-nt,len(ci))
            ids=np.r_[ti[coverage_sample(p[ti],nt)],ci[coverage_sample(p[ci],nc)]]
            center=p.mean(0);scale=np.linalg.norm(p-center,axis=1).max()
            np.savez_compressed(folder/'point.npz',points_xyz=((p[ids]-center)/scale).astype(np.float32),centroid_xyz=center,scale=scale,metric_units='m')
            np.savez_compressed(folder/'cleaned_local.npz',xyz=p);views(p,folder/'point_views.jpg',floor)
            canvas=im.copy();canvas.thumbnail((1280,640));draw=ImageDraw.Draw(canvas);puv=project(p,row)[0]
            for i in balanced_ids(p,5000):
                u,v=puv[i];x,y=u*canvas.width,v*canvas.height;draw.ellipse((x-1,y-1,x+1,y+1),fill='#ff942e' if p[i,2]<floor else '#19e9ae')
            canvas.save(folder/'projection.jpg',quality=94)
            survey=Path(row['source_image_relpath']).parts[0]
            repeated=any(s==survey and np.linalg.norm(axis-a)<5 for s,a in seen[item['species']]);seen[item['species']].append((survey,axis))
            r.update(status='pair_candidate_unreviewed',axis_local_xy=axis.tolist(),ground=ground,tracking=stats,
                     trunk_points=len(ti),crown_points=len(ci),sample_unique_points=len(np.unique(p[ids],axis=0)),
                     nearby_same_species_view=repeated,point=str(folder/'point.npz'))
        except (ValueError,IndexError) as exc:r['reason']=str(exc)
        results.append(r);write(folder/'record.json',r);print(item['id'],item['species'],r['status'],r['image_evidence']['top1'],flush=True)
        write(OUT/'results.progress.json',results)
    write(OUT/'results.json',results);write(OUT/'summary.json',{'attempts':len(results),'statuses':dict(Counter(r['status'] for r in results)),
      'by_species':{n:dict(Counter(r['status'] for r in results if r['species']==n)) for n in NAMES},'model_changed':False,'accepted_pairs':0})

if __name__=='__main__':main()
