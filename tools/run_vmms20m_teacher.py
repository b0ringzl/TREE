"""Logged, inference-only VMMS survey. Predictions never replace human labels.

20 m anchor grid + quality-triggered nearby cameras. Broad BioCLIP vocabulary,
original pixels, bounded shadow lifting and appearance/geometry-gated association.
No Google image downloads, no training, no deployed-file writes.
"""
import argparse
import bisect
import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from focus8_data import LABELS, TAXA
from training_safety import sha256, deployed_artifacts
from distillation_core import read_json, save_json
from tree_species_teacher_demo.engine import Engine, EXPERIMENT

VMMS = ROOT.parent / 'vmms'
COORDS = ROOT / 'hk-tree-vmms/outputs/coordinates/frame_coordinates.csv'
WEIGHT = ROOT / 'hk-tree-vmms/derived/training_runs/combined_tree_segmentation/yolo11m_seg_1024_20260901_104757/weights/best.pt'
YAW = {'hewentian_pano': -177., 'stubbs_pano_1': -178.}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_suffix('.tmp')
    save_json(tmp,value);tmp.replace(path)


def stamp():
    return datetime.now(timezone.utc).isoformat()


def quality(im):
    gray=np.asarray(im.convert('L').resize((256,256)),dtype=np.uint8)
    dark=float((gray<35).mean());black=float((gray<8).mean())
    detail=float(cv2.Laplacian(gray,cv2.CV_32F).var())
    flags=[]
    if dark>.4:flags.append('underexposed')
    if black>.4:flags.append('clipped_shadows')
    if min(im.size)<160:flags.append('low_native_resolution')
    if detail<25:flags.append('little_detail')
    score=max(.05,min(1.,min(im.size)/320))*(1-.8*dark)
    return {'width':im.width,'height':im.height,'dark_fraction_lt35':dark,'black_fraction_lt8':black,
        'laplacian_variance':detail,'score':score,'flags':flags}


def lift_shadows(im):
    rgb=np.asarray(im)
    lab=cv2.cvtColor(rgb,cv2.COLOR_RGB2LAB)
    changed=lab.copy()
    changed[:,:,0]=cv2.createCLAHE(clipLimit=1.5,tileGridSize=(8,8)).apply(lab[:,:,0])
    local=cv2.cvtColor(changed,cv2.COLOR_LAB2RGB).astype(np.float32)/255
    # Bound correction, retain chroma and blend with unmodified pixels.
    lifted=np.clip(local**.8,0,1)
    return Image.fromarray(np.uint8(np.clip(.65*rgb/255+.35*lifted,0,1)*255))


def bbox_iou(a,b):
    inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/max(union,1e-8)


def plan(out):
    out.mkdir(parents=True,exist_ok=False)
    grouped=defaultdict(list)
    with COORDS.open(encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):grouped[row['stream_id']].append(row)
    frames={};anchors=[];neighbors={};routes=[]
    for stream,rows in grouped.items():
        rows.sort(key=lambda r:int(r['seq_id']))
        distance=0.;previous=None
        for row in rows:
            xy=[float(row['hk80_easting']),float(row['hk80_northing'])]
            if previous is not None:distance+=math.dist(xy,previous)
            previous=xy
            row.update(key=stream+'__'+row['frame_id'],distance_m=distance,xy=xy,
                source=str((VMMS/row['source_image_relpath']).resolve()))
            frames[row['key']]=row
        distances=[r['distance_m'] for r in rows]
        selected=[]
        for requested in np.arange(0,distance+.001,20.):
            j=bisect.bisect_left(distances,requested)
            candidates=[max(0,j-1),min(j,len(rows)-1)]
            k=min(candidates,key=lambda i:abs(distances[i]-requested))
            if rows[k]['key'] not in selected:
                selected.append(rows[k]['key'])
        for key in selected:
            center=frames[key]['distance_m'];extra=[]
            for offset in (-7.,7.):
                j=bisect.bisect_left(distances,center+offset)
                candidates=[max(0,j-1),min(j,len(rows)-1)]
                k=min(candidates,key=lambda i:abs(distances[i]-(center+offset)))
                if abs(distances[k]-center)>=3 and rows[k]['key']!=key:extra.append(rows[k]['key'])
            neighbors[key]=list(dict.fromkeys(extra))
        anchors+=selected
        routes.append({'stream':stream,'indexed_frames':len(rows),'length_m':distance,'anchors':len(selected)})
    missing=[k for k in set(anchors+[n for ns in neighbors.values() for n in ns]) if not Path(frames[k]['source']).is_file()]
    if missing:raise RuntimeError('Missing source panoramas: '+str(missing[:5]))
    write(out/'frames.json',frames)
    write(out/'plan.json',{'created_at':stamp(),'routes':routes,'anchors':anchors,'neighbors':neighbors,'missing_sources':missing,
        'spacing_m':20,'nearest_actual_exposure_to_grid':True,'neighbor_offsets_m':[-7,7]})
    write(out/'policy.json',{'training':False,'deployment':False,'detector_threshold':.4,
        'classification':'Original BioCLIP 2.5, full raw 148-species vocabulary plus foxtail; no forced eight-class softmax',
        'teacher_selection_limit':'Original teacher beat the new focus-eight head on inspected project diagnostics; not a claim of globally best accuracy.',
        'uncertain_trigger':'top1<0.65 OR margin<0.20 OR any native quality flag',
        'enhancement':'Whole original + native detail crop; add bounded CLAHE/gamma whole view when underexposed. Augmentations not independent views.',
        'multiview':'Nearby camera detection + independent SIFT spatial verification. Labels never used for association.',
        'human_correction':'Manual sheep-hoof/Bauhinia labels requested as blakeana: correction overlay only, never web taxonomy merge. Palm labels self-rated 70%: provisional.',
        'google':'No API key found in configured Google key variables. Queue official viewpoint/heading links; no Google images downloaded or claimed verified.',
        'reporting':'All frames, detector misses, low quality and unresolved cases retained. Highlight filter is not accuracy evidence.'})
    protected=[WEIGHT,*deployed_artifacts(),ROOT/'tree_species_teacher_demo/engine.py']
    for row in read_json(EXPERIMENT/'selection_lock.json')['selected_runs']:protected.append(Path(row['run'])/'best.pt')
    protected.append(Path(read_json(EXPERIMENT/'protocol.json')['teacher_weight']))
    write(out/'protected_artifacts.json',{str(p):sha256(p) for p in protected})
    write(out/'status.json',{'stage':'planned','anchors':len(anchors),'routes':routes})
    print('Planned '+str(len(anchors))+' anchors',flush=True)


class Survey:
    def __init__(self,out):
        self.out=out;self.frames=read_json(out/'frames.json');self.plan=read_json(out/'plan.json')
        self.teacher=Engine();self.teacher.load()
        self.labels=self.teacher.classes['original_names']+['Wodyetia bifurcata']
        self.bank=torch.cat([self.teacher.bank,self.teacher.extra_bank],0)
        from ultralytics import YOLO
        self.detector=YOLO(str(WEIGHT));self.detector.to('cuda')
        self.feature_cache={}
        torch.set_num_threads(6)
        write(out/'class_names.json',self.labels)

    def encode(self,images):
        features=[]
        with torch.inference_mode():
            for start in range(0,len(images),12):
                pixels=[]
                for im in images[start:start+12]:
                    square=ImageOps.pad(im,(224,224),method=Image.Resampling.BICUBIC,color=(124,116,104))
                    pixels.append(torch.from_numpy(np.array(square).copy()).permute(2,0,1).float()/255)
                x=torch.stack(pixels).cuda()
                x=((x-self.teacher.mean)/self.teacher.std).to(self.teacher.dtype)
                features.append(self.teacher.model.encode_image(x,normalize=True).float())
            x=torch.cat(features)
            z=x@self.bank.T*self.teacher.model.logit_scale.exp().float()
            p=(z/self.teacher.choice['calibration_temperature']).softmax(-1)
        return x.cpu().numpy(),p.cpu().numpy()

    def top(self,p,n=3):
        return [{'species':self.labels[int(i)],'score':float(p[i])} for i in np.argsort(-p)[:n]]

    def frame(self,key):
        directory=self.out/'frames'/key
        if (directory/'result.json').exists():return read_json(directory/'result.json')
        directory.mkdir(parents=True,exist_ok=True)
        frame=self.frames[key]
        with Image.open(frame['source']) as source:image=source.convert('RGB')
        w,h=image.size;proposals=[]
        for left,right in [(0,1),(0,.55),(.225,.775),(.45,1)]:
            x0,x1=round(left*w),round(right*w)
            det=self.detector.predict(image.crop((x0,0,x1,h)),imgsz=1024,conf=.10,iou=.5,
                agnostic_nms=True,max_det=100,device=0,verbose=False)[0]
            for box,score in zip(det.boxes.xyxy.cpu().tolist(),det.boxes.conf.cpu().tolist()):
                a,b,c,d=box
                proposals.append({'box':[max(0,(a+x0)/w),max(0,b/h),min(1,(c+x0)/w),min(1,d/h)],'score':score})
        keep=[]
        for p in sorted(proposals,key=lambda x:-x['score']):
            if p['score']>=.4 and not any(bbox_iou(p['box'],q['box'])>.5 for q in keep):keep.append(p)
        instances=[];images=[];groups=[]
        for i,p in enumerate(keep):
            b=p['box'];box=[round(b[0]*w),round(b[1]*h),round(b[2]*w),round(b[3]*h)]
            if box[2]-box[0]<12 or box[3]-box[1]<16:continue
            crop=image.crop(box);q=quality(crop)
            if b[2]-b[0]>.25:q['flags'].append('possible_multi_tree_box')
            path=directory/f'{i:03d}_original.jpg';crop.save(path,quality=95)
            variants=[('whole_original',crop)]
            side=min(crop.size)
            # Native detail: top-centre square for tall trees, centre square for wide crowns.
            ox=max(0,(crop.width-side)//2)
            detail=crop.crop((ox,0,ox+side,side))
            if crop.width!=crop.height:variants.append(('native_detail',detail))
            if 'underexposed' in q['flags']:
                enhanced=lift_shadows(crop);enhanced.save(directory/f'{i:03d}_shadow_lift.jpg',quality=95)
                variants.append(('bounded_shadow_lift',enhanced))
            group=[]
            for name,im in variants:group.append((len(images),name));images.append(im)
            groups.append(group)
            bearing=(float(frame['heading_deg'])+(sum(b[::2])/2-.5)*360-YAW.get(frame['stream_id'],-180))%360
            instances.append({'id':key+f'__tree_{i:03d}','box':b,'native_box_xyxy':box,'crop':str(path),
                'quality':q,'detector_score':p['score'],'bearing_deg':bearing,
                'google_viewpoint_url':f"https://www.google.com/maps/@?api=1&map_action=pano&viewpoint={frame['wgs84_latitude']},{frame['wgs84_longitude']}&heading={bearing:.2f}&pitch=5&fov=50",
                'google_verified':False,'human_review':'pending','training_eligible':False})
        if images:
            x,p=self.encode(images);embeddings=[];probabilities=[]
            for row,group in zip(instances,groups):
                indices=[g[0] for g in group]
                pooled=p[indices].mean(0);feature=x[indices[0]]
                row['top3']=self.top(pooled);row['original_top3']=self.top(p[indices[0]])
                row['augmentation_predictions']=[{'variant':name,'top3':self.top(p[idx])} for idx,name in group]
                scores=np.sort(pooled)[::-1]
                row['uncertain']=bool(scores[0]<.65 or scores[0]-scores[1]<.20 or row['quality']['flags'])
                row['within_frame_agreement']=len({int(p[j].argmax()) for j in indices})==1
                row['focus_eight_candidate']=row['top3'][0]['species'] in LABELS
                row['probability_index']=len(probabilities)
                embeddings.append(feature);probabilities.append(pooled)
            np.savez(directory/'features.npz',embedding=np.array(embeddings),probabilities=np.array(probabilities))
        preview=image.copy();preview.thumbnail((1600,900))
        draw=ImageDraw.Draw(preview)
        for row in instances:
            b=row['box'];coords=[b[0]*preview.width,b[1]*preview.height,b[2]*preview.width,b[3]*preview.height]
            draw.rectangle(coords,outline=(255,210,50),width=2)
            draw.text((coords[0]+3,coords[1]+3),row['id'].split('__tree_')[-1]+' '+str(round(row['detector_score']*100))+'%',fill='yellow')
        preview.save(directory/'overview.jpg',quality=90)
        result={'frame_key':key,'source':frame['source'],'source_sha256':sha256(frame['source']),'size':[w,h],
            'camera':{'latitude':float(frame['wgs84_latitude']),'longitude':float(frame['wgs84_longitude']),'heading':float(frame['heading_deg']),'distance_m':frame['distance_m']},
            'instances':instances,'raw_proposal_count':len(proposals),'completed_at':stamp(),'errors':[]}
        write(directory/'result.json',result)
        return result

    def feature(self,key):
        if key not in self.feature_cache:
            path=self.out/'frames'/key/'features.npz'
            if not path.exists():return None
            with np.load(path) as d:self.feature_cache[key]=(d['embedding'].copy(),d['probabilities'].copy())
        return self.feature_cache[key]

    def spatial_matches(self,a,b):
        # Local image correspondences are stronger evidence than species agreement.
        def pixels(path):
            im=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
            scale=min(1,600/max(im.shape));return cv2.resize(im,None,fx=scale,fy=scale)
        sift=cv2.SIFT_create(nfeatures=500)
        ka,da=sift.detectAndCompute(pixels(a['crop']),None);kb,db=sift.detectAndCompute(pixels(b['crop']),None)
        if da is None or db is None or min(len(da),len(db))<8:return 0
        pairs=cv2.BFMatcher().knnMatch(da,db,k=2)
        good=[m[0] for m in pairs if len(m)==2 and m[0].distance<.7*m[1].distance]
        if len(good)<8:return 0
        pa=np.float32([ka[m.queryIdx].pt for m in good]);pb=np.float32([kb[m.trainIdx].pt for m in good])
        _,mask=cv2.findHomography(pa,pb,cv2.RANSAC,4.)
        return int(mask.sum()) if mask is not None else 0

    def associate(self,anchor,neighbor):
        aa=self.feature(anchor['frame_key']);bb=self.feature(neighbor['frame_key'])
        if aa is None or bb is None:return []
        similarities=aa[0]@bb[0].T;links=[];used=set()
        for a in anchor['instances']:
            i=a['probability_index'];order=np.argsort(-similarities[i])
            if not len(order):continue
            j=int(order[0]);b=neighbor['instances'][j]
            if j in used or similarities[i,j]<.86:continue
            if len(order)>1 and similarities[i,j]-similarities[i,order[1]]<.035:continue
            if abs((a['bearing_deg']-b['bearing_deg']+180)%360-180)>45:continue
            if int(similarities[:,j].argmax())!=i:continue
            inliers=self.spatial_matches(a,b)
            if inliers<8:continue
            used.add(j)
            links.append({'anchor_id':a['id'],'neighbor_id':b['id'],'neighbor_frame':neighbor['frame_key'],
                'appearance_cosine':float(similarities[i,j]),'sift_homography_inliers':inliers,
                'status':'tentative_same_tree_or_shared_context_needs_review','species_used_for_matching':False})
        return links

    def run(self,limit=None):
        anchors=self.plan['anchors'][:limit] if limit else self.plan['anchors']
        for ordinal,key in enumerate(anchors,1):
            if (self.out/'anchors'/f'{key}.json').exists():continue
            write(self.out/'status.json',{'stage':'inference','anchor':ordinal,'total_anchors':len(anchors),'current':key,'updated_at':stamp(),'pid':os.getpid()})
            main=self.frame(key);neighbor_results=[];links=[]
            if not main['instances'] or any(r['uncertain'] for r in main['instances']):
                for nkey in self.plan['neighbors'][key]:
                    near=self.frame(nkey);neighbor_results.append(near);links.extend(self.associate(main,near))
            final=[]
            for row in main['instances']:
                features=self.feature(key);ps=[features[1][row['probability_index']]];weights=[row['quality']['score']]
                evidence=[{'frame_key':key,'instance_id':row['id'],'quality_score':weights[0],'crop':row['crop']}]
                for link in links:
                    if link['anchor_id']!=row['id']:continue
                    near=next(r for r in neighbor_results if r['frame_key']==link['neighbor_frame'])
                    other=next(r for r in near['instances'] if r['id']==link['neighbor_id'])
                    ps.append(self.feature(near['frame_key'])[1][other['probability_index']]);weights.append(other['quality']['score'])
                    evidence.append({'frame_key':near['frame_key'],'instance_id':other['id'],'quality_score':weights[-1],'crop':other['crop'],'association':link})
                pooled=np.average(ps,axis=0,weights=weights);order=np.argsort(-pooled)
                agree=len({int(p.argmax()) for p in ps})==1
                strong=bool(pooled[order[0]]>=.65 and pooled[order[0]]-pooled[order[1]]>=.20 and agree and max(weights)>=.55)
                label=self.labels[int(order[0])]
                final.append({**row,'fused_top3':self.top(pooled),'linked_camera_views':len(evidence),'view_agreement':agree,'evidence':evidence,
                    'best_quality_view':max(evidence,key=lambda r:r['quality_score']),
                    'status':'strong_candidate_pending_review' if strong else 'unresolved_pending_review',
                    'showcase_candidate':strong and label in LABELS and not ('possible_multi_tree_box' in row['quality']['flags']),
                    'special_checks':(['Inspect papery bark and leaves; do not rely on dark crown alone'] if label=='Melaleuca cajuputi' else ['Inspect phyllodes; compare non-acacia alternatives'] if label=='Acacia confusa' else [])})
            write(self.out/'anchors'/f'{key}.json',{'frame_key':key,'instances':final,'neighbor_frames':[r['frame_key'] for r in neighbor_results],
                'links':links,'empty_detector_result':not bool(main['instances']),'created_at':stamp()})
            print(f'{ordinal}/{len(anchors)} {key}: {len(final)} candidates; {len(links)} tentative links',flush=True)
        self.summarize()

    def summarize(self):
        results=[read_json(p) for p in sorted((self.out/'anchors').glob('*.json'))]
        instances=[r for f in results for r in f['instances']]
        write(self.out/'summary.json',{'completed_anchor_frames':len(results),'planned_anchor_frames':len(self.plan['anchors']),
            'processed_camera_frames':len(list((self.out/'frames').glob('*/result.json'))),'anchor_candidates':len(instances),
            'empty_anchor_frames':sum(f['empty_detector_result'] for f in results),'tentatively_linked_multiview':sum(r['linked_camera_views']>1 for r in instances),
            'strong_focus_candidates':sum(r['showcase_candidate'] for r in instances),'status_counts':dict(Counter(r['status'] for r in instances)),
            'focus_counts':{s:sum(r['fused_top3'][0]['species']==s for r in instances) for s in LABELS},
            'google_images_verified':0,'metrics_are_prediction_counts_not_accuracy':True})
        write(self.out/'google_review_queue.json',[{'instance_id':r['id'],'url':r['google_viewpoint_url'],'candidate':r['fused_top3'],
            'status':'not_accessed_no_api_credentials','note':'Camera coordinate and approximate panorama bearing, not verified tree location; check date/identity/viewpoint before comparison'}
            for r in instances if r['status']=='unresolved_pending_review'])
        for path,expected in read_json(self.out/'protected_artifacts.json').items():
            if sha256(path)!=expected:raise RuntimeError('Protected artifact changed: '+path)
        write(self.out/'status.json',{'stage':'complete_local_inference' if len(results)==len(self.plan['anchors']) else 'partial_local_inference',
            'completed_anchor_frames':len(results),'production_unchanged':True,'google_verification':'pending','updated_at':stamp()})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--stage',choices=['prepare','run'],required=True);p.add_argument('--limit',type=int)
    args=p.parse_args();out=args.out.resolve()
    if args.stage=='prepare':plan(out)
    else:Survey(out).run(args.limit)
