"""Verify real-point assets, attach existing image-model evidence and review shortlist.

No human decision or training flag is changed. Automatic shortlist is NOT visual QA.
"""
from collections import Counter
from pathlib import Path
import hashlib
import numpy as np
from PIL import Image,ImageDraw
from scipy.spatial import cKDTree
from ultralytics import YOLO
from build_hk_five_species_pairs import OUT,NAMES
from build_hk_dense_dataset import coordinates
from build_hk_lidar_pairs import read,write
from expand_hk_species_pilot import MODEL

def main():
    rows=read(OUT/'manifest.json');coords=coordinates()
    model=YOLO(str(MODEL));names=read(MODEL.parent.parent/'final_metrics.json')['class_names']
    valid=[r for r in rows if r['status']=='geometry_candidate_pending_review'];verification=[]
    for r in valid:
        folder=OUT/'instances'/r['id'];p=np.load(folder/'point.npz');xyz=p['points_xyz']
        cleaned=np.load(folder/'cleaned_local.npz')['xyz'];real=xyz*p['scale']+p['centroid_xyz']
        assert np.isfinite(xyz).all() and 0<len(xyz)<=2048
        assert len(np.unique(xyz,axis=0))==len(xyz)
        error=float(cKDTree(cleaned).query(real)[0].max());assert error<1e-4
        assert float(np.linalg.norm(xyz,axis=1).max())<=1.00001
        verification.append({'id':r['id'],'independent_points':len(xyz),'max_reconstruction_error_m':error})
        pred=model.predict(str(folder/'image.jpg'),verbose=False,device=0)[0]
        prob=pred.probs.data.cpu().numpy();rank=np.argsort(-prob);top=int(rank[0])
        assert int(pred.names[top].split('_',1)[0])==top
        r['image_evidence']={'top1':names[top],'score':float(prob[top]),'calibrated':False,
            'label_agreement':names[top]==r['species'],'model_has_target_class':r['species'] in names,
            'target_score':float(prob[names.index(r['species'])]) if r['species'] in names else None,
            'top3':[{'species':names[int(i)],'score':float(prob[i])} for i in rank[:3]]}
        r['total_height_verified']=False
        write(folder/'record.json',r)
    selected=[];seen_groups=set();positions={};shortlist={}
    for r in sorted(valid,key=lambda r:(not r['image_evidence']['label_agreement'],-r['image_evidence']['score'],r['id'])):
        row=coords[r['frame_key']];pos=np.array([float(row['local_'+k]) for k in 'xy']);survey=r['survey']
        group=r.get('tree_group',r['id']);near=any(k!=r['frame_key'] and np.linalg.norm(pos-p)<15 for k,p in positions.get(survey,[]))
        if group in seen_groups:reason='alternate_same_tree_group'
        elif near:reason='alternate_view_within_15m'
        else:
            reason='shortlist_not_visually_checked';selected.append(r);seen_groups.add(group)
            positions.setdefault(survey,[]).append((r['frame_key'],pos))
        shortlist[r['id']]={'disposition':reason,'training_eligible':False}
    write(OUT/'automated_shortlist.json',shortlist)
    write(OUT/'manifest.json',rows)
    protected=read(OUT/'protected_reviews.json')
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in protected.items())
    write(OUT/'verification.json',{'verified_assets':len(verification),'asset_checks':verification,
            'protected_review_files_unchanged':len(protected),'new_auto_acceptances':0})
    preview=OUT/'precheck';preview.mkdir(exist_ok=True)
    for name in NAMES:
        subset=[r for r in selected if r['species']==name]
        for start in range(0,len(subset),6):
            batch=subset[start:start+6];sheet=Image.new('RGB',(1200,len(batch)*290),'white');draw=ImageDraw.Draw(sheet)
            for j,r in enumerate(batch):
                y=j*290;draw.text((8,y+4),r['id']+'  model='+r['image_evidence']['top1'][:30]+f" {r['image_evidence']['score']:.2f}",fill='black')
                for x,file in ((0,'projection.jpg'),(660,'point_views.jpg')):
                    im=Image.open(OUT/'instances'/r['id']/file);im.thumbnail((650 if x==0 else 530,255));sheet.paste(im,(x,y+25))
            sheet.save(preview/(name.split()[0]+'_'+str(start//6)+'.jpg'),quality=92)
    write(OUT/'delivery_status.json',{'effective_annotations':len(rows),'geometry_candidates':len(valid),
        'candidate_counts':dict(Counter(r['species'] for r in valid)),
        'automated_shortlist_counts':dict(Counter(r['species'] for r in selected)),
        'visual_qa_complete':False,'trained_new_model':False})
    print('Verified',len(valid),'assets; automatic shortlist',len(selected),flush=True)

if __name__=='__main__':main()
