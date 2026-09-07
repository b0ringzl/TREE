"""WHU PTv2 + YOLO transfer heads; grouped, buffered weak-label validation.

Default requires accepted paired review. Explicit --experimental-weak-labels
creates an isolated experiment, never replaces deployed model weights.
"""
import argparse,hashlib,json,sys
from pathlib import Path
from collections import Counter
import numpy as np
import torch
import joblib
from PIL import Image
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score,balanced_accuracy_score,classification_report,confusion_matrix
from scipy.spatial import cKDTree
from build_hk_lidar_pairs import read,write,TARGETS

WORK=Path(__file__).resolve().parents[2];sys.path.insert(0,str(WORK/'tree_species_demo'))
from hk_three_species import Features,normalize,NAMES,IMAGE,POINT
DATA=WORK/'hk-tree-vmms/derived/hk_dualsource_three_species_20260907'

def main():
    p=argparse.ArgumentParser();p.add_argument('--experimental-weak-labels',action='store_true');a=p.parse_args()
    out=DATA/('experiment' if a.experimental_weak_labels else 'accepted_training');out.mkdir(exist_ok=True)
    manifest=read(DATA/'manifest.json');review_path=DATA/'pair_reviews.json';reviews=read(review_path) if review_path.exists() else {}
    candidates=[]
    for r in manifest:
        review=reviews.get(r['id'],{})
        if r['status']!='geometry_candidate_pending_review' or review.get('decision')=='reject':continue
        if r.get('quality',{}).get('sample_unique_points',0)<2048:continue
        if r.get('group_label_conflict') and review.get('decision')!='accept':continue
        if not a.experimental_weak_labels and review.get('decision')!='accept':continue
        candidates.append({**r,'species':review.get('species',r['species'])})
    conflicted={g for g in {r['tree_group'] for r in candidates} if len({r['species'] for r in candidates if r['tree_group']==g})>1}
    candidates=[r for r in candidates if r['tree_group'] not in conflicted]
    # At most two views per inferred tree; prefer structural coverage/real points,
    # never choose based on classifier correctness.
    selected=[]
    for group in sorted({r['tree_group'] for r in candidates}):
        members=[r for r in candidates if r['tree_group']==group]
        selected+=sorted(members,key=lambda r:(-r['quality']['trunk_bin_occupancy'],-r['quality']['sample_unique_points'],-r['quality']['clean_points'],r['id']))[:2]
    if len(selected)<12:raise ValueError('Too few eligible pairs for three-class training')
    y=np.array([TARGETS.index(r['species']) for r in selected]);pos=np.array([r['quality']['axis_local_xy'] for r in selected]);surveys=np.array([r['survey'] for r in selected])
    # Merge close stems conservatively (including nearby same-tree center drift).
    spatial=[]
    for survey in sorted(set(surveys)):
        ids=np.where(surveys==survey)[0];groups=DBSCAN(eps=5,min_samples=1).fit_predict(pos[ids])
        spatial.extend((int(i),survey+'_'+str(g)) for i,g in zip(ids,groups))
    groups=np.array([g for _,g in sorted(spatial)])
    # 25m exceeds twice the largest permitted crown radius (11m), reducing
    # shared-canopy leakage as well as duplicate-view leakage across folds.
    split=None
    for train,test in GroupShuffleSplit(n_splits=100,test_size=.25,random_state=20260907).split(pos,y,groups):
        keep=[]
        for i in train:
            other=test[surveys[test]==surveys[i]]
            if not len(other) or np.linalg.norm(pos[other]-pos[i],axis=1).min()>=25:keep.append(i)
        train=np.array(keep,dtype=int)
        if all((y[train]==c).sum()>=4 and (y[test]==c).sum()>=2 for c in range(3)):
            split=(train,test);break
    if split is None:
        write(out/'blocked.json',{'reason':'insufficient independent buffered groups for all three classes','candidate_counts':dict(Counter(map(int,y))),'groups':len(set(groups))})
        raise ValueError('Insufficient independent groups; refusing train/test leakage')
    train,test=split;device='cuda' if torch.cuda.is_available() else 'cpu';encoder=Features(device);features=[]
    cache=out/'features';cache.mkdir(exist_ok=True)
    for idx,r in enumerate(selected):
        fingerprint=hashlib.sha256(Path(r['point']).read_bytes()+Path(r['image']).read_bytes()).hexdigest()[:16];path=cache/(r['id']+'_'+fingerprint+'.npz')
        if path.exists():
            with np.load(path) as f:iv=f['image'];pv=f['point']
        else:
            with Image.open(r['image']) as im:iv=encoder.encode_image(im.convert('RGB'))
            xyz=np.load(r['point'])['points_xyz'];pv=encoder.encode_point(xyz)
            np.savez_compressed(path,image=iv,point=pv)
        features.append((iv,pv))
        if idx%10==0:print('features',idx+1,'/',len(selected),flush=True)
    xi=normalize(np.array([v[0] for v in features]));xp=normalize(np.array([v[1] for v in features]));xf=normalize(np.c_[xi,xp]);xs={'image':xi,'point':xp,'fusion':xf};heads={};metrics={};references={};limits={}
    for mode,x in xs.items():
        # Fixed regularization, no selection on held-out predictions.
        clf=LogisticRegression(C=5,class_weight='balanced',max_iter=3000,random_state=7)
        counts=Counter(groups[train]);weights=np.array([1/counts[g] for g in groups[train]])
        clf.fit(x[train],y[train],sample_weight=weights);prediction=clf.predict(x[test]);heads[mode]=clf
        metrics[mode]={'accuracy':float(accuracy_score(y[test],prediction)),'balanced_accuracy':float(balanced_accuracy_score(y[test],prediction)),'per_class':classification_report(y[test],prediction,labels=[0,1,2],target_names=NAMES,output_dict=True,zero_division=0),'confusion_matrix':confusion_matrix(y[test],prediction,labels=[0,1,2]).tolist()}
        references[mode]=x[train];similarity=x[train]@x[train].T;np.fill_diagonal(similarity,-1)
        limits[mode]=float(max(.03,min(.7,np.quantile(1-similarity.max(1),.95)*1.5)))
    assignments=[{'id':r['id'],'species':r['species'],'tree_group':groups[i],'split':'train' if i in train else 'heldout' if i in test else 'buffer_excluded','annotation_provenance':r['annotation_provenance'],'point':r['point'],'image':r['image']} for i,r in enumerate(selected)]
    bundle={'version':1,'classes':NAMES,'heads':heads,'references':references,'distance_limits':limits,'experimental':True,'label_policy':'human-image-labelled / unaccepted-3D-transfer' if a.experimental_weak_labels else 'accepted-paired-review','image_encoder':str(IMAGE),'point_encoder':str(POINT),'point_sampling':2048}
    joblib.dump(bundle,out/'model.joblib')
    report={'status':'experimental_only','supervision':bundle['label_policy'],'encoder_training':'frozen YOLO image encoder and frozen WHU PTv2; trained three regularized classification/fusion heads, not end-to-end fine-tuning','classes':NAMES,'selected_pairs':len(selected),'train_pairs':len(train),'heldout_pairs':len(test),'buffer_excluded':len(selected)-len(train)-len(test),'train_groups':len(set(groups[train])),'heldout_groups':len(set(groups[test])),'metrics_scope':'Agreement with provisional image-transfer labels on group-disjoint 25m-buffered holdout, NOT human-accepted real-world accuracy','metrics':metrics,'promotion':'not promoted; existing Wuhan and general HK weights unchanged'}
    write(out/'report.json',report);write(out/'split.json',assignments);print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
