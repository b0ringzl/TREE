"""Auditable five-class pilot: latest accepted assets, buffered spatial holdout.
Frozen YOLO/WHU PTv2 encoders; experimental heads only, no demo promotion.
"""
import argparse,hashlib,sys
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.spatial import cKDTree
from sklearn.model_selection import GroupShuffleSplit
from build_hk_lidar_pairs import read,write
from build_hk_dense_dataset import coordinates
from build_hk_five_species_pairs import NAMES

WORK=Path(__file__).resolve().parents[2];sys.path.insert(0,str(WORK/'tree_species_demo'))
import albizia_conflict_api as conflict
import leucaena_recovery_api as recovery
BASE=WORK/'hk-tree-vmms/derived';OLD=BASE/'hk_dualsource_five_species_20260907'
DEL=BASE/'delonix_refined_v1_20260907';OUT=BASE/'reviewed_five_species_pilot_v1_20260907'
CLASSES=list(NAMES)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def ensure_sources():
    for path,digest in read(OUT/'protected_sources.json').items():
        if sha(path)!=digest:raise ValueError('Source changed during experiment: '+path)

def prepare():
    if OUT.exists():raise ValueError('Preserve existing revision; use --stage train to resume feature extraction only.')
    if recovery.stale():raise ValueError('Recovery source adjudication changed')
    initial=conflict.candidates()['items'];old={r['id']:r for r in read(OLD/'manifest.json')};rows=coordinates()
    paths=[OLD/'manifest.json',OLD/'reviews.json',conflict.ROOT/'manifest.json',conflict.ROOT/'adjudications.json',
           recovery.ROOT/'manifest.json',recovery.ROOT/'reviews.json',DEL/'manifest.json',DEL/'reviews.json']
    OUT.mkdir();protected={str(p):sha(p) for p in paths};write(OUT/'protected_sources.json',protected)
    merged={};history=[]
    for r in initial:
        o=old[r['id']];merged[r['id']]={**o,**r,'source_version':OLD.name,'asset_dir':str(OLD/'instances'/r['id']),
          'identity_group':r.get('split_group',r['tree_group']),'coverage_scope':'accepted_pair_completeness_not_verified'}
    for root in [DEL,recovery.ROOT]:
        reviews=read(root/'reviews.json') if (root/'reviews.json').exists() else {}
        manifest={r['id']:r for r in read(root/'manifest.json')}
        for ident,review in reviews.items():
            r=manifest[ident]
            # A rejection of a new asset does not revoke an older accepted version.
            if review['decision']!='accept':continue
            if r['status']!='candidate_pending_review':raise ValueError('Accepted review lacks an asset: '+ident)
            if ident in merged:history.append({'id':ident,'old_version':merged[ident]['source_version'],'accepted_replacement':root.name})
            original=old.get(ident,{})
            merged[ident]={**r,'species':review['species'],'source_version':root.name,'accepted_review':review,
                'asset_dir':str(root/'instances'/ident),'image':str(root/'instances'/ident/'image.jpg'),
                'point':str(root/'instances'/ident/'point.npz'),
                'identity_group':review.get('split_group') or original.get('tree_group') or ident,
                'coverage_scope':review.get('coverage_scope','accepted_refined_pair_completeness_not_verified')}
    records=sorted(merged.values(),key=lambda r:r['id']);checks=[]
    for r in records:
        row=rows[r['frame_key']];r['survey']=Path(row['source_image_relpath']).parts[0]
        r['class_index']=CLASSES.index(r['species']);r['training_eligible']=False
        r['image_sha256']=sha(r['image']);r['point_sha256']=sha(r['point'])
        with np.load(r['point'],allow_pickle=False) as p:
            xyz=p['points_xyz'];assert xyz.ndim==2 and xyz.shape[1]==3 and 8<=len(xyz)<=2048 and np.isfinite(xyz).all()
            assert len(np.unique(xyz,axis=0))==len(xyz)
            checks.append({'id':r['id'],'point_count':len(xyz),'finite_unique':True})
    # Leakage blocks are conservative partitions, NOT assertions of physical identity.
    parent=list(range(len(records)))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    def union(i,j):parent[find(j)]=find(i)
    clouds=[np.load(Path(r['asset_dir'])/'cleaned_local.npz',allow_pickle=False)['xyz'] for r in records]
    boxes=[(p.min(0),p.max(0)) for p in clouds];links=[]
    for i,a in enumerate(records):
        for j in range(i+1,len(records)):
            b=records[j]
            if a['survey']!=b['survey']:continue
            distance=float(np.linalg.norm(np.array(a['quality']['axis_local_xy'])-b['quality']['axis_local_xy']))
            why=[]
            if a['identity_group']==b['identity_group']:why.append('shared_identity_or_original_algorithm_group')
            if distance<=5:why.append('conservative_5m_axis_buffer')
            lo1,hi1=boxes[i];lo2,hi2=boxes[j]
            if np.all(lo1<=hi2+.2) and np.all(lo2<=hi1+.2):
                ab=float((cKDTree(clouds[j]).query(clouds[i])[0]<.2).mean());ba=float((cKDTree(clouds[i]).query(clouds[j])[0]<.2).mean())
                if max(ab,ba)>.2:why.append('shared_measured_support_20pct_at_20cm')
            else:ab=ba=0.
            if why:union(i,j);links.append({'a':a['id'],'b':b['id'],'reasons':why,'axis_distance_m':distance,'overlap_a_to_b':ab,'overlap_b_to_a':ba,'identity_verified':False})
    for i,r in enumerate(records):r['leakage_group']='block_'+str(find(i))
    # Cap correlated views at two per species/block without using model predictions.
    selected=[]
    for key in sorted({(r['species'],r['leakage_group']) for r in records}):
        group=[r for r in records if (r['species'],r['leakage_group'])==key]
        selected+=sorted(group,key=lambda r:(-r['quality']['clean_points'],r['id']))[:2]
    selected.sort(key=lambda r:r['id']);y=np.array([r['class_index'] for r in selected]);groups=np.array([r['leakage_group'] for r in selected]);surveys=np.array([r['survey'] for r in selected]);pos=np.array([r['quality']['axis_local_xy'] for r in selected])
    support={c:len(set(groups[y==c])) for c in range(len(CLASSES))};evaluated=[c for c,n in support.items() if n>=3];rare=set(range(len(CLASSES)))-set(evaluated)
    split=None
    if evaluated:
        # Test grouping chosen using labels/geometry only, never scores; fixed seed.
        for train,test in GroupShuffleSplit(n_splits=3000,test_size=.22,random_state=20260907).split(pos,y,groups):
            if set(y[test])&rare or any(not (y[test]==c).any() for c in evaluated):continue
            keep=[i for i in train if all(surveys[i]!=surveys[j] or np.linalg.norm(pos[i]-pos[j])>=25 for j in test)]
            train=np.array(keep,int)
            if any(len(set(groups[train][y[train]==c]))<(2 if c in evaluated else 1) for c in range(len(CLASSES))):continue
            split=(train,test);break
    if split is None:
        train=np.arange(len(selected));test=np.array([],int);evaluated=[]
    else:train,test=split
    for i,r in enumerate(selected):
        r['split']='train' if i in train else 'heldout' if i in test else 'buffer_excluded'
        r['training_eligible']=r['split']=='train'
    assert not (set(groups[train])&set(groups[test]))
    if len(test):assert all(surveys[i]!=surveys[j] or np.linalg.norm(pos[i]-pos[j])>=25 for i in train for j in test)
    write(OUT/'accepted_manifest.json',records);write(OUT/'accepted_asset_replacements.json',history);write(OUT/'asset_checks.json',checks)
    write(OUT/'leakage_links.json',links);write(OUT/'split.json',selected)
    plan={'accepted_views':len(records),'accepted_by_class':dict(Counter(r['species'] for r in records)),
          'selected_views':len(selected),'split_counts':dict(Counter(r['split'] for r in selected)),
          'conservative_blocks_by_class':{CLASSES[c]:n for c,n in support.items()},
          'evaluated_class_indices':evaluated,'not_evaluable_classes':[CLASSES[c] for c in range(len(CLASSES)) if c not in evaluated],
          'split_policy':'5m/point-overlap leakage blocks; max 2 views per species/block; 25m train-holdout axis buffer within each survey; fixed preprediction seed.',
          'holdout_scope':'Small selected accepted dataset; no independent operational accuracy claim. Non-heldout classes cannot be evaluated.',
          'no_holdout_reason':None if len(test) else 'No class-complete eligible buffered split; fit-only pilot, no accuracy estimate.'}
    write(OUT/'plan.json',plan);ensure_sources();print(plan,flush=True)

def train():
    import torch,joblib
    from PIL import Image
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
    from hk_three_species import Features,normalize,IMAGE,POINT
    ensure_sources();records=read(OUT/'split.json');plan=read(OUT/'plan.json')
    if (OUT/'model.joblib').exists():raise ValueError('Existing experimental model preserved; choose a new revision.')
    model_hashes={str(p):sha(p) for p in [IMAGE,POINT]};write(OUT/'encoder_hashes.json',model_hashes)
    cache=OUT/'features';cache.mkdir(exist_ok=True);encoder=Features('cuda' if torch.cuda.is_available() else 'cpu');encoded=[]
    for n,r in enumerate(records):
        if sha(r['image'])!=r['image_sha256'] or sha(r['point'])!=r['point_sha256']:raise ValueError('Asset changed: '+r['id'])
        fingerprint=hashlib.sha256((r['image_sha256']+r['point_sha256']+''.join(model_hashes.values())).encode()).hexdigest()[:16]
        p=cache/(r['id']+'_'+fingerprint+'.npz')
        if p.exists():
            with np.load(p,allow_pickle=False) as data:iv=data['image'];pv=data['point']
        else:
            with Image.open(r['image']) as im:iv=encoder.encode_image(im.convert('RGB'))
            with np.load(r['point'],allow_pickle=False) as data:pv=encoder.encode_point(data['points_xyz'])
            assert np.isfinite(iv).all() and np.isfinite(pv).all()
            np.savez_compressed(p,image=iv,point=pv)
        encoded.append((iv,pv));print('features',n+1,'/',len(records),r['id'],flush=True)
    xi=normalize(np.array([v[0] for v in encoded]));xp=normalize(np.array([v[1] for v in encoded]));xf=normalize(np.c_[xi,xp])
    y=np.array([r['class_index'] for r in records]);train=np.array([i for i,r in enumerate(records) if r['split']=='train']);test=np.array([i for i,r in enumerate(records) if r['split']=='heldout'],int)
    block_counts=Counter((r['species'],r['leakage_group']) for r in records if r['split']=='train')
    weights=np.array([1/block_counts[(records[i]['species'],records[i]['leakage_group'])] for i in train]);heads={};metrics={};predictions=[]
    for mode,x in {'image':xi,'point':xp,'fusion':xf}.items():
        head=LogisticRegression(C=5,class_weight='balanced',max_iter=3000,random_state=7);head.fit(x[train],y[train],sample_weight=weights);heads[mode]=head
        if len(test):
            pred=head.predict(x[test]);prob=head.predict_proba(x[test]);evaluated=plan['evaluated_class_indices']
            full=classification_report(y[test],pred,labels=list(range(len(CLASSES))),target_names=CLASSES,output_dict=True,zero_division=0)
            metrics[mode]={'subset_accuracy':float(accuracy_score(y[test],pred)),
                'macro_f1_evaluated_classes_only':float(np.mean([full[CLASSES[c]]['f1-score'] for c in evaluated])),
                'per_evaluated_class':{CLASSES[c]:full[CLASSES[c]] for c in evaluated},
                'not_evaluated_classes':plan['not_evaluable_classes'],
                'confusion_matrix_5_columns':confusion_matrix(y[test],pred,labels=list(range(len(CLASSES)))).tolist(),
                'warning':'Metrics cover only classes with holdout support, NOT five-class overall or real-world accuracy.'}
            predictions += [{'id':records[i]['id'],'mode':mode,'truth':CLASSES[y[i]],'prediction':CLASSES[pred[j]],'score':float(prob[j].max()),'score_calibrated':False} for j,i in enumerate(test)]
    ensure_sources();assert all(sha(p)==h for p,h in model_hashes.items())
    bundle={'version':1,'classes':CLASSES,'chinese_names':[NAMES[c] for c in CLASSES],'heads':heads,'experimental':True,'deployed':False,
        'image_encoder':str(IMAGE),'point_encoder':str(POINT),'encoder_hashes':model_hashes,'feature_normalization':'L2 each source; L2 concatenated image+point',
        'not_evaluable_classes':plan['not_evaluable_classes'],'calibrated':False,'unknown_rejection_validated':False}
    joblib.dump(bundle,OUT/'model.joblib');write(OUT/'heldout_predictions.json',predictions)
    report={'status':'experimental_only_not_deployed','encoder_training':'Frozen YOLO and WHU PTv2; trained 3 regularized classification heads, not end-to-end fine-tuning.',
            'plan':plan,'metrics':metrics,'formal_five_class_accuracy':None,'production_model_changed':False,
            'limitations':['Small reviewed convenience sample','Leakage groups except explicit adjudication are conservative inferred blocks','Silver-leadtree samples share one local scene','Partial crowns are not complete-tree or height truth','No calibrated confidence or validated unknown rejection']}
    write(OUT/'report.json',report);print(report,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['prepare','train','all'],default='all');args=parser.parse_args()
    if args.stage in ('prepare','all'):prepare()
    if args.stage in ('train','all'):train()
