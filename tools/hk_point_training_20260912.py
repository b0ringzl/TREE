"""Versioned, source-preserving point-only experiment. No deployment writes."""
import argparse, json, sys, hashlib, shutil
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from training_safety import sha256, backup_before_training, deployed_artifacts, verify_backup
from rebuild_banyan_coverage_preview import sample_ids, orthographic
OUT=ROOT/'outputs/hk_point_training_20260912_v1'
CAT=ROOT/'outputs/hk_point_assist_20260911_v1/catalog.json'
REV=CAT.with_name('geometry_reviews.json')
SEED=20260912
NAMES={'Ficus microcarpa':'榕樹（細葉榕）','Livistona chinensis':'蒲葵','Wodyetia bifurcata':'狐尾椰子','Delonix regia':'鳳凰木','Aleurites moluccana':'石栗','Leucaena leucocephala':'銀合歡','Albizia lebbeck':'大葉合歡','Araucaria columnaris(Araucaria heterophylla)':'南洋杉（原標籤待細分）'}
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v): Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def clean(x):
    x=np.asarray(x,dtype=np.float64); total=len(x); finite=np.isfinite(x).all(1); x=np.unique(x[finite],axis=0); unique=len(x)
    # Conservative sparse-outlier removal; never discard merely sparse trunks.
    d=cKDTree(x).query(x,k=min(9,len(x)),workers=4)[0][:,-1]
    med=float(np.median(d)); mad=float(np.median(abs(d-med)))
    threshold=max(1.,med+12*1.4826*mad)
    keep=d<=threshold; removed=int((~keep).sum()); x=x[keep]
    ids=sample_ids(x,2048); metric=x[ids]; center=(x.min(0)+x.max(0))/2; scale=max(np.linalg.norm(x-center,axis=1).max(),1e-8)
    return x,metric,(metric-center)/scale,dict(input_points=total,finite_points=int(finite.sum()),unique_points=unique,isolated_removed=removed,threshold_m=threshold,clean_points=len(x),sample_points=len(ids),extent_m=np.ptp(x,axis=0).tolist(),centroid_xyz=center.tolist(),scale=float(scale),whole_tree_verified=False,ground_removed_this_run=False)
def save_cloud(folder,x):
    folder.mkdir(parents=True,exist_ok=False); x,m,p,q=clean(x)
    np.savez_compressed(folder/'cleaned.npz',xyz=x.astype(np.float32))
    np.savez_compressed(folder/'point.npz',points_xyz=p.astype(np.float32),metric_xyz=m.astype(np.float32),centroid_xyz=q['centroid_xyz'],scale=q['scale'])
    orthographic(x,sample_ids(x,6000),folder/'views.jpg',float(np.quantile(x[:,2],.25)))
    write(folder/'quality.json',q)
    return x,q
def prepare():
    OUT.mkdir(exist_ok=False); rows=read(CAT); reviews=read(REV); write(OUT/'accepted_catalog_snapshot.json',rows); write(OUT/'geometry_reviews_snapshot.json',reviews)
    records=[]; clouds=[]; trees=[]
    for i,r in enumerate(rows):
        assert r['current_accepted'] and r['review_decision']=='accept'
        for kind in ['cleaned','point','image']:
            assert sha256(r['asset_paths'][kind])==r['sha256'][kind],r['id']+kind
        folder=OUT/'data'/r['key']; x=np.load(r['asset_paths']['cleaned'])['xyz']; x,q=save_cloud(folder,x)
        rv=reviews.get(r['key'],{}); reasons=[]
        if rv and (rv.get('same_tree') is False or rv.get('isolated_tree') is False):reasons.append('latest_geometry_review_does_not_confirm_same_isolated_tree')
        if q['sample_points']<512:reasons.append('fewer_than_512_real_points')
        records.append(dict(key=r['key'],id=r['id'],species=r['species'],chinese=NAMES[r['species']],survey=r['survey'],axis=r['quality']['axis_local_xy'],provisional_tree_group=r['provisional_tree_group'],source_assets=r['asset_paths'],source_sha256=r['sha256'],point=str(folder/'point.npz'),cleaned=str(folder/'cleaned.npz'),views=str(folder/'views.jpg'),quality=q,label_evidence='explicit_accepted_pair_review',review=r['review'],exclusion_reasons=reasons,legacy_group_conflict_flag=r['group_label_conflict']))
        # Fixed real subset for overlap, not a learned selection.
        a=x[sample_ids(x,5000)]; clouds.append(a);trees.append(cKDTree(a))
        print('cleaned accepted',i+1,len(rows),r['id'],flush=True)
    parent=list(range(len(records)))
    def find(a):
        while parent[a]!=a:parent[a]=parent[parent[a]];a=parent[a]
        return a
    links=[]
    for i,a in enumerate(records):
        for j in range(i):
            b=records[j]
            if a['survey']!=b['survey']:continue
            distance=float(np.linalg.norm(np.array(a['axis'])-b['axis']))
            if distance>35:continue
            # Dense original cloud membership avoids false non-overlap from two independent sparse samples.
            xa=np.load(a['cleaned'])['xyz']; xb=np.load(b['cleaned'])['xyz']
            ab=float((cKDTree(xb).query(clouds[i],workers=4)[0]<.2).mean()); ba=float((cKDTree(xa).query(clouds[j],workers=4)[0]<.2).mean())
            same=a['provisional_tree_group']==b['provisional_tree_group'] or distance<5 or max(ab,ba)>=.2
            if same:
                parent[find(i)]=find(j);links.append(dict(a=a['key'],b=b['key'],axis_distance_m=distance,overlap_ab=ab,overlap_ba=ba))
    groups=defaultdict(list)
    for i,r in enumerate(records):r['group']='block_'+str(find(i));groups[r['group']].append(r)
    for group,rs in groups.items():
        if len({r['species'] for r in rs})>1:
            for r in rs:r['exclusion_reasons'].append('mixed_species_conservative_overlap_block')
    # Audit new unlabeled LAS. Match only exact original coordinates, not approximate nearby trees.
    import laspy
    new=read(ROOT/'reports/hk_data_inventory_20260912_v1/content_audit_v2.json')['single_tree_content']; associations=[]
    for n,s in enumerate(new):
        assert sha256(s['path'])==s['sha256']; las=laspy.read(s['path']); x=np.c_[las.x,las.y,las.z]; folder=OUT/'new_las'/(s['block']+'_'+s['id']);x,q=save_cloud(folder,x)
        sample=x[sample_ids(x,5000)]; hits=[]
        for r in records:
            if np.linalg.norm(np.array(r['axis'])-np.median(x[:,:2],axis=0))>30:continue
            old=np.load(r['source_assets']['cleaned'])['xyz']; nearest=cKDTree(old).query(sample,workers=4)[0]
            coverage=float((nearest<.005).mean())
            if coverage>.05:hits.append(dict(key=r['key'],species=r['species'],group=r['group'],exact_coordinate_fraction=coverage))
        hits.sort(key=lambda h:-h['exact_coordinate_fraction'])
        # Correspondence evidence is recorded, but automated label transfer is not treated as human truth.
        associations.append(dict(id=s['block']+'_'+s['id'],source=s['path'],sha256=s['sha256'],quality=q,matches=hits[:8],status='needs_image_and_instance_confirmation',training_eligible=False,point=str(folder/'point.npz'),views=str(folder/'views.jpg')))
        print('new LAS',n+1,len(new),associations[-1]['id'],hits[:1],flush=True)
    write(OUT/'new_las_associations.json',associations);write(OUT/'leakage_links.json',links)
    # Cap repeated views. Exclusions are predetermined geometry/review rules, not correctness.
    selected=[]
    for group,rs in sorted(groups.items()):
        eligible=[r for r in rs if not r['exclusion_reasons']]
        chosen=sorted(eligible,key=lambda r:(-r['quality']['clean_points'],r['key']))[:2]
        for r in eligible:
            if r not in chosen:r['exclusion_reasons'].append('repeated_view_cap_two_per_block')
        selected+=chosen
    selected.sort(key=lambda r:r['key']); classes=sorted({r['species'] for r in selected}); support={s:len({r['group'] for r in selected if r['species']==s}) for s in classes}
    eligible_classes=[s for s in classes if support[s]>=4]; rare=set(classes)-set(eligible_classes)
    rng=np.random.default_rng(SEED); assignment=None
    # Holdout by trees, 10 m spatial separation. Fixed labels/geometry-only search.
    for attempt in range(10000):
        val=set();test=set()
        for s in eligible_classes:
            gs=sorted({r['group'] for r in selected if r['species']==s});order=rng.permutation(gs);val.add(order[0]);test.add(order[1])
        for r in selected:r['split']='validation' if r['group'] in val else 'test_regression' if r['group'] in test else 'train'
        holds=[r for r in selected if r['split']!='train']
        for r in selected:
            if r['split']=='train' and any(r['survey']==h['survey'] and np.linalg.norm(np.array(r['axis'])-h['axis'])<10 for h in holds):r['split']='spatial_buffer_excluded'
        if all(len({r['group'] for r in selected if r['species']==s and r['split']=='train'})>=(2 if s in eligible_classes else 1) for s in classes):
            assignment=attempt;break
    if assignment is None:raise RuntimeError('No defensible split; no fitting allowed')
    for r in records:
        if r not in selected:r['split']='quarantine'
        r['class_index']=classes.index(r['species']) if r['species'] in classes else None
    plan=dict(seed=SEED,classes=classes,names={s:NAMES[s] for s in classes},group_support=support,evaluated_classes=eligible_classes,train_only_classes=sorted(rare),split_counts=dict(Counter(r['split'] for r in records)),selected_attempt=assignment,split_policy='same survey; 5 m axes / >=20% overlap at 20 cm / original group union; mixed-label blocks excluded; max 2 views/block; >=10m training-holdout buffer; one validation and one test block per class with >=4 blocks',test_scope='Previously reviewed convenience samples: regression evidence only, not fresh unbiased field accuracy',cleaning_policy='finite + exact deduplication + conservative isolated-point rejection; height/azimuth 2048 real point subset; no invented roots/crowns and no unverified ground stripping',new_las_training=0,source_hashes={str(CAT):sha256(CAT),str(REV):sha256(REV)})
    write(OUT/'manifest.json',records);write(OUT/'plan.json',plan)
    print(json.dumps(plan,ensure_ascii=False,indent=2),flush=True)
    from PIL import Image,ImageDraw,ImageOps
    for page in range((len(records)+14)//15):
        sheet=Image.new('RGB',(1500,1200),'white');draw=ImageDraw.Draw(sheet)
        for cell,r in enumerate(records[page*15:(page+1)*15]):
            x=(cell%5)*300;y=(cell//5)*400
            with Image.open(r['source_assets']['image']) as im:sheet.paste(ImageOps.contain(im.convert('RGB'),(290,320)),(x,y))
            draw.text((x+4,y+323),str(page*15+cell)+' '+r['key'][:8]+'\n'+r['species'][:39]+'\n'+r['split'],fill='black')
        sheet.save(OUT/f'label_contact_{page+1}.jpg')
def descriptors(metric):
    x=np.asarray(metric); ext=np.ptp(x,axis=0); center=np.median(x,axis=0); radius=np.linalg.norm(x[:,:2]-center[:2],axis=1); h=(x[:,2]-x[:,2].min())/max(ext[2],1e-8)
    vals=list(np.log1p(ext))+[float(ext[2]/max(ext[:2].max(),.1))]
    for a,b in [(0,.2),(.2,.4),(.4,.6),(.6,.8),(.8,1.001)]:
        ids=(h>=a)&(h<b);v=radius[ids];vals.extend([float(ids.mean()),float(np.quantile(v,.5)) if len(v) else 0.,float(np.quantile(v,.95)) if len(v) else 0.])
    return np.array(vals,dtype=np.float32)
def features():
    import torch
    sys.path.insert(0,str(ROOT/'tree_species_demo'))
    from hk_three_species import Features,POINT
    torch.set_num_threads(4);encoder=Features('cuda' if torch.cuda.is_available() else 'cpu');rows=read(OUT/'manifest.json');dest=OUT/'features';dest.mkdir(exist_ok=False)
    hashes={str(POINT):sha256(POINT)};feats=[];geoms=[];keys=[]
    for n,r in enumerate(rows):
        if r['split'] not in ['train','validation','test_regression']:continue
        with np.load(r['point']) as f:points=f['points_xyz'];metric=f['metric_xyz']
        encoded=encoder.encode_point(points);geom=descriptors(metric);assert np.isfinite(encoded).all();feats.append(encoded);geoms.append(geom);keys.append(r['key'])
        print('point features',n+1,len(rows),r['key'],flush=True)
    np.savez_compressed(dest/'all.npz',point=np.array(feats),geometry=np.array(geoms),keys=np.array(keys))
    write(dest/'provenance.json',dict(encoder_hashes=hashes,manifest_sha256=sha256(OUT/'manifest.json'),point_hashes={r['key']:sha256(r['point']) for r in rows if r['key'] in keys},images_used_for_features=False))
def metrics(y,p,rs,classes):
    from sklearn.metrics import classification_report,confusion_matrix
    pred=p.argmax(1);support=sorted(set(y.tolist()));report=classification_report(y,pred,labels=list(range(len(classes))),target_names=classes,output_dict=True,zero_division=0)
    groups=sorted({r['group'] for r in rs});gy=[];gp=[]
    for g in groups:
        ids=[i for i,r in enumerate(rs) if r['group']==g];gy.append(int(y[ids[0]]));gp.append(int(p[ids].mean(0).argmax()))
    return dict(n_views=len(y),correct_views=int((y==pred).sum()),view_accuracy=float((y==pred).mean()),n_groups=len(groups),correct_groups=int((np.array(gy)==gp).sum()),group_accuracy=float((np.array(gy)==gp).mean()),macro_f1_supported_only=float(np.mean([report[classes[c]]['f1-score'] for c in support])),per_class=report,confusion_matrix=confusion_matrix(y,pred,labels=list(range(len(classes)))).tolist(),not_evaluated_classes=[classes[i] for i in range(len(classes)) if i not in support])
def train():
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from hk_point_model import CentroidHead
    plan=read(OUT/'plan.json');allrows=read(OUT/'manifest.json');lookup={r['key']:r for r in allrows};f=np.load(OUT/'features/all.npz');rows=[lookup[k] for k in f['keys']];raw=f['point'];x=raw/np.maximum(np.linalg.norm(raw,axis=1,keepdims=True),1e-8);geo=f['geometry'];classes=plan['classes'];y=np.array([r['class_index'] for r in rows]);train=np.array([i for i,r in enumerate(rows) if r['split']=='train']);val=np.array([i for i,r in enumerate(rows) if r['split']=='validation']);test=np.array([i for i,r in enumerate(rows) if r['split']=='test_regression'])
    provenance=read(OUT/'features/provenance.json')
    assert provenance['manifest_sha256']==sha256(OUT/'manifest.json')
    for path,h in provenance['encoder_hashes'].items():assert sha256(path)==h
    for r in rows:assert sha256(r['point'])==provenance['point_hashes'][r['key']]
    configs=[dict(name='00_centroid_baseline',kind='centroid',feature='point'),dict(name='01_point_linear_c1',kind='linear',feature='point',C=1.),dict(name='02_point_linear_c10',kind='linear',feature='point',C=10.),dict(name='03_point_geometry_c1',kind='linear',feature='point_geometry',C=1.)]
    runs=[];baseline=None
    for cfg in configs:
        run=OUT/'runs'/cfg['name'];run.mkdir(parents=True,exist_ok=False)
        cfg.update(seed=SEED,classes=classes,point_encoder=list(provenance['encoder_hashes'])[0],selection='validation macro F1; then validation group accuracy; then earlier simpler run',images_used_for_model=False,backbone='frozen WHU PointTransformerV2; new point-only classification head',training_examples=len(train))
        write(run/'config.json',cfg)
        # Explicit zero initialization snapshot for head parameters (sklearn optimizes from zeros).
        dims=x.shape[1]+(geo.shape[1] if cfg['feature']=='point_geometry' else 0)
        np.savez_compressed(run/'initial_head.npz',coef=np.zeros((len(classes),dims)),intercept=np.zeros(len(classes)))
        write(run/'original_evaluation.json',baseline or dict(baseline_status='No compatible eight-label deployed head; this run fits the fixed-feature centroid baseline',existing_published_point_head_5class_regression_accuracy=.4,warning='Historical 4/10 is a different split and is not used as this experiment baseline'))
        inputs=deployed_artifacts()+[run/'initial_head.npz',run/'config.json',run/'original_evaluation.json',OUT/'manifest.json',OUT/'plan.json',OUT/'accepted_catalog_snapshot.json',OUT/'geometry_reviews_snapshot.json',OUT/'features/provenance.json',Path(__file__),ROOT/'tools/hk_point_model.py']
        backup=backup_before_training(run,inputs,dict(training=cfg['name'],deployed_weights_unchanged=True));print('BACKUP VERIFIED',backup,flush=True)
        # Every learned preprocessing transform is fit only after the backup and on train only.
        scaler=None;z=x
        if cfg['feature']=='point_geometry':
            scaler=StandardScaler().fit(geo[train]);z=np.c_[x,.15*scaler.transform(geo)]
        counts=Counter(r['group'] for r in rows if r['split']=='train');w=np.array([1/counts[rows[i]['group']] for i in train])
        head=CentroidHead() if cfg['kind']=='centroid' else LogisticRegression(C=cfg['C'],class_weight='balanced',max_iter=5000,random_state=SEED)
        head.fit(z[train],y[train],sample_weight=w)
        bundle=dict(head=head,scaler=scaler,feature=cfg['feature'],classes=classes,names=plan['names'],encoder=cfg['point_encoder'],encoder_sha256=provenance['encoder_hashes'][cfg['point_encoder']],experimental=True,deployed=False,calibrated=False,geometry_weight=.15,manifest_sha256=sha256(OUT/'manifest.json'))
        joblib.dump(bundle,run/'model.joblib');v=head.predict_proba(z[val]);t=head.predict_proba(z[train]);result=dict(config=cfg,run=str(run),validation=metrics(y[val],v,[rows[i] for i in val],classes),training=metrics(y[train],t,[rows[i] for i in train],classes))
        write(run/'validation.json',result);verify_backup(backup);runs.append((result,bundle,z))
        if baseline is None:baseline=result
        print(cfg['name'],'validation',result['validation']['view_accuracy'],result['validation']['macro_f1_supported_only'],flush=True)
    selected=max(range(len(runs)),key=lambda i:(runs[i][0]['validation']['macro_f1_supported_only'],runs[i][0]['validation']['group_accuracy'],-i))
    # Seal model choice before evaluating any test prediction.
    selection=dict(selected_run=runs[selected][0]['run'],selected_index=selected,selection_metric='validation macro_f1_supported_only, then group_accuracy, then earlier run',test_used_for_selection=False)
    write(OUT/'selection.json',selection);predictions=[];testresults={}
    for idx in sorted({0,selected}):
        result,bundle,z=runs[idx];p=bundle['head'].predict_proba(z[test]);testresults[result['config']['name']]=metrics(y[test],p,[rows[i] for i in test],classes)
    for split,ids in [('train',train),('validation',val),('test_regression',test)]:
        bundle=runs[selected][1];z=runs[selected][2];p=bundle['head'].predict_proba(z[ids]);pb=runs[0][1]['head'].predict_proba(runs[0][2][ids])
        for j,i in enumerate(ids):
            order=np.argsort(-p[j]);predictions.append(dict(key=rows[i]['key'],split=split,truth=classes[y[i]],prediction=classes[order[0]],correct=bool(order[0]==y[i]),baseline_prediction=classes[pb[j].argmax()],baseline_correct=bool(pb[j].argmax()==y[i]),top3=[dict(species=classes[k],score=float(p[j,k])) for k in order[:3]]))
    report=dict(status='completed_experimental_not_deployed',plan=plan,selection=selection,runs=[r[0] for r in runs],test_results=testresults,images_used_for_training=False,labels_source='accepted paired image review, SHA-verified; no model-generated replacement labels',limitations=['Small convenience regression set: only 5 test tree blocks; no unbiased operational accuracy claim','Eight output labels, only five labels evaluated; three train-only classes lack enough independent groups','Araucaria legacy combined taxon is retained, not confirmed species-level separation','Source clouds remain partial; observed height is not surveyed whole-tree height','Frozen backbone; trained classification heads, not end-to-end PTv2 fine-tuning','New 72 LAS cleaned and correspondence-audited but not silently admitted as labeled truth'],production_unchanged=True)
    write(OUT/'predictions.json',predictions);write(OUT/'report.json',report)
    print(json.dumps(dict(selection=selection,test_results=testresults),ensure_ascii=False),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','features','train']);args=p.parse_args()
    {'prepare':prepare,'features':features,'train':train}[args.stage]()
