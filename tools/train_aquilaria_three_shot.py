"""991 user-confirmed Aquilaria: frozen PTv2, grouped 2-shot audit + final 3-shot.
Every fit has its own verified physical backup. Never deploys or rewrites labels.
"""
import json,sys,uuid,itertools
from pathlib import Path
import numpy as np,joblib,torch
from scipy.spatial import cKDTree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'tree_species_demo')]
from training_safety import sha256,backup_before_training,verify_backup,deployed_artifacts
from rebuild_banyan_coverage_preview import sample_ids
from hk_three_species import Features,POINT
BASE=ROOT/'outputs/bauhinia_ptv3_fewshot_20260912_v1'
SEL=ROOT/'outputs/tree991_manual_boxes_20260913_v1'
SPECIES='Aquilaria sinensis';SEED=20260913

def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def metrics(y,p,keys,classes):
 pred=p.argmax(1);r=classification_report(y,pred,labels=np.arange(len(classes)),target_names=classes,output_dict=True,zero_division=0)
 return dict(n=len(y),correct=int((y==pred).sum()),accuracy=float((y==pred).mean()),macro_f1_supported=float(np.mean([r[classes[i]]['f1-score'] for i in sorted(set(y))])),per_class=r,confusion_matrix=confusion_matrix(y,pred,labels=np.arange(len(classes))).tolist(),predictions=[dict(key=k,truth=classes[int(t)],prediction=classes[int(z)],scores={c:float(v) for c,v in zip(classes,pr)}) for k,t,z,pr in zip(keys,y,pred,p)])

def main():
 torch.set_num_threads(4);torch.manual_seed(SEED);np.random.seed(SEED)
 out=ROOT/'outputs'/('aquilaria_ptv2_fewshot_20260913_'+uuid.uuid4().hex[:8]);out.mkdir(exist_ok=False);(out/'clouds').mkdir();print('OUTPUT',out,flush=True)
 oldrows=read(BASE/'manifest.json');oldcfg=read(BASE/'runs/ptv2_fold_1_v1/config.json');baselinepath=BASE/'runs/ptv2_fold_1_v1/model.joblib';baseline=joblib.load(baselinepath)
 oldfeatures=np.load(BASE/'ptv2_features.npz');prov=read(BASE/'ptv2_feature_provenance.json')
 assert oldfeatures['keys'].tolist()==[r['key'] for r in oldrows]
 assert sha256(BASE/'manifest.json')==prov['manifest_sha256'] and sha256(POINT)==prov['weight_sha256']==baseline['encoder_sha256']
 assert sha256(BASE/'ptv2_features.npz')==oldcfg['feature_sha256']
 livepath=ROOT/'outputs/hk_point_assist_20260911_v1/catalog.json';live=read(livepath);accepted={r['key'] for r in live if r['current_accepted'] and r['review_decision']=='accept'}
 for r in oldrows:
  if 'legacy_key' in r:assert r['legacy_key'] in accepted,'Legacy acceptance changed: '+r['key']
  assert sha256(r['source'])==r['source_sha256']
 save(out/'accepted_catalog_snapshot.json',live);save(out/'legacy_manifest_snapshot.json',oldrows)
 save(out/'bauhinia_confirmation_snapshot.json',read(BASE/'user_species_confirmation.json'))
 state=read(SEL/'state.json');save(out/'selection_snapshot.json',state)
 correction=read(SEL/'orientation_audit_66665315b9/report.json');save(out/'reference_mapping_correction.json',correction['user_confirmed_reference_mapping'])
 rows=[dict(r,split=next(name for name,key in [('train','training_keys'),('validation','validation_keys'),('test_regression','test_keys')] if r['key'] in oldcfg[key])) for r in oldrows]
 clouds={};sourceids={};newrecords=[]
 for k in 'ABC':
  assert state['slots'][k]['confirmed'],'Unconfirmed tree '+k
  choices=[(p,read(p)) for p in (SEL/'exports').glob('*/record.json')]
  p,r=max([(p,r) for p,r in choices if r['slot']==k and r['selection']==state['slots'][k]],key=lambda v:v[1]['created_at'])
  assert r['species']==SPECIES and all(sha256(p.parent/n)==v['sha256'] for n,v in r['files'].items())
  with np.load(p.parent/'full.npz') as f:clouds[k]=f['xyz'];sourceids[k]=f['source_indices']
  newrecords.append(dict(key='991_old_'+k,id='991_old_'+k,group='991_old_'+k,scene_group='tree991_same_acquisition',species=SPECIES,split='fewshot_positive',source=str(p.parent/'full.npz'),source_sha256=sha256(p.parent/'full.npz'),photo_correspondence=correction['user_confirmed_reference_mapping']['old_'+k],label_evidence='User species confirmation, saved revision '+str(state['revision'])+' explicit 3D confirmation, and user A/C correction'))
 remove={k:np.zeros(len(x),bool) for k,x in clouds.items()};audit=[]
 for a,b in itertools.combinations(clouds,2):
  da=cKDTree(clouds[b]).query(clouds[a],workers=4)[0]<.005;db=cKDTree(clouds[a]).query(clouds[b],workers=4)[0]<.005
  shared=np.intersect1d(sourceids[a],sourceids[b]);da|=np.isin(sourceids[a],shared);db|=np.isin(sourceids[b],shared)
  remove[a]|=da;remove[b]|=db;audit.append(dict(pair=a+b,shared_indices=len(shared),a_points_removed=int(da.sum()),b_points_removed=int(db.sum()),policy='Remove shared original indices AND both sides of pairs closer than 5 mm'))
 save(out/'overlap_cleaning.json',audit)
 newfeat=[];encoder=Features('cuda' if torch.cuda.is_available() else 'cpu')
 for k,r in zip('ABC',newrecords):
  valid=~remove[k]&np.isfinite(clouds[k]).all(1);idx=np.flatnonzero(valid);_,unique=np.unique(clouds[k][idx],axis=0,return_index=True);idx=idx[np.sort(unique)];x=clouds[k][idx]
  path=out/'clouds'/(r['key']+'.npz');m=x[sample_ids(x,2048)].astype('float32');center=(x.min(0)+x.max(0))/2;scale=max(float(np.linalg.norm(x-center,axis=1).max()),1e-6);points=((m-center)/scale).astype('float32')
  np.savez_compressed(path,xyz=x,source_indices=sourceids[k][idx],metric=m,points=points)
  r.update(derived=str(path),derived_sha256=sha256(path),points_before=len(clouds[k]),points_after=len(x),sample_points=len(m),geometry_caveat='Manual ROI; supports and lower vegetation not certified removed. No invented tree completion.')
  features=[]
  for degree in [0,90,180,270]:
   a=np.deg2rad(degree);rot=np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]],dtype='float32')
   with torch.inference_mode():features.append(encoder.encode_point(points@rot.T))
  newfeat.append(np.mean(features,0));print('FEATURES',k,len(x),flush=True)
 rows+=newrecords;classes=sorted(oldcfg['classes']+[SPECIES]);keys=[r['key'] for r in rows];y=np.array([classes.index(r['species']) for r in rows]);raw=np.vstack([oldfeatures['features'],newfeat]);assert np.isfinite(raw).all();x=raw/np.maximum(np.linalg.norm(raw,axis=1,keepdims=True),1e-8)*10
 np.savez_compressed(out/'features.npz',features=raw,keys=keys)
 save(out/'manifest.json',rows);save(out/'label_map.json',{str(i):c for i,c in enumerate(classes)})
 save(out/'confirmation.json',dict(species=SPECIES,chinese='土沉香',confirmed_ids=[r['key'] for r in newrecords],request='再根据这三个点云实例进行小样本训练，实现土沉香树种的识别精度提升',scope='Experimental training only; not production deployment or perfect segmentation certification'))
 protocol=dict(seed=SEED,encoder='Existing PTv2 frozen, 384-dimensional feature',encoder_sha256=sha256(POINT),sampling='2048 measured points, centered unit sphere, four fixed Z rotations mean feature; same legacy pipeline',head='LogisticRegression C=1 balanced max_iter=3000; zero initialization; fixed before scoring',classes=classes,baseline=str(baselinepath),baseline_scope='Existing six-class PTv2 head lacks Aquilaria. Aquilaria recall=0 is missing-label capability, not fair equal-capacity evidence of accuracy gain.',validation='3 predeclared leave-one-tree-out 2-shot folds, same 6 old validation and 5 old regression trees; no tuning or best-fold selection',final='All three Aquilaria plus unchanged legacy train groups; no independent positive test for final model',limitations=['Three positives same scene; source/site/background confounded with new class','Human ROI supports and low vegetation may remain','No new independent test; all evaluation trees previously inspected','Point-only scores do not establish image/cloud fusion improvements'],production_unchanged=True,raw_images_used=False,absolute_coordinates_as_feature=False)
 save(out/'protocol.json',protocol)
 save(out/'feature_provenance.json',dict(encoder_sha256=sha256(POINT),feature_sha256=sha256(out/'features.npz'),legacy_feature_sha256=sha256(BASE/'ptv2_features.npz'),legacy_provenance=prov,manifest_sha256=sha256(out/'manifest.json')))
 train=[i for i,r in enumerate(rows) if r['split']=='train'];val=[i for i,r in enumerate(rows) if r['split']=='validation'];test=[i for i,r in enumerate(rows) if r['split']=='test_regression'];positive=[i for i,r in enumerate(rows) if r['split']=='fewshot_positive']
 bp=np.zeros((len(rows),len(classes)));oldp=baseline['head'].predict_proba(x)
 for j,c in enumerate(baseline['classes']):bp[:,classes.index(c)]=oldp[:,j]
 baseline_eval={name:metrics(y[ids],bp[ids],[keys[i] for i in ids],classes) for name,ids in [('old_validation',val),('old_regression',test),('aquilaria',positive)]};save(out/'baseline_evaluation.json',baseline_eval)
 results=[]
 for fold in range(4):
  final=fold==3;run=out/'runs'/('final_3shot' if final else 'fold_'+str(fold+1));run.mkdir(parents=True,exist_ok=False)
  ti=train+[i for j,i in enumerate(positive) if final or j!=fold];vi=val+([] if final else [positive[fold]])
  assert not set(ti)&set(vi+test);assert len(set(rows[i]['group'] for i in ti+vi+test))==len(ti+vi+test)
  config=dict(protocol,training_keys=[keys[i] for i in ti],validation_keys=[keys[i] for i in vi],regression_keys=[keys[i] for i in test],fold=fold,final=final);save(run/'config.json',config)
  save(run/'original_evaluation.json',dict(validation=metrics(y[vi],bp[vi],[keys[i] for i in vi],classes),regression=baseline_eval['old_regression']))
  np.savez_compressed(run/'initial_head.npz',coef=np.zeros((len(classes),x.shape[1])),intercept=np.zeros(len(classes)))
  backup_inputs=deployed_artifacts()+[POINT,baselinepath,run/'initial_head.npz',run/'config.json',run/'original_evaluation.json',Path(__file__)]+list(out.glob('*.json'))
  backup_inputs+=list((BASE/'runs').glob('*/model.joblib'))
  backup=backup_before_training(run,backup_inputs,dict(attempt=run.name,encoder_frozen=True,production_unchanged=True));verify_backup(backup);print('BACKUP VERIFIED',run.name,flush=True)
  head=LogisticRegression(C=1,class_weight='balanced',max_iter=3000,random_state=SEED);head.fit(x[ti],y[ti]);assert head.classes_.tolist()==list(range(len(classes)))
  joblib.dump(dict(head=head,classes=classes,encoder='ptv2',encoder_sha256=sha256(POINT),feature_normalization='l2_then_times10',input_config=protocol['sampling'],training_keys=config['training_keys'],experimental=True,deployed=False,calibrated=False),run/'model.joblib')
  result=dict(run=str(run),backup=str(backup),final=final,training=metrics(y[ti],head.predict_proba(x[ti]),[keys[i] for i in ti],classes),validation=metrics(y[vi],head.predict_proba(x[vi]),[keys[i] for i in vi],classes),old_validation=metrics(y[val],head.predict_proba(x[val]),[keys[i] for i in val],classes),old_regression=metrics(y[test],head.predict_proba(x[test]),[keys[i] for i in test],classes),held_out_positive=None if final else metrics(y[[positive[fold]]],head.predict_proba(x[[positive[fold]]]),[keys[positive[fold]]],classes))
  save(run/'result.json',result);verify_backup(backup);results.append(result);print('RESULT',run.name,'val',result['validation']['correct'],'/',result['validation']['n'],'reg',result['old_regression']['correct'],'/',len(test),'heldout',result['held_out_positive'],flush=True)
 save(out/'results.json',results)
 held=[r['held_out_positive']['predictions'][0] for r in results if not r['final']]
 summary=dict(output=str(out),model=str(out/'runs/final_3shot/model.joblib'),classes=classes,trees=len(rows),held_out_aquilaria_correct=sum(r['prediction']==SPECIES for r in held),held_out_aquilaria_total=3,held_out_predictions=held,baseline_old_validation=baseline_eval['old_validation']['correct'],baseline_old_regression=baseline_eval['old_regression']['correct'],final_old_validation=results[-1]['old_validation']['correct'],final_old_regression=results[-1]['old_regression']['correct'],backup_verified=True,production_unchanged=True)
 save(out/'summary.json',summary);print('SUMMARY',json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
