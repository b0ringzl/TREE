"""Each Z rotation is a training example; original tree remains the split unit."""
import sys,json,uuid
from pathlib import Path
import numpy as np,torch,joblib
from sklearn.linear_model import LogisticRegression
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'tree_species_demo')]
from train_aquilaria_three_shot import read,save,metrics
from training_safety import backup_before_training,verify_backup,deployed_artifacts,sha256
from hk_three_species import Features,POINT
PREV=ROOT/'outputs/aquilaria_ptv2_fewshot_20260913_44056021'
LEGACY=ROOT/'outputs/bauhinia_ptv3_fewshot_20260912_v1'
SEED=20260913;DEGREES=[0,90,180,270]

def main():
 out=ROOT/'outputs'/('aquilaria_rotation_samples_20260913_'+uuid.uuid4().hex[:8]);out.mkdir(exist_ok=False);print('OUTPUT',out,flush=True)
 torch.set_num_threads(4);torch.manual_seed(SEED);np.random.seed(SEED)
 rows=read(PREV/'manifest.json');previous=read(PREV/'results.json');classes=read(PREV/'label_map.json');classes=[classes[str(i)] for i in range(len(classes))]
 live=read(ROOT/'outputs/hk_point_assist_20260911_v1/catalog.json');accepted={r['key'] for r in live if r['current_accepted'] and r['review_decision']=='accept'}
 save(out/'accepted_catalog_snapshot.json',live)
 for name in ['manifest.json','confirmation.json','selection_snapshot.json','reference_mapping_correction.json','overlap_cleaning.json','label_map.json','bauhinia_confirmation_snapshot.json']:save(out/name,read(PREV/name))
 protocol=dict(seed=SEED,degrees=DEGREES,classes=classes,encoder='Frozen PTv2',encoder_sha256=sha256(POINT),training='Each rotation contributes a separate loss example, L2 feature x10, LogisticRegression C=1 balanced max_iter=3000',split_unit='original tree group; all rotations inherit its train/validation/regression assignment',evaluation='Primary: average the four per-rotation probabilities per held-out tree; fixed 0-degree result secondary, no angle picking',baseline='Previous seven-class mean-feature head, same tree folds and same cleaned inputs',positive_trees=3,positive_augmented_examples=12,previous_results_inspected=True,no_hyperparameter_search=True,production_unchanged=True,limitations=['Augmentation does not create independent trees','All three positives same site and acquisition','Previously inspected trees are regression evidence, not fresh blind test','Supports/vegetation may remain; no improvement in physical tree geometry'])
 save(out/'protocol.json',protocol)
 assert protocol['encoder_sha256']==read(PREV/'protocol.json')['encoder_sha256']
 encoder=Features('cuda' if torch.cuda.is_available() else 'cpu');features=[];samples=[]
 for row in rows:
  if 'legacy_key' in row:assert row['legacy_key'] in accepted,row['key']
  assert sha256(row['source'])==row['source_sha256']
  p=Path(row['derived']);p=p if p.is_absolute() else LEGACY/p
  assert sha256(p)==row['derived_sha256'];data=np.load(p);points=data['points'];v=[]
  for degree in DEGREES:
   a=np.deg2rad(degree);rot=np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]],dtype='float32')
   with torch.inference_mode():v.append(encoder.encode_point(points@rot.T))
   samples.append(dict(sample_id=row['key']+'_rot'+str(degree),tree_key=row['key'],group=row['group'],degrees=degree,species=row['species'],split=row['split'],derived=str(p),derived_sha256=sha256(p)))
  features.append(v)
 features=np.asarray(features);assert np.isfinite(features).all();np.savez_compressed(out/'rotation_features.npz',features=features,keys=[r['key'] for r in rows],degrees=DEGREES)
 # Check no hidden change to inputs or the frozen encoder relative to prior experiment.
 old=np.load(PREV/'features.npz');assert old['keys'].tolist()==[r['key'] for r in rows]
 delta=float(np.max(abs(features.mean(1)-old['features'])));assert delta<1e-4,delta
 save(out/'feature_provenance.json',dict(feature_sha256=sha256(out/'rotation_features.npz'),encoder_sha256=sha256(POINT),previous_mean_feature_max_error=delta,trees=len(rows),examples=len(samples),rotations=DEGREES))
 save(out/'augmentation_manifest.json',samples)
 x=features/np.maximum(np.linalg.norm(features,axis=2,keepdims=True),1e-8)*10;y=np.array([classes.index(r['species']) for r in rows]);keys=[r['key'] for r in rows]
 train=[i for i,r in enumerate(rows) if r['split']=='train'];val=[i for i,r in enumerate(rows) if r['split']=='validation'];test=[i for i,r in enumerate(rows) if r['split']=='test_regression'];pos=[i for i,r in enumerate(rows) if r['split']=='fewshot_positive']
 results=[]
 for fold in range(4):
  final=fold==3;tag='final_3shot' if final else 'fold_'+str(fold+1);run=out/'runs'/tag;run.mkdir(parents=True,exist_ok=False)
  ti=train+[i for j,i in enumerate(pos) if final or j!=fold];vi=val+([] if final else [pos[fold]])
  assert not set(rows[i]['group'] for i in ti)&set(rows[i]['group'] for i in vi+test)
  tr=x[ti].reshape(-1,x.shape[2]);labels=np.repeat(y[ti],4)
  oldrun=Path(previous[fold]['run']);oldmodel=joblib.load(oldrun/'model.joblib');mean=features.mean(1);mean=mean/np.maximum(np.linalg.norm(mean,axis=1,keepdims=True),1e-8)*10;bp=oldmodel['head'].predict_proba(mean)
  config=dict(protocol,training_tree_keys=[keys[i] for i in ti],validation_tree_keys=[keys[i] for i in vi],regression_tree_keys=[keys[i] for i in test],training_examples=len(tr),training_tree_count=len(ti),fold=tag)
  save(run/'config.json',config);save(run/'split_assignment.json',[dict(**s,fold_split='train' if keys.index(s['tree_key']) in ti else 'validation' if keys.index(s['tree_key']) in vi else 'regression' if keys.index(s['tree_key']) in test else 'excluded') for s in samples])
  original={name:metrics(y[ids],bp[ids],[keys[i] for i in ids],classes) for name,ids in [('validation',vi),('old_validation',val),('old_regression',test)]};save(run/'original_evaluation.json',original)
  np.savez_compressed(run/'initial_head.npz',coef=np.zeros((len(classes),x.shape[2])),intercept=np.zeros(len(classes)))
  inputs=deployed_artifacts()+[POINT,oldrun/'model.joblib',Path(__file__),run/'initial_head.npz',run/'config.json',run/'split_assignment.json',run/'original_evaluation.json']+list(out.glob('*.json'))+list((PREV/'runs').glob('*/model.joblib'))
  backup=backup_before_training(run,inputs,dict(attempt=tag,rotation_examples=True,grouped_by_original_tree=True));verify_backup(backup);print('BACKUP VERIFIED',tag,flush=True)
  head=LogisticRegression(C=1,class_weight='balanced',max_iter=3000,random_state=SEED);head.fit(tr,labels);assert head.classes_.tolist()==list(range(len(classes)))
  bundle=dict(head=head,classes=classes,encoder='ptv2',encoder_sha256=sha256(POINT),feature_normalization='l2_then_times10',rotation_degrees=DEGREES,aggregation='mean_probabilities_after_per_rotation_feature_normalization',training_tree_keys=config['training_tree_keys'],experimental=True,deployed=False,calibrated=False)
  joblib.dump(bundle,run/'model.joblib');per=head.predict_proba(x.reshape(-1,x.shape[2])).reshape(len(rows),4,len(classes));prob=per.mean(1)
  result=dict(run=str(run),backup=str(backup),final=final,training_examples=len(tr),training_tree_count=len(ti),baseline=original)
  for name,ids in [('training',ti),('validation',vi),('old_validation',val),('old_regression',test)]:result[name]=metrics(y[ids],prob[ids],[keys[i] for i in ids],classes)
  if not final:
   i=pos[fold];result['held_out_positive']=metrics(y[[i]],prob[[i]],[keys[i]],classes);result['held_out_zero_degree']=metrics(y[[i]],per[[i],0],[keys[i]],classes)
  save(run/'result.json',result);verify_backup(backup);results.append(result)
  print('RESULT',tag,'val',result['validation']['correct'],'/',len(vi),'reg',result['old_regression']['correct'],'/',len(test),'positive',None if final else result['held_out_positive']['predictions'][0]['prediction'],flush=True)
 save(out/'results.json',results)
 held=[r['held_out_positive']['predictions'][0] for r in results if not r['final']]
 summary=dict(output=str(out),model=str(out/'runs/final_3shot/model.joblib'),positive_real_trees=3,positive_augmented_samples=12,total_real_trees=len(rows),total_augmented_samples=len(samples),correct=sum(r['prediction']=='Aquilaria sinensis' for r in held),total=3,held_out_predictions=held,previous_correct=2,final_old_validation=results[-1]['old_validation']['correct'],final_old_regression=results[-1]['old_regression']['correct'],backup_verified=True,deployed_unchanged=True)
 save(out/'summary.json',summary);print('SUMMARY',json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
