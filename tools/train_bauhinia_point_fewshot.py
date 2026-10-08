"""Fail-closed, same-data PTv2/PTv3 linear-head trials. Never deploy."""
import sys,json,argparse
from pathlib import Path
import numpy as np,joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
from training_safety import backup_before_training,verify_backup,deployed_artifacts,sha256
OUT=ROOT/'outputs/bauhinia_ptv3_fewshot_20260912_v1'
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def metrics(y,p,rows,classes):
 pred=p.argmax(1);report=classification_report(y,pred,labels=list(range(len(classes))),target_names=classes,output_dict=True,zero_division=0)
 return dict(n=len(y),correct=int((pred==y).sum()),accuracy=float((pred==y).mean()),macro_f1_supported=float(np.mean([report[classes[c]]['f1-score'] for c in sorted(set(y))])),per_class=report,confusion_matrix=confusion_matrix(y,pred,labels=range(len(classes))).tolist(),predictions=[dict(id=r['id'],key=r['key'],truth=classes[int(t)],prediction=classes[int(z)],correct=bool(t==z),scores={classes[c]:float(v) for c,v in enumerate(probs)}) for r,t,z,probs in zip(rows,y,pred,p)])
def main():
 parser=argparse.ArgumentParser();parser.add_argument('model',choices=['ptv2','ptv3']);args=parser.parse_args();config=read(OUT/'config.json');rows=read(OUT/'manifest.json');classes=config['classes'];f=np.load(OUT/(args.model+'_features.npz'));assert f['keys'].tolist()==[r['key'] for r in rows];prov=read(OUT/(args.model+'_feature_provenance.json'));assert prov['manifest_sha256']==sha256(OUT/'manifest.json');raw=f['features'];assert np.isfinite(raw).all();x=raw/np.maximum(np.linalg.norm(raw,axis=1,keepdims=True),1e-8)*10;y=np.array([r['class_index'] for r in rows]);positive=[i for i,r in enumerate(rows) if r['split']=='fewshot_positive'];base_train=[i for i,r in enumerate(rows) if r['split']=='train'];val=[i for i,r in enumerate(rows) if r['split']=='validation'];test=[i for i,r in enumerate(rows) if r['split']=='test_regression']
 weight=ROOT/('models/sonata_official_v1/sonata.pth' if args.model=='ptv3' else 'experiments/whu18_yolo11_ptv2_fusion_20260826/ptv2/run/best.pt');assert sha256(weight)==prov['weight_sha256'];runs=[]
 for fold in list(range(3))+([3] if args.model=='ptv3' else []):
  final=fold==3;tag='final_3shot' if final else f'fold_{fold+1}';run=OUT/'runs'/(args.model+'_'+tag+'_v1');run.mkdir(parents=True,exist_ok=False)
  training=base_train+[i for j,i in enumerate(positive) if final or j!=fold];validation=val+([] if final else [positive[fold]])
  assert not set(training)&set(validation+test)
  cfg=dict(config,feature_model=args.model,feature_dimensions=x.shape[1],fold=tag,training_keys=[rows[i]['key'] for i in training],validation_keys=[rows[i]['key'] for i in validation],test_keys=[rows[i]['key'] for i in test],model_selection='fixed C=1 before all evaluation; no search or best-fold selection',feature_sha256=sha256(OUT/(args.model+'_features.npz')),encoder_sha256=sha256(weight));save(run/'config.json',cfg)
  np.savez_compressed(run/'initial_head.npz',coef=np.zeros((len(classes),x.shape[1])),intercept=np.zeros(len(classes)))
  original=dict(status='Zero-initialized six-class head; no compatible deployed Bauhinia point head',initial_validation=metrics(y[validation],np.ones((len(validation),len(classes)))/len(classes),[rows[i] for i in validation],classes),historical_report_scope='Previous model report has different label space and split; preserved for rollback, not used as an accuracy comparator')
  baseline=OUT/'runs'/('ptv2_'+tag+'_v1')/'result.json'
  if args.model=='ptv3' and baseline.exists():original['compatible_ptv2_same_split']=read(baseline)
  save(run/'original_evaluation.json',original)
  inputs=deployed_artifacts()+[weight,run/'initial_head.npz',run/'config.json',run/'original_evaluation.json',OUT/'manifest.json',OUT/'config.json',OUT/'label_map.json',OUT/'split_assignment.json',OUT/'user_species_confirmation.json',OUT/'current_accepted_catalog_snapshot.json',OUT/'geometry_reviews_snapshot.json',OUT/(args.model+'_feature_provenance.json'),OUT/'new_tree_overlap_audit.json',ROOT/'outputs/hk_point_training_20260912_v1/report.json',Path(__file__),ROOT/'tools/extract_bauhinia_point_features.py']
  inputs+=[p for p in (ROOT/'outputs/hk_point_training_20260912_v1/runs').glob('*/model.joblib')]
  backup=backup_before_training(run,inputs,dict(experiment='three user-confirmed Bauhinia trees',attempt=run.name,backbone_frozen=True,deployed_unchanged=True));verify_backup(backup);print('BACKUP VERIFIED',run.name,flush=True)
  # All learned fitting occurs after verified physical backups.
  head=LogisticRegression(C=1.,class_weight='balanced',max_iter=3000,random_state=config['seed']);head.fit(x[training],y[training]);assert head.classes_.tolist()==list(range(len(classes)))
  joblib.dump(dict(head=head,classes=classes,encoder=args.model,encoder_sha256=sha256(weight),feature_normalization='l2_then_times10',input_config=config['input'],experimental=True,deployed=False,calibrated=False,training_keys=cfg['training_keys']),run/'model.joblib')
  result=dict(run=str(run),model=args.model,fold=tag,backup=str(backup),training=metrics(y[training],head.predict_proba(x[training]),[rows[i] for i in training],classes),validation=metrics(y[validation],head.predict_proba(x[validation]),[rows[i] for i in validation],classes),test_regression=metrics(y[test],head.predict_proba(x[test]),[rows[i] for i in test],classes),production_unchanged=True)
  save(run/'result.json',result);verify_backup(backup);runs.append(result);print(args.model,tag,'val',result['validation']['correct'],'/',result['validation']['n'],'test',result['test_regression']['correct'],'/',result['test_regression']['n'],flush=True)
 save(OUT/(args.model+'_results.json'),runs)
if __name__=='__main__':main()
