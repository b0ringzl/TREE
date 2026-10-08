"""Two visually linked LAS augment existing TRAIN groups only; fixed C=10 trial."""
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'));sys.path.insert(0,str(ROOT/'tree_species_demo'))
from hk_point_training_20260912 import OUT,read,write,metrics,SEED
from training_safety import backup_before_training,deployed_artifacts,verify_backup,sha256
from hk_three_species import Features,POINT
def main():
    import joblib,torch
    from sklearn.linear_model import LogisticRegression
    from collections import Counter
    torch.set_num_threads(4);run=OUT/'runs/05_confirmed_las_c10';run.mkdir(exist_ok=False);review=read(OUT/'agent_label_review.json');rows=read(OUT/'manifest.json');lookup={r['key']:r for r in rows};associations={r['id']:r for r in read(OUT/'new_las_associations.json')};f=np.load(OUT/'features/all.npz');rows=[lookup[k] for k in f['keys']];features=list(f['point']);encoder=Features('cuda' if torch.cuda.is_available() else 'cpu')
    for label in review['new_labels']:
        parent=lookup[label['same_tree_key']];assert parent['split']=='train' and parent['group']==label['group'] and parent['species']==label['species'];new=associations[label['id']]
        x=np.load(new['point'])['points_xyz'];features.append(encoder.encode_point(x));rows.append(dict(key=new['id'],id=new['id'],group=parent['group'],species=parent['species'],class_index=parent['class_index'],split='train',point=new['point'],views=new['views'],source_assets={'image':parent['source_assets']['image']},label_evidence=label,quality=new['quality']))
    classes=read(OUT/'plan.json')['classes'];x=np.array(features);x/=np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-8);y=np.array([r['class_index'] for r in rows]);train=np.array([i for i,r in enumerate(rows) if r['split']=='train']);val=np.array([i for i,r in enumerate(rows) if r['split']=='validation']);test=np.array([i for i,r in enumerate(rows) if r['split']=='test_regression'])
    write(run/'manifest.json',rows);np.savez_compressed(run/'features.npz',features=x,keys=np.array([r['key'] for r in rows]));np.savez_compressed(run/'initial_head.npz',coef=np.zeros((8,384)),intercept=np.zeros(8))
    cfg=dict(seed=SEED,C=10.,training_views=len(train),added_las=2,added_independent_trees=0,backbone='frozen PTv2',head='L2 regularized balanced logistic regression',hyperparameters='fixed to first-phase C10; no tuning on test',test_scope='same previously inspected regression set',label_source='explicit accepted pair plus recorded agent panorama/LAS correspondence verification')
    write(run/'config.json',cfg);backup=backup_before_training(run,deployed_artifacts()+[run/'manifest.json',run/'initial_head.npz',run/'config.json',OUT/'agent_label_review.json',OUT/'plan.json',OUT/'report.json',Path(__file__)]+[Path(associations[l['id']]['point']) for l in review['new_labels']],dict(attempt='added_las',no_deployment=True));print('ADDED LAS BACKUP VERIFIED',flush=True)
    counts=Counter(rows[i]['group'] for i in train);w=np.array([1/counts[rows[i]['group']] for i in train]);head=LogisticRegression(C=10,class_weight='balanced',max_iter=5000,random_state=SEED);head.fit(x[train],y[train],sample_weight=w)
    bundle=dict(head=head,scaler=None,feature='point',classes=classes,names=read(OUT/'plan.json')['names'],encoder=str(POINT),encoder_sha256=sha256(POINT),experimental=True,deployed=False,calibrated=False);joblib.dump(bundle,run/'model.joblib')
    v=metrics(y[val],head.predict_proba(x[val]),[rows[i] for i in val],classes);write(run/'validation_before_test.json',v);report=dict(config=cfg,validation=v,test_regression=metrics(y[test],head.predict_proba(x[test]),[rows[i] for i in test],classes),training=metrics(y[train],head.predict_proba(x[train]),[rows[i] for i in train],classes),production_unchanged=True);write(run/'report.json',report)
    p=head.predict_proba(x);preds=[]
    for i,r in enumerate(rows):
        order=np.argsort(-p[i]);preds.append(dict(key=r['key'],split=r['split'],truth=classes[y[i]],prediction=classes[order[0]],correct=bool(order[0]==y[i]),top3=[dict(species=classes[k],score=float(p[i,k])) for k in order[:3]]))
    write(run/'predictions.json',preds);verify_backup(backup);print('validation',v['view_accuracy'],'test',report['test_regression']['view_accuracy'],flush=True)
if __name__=='__main__':main()
