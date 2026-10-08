"""One additional preregistered neural-head experiment; no test-driven tuning."""
import sys,json
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'));sys.path.insert(0,str(ROOT/'tree_species_demo'))
from hk_point_training_20260912 import OUT,read,write,metrics,SEED
from training_safety import backup_before_training,deployed_artifacts,verify_backup,sha256
from hk_three_species import Features,POINT

def main():
    torch.set_num_threads(4);torch.manual_seed(SEED);np.random.seed(SEED)
    run=OUT/'runs/04_rotation_mlp';run.mkdir(exist_ok=False)
    rows=[r for r in read(OUT/'manifest.json') if r['split'] in ['train','validation','test_regression']];plan=read(OUT/'plan.json');classes=plan['classes']
    cfg=dict(seed=SEED,rotation_degrees=[0,90,180,270],architecture=[384,128,len(classes)],dropout=.3,optimizer='AdamW',lr=.001,weight_decay=.05,epochs=200,patience=45,selection='validation macro F1 then group accuracy then validation loss; never test',test_status='Existing 7-view/5-group regression set has been inspected in first phase; not fresh evidence',backbone='frozen PTv2',training_modality='XYZ only')
    write(run/'config.json',cfg);model=torch.nn.Sequential(torch.nn.Linear(384,128),torch.nn.ReLU(),torch.nn.Dropout(.3),torch.nn.Linear(128,len(classes)))
    torch.save(model.state_dict(),run/'initial_head.pt')
    backup=backup_before_training(run,deployed_artifacts()+[run/'initial_head.pt',run/'config.json',OUT/'manifest.json',OUT/'plan.json',OUT/'report.json',OUT/'accepted_catalog_snapshot.json',OUT/'geometry_reviews_snapshot.json',Path(__file__)],dict(attempt='rotation_neural_head',no_deployment=True));print('ROTATION BACKUP VERIFIED',flush=True)
    # All four rotations are deterministic and label preserving around the vertical Z axis.
    encoder=Features('cuda' if torch.cuda.is_available() else 'cpu');features=[]
    for i,r in enumerate(rows):
        x=np.load(r['point'])['points_xyz'];vs=[]
        for degree in cfg['rotation_degrees']:
            a=np.deg2rad(degree);rot=np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]],np.float32);v=encoder.encode_point(x@rot.T);vs.append(v/max(np.linalg.norm(v),1e-8))
        features.append(vs);print('rotation features',i+1,len(rows),flush=True)
    x=np.array(features);np.savez_compressed(run/'features.npz',features=x,keys=np.array([r['key'] for r in rows]));del encoder
    y=np.array([r['class_index'] for r in rows]);train=np.array([i for i,r in enumerate(rows) if r['split']=='train']);val=np.array([i for i,r in enumerate(rows) if r['split']=='validation']);test=np.array([i for i,r in enumerate(rows) if r['split']=='test_regression']);xt=torch.tensor(x[train].reshape(-1,384));yt=torch.tensor(np.repeat(y[train],4),dtype=torch.long)
    from collections import Counter
    group_counts=Counter(rows[i]['group'] for i in train);class_groups={s:len({rows[i]['group'] for i in train if y[i]==s}) for s in range(len(classes))}
    weights=torch.tensor(np.repeat([1/(group_counts[rows[i]['group']]*class_groups[y[i]]) for i in train],4),dtype=torch.float32);weights/=weights.mean()
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.05);lossfn=torch.nn.CrossEntropyLoss(reduction='none');best=None;history=[];best_epoch=-1
    def probabilities(ids):
        model.eval()
        with torch.no_grad():return torch.softmax(model(torch.tensor(x[ids].reshape(-1,384))),1).reshape(len(ids),4,-1).mean(1).numpy()
    for epoch in range(200):
        model.train();opt.zero_grad();loss=(lossfn(model(xt),yt)*weights).mean();loss.backward();opt.step();p=probabilities(val);m=metrics(y[val],p,[rows[i] for i in val],classes);vl=float(-np.log(np.maximum(p[np.arange(len(val)),y[val]],1e-9)).mean());score=(m['macro_f1_supported_only'],m['group_accuracy'],-vl)
        history.append(dict(epoch=epoch,train_loss=float(loss),validation_loss=vl,validation_accuracy=m['view_accuracy'],validation_macro_f1=m['macro_f1_supported_only']))
        if best is None or score>best:
            best=score;best_epoch=epoch;torch.save(dict(state_dict=model.state_dict(),classes=classes,config=cfg,encoder=str(POINT),encoder_sha256=sha256(POINT),experimental=True),run/'best.pt')
        if epoch-best_epoch>=45:break
    write(run/'history.json',history);write(run/'selection_before_test.json',dict(epoch=best_epoch,validation_score=best,test_used=False))
    model.load_state_dict(torch.load(run/'best.pt',weights_only=False)['state_dict']);report=dict(config=cfg,best_epoch=best_epoch,validation=metrics(y[val],probabilities(val),[rows[i] for i in val],classes),test_regression=metrics(y[test],probabilities(test),[rows[i] for i in test],classes),training=metrics(y[train],probabilities(train),[rows[i] for i in train],classes),production_unchanged=True)
    preds=[]
    for ids in [train,val,test]:
        p=probabilities(ids)
        for j,i in enumerate(ids):
            order=np.argsort(-p[j]);preds.append(dict(key=rows[i]['key'],split=rows[i]['split'],truth=classes[y[i]],prediction=classes[order[0]],correct=bool(order[0]==y[i]),top3=[dict(species=classes[k],score=float(p[j,k])) for k in order[:3]]))
    write(run/'predictions.json',preds);write(run/'report.json',report);verify_backup(backup);print(json.dumps(report,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
