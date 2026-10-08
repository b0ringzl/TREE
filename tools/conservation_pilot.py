"""Prospective, binary Aquilaria verifier; no production deployment or pseudo-labels."""
import argparse
import os
import sys
from pathlib import Path
from collections import Counter
os.environ['HF_HUB_OFFLINE']='1'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import torch.nn.functional as F
from training_safety import sha256,backup_before_training,verify_backup
from distillation_core import read_json,save_json,digest
from train_sstv1_incremental import load_old,logits,prepare_pixels
from train_aquilaria_sstv1_conservative import decisions
from bioclip_probe import ResidualHead,TEACHER
PRIOR=ROOT/'outputs/SSTv1_Aquilaria_User_20260914_v1'
DEPLOY=ROOT/'tree_species_teacher_demo/models/SSTv1_Aquilaria_User_20260915_v1/deployment.json'
AQ=11
SEED=20260917

def baseline_heads(device):
    manifest=read_json(DEPLOY);heads=[]
    for m in manifest['members']:
        path=DEPLOY.parent/m['file'];assert sha256(path)==m['sha256']
        state=torch.load(path,map_location='cpu',weights_only=False)
        h=ResidualHead('mlp').to(device).eval().requires_grad_(False);h.load_state_dict(state['model_state']);heads.append(h)
    return heads

def prepare(out):
    out.mkdir(parents=True,exist_ok=False)
    manifest=read_json(DEPLOY)
    cfg={'model':'SST-Conserve-v1-pilot','seed':SEED,'target':'Aquilaria sinensis','target_index':AQ,
         'baseline_version':manifest['version'],'architecture':'binary linear residual on current SSTv1 log odds, frozen 1024-dimensional BioCLIP',
         'recipes_l2':[.001,.01,.1],'epochs':160,'lr':.03,'validation_fpr_budget':.01,
         'selection':'Each model/epoch threshold maximizes photo recall at <=1% negative FPR on planning only; best recall then average precision then smallest residual. Test unopened until selection_lock.',
         'baseline_comparator':'Current SSTv1 Aquilaria probability at a planning-tuned <=1% FPR threshold; also report original 148-way argmax. Not compare unlike top1 vs threshold alone.',
         'preprocessing':'All newly extracted features use exact Demo prepare gray padding JPEG95 then decode224, batch1. Reused positive features verified frozen-cache original brightness1 only.',
         'scope':'Previously inspected web/field regression, not new independent physical trees; novel pending review excluded from fitting.',
         'human_review_required_before_new_web_data':True,'automatic_deployment':False}
    save_json(out/'config.json',cfg)
    hd=decisions(ROOT/'external_datasets/herbarium_training_candidates_20260907/human_review.sqlite3')
    pd=decisions(ROOT/'external_datasets/sstv1_incremental_20260914_v1/human_review.sqlite3')
    save_json(out/'human_review_snapshot.json',{'herbarium':hd,'incremental':pd})
    rows,x,z=load_old(PRIOR,'train');heads=baseline_heads('cuda')
    classes=read_json(PRIOR/'classes.json');choice=read_json(PRIOR/'teacher_choice.json')
    with np.load(PRIOR/'features.npz') as f:extra=torch.tensor(f['extra_bank'],device='cuda');scale=float(f['logit_scale'])
    scores=[]
    with torch.inference_mode():
        for start in range(0,len(rows),512):
            scores.extend(torch.stack([logits(h,x[start:start+512],z[start:start+512],classes['groups'],choice['calibration_temperature'],extra,scale).softmax(-1)[:,AQ] for h in heads]).mean(0).cpu().tolist())
    negative=[]
    for i,r in enumerate(rows):
        if r['target']==AQ:continue
        if r.get('auxiliary'):
            d=hd.get(r['sample_id'].removeprefix('herbarium_'),{})
            if d.get('status')!='accepted' or d.get('sha256')!=r['source_sha256']:continue
        negative.append(i)
    rng=np.random.default_rng(SEED)
    hard=sorted(negative,key=lambda i:scores[i],reverse=True)[:256]
    sampled=rng.choice(negative,min(2000,len(negative)),replace=False).tolist()
    ids=sorted(set(hard+sampled))
    train=[{**rows[i],'phase':'train','binary_target':0,'selection_reason':'train_only_hard_negative' if i in hard else 'seeded_train_negative'} for i in ids]
    positive=[r for r in read_json(PRIOR/'feature_rows.json') if r['phase']=='train' and r['brightness']==1.]
    train += [{**r,'binary_target':1,'selection_reason':'previously_accepted_aquilaria'} for r in positive]
    new=read_json(PRIOR/'accepted_new_manifest.json')
    held={}
    for phase in ('planning','test'):
        source=read_json(PRIOR/('old_test_source_audited_rows.json' if phase=='test' else 'old_planning_rows.json'))
        audit=read_json(PRIOR/'old_source_audit.json')
        reliable={str(r['observation']) for r in audit if r['current_name']=='Aquilaria sinensis' and r['quality_grade']=='research'}
        kept=[r for r in source if r['target']!=AQ or r['group_id'].split('observation:')[-1] in reliable]
        kept += [r for r in new if r['phase']==phase]
        held[phase]=[{**r,'phase':phase,'binary_target':int(r['target']==AQ)} for r in kept]
    def key(r):return r.get('sha256') or r.get('source_sha256')
    held_hash={key(r) for rs in held.values() for r in rs};held_groups={r['group_id'] for rs in held.values() for r in rs}
    train=[r for r in train if key(r) not in held_hash and r['group_id'] not in held_groups]
    seen=set();dedup=[]
    for r in train:
        if key(r) in seen:continue
        seen.add(key(r));dedup.append(r)
    datasets={'train':dedup,**held}
    for phase,rs in datasets.items():
        for r in rs:
            assert sha256(r['source_image'])==key(r)
            r['label_authority']=r.get('review_authority','retained_source_species_not_model_prediction')
        save_json(out/f'{phase}_rows.json',rs)
    for p in ('planning','test'):
        assert not {key(r) for r in datasets['train']}&{key(r) for r in datasets[p]}
    protected=[DEPLOY,TEACHER,*[DEPLOY.parent/m['file'] for m in manifest['members']],
       ROOT/'tree_species_teacher_demo/engine.py',ROOT/'spacescan_trees_demo_v1/server.py',
       ROOT/'spacescan_trees_demo_v1/models/ptv2_seven_20260913_v1/model.joblib',
       ROOT/'spacescan_trees_demo_v1/models/ptv2_seven_20260913_v1/deployment.json',
       ROOT/'experiments/whu18_yolo11_ptv2_fusion_20260826/ptv2/run/best.pt']
    save_json(out/'contract.json',{'artifacts':[{'path':str(p),'sha256':sha256(p)} for p in protected]})
    save_json(out/'data_summary.json',{p:{'photos':len(rs),'positive_photos':sum(r['binary_target'] for r in rs),'source_groups':len({r['group_id'] for r in rs})} for p,rs in datasets.items()})
    print('PREPARED',read_json(out/'data_summary.json'),flush=True)

def extract(out,phases):
    from tree_species_teacher_demo.engine import Engine
    cfg=read_json(out/'config.json');engine=Engine();engine.load()
    for a in read_json(out/'contract.json')['artifacts']:assert sha256(a['path'])==a['sha256']
    previous_rows=read_json(PRIOR/'feature_rows.json')
    reuse={r['source_image']:i for i,r in enumerate(previous_rows) if r['brightness']==1.}
    with np.load(PRIOR/'features.npz') as f:
        assert str(f['rows_hash'])==digest(previous_rows)
        px=f['x'].astype(np.float32);pz=f['z'].copy()
    def infer(x,z):
        return torch.stack([logits(h,x,z,engine.classes['groups'],engine.choice['calibration_temperature'],engine.extra_bank,engine.model.logit_scale.exp().float()).softmax(-1) for h in engine.sstv1_heads]).mean(0)
    for phase in phases:
        target=out/f'{phase}_features.npz'
        if target.exists():raise FileExistsError(target)
        if phase=='test' and not (out/'selection_lock.json').exists():raise RuntimeError('Test requires selection lock')
        rows=read_json(out/f'{phase}_rows.json');xx=[];pp=[];reused=0
        with torch.inference_mode():
            for i,r in enumerate(rows):
                assert sha256(r['source_image'])==(r.get('sha256') or r.get('source_sha256'))
                j=reuse.get(r['source_image'])
                if j is not None:
                    x=torch.tensor(px[j:j+1],device='cuda');z=torch.tensor(pz[j:j+1],device='cuda');reused+=1
                else:
                    pixels=prepare_pixels(r['source_image'])[None].cuda()
                    e=engine.model.encode_image(((pixels-engine.mean)/engine.std).to(engine.dtype),normalize=True).float()
                    z=e@engine.bank.T*engine.model.logit_scale.exp().float();x=e.half().float()
                xx.append(x[0].cpu().numpy());pp.append(infer(x,z)[0].cpu().numpy())
                if i%200==0:print('EXTRACT',phase,i,len(rows),flush=True)
        np.savez_compressed(target,x=np.array(xx,dtype=np.float32),p=np.array(pp,dtype=np.float32),rows_hash=digest(rows),encoder_sha256=engine.encoder_sha256)
        save_json(out/f'{phase}_extraction.json',{'n':len(rows),'reused_original_brightness_features':reused,'new_batch_size':1,'preprocessing':cfg['preprocessing']})

def arrays(out,phase):
    rows=read_json(out/f'{phase}_rows.json')
    with np.load(out/f'{phase}_features.npz') as f:
        assert str(f['rows_hash'])==digest(rows)
        x=f['x'].copy();p=f['p'].copy()
    return rows,x,p,np.array([r['binary_target'] for r in rows],dtype=np.int64)

def binary_metrics(y,s,t):
    pred=np.asarray(s,dtype=np.float64)>=t;pos=y==1;neg=~pos
    tp=int((pred&pos).sum());fp=int((pred&neg).sum());fn=int((~pred&pos).sum());tn=int((~pred&neg).sum())
    return {'n':len(y),'positive':int(pos.sum()),'negative':int(neg.sum()),'tp':tp,'fn':fn,'fp':fp,'tn':tn,'recall':tp/max(1,tp+fn),'fpr':fp/max(1,fp+tn),'precision':tp/max(1,tp+fp),'threshold':float(t)}

def threshold(y,s,budget):
    # Choose the lowest score excluding all but the allowed negative count; ties excluded together.
    neg=np.sort(s[y==0])[::-1];allowed=int(np.floor(len(neg)*budget))
    return float(np.nextafter(float(neg[min(allowed,len(neg)-1)]),np.inf))

def odds(p):return np.log(np.clip(p,1e-7,1-1e-7)/np.clip(1-p,1e-7,1))

def train(out):
    from sklearn.metrics import average_precision_score
    cfg=read_json(out/'config.json');torch.set_num_threads(4)
    tr,x,p,y=arrays(out,'train');va,vx,vp,vy=arrays(out,'planning')
    bt=threshold(vy,vp[:,AQ],cfg['validation_fpr_budget'])
    baseline=binary_metrics(vy,vp[:,AQ],bt)
    save_json(out/'baseline_evaluation.json',{'planning_budget':baseline,'planning_argmax':binary_metrics(vy,(vp.argmax(1)==AQ).astype(float),.5),'baseline_version':cfg['baseline_version']})
    X=torch.tensor(x);Y=torch.tensor(y,dtype=torch.float32);offset=torch.tensor(odds(p[:,AQ]));VX=torch.tensor(vx);vo=torch.tensor(odds(vp[:,AQ]))
    groups=Counter((int(r['binary_target']),r.get('split_group',r['group_id'])) for r in tr)
    w=np.array([1/groups[(int(r['binary_target']),r.get('split_group',r['group_id']))] for r in tr],dtype=np.float32)
    for c in (0,1):w[y==c]/=w[y==c].sum()
    w=torch.tensor(w/2)
    protected=[Path(r['path']) for r in read_json(out/'contract.json')['artifacts']]
    candidates=[]
    for i,l2 in enumerate(cfg['recipes_l2']):
        run=out/f'linear_l2_{l2}';run.mkdir(exist_ok=False)
        torch.manual_seed(SEED+i);head=torch.nn.Linear(1024,1);torch.nn.init.zeros_(head.weight);torch.nn.init.zeros_(head.bias)
        torch.save({'model_state':head.state_dict(),'input':1024,'initialization':'zero residual on deployed logodds'},run/'initial.pt')
        save_json(run/'config.json',{**cfg,'l2':l2,'seed':SEED+i})
        backup=backup_before_training(run,[run/'initial.pt',run/'config.json',*protected,Path(__file__),ROOT/'tools/training_safety.py',out/'train_features.npz',out/'planning_features.npz',*[p for p in out.glob('*.json')]],{'purpose':'binary specialist fit','seed':SEED+i,'no_pending_images':True})
        optimizer=torch.optim.Adam(head.parameters(),lr=cfg['lr']);history=[];best=None
        for epoch in range(cfg['epochs']+1):
            if epoch:
                head.train();optimizer.zero_grad();s=offset+head(X).squeeze(1)
                loss=(F.binary_cross_entropy_with_logits(s,Y,reduction='none')*w).sum()+l2*head.weight.square().sum()
                loss.backward();optimizer.step()
            head.eval()
            with torch.inference_mode():vs=torch.sigmoid(vo+head(VX).squeeze(1)).numpy().astype(np.float64)
            t=threshold(vy,vs,cfg['validation_fpr_budget']);m=binary_metrics(vy,vs,t);ap=float(average_precision_score(vy,vs));norm=float(head.weight.detach().square().sum());key=(m['recall'],ap,-norm)
            history.append({'epoch':epoch,**m,'average_precision':ap,'weight_norm_squared':norm})
            if best is None or key>best:
                best=key;torch.save({'model_state':head.state_dict(),'config':cfg,'l2':l2,'epoch':epoch,'threshold':t,'baseline_threshold':bt,'encoder_sha256':read_json(DEPLOY)['encoder_sha256'],'baseline_deployment':read_json(DEPLOY),'planning':m},run/'best.pt')
            if epoch%40==0:print('FIT',l2,epoch,'recall',m['recall'],'FP',m['fp'],flush=True)
        save_json(run/'history.json',history);verify_backup(backup)
        saved=torch.load(run/'best.pt',map_location='cpu',weights_only=False)
        candidates.append({'path':str(run/'best.pt'),'sha256':sha256(run/'best.pt'),'key':list(best),'epoch':saved['epoch'],'planning':saved['planning'],'backup':str(backup)})
    chosen=max(candidates,key=lambda r:tuple(r['key']))
    save_json(out/'selection_lock.json',{'chosen':chosen,'candidates':candidates,'baseline_threshold':bt,'baseline_planning':baseline,'heldout_not_yet_scored':True,'deployed':False})
    print('LOCKED',chosen,flush=True)

def evaluate(out):
    from sklearn.metrics import average_precision_score,roc_auc_score
    lock=read_json(out/'selection_lock.json');chosen=lock['chosen'];assert sha256(chosen['path'])==chosen['sha256']
    saved=torch.load(chosen['path'],map_location='cpu',weights_only=False);head=torch.nn.Linear(1024,1).eval();head.load_state_dict(saved['model_state'])
    reports={}
    for phase in ('planning','test'):
        rows,x,p,y=arrays(out,phase)
        with torch.inference_mode():s=torch.sigmoid(torch.tensor(odds(p[:,AQ]))+head(torch.tensor(x)).squeeze(1)).numpy().astype(np.float64)
        b=p[:,AQ].astype(np.float64);m={'baseline':binary_metrics(y,b,lock['baseline_threshold']),'specialist':binary_metrics(y,s,saved['threshold']),
            'baseline_argmax':binary_metrics(y,(p.argmax(1)==AQ).astype(float),.5),'ap_baseline':float(average_precision_score(y,b)),'ap_specialist':float(average_precision_score(y,s)),
            'auc_baseline':float(roc_auc_score(y,b)),'auc_specialist':float(roc_auc_score(y,s))}
        save_json(out/f'{phase}_evaluation.json',m);reports[phase]=m
        rec=[{'sample_id':r['sample_id'],'source_image':r['source_image'],'group_id':r['group_id'],'truth':int(y[i]),'source_species':r.get('source_species',r.get('species')),'baseline':float(b[i]),'specialist':float(s[i]),'baseline_positive':bool(b[i]>=lock['baseline_threshold']),'specialist_positive':bool(s[i]>=saved['threshold'])} for i,r in enumerate(rows)]
        save_json(out/f'{phase}_predictions.json',rec)
    for r in lock['candidates']:verify_backup(r['backup'])
    for r in read_json(out/'contract.json')['artifacts']:assert sha256(r['path'])==r['sha256']
    m=reports['test'];qualified=m['specialist']['recall']>=m['baseline']['recall']+.1 and m['specialist']['fpr']<=m['baseline']['fpr']
    save_json(out/'COMPLETE.json',{'trained':True,'regression_target_met':qualified,'fresh_independent_validation':False,'deployment_authorized':False,'production_unchanged':True,'selected':chosen,'reports':reports})
    print('RESULT',reports,'target_met',qualified,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--stage',choices=['prepare','extract-development','train','extract-test','evaluate'],required=True);a=p.parse_args();out=a.out.resolve()
    if a.stage=='prepare':prepare(out)
    elif a.stage=='extract-development':extract(out,['train','planning'])
    elif a.stage=='extract-test':extract(out,['test'])
    elif a.stage=='train':train(out)
    else:evaluate(out)
