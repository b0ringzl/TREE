"""Two predeclared local-kernel precision fits, per-fit backups, one study lock.

No source image relabelling, training/test leakage, production mutation or
test-driven recipe choice. The old inspected test remains regression only.
"""
import argparse
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import train_conservation_browser_20260917 as base
from distillation_core import digest, read_json, save_json, metrics
from training_safety import sha256, backup_before_training, verify_backup

ROOT=base.ROOT
PREVIOUS=ROOT/'outputs/SST_Conserve_Browser_20260917_v3'
SEED=20260917
FOCUS=[16,148,149]
RECIPES=[
    dict(name='local_kernel_beta8',beta=8.,cap=2.,ce=.05,focus=1.,negative=2.,kl=1.,l2=.02),
    dict(name='local_kernel_beta20',beta=20.,cap=2.,ce=.05,focus=1.,negative=2.,kl=1.,l2=.02),
]

class BoundedResidual(torch.nn.Module):
    def __init__(self,cap,support=None,beta=8.):
        super().__init__()
        if support is None:support=torch.zeros(1,1024)
        self.register_buffer('support',F.normalize(support.detach().float().clone(),dim=-1))
        self.linear=torch.nn.Linear(len(support),3,bias=False)
        self.cap=float(cap)
        self.beta=float(beta)
        self._feature_cache={}
        torch.nn.init.zeros_(self.linear.weight)
    def forward(self,x):
        # Frozen exact training embeddings, not fitted centroids. No global bias.
        key=id(x)
        if key not in self._feature_cache:
            assert not x.requires_grad
            with torch.no_grad():
                cosine=F.normalize(x,dim=-1)@self.support.T
                features=torch.exp(self.beta*(cosine.clamp(-1,1)-1.))
            self._feature_cache[key]=(x,features)
        return self.cap*torch.tanh(self.linear(self._feature_cache[key][1])/self.cap)

def logp(head,d):
    residual=head(d['x'])
    delta=F.pad(residual[:,0:1],(16,133))+F.pad(residual[:,1:3],(148,0))
    return torch.logsumexp(F.log_softmax(d['b']+delta[None],-1),dim=0)-np.log(3.)

def scores(head,d,native=False):
    with torch.inference_mode():
        return (d['b'][:,:,:148].softmax(-1).mean(0) if native else logp(head,d).exp()).cpu().numpy()

def class_source_weights(rows):
    groups=Counter((r['target'],r['group_id']) for r in rows)
    perclass=Counter(y for y,g in groups)
    w=np.array([1/(groups[(r['target'],r['group_id'])]*perclass[r['target']]) for r in rows],dtype=np.float32)
    return torch.tensor(w/w.sum(),device='cuda')

def margin(lp):
    values,indices=lp.topk(2,dim=-1)
    competitor=torch.stack([torch.where(indices[:,0]==k,values[:,1],values[:,0]) for k in FOCUS],dim=1)
    return lp[:,FOCUS]-competitor

def counts(d,p,k):
    y=d['y'].cpu().numpy();pred=p.argmax(1)
    tp=int(((y==k)&(pred==k)).sum());fp=int(((y!=k)&(pred==k)).sum())
    fn=int(((y==k)&(pred!=k)).sum());tn=int(((y!=k)&(pred!=k)).sum())
    return dict(tp=tp,fp=fp,fn=fn,tn=tn,f1=2*tp/max(1,2*tp+fp+fn))

def compact(d,p):
    m=metrics(d['y'].cpu().numpy(),p)
    return {**{k:m[k] for k in ('n','top1','macro_f1','balanced_accuracy')},
            'correct':int(round(m['n']*m['top1'])),
            'targets':{str(k):counts(d,p,k) for k in FOCUS},'aquilaria':counts(d,p,11)}

def gate(ov,nv,old0,new0,native0):
    # Counts rather than rounded percentages; all thresholds fixed pre-fit.
    # Compare accuracy in the same 150-class space. Native148 remains a reported
    # reference: adding two classes already changes one planning prediction.
    ok=ov['correct']>=old0['correct']
    ok=ok and ov['macro_f1']>=max(old0['macro_f1'],native0['macro_f1'])-.001
    for k in FOCUS:
        key=str(k)
        ok=ok and ov['targets'][key]['tp']>=old0['targets'][key]['tp']
        ok=ok and ov['targets'][key]['fp']<=old0['targets'][key]['fp']
        ok=ok and nv['targets'][key]['tp']>=new0['targets'][key]['tp']
    ok=ok and ov['targets']['16']['f1']>=old0['targets']['16']['f1']
    ok=ok and ov['aquilaria']['tp']>=native0['aquilaria']['tp']
    ok=ok and ov['aquilaria']['fp']<=native0['aquilaria']['fp']
    return bool(ok)

def ranking(ov,nv,norm):
    # Retain/increase new-class recall first, then improve old-class precision.
    return (nv['balanced_accuracy'],ov['targets']['16']['f1'],
            -sum(ov['targets'][str(k)]['fp'] for k in FOCUS),
            ov['correct'],ov['macro_f1'],-norm)

def prepare(out):
    out.mkdir(parents=True,exist_ok=False)
    base.verify_contract(PREVIOUS)
    data=out/'data';base.prepare(data)
    matches=[]
    for name in ('accepted_replay_manifest.json','new_manifest.json','replay_source_indices.json',
                 'labels.json','classes.json','contract.json','human_review_snapshot.json'):
        expected=sha256(PREVIOUS/name)
        assert sha256(data/name)==expected, 'Audited data or human reviews changed: '+name
        matches.append(dict(file=name,sha256=expected,source=str(PREVIOUS/name)))
    source=PREVIOUS/'new_features.npz'
    shutil.copy2(source,data/'new_features.npz')
    assert sha256(source)==sha256(data/'new_features.npz')
    save_json(data/'audited_data_equivalence.json',dict(files=matches,previous= str(PREVIOUS),
              feature_sha256=sha256(source),physically_copied=True))
    cfg=dict(model='SST-Conserve-LocalKernel-v1',request='你来优化一下模型',seed=SEED,
             epochs=200,lr=.01,weight_decay=.01,recipes=RECIPES,output_count=150,
             architecture='No-global-bias RBF kernel residual; supports are fixed exact training vectors of all target positives and the union of training-only hard negatives; cap*tanh(K(x)@weights/cap), beta8 or20.',
             reason='Three registered linear loss variants all fell back to epoch0 on development. Their regression was not opened. This follow-up uses the same development gate and a new local nonlinear architecture; no test-driven settings.',
             initialization='New zero residual for every fit; frozen deployed BioCLIP2.5+3MLP, not earlier failed residual.',
             sampling='Combined old23906+new38, equal mass per supported class, then source group, then image.',
             focus_loss='Training-only frozen baseline margins choose 256 hardest negatives per target. Positive goal min(m0,1) if m0>0 else .25; squared hinge stops above goal. Negative goal <=-.25. Positive and negative source groups normalized separately.',
             regularization='Old replay class/source-weighted KL to unchanged baseline + combined class/source-weighted actual bounded residual squared.',
             selection='Oldplanning correct >= same-vocabulary base150; macro>=max(native148,base150)-.001; each target TP>=base and FP<=base; new15 each target TP>=base; Aquilaria TP>=native and FP<=native. Rank lexicographically new balanced recall, old white F1, -targetFP, old correct, oldmacro, -residual norm. Native148 has one extra planning correct before fitting; report separately. Initial zero can win.',
             study_rule='Both recipes registered before fit. No historical regression evaluated until global recipe/epoch lock. Never adjust recipes based on regression.',
             evaluation='15 new development photos repeatedly used for selection; no fresh unbiased test. Old selection/test/field and human92 are historical regression only.',
             preprocessing='Existing feature caches reused as frozen inputs. Legacy distillation cache uses center crop; newer source images and human92 use gray letterbox prepare_pixels. Each comparison uses identical inputs, but legacy regression is not a full current-Demo upload replay.',
             constraints='No rejected/pending/uncertain actual human-review data; no augmentation-as-independent-tree claim; no production switch.',
             no_deployment=True)
    save_json(out/'protocol.json',cfg)
    print('PREPARED OPTIMIZATION',out,flush=True)

def load_dev(data):
    model=base.models(data)
    tr=base.tensor_data(data,base.replay(data),model)
    va=base.tensor_data(data,base.load_cached(data,'planning'),model)
    rows=read_json(data/'new_manifest.json')
    with np.load(data/'new_features.npz') as f:
        assert str(f['rows_hash'])==digest(rows)
        x=f['x'].astype(np.float32);z=f['z'].copy()
    def subset(phase):
        ids=[i for i,r in enumerate(rows) if r['phase']==phase]
        return base.tensor_data(data,([rows[i] for i in ids],x[ids],z[ids]),model)
    nt,nv=subset('train'),subset('planning')
    combined={'rows':tr['rows']+nt['rows'],'x':torch.cat([tr['x'],nt['x']]),
              'y':torch.cat([tr['y'],nt['y']]),'b':torch.cat([tr['b'],nt['b']],dim=1)}
    assert len(combined['rows'])==23944
    return model,tr,va,nt,nv,combined

def build_focus(d,zero):
    with torch.no_grad():m0=margin(logp(zero,d))
    train_y=d['y'];specs=[];audit=[]
    for j,k in enumerate(FOCUS):
        positive=(train_y==k).nonzero().flatten()
        negative=(train_y!=k).nonzero().flatten()
        negative=negative[torch.argsort(m0[negative,j],descending=True,stable=True)[:256]]
        goal=torch.where(m0[positive,j]>0,m0[positive,j].clamp_max(1.),.25).detach()
        pr=[d['rows'][i] for i in positive.cpu().tolist()]
        nr=[d['rows'][i] for i in negative.cpu().tolist()]
        specs.append((positive,negative,goal,class_source_weights(pr),class_source_weights(nr)))
        audit.append(dict(target=k,positive_rows=pr,negative_rows=nr,positive_goals=goal.cpu().tolist(),
                          frozen_negative_margin=m0[negative,j].cpu().tolist(),mining_scope='training_only'))
    return specs,audit

def focus_loss(lp,specs,negative_factor):
    mm=margin(lp);loss=0.
    for j,(pos,neg,goal,pw,nw) in enumerate(specs):
        posloss=(F.relu(goal-mm[pos,j]).square()*pw).sum()
        negloss=(F.relu(mm[neg,j]+.25).square()*nw).sum()
        loss=loss+(posloss+negative_factor*negloss)/(1.+negative_factor)
    return loss/len(specs)

def fit_one(out,recipe,tr,va,nt,nv,combined,specs,baseline,support):
    data=out/'data';protocol=read_json(out/'protocol.json')
    run=out/'runs'/recipe['name'];run.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    head=BoundedResidual(recipe['cap'],support,recipe['beta']).cuda()
    config={**protocol,'recipe':recipe}
    save_json(run/'config.json',config)
    initial=run/'initial.pt'
    torch.save(dict(model_state=head.state_dict(),config=config),initial)
    save_json(run/'baseline_evaluation.json',baseline)
    protected=[Path(r['path']) for r in read_json(data/'contract.json')['artifacts']]
    snapshots=[p for p in data.glob('*.json')]+[data/'new_features.npz',out/'protocol.json',out/'focus_training_manifest.json',out/'data_weights.json',out/'human_feature_contract.json',out/'support_manifest.json']
    backup=backup_before_training(run,[initial,*protected,*snapshots,run/'config.json',run/'baseline_evaluation.json',
        Path(__file__),Path(base.__file__),ROOT/'tools/training_safety.py',ROOT/'tools/bioclip_probe.py',ROOT/'tools/distillation_core.py',
        ROOT/'outputs/conservation_browser_training_audit_20260917_v1/prepared_audit.json',ROOT/'outputs/conservation_browser_training_audit_20260917_v1/prepared_queue_resolution.json'],
        dict(seed=SEED,recipe=recipe['name'],purpose='Precision optimization; physical copies before any optimizer update'))
    print('BACKUP VERIFIED',recipe['name'],backup,flush=True)
    cw=class_source_weights(combined['rows']);rw=class_source_weights(tr['rows'])
    with torch.no_grad():anchor=tr['b'].softmax(-1).mean(0).detach()
    old0,new0,native0=baseline['old150'],baseline['new150'],baseline['native148']
    bestkey=ranking(old0,new0,0.);bestepoch=0
    best={k:v.detach().cpu().clone() for k,v in head.state_dict().items()};history=[]
    opt=torch.optim.AdamW(head.parameters(),lr=protocol['lr'],weight_decay=protocol['weight_decay'])
    for epoch in range(1,protocol['epochs']+1):
        head.train();opt.zero_grad(set_to_none=True)
        lp=logp(head,combined)
        ce=(-lp.gather(1,combined['y'][:,None]).flatten()*cw).sum()
        kl=(F.kl_div(lp[:len(tr['rows'])],anchor,reduction='none').sum(-1)*rw).sum()
        norm=(head(combined['x']).square().mean(-1)*cw).sum()
        contrast=focus_loss(lp,specs,recipe['negative']) if recipe['focus'] else lp.sum()*0.
        loss=recipe['ce']*ce+recipe['kl']*kl+recipe['l2']*norm+recipe['focus']*contrast
        assert torch.isfinite(loss)
        loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),5,error_if_nonfinite=True);opt.step()
        head.eval()
        old=compact(va,scores(head,va));new=compact(nv,scores(head,nv))
        safe=gate(old,new,old0,new0,native0)
        with torch.no_grad():actualnorm=float((head(combined['x']).square().mean(-1)*cw).sum())
        key=ranking(old,new,actualnorm)
        if safe and key>bestkey:
            bestkey=key;bestepoch=epoch;best={k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
        history.append(dict(epoch=epoch,loss=float(loss.detach()),ce=float(ce.detach()),kl=float(kl.detach()),
                    contrast=float(contrast.detach()),norm=actualnorm,old=old,new=new,safe=safe))
        if epoch%20==0:print('FIT',recipe['name'],epoch,'best',bestepoch,'new',new['correct'],'white',old['targets']['16'],flush=True)
    torch.save(dict(model_state=head.state_dict(),optimizer_state=opt.state_dict(),epoch=protocol['epochs'],config=config),run/'last.pt')
    torch.save(dict(model_state=best,epoch=bestepoch,config=config,labels=read_json(data/'labels.json'),encoder_sha256=read_json(base.DEPLOY)['encoder_sha256']),run/'best.pt')
    save_json(run/'history.json',history)
    head.load_state_dict(best);head.eval()
    old=compact(va,scores(head,va));new=compact(nv,scores(head,nv))
    safe=gate(old,new,old0,new0,native0)
    strict=bestkey[:5]>ranking(old0,new0,0.)[:5]
    result=dict(recipe=recipe,epoch=bestepoch,key=bestkey,old=old,new=new,safe=safe,strict_gain=strict,
                checkpoint=str(run/'best.pt'),sha256=sha256(run/'best.pt'),backup=str(backup),deployed=False)
    save_json(run/'RESULT.json',result)
    verify_backup(backup);base.verify_contract(data)
    print('FIT COMPLETE',recipe['name'],bestepoch,'safe',safe,'gain',strict,flush=True)
    return result

def fit(out):
    assert not (out/'selection_lock.json').exists(), 'Immutable completed study; prepare a new version for another attempt'
    torch.set_num_threads(4)
    protocol=read_json(out/'protocol.json');assert protocol['recipes']==RECIPES
    human_source=ROOT/'outputs/SST_Conserve_Browser_20260917_v2/post_training_human_regression/features.npz'
    save_json(out/'human_feature_contract.json',{'path':str(human_source),'sha256':sha256(human_source),
        'rows_sha256':sha256(human_source.parent/'challenge_rows.json'),'stage':'hash only before fitting; no regression arrays loaded'})
    data=out/'data';model,tr,va,nt,nv,combined=load_dev(data)
    zero=BoundedResidual(2.).cuda().eval()
    baseline=dict(old150=compact(va,scores(zero,va)),new150=compact(nv,scores(zero,nv)),native148=compact(va,scores(zero,va,True)))
    specs,audit=build_focus(combined,zero)
    save_json(out/'focus_training_manifest.json',audit)
    support_ids=sorted({int(i) for spec in specs for indices in spec[:2] for i in indices.cpu().tolist()})
    support=combined['x'][support_ids].detach()
    save_json(out/'support_manifest.json',dict(rows=[combined['rows'][i] for i in support_ids],
              indices=support_ids,n=len(support_ids),origin='Exact immutable train embeddings only; no clustering or fit during support construction. Included in every initialization backup.'))
    cw=class_source_weights(combined['rows']).cpu().numpy()
    save_json(out/'data_weights.json',dict(scheme='equal supported class, then (label,group_id), then image',
              weights=cw.tolist(),rows_hash=digest(combined['rows']),
              new_target_weight={str(k):float(sum(w for r,w in zip(combined['rows'],cw) if r in nt['rows'] and r['target']==k)) for k in FOCUS}))
    save_json(out/'baseline_evaluation.json',baseline)
    results=[]
    for recipe in RECIPES:
        results.append(fit_one(out,recipe,tr,va,nt,nv,combined,specs,baseline,support))
    chosen=max(results,key=lambda r:tuple(r['key']) if r['safe'] else (-1.,)*6)
    assert chosen['safe'], 'No safe candidate including epoch0; do not report a valid replacement'
    lock={**chosen,'selection_uses_validation_only':True,'all_recipes_finished_before_regression':True,
          'regression_not_opened_in_this_study':True,'production_unchanged':True,'deployment_authorized':False,
          'consider_for_demo_trial':bool(chosen['strict_gain'] and chosen['epoch']>0)}
    save_json(out/'selection_lock.json',lock)
    save_json(out/'all_recipes.json',results)
    print('GLOBAL SELECTION LOCKED',chosen['recipe']['name'],chosen['epoch'],flush=True)

def load_selected(out):
    lock=read_json(out/'selection_lock.json')
    assert lock['all_recipes_finished_before_regression'] is True
    assert sha256(lock['checkpoint'])==lock['sha256']
    state=torch.load(lock['checkpoint'],weights_only=False,map_location='cpu')
    assert state['encoder_sha256']==read_json(base.DEPLOY)['encoder_sha256']
    assert state['labels']==read_json(out/'data/labels.json')
    assert state['config']['recipe']==lock['recipe']
    head=BoundedResidual(lock['recipe']['cap'],state['model_state']['support'],lock['recipe']['beta']).cuda().eval().requires_grad_(False)
    head.load_state_dict(state['model_state'])
    return head,lock

def evaluate(out):
    destination=out/'regression';destination.mkdir(exist_ok=False)
    head,lock=load_selected(out);data=out/'data'
    # Let the immutable legacy feature loader enforce this globally locked model.
    data_lock={'selected':lock['checkpoint'],'sha256':lock['sha256']}
    assert not (data/'selection_lock.json').exists()
    save_json(data/'selection_lock.json',data_lock)
    model,tr,va,nt,nv,combined=load_dev(data)
    zero=BoundedResidual(2.).cuda().eval();results={}
    def compare(name,d):
        bp=scores(zero,d);cp=scores(head,d);np_=scores(zero,d,True);y=d['y'].cpu().numpy()
        result=dict(baseline150=metrics(y,bp),candidate150=metrics(y,cp),
          baseline_details=compact(d,bp),candidate_details=compact(d,cp),
          corrected=int(((bp.argmax(1)!=y)&(cp.argmax(1)==y)).sum()),
          regressed=int(((bp.argmax(1)==y)&(cp.argmax(1)!=y)).sum()))
        if y.max()<148:result['native148']=metrics(y,np_)
        save_json(destination/f'{name}_evaluation.json',result)
        save_json(destination/f'{name}_rows.json',d['rows'])
        np.savez_compressed(destination/f'{name}_predictions.npz',baseline150=bp,candidate150=cp,target=y)
        print('EVAL',name,base.compact(result['baseline150']),base.compact(result['candidate150']),flush=True)
        results[name]=result
    compare('new_train',nt);compare('new_development',nv);compare('old_planning',va)
    for phase in ('selection','test','field'):
        compare('old_'+phase,base.tensor_data(data,base.load_cached(data,phase),model))
    # Reuse last turn's exact-encoder fresh letterbox features, binary truth only.
    import evaluate_conservation_browser_challenge_20260917 as human
    human.preflight(ROOT/'outputs/SST_Conserve_Browser_20260917_v2')
    source=ROOT/'outputs/SST_Conserve_Browser_20260917_v2/post_training_human_regression'
    rows=read_json(source/'challenge_rows.json')
    features=source/'features.npz'
    expected=read_json(out/'human_feature_contract.json')
    assert sha256(features)==expected['sha256']
    assert sha256(source/'challenge_rows.json')==expected['rows_sha256']
    with np.load(features) as f:
        assert str(f['rows_hash'])==digest(rows)
        assert str(f['encoder_sha256'])==read_json(base.DEPLOY)['encoder_sha256']
        x=f['x'].astype(np.float32);z=f['z'].copy()
    assert not {r.get('sha256',r.get('source_sha256')) for r in combined['rows']} & {r['sha256'] for r in rows}
    assert not {r['group_id'] for r in combined['rows']} & {r['group_id'] for r in rows}
    d=base.tensor_data(data,([dict(r,target=11 if r['binary_target'] else -1) for r in rows],x,z),model)
    truth=np.array([r['binary_target'] for r in rows])
    ps={'native148':scores(zero,d,True),'baseline150':scores(zero,d),'candidate150':scores(head,d)}
    human_result={k:human.binary_argmax_metrics(truth,v) for k,v in ps.items()}
    save_json(destination/'human_aquilaria_evaluation.json',dict(branches=human_result,source=str(source),feature_sha256=sha256(features),binary_truth_only=True,trained_or_selected_on_these=False))
    np.savez_compressed(destination/'human_aquilaria_predictions.npz',**ps,binary_target=truth)
    for r in read_json(out/'all_recipes.json'):verify_backup(r['backup'])
    base.verify_contract(data)
    save_json(out/'COMPLETE.json',dict(trained=True,recipes=2,selected=lock,production_unchanged=True,deployed=False,
              evaluation_scope='Development-selected; previously inspected historical regression only.',
              human_aquilaria=human_result,results={k:{'baseline150':base.compact(v['baseline150']),'candidate150':base.compact(v['candidate150']),'corrected':v['corrected'],'regressed':v['regressed']} for k,v in results.items()}))
    print('COMPLETE no deployment',human_result,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--stage',choices=['prepare','fit','evaluate'],required=True)
    args=parser.parse_args();{'prepare':prepare,'fit':fit,'evaluate':evaluate}[args.stage](args.out.resolve())
