"""Second controlled stage: matched dual-view training and TRAIN-only negatives.

Every fit starts at deployed weights and has its own physical verified backup.
No test labels/predictions select negatives, a checkpoint, or hyperparameters.
"""
import argparse
import math
import shutil
import sys
import time
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import numpy as np
import torch
import torch.nn.functional as F
from distillation_core import canonical_image,digest,read_json,save_json,bootstrap_gain
from training_safety import sha256,backup_before_training,verify_backup
from train_artocarpus_user_20260918 import (protected,heads_from_deployment,logprob,predict,measure,
                                         cpu_state,Engine,SSTV1_MANIFEST)
from train_aq_ar_diagnostic_20260919 import (verify_images,evaluate_dev,rank,unchanged,load_data)
from import_artocarpus_user_20260918 import current_human_review_exclusions
from prepare_artocarpus_fit_20260918 import identities,row_ids

OUT=ROOT/'outputs/SSTv1_AQ_AR_Dualview_20260919_v1'
PREV=ROOT/'outputs/SSTv1_Artocarpus_User_20260918_v1'
DIAG=ROOT/'outputs/SSTv1_AQ_AR_Diagnostic_20260919_v1'
AUDIT=ROOT/'outputs/artocarpus_preprocess_audit_20260917_v1'
SEED=20260920
RECIPES=[dict(name='D_matched_views',focus=1.,negative_weight=0.),
         dict(name='E_focus2',focus=2.,negative_weight=0.),
         dict(name='F_focus2_hardneg',focus=2.,negative_weight=.1)]


def tensor(a):
    return torch.tensor(np.asarray(a,dtype=np.float32),device='cuda')


def prepare(out):
    out.mkdir(parents=True,exist_ok=False)
    rows=read_json(PREV/'train_rows.json');review=current_human_review_exclusions()
    assert not any(r['sha256'] in review['blocked_sha256'] or row_ids(r)&set(review['blocked_ids']) for r in rows)
    held=read_json(PREV/'heldout_metadata.json')
    tokens=set().union(*(set(r['identity_tokens']) for r in held['records']))
    assert not any(identities(r)&tokens for r in rows)
    verify_images(rows)
    old=read_json(PREV/'fit/pre_training_backup/backup_manifest.json')
    hashes={str(Path(r['source']).resolve()):r['sha256'] for r in old['files']}
    assert sha256(PREV/'train_features.npz')==hashes[str((PREV/'train_features.npz').resolve())]
    for name in ('train_rows.json','classes.json','labels.json','teacher_choice.json','heldout_metadata.json','train_features.npz'):
        shutil.copy2(PREV/name,out/name)
    save_json(out/'human_review_snapshot.json',review)
    dep=read_json(SSTV1_MANIFEST)
    protocol=dict(created_at=datetime.now(timezone.utc).isoformat(),seed=SEED,steps=120,eval_every=10,
        recipes=RECIPES,lr=3e-5,weight_decay=.01,clip_norm=5.,kl_weight=.5,negative_margin=.3,
        train_count=len(rows),deployment_manifest=dep,temperature=dep['temperature'],
        views='Exact Demo letterbox plus exact short-side224 center crop; .5/.5 probability mean during BOTH fit and inference',
        independent_samples='4198 parent photographs, not 8396 independent trees; retain original source groups',
        loss='Source/class-balanced NLL of dual-view mean, plus KL .5 to initial dual mean; F adds .1 source-balanced hard-negative true-vs-focal softplus margin .3',
        negatives='Frozen from TRAIN only: top160 non-focal rows per focal species by starting dual p(target), max3/source group; no label changes',
        selection='Compared against BOTH initial single and dual: focal TP>=each, FP<=each and F1>=each within old/new/combined dev; overall correct>=each and macroF1>=each-.001. Maximize mean focalF1/min focalrecall/overall correct/macroF1. Initial dual is fallback.',
        test='Historical regression only after lock. Fresh acquisitions quarantined, never used in fitting or checkpoint selection.',
        trainable='Only three existing residual MLPs, native148 compatibility. Frozen BioCLIP/text banks/temperature.',
        deployed=False,protected_sha256={str(p.resolve()):sha256(p) for p in protected()},
        external_cache_sha256={str(p.resolve()):sha256(p) for p in [AUDIT/'paired_features_predictions.npz',
            DIAG/'exact_regression/center_features.npz',PREV/'test_features.npz']})
    save_json(out/'protocol.json',protocol)
    print('PREPARED',len(rows),flush=True)


def features(out):
    protocol=read_json(out/'protocol.json');unchanged(protocol)
    assert not (out/'train_center_features.npz').exists()
    chunks=out/'center_chunks';chunks.mkdir(exist_ok=False)
    rows=read_json(out/'train_rows.json');verify_images(rows)
    torch.set_num_threads(4);engine=Engine();engine.load()
    assert engine.sstv1_version==protocol['deployment_manifest']['version']
    xx=[];zz=[];started=time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as pool,torch.inference_mode():
        for start in range(0,len(rows),64):
            part=rows[start:start+64]
            pixels=list(pool.map(lambda r:canonical_image(r['source_image']),part))
            x=[];z=[]
            for pixel in pixels:
                e=engine.model.encode_image(((pixel[None].cuda()-engine.mean)/engine.std).to(engine.dtype),normalize=True).float()
                raw=e@engine.bank.T*engine.model.logit_scale.exp().float()
                x.append(e.half().cpu().numpy());z.append(raw.cpu().numpy())
            x=np.concatenate(x);z=np.concatenate(z);xx.append(x);zz.append(z)
            np.savez_compressed(chunks/f'{start:06d}.npz',x=x,z=z,rows_hash=digest(part))
            print('TRAIN_CENTER',min(start+64,len(rows)),len(rows),'seconds',round(time.perf_counter()-started),flush=True)
    np.savez_compressed(out/'train_center_features.npz',x=np.concatenate(xx),z=np.concatenate(zz),rows_hash=digest(rows),encoder_sha256=engine.encoder_sha256)
    unchanged(protocol)
    save_json(out/'FEATURES_COMPLETE.json',dict(n=len(rows),batch=1,encoder_sha256=engine.encoder_sha256,
        center_sha256=sha256(out/'train_center_features.npz'),pad_sha256=sha256(out/'train_features.npz')))


def load_pair(out,phase):
    if phase=='train':
        rows,x,z=load_data(out,'train');path=out/'train_center_features.npz'
        with np.load(path) as f:
            assert str(f['rows_hash'])==digest(rows)
            xc,zc=tensor(f['x']),tensor(f['z'])
    elif phase=='dev':
        rows=read_json(AUDIT/'rows.json')
        with np.load(AUDIT/'paired_features_predictions.npz') as f:
            assert str(f['rows_hash'])==digest(rows)
            x,z,xc,zc=[tensor(f[k]) for k in ('demo_letterbox_x','demo_letterbox_z','center_crop_x','center_crop_z')]
    elif phase=='test':
        assert (out/'selection_lock.json').exists()
        rows,x,z=load_data(PREV,'test')
        with np.load(DIAG/'exact_regression/center_features.npz') as f:
            assert str(f['rows_hash'])==digest(rows)
            xc,zc=tensor(f['x']),tensor(f['z'])
    return rows,x,z,xc,zc


def paired_logprob(heads,data,params):
    return torch.logaddexp(logprob(heads,data[1],data[2],**params),
                          logprob(heads,data[3],data[4],**params))-math.log(2.)


def paired_predict(heads,data,params):
    heads.eval()
    with torch.inference_mode():
        return torch.cat([paired_logprob(heads,(data[0][i:i+256],*[a[i:i+256] for a in data[1:]]),params).exp()
            for i in range(0,len(data[0]),256)]).cpu().numpy()


def allowed(m,baselines):
    for b in baselines:
        for cohort in ('combined','old','new'):
            for sp in ('aquilaria','white'):
                a,c=m[cohort][sp],b[cohort][sp]
                if a['tp']<c['tp'] or a['fp']>c['fp'] or a['f1']<c['f1']-1e-12:return False
        if m['combined']['all']['correct']<b['combined']['all']['correct']:return False
        if m['combined']['all']['macro_f1']<b['combined']['all']['macro_f1']-.001:return False
    return True


def fit(out):
    assert not (out/'baseline_evaluation.json').exists(), 'New version and backup required for retries'
    protocol=read_json(out/'protocol.json');unchanged(protocol)
    for path,h in protocol['external_cache_sha256'].items():assert sha256(path)==h
    done=read_json(out/'FEATURES_COMPLETE.json')
    assert sha256(out/'train_center_features.npz')==done['center_sha256']
    assert sha256(out/'train_features.npz')==done['pad_sha256']
    review=current_human_review_exclusions();rows=read_json(out/'train_rows.json')
    assert not any(r['sha256'] in review['blocked_sha256'] or row_ids(r)&set(review['blocked_ids']) for r in rows)
    save_json(out/'human_review_pre_fit.json',review)
    torch.set_num_threads(4)
    train,dev=[load_pair(out,p) for p in ('train','dev')]
    with np.load(out/'train_features.npz') as f:
        params=dict(groups=read_json(out/'classes.json')['groups'],temp=protocol['temperature'],extra=tensor(f['extra_bank']),scale=float(f['scale']))
    heads=heads_from_deployment(protocol);initial=cpu_state(heads)
    bt=paired_predict(heads,train,params);bd=paired_predict(heads,dev,params);bs=predict(heads,dev[:3],params)
    bases=[evaluate_dev(dev[0],p) for p in (bs,bd)]
    save_json(out/'baseline_evaluation.json',dict(train=measure(rows,bt),single=bases[0],dual=bases[1]))
    np.savez_compressed(out/'baseline_predictions.npz',train_dual=bt,dev_dual=bd,dev_single=bs)
    torch.save(dict(model_state=initial),out/'initial.pt')
    hard=[]
    for target in (11,16):
        counts=Counter();selected=0
        for i in np.argsort(-bt[:,target],kind='stable'):
            r=rows[i]
            if r['target'] in (11,16) or counts[r['group_id']]>=3:continue
            hard.append(dict(**r,training_index=int(i),negative_for=target,initial_focal_probability=float(bt[i,target]),
                             mining_scope='TRAIN ONLY; original source label unchanged'))
            counts[r['group_id']]+=1;selected+=1
            if selected==160:break
    assert len(hard)==320
    save_json(out/'hard_negative_manifest.json',hard)
    hard_i=torch.tensor([r['training_index'] for r in hard],device='cuda')
    hard_t=torch.tensor([r['negative_for'] for r in hard],device='cuda')
    hcounts=Counter((r['negative_for'],r['group_id']) for r in hard)
    hw=tensor([1/hcounts[(r['negative_for'],r['group_id'])] for r in hard]);hw/=hw.sum()
    gc=Counter((r['target'],r['group_id']) for r in rows);cg=defaultdict(set)
    for r in rows:cg[r['target']].add(r['group_id'])
    w0=np.array([1/(len(cg[r['target']])*gc[(r['target'],r['group_id'])]) for r in rows])
    y=torch.tensor([r['target'] for r in rows],device='cuda');anchor=tensor(bt)
    best=dict(recipe='initial_dual',step=0,checkpoint=str(out/'initial.pt'),metrics=bases[1])
    for recipe in protocol['recipes']:
        run=out/recipe['name'];run.mkdir(exist_ok=False)
        save_json(run/'config.json',dict(**recipe,seed=SEED,steps=120,lr=protocol['lr'],no_resume=True))
        backup=backup_before_training(run,[*protected(),*sorted(out.glob('*.json')),
            out/'initial.pt',*sorted(out.glob('*.npz')),AUDIT/'paired_features_predictions.npz',AUDIT/'rows.json',
            run/'config.json',Path(__file__),ROOT/'tools/training_safety.py'],dict(recipe=recipe,seed=SEED))
        print('BACKUP_VERIFIED',recipe['name'],flush=True)
        torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED);np.random.seed(SEED)
        heads=heads_from_deployment(protocol)
        w=w0*np.where(np.isin(y.cpu().numpy(),[11,16]),recipe['focus'],1.);weights=tensor(w/w.sum())
        opt=torch.optim.AdamW(heads.parameters(),lr=protocol['lr'],weight_decay=protocol['weight_decay'])
        history=[]
        for step in range(1,121):
            heads.train();opt.zero_grad(set_to_none=True)
            lp=paired_logprob(heads,train,params)
            ce=(F.nll_loss(lp,y,reduction='none')*weights).sum()
            kl=F.kl_div(lp,anchor,reduction='batchmean')
            negative=(F.softplus(lp[hard_i,hard_t]-lp[hard_i,y[hard_i]]+protocol['negative_margin'])*hw).sum()
            loss=ce+protocol['kl_weight']*kl+recipe['negative_weight']*negative
            assert bool(torch.isfinite(loss))
            loss.backward();torch.nn.utils.clip_grad_norm_(heads.parameters(),5.,error_if_nonfinite=True);opt.step()
            if step%10:continue
            pred=paired_predict(heads,dev,params);m=evaluate_dev(dev[0],pred);ok=allowed(m,bases)
            if ok and rank(m)>rank(best['metrics']):
                checkpoint=run/f'step_{step:03d}.pt'
                torch.save(dict(model_state=cpu_state(heads),step=step,recipe=recipe),checkpoint)
                best=dict(recipe=recipe['name'],step=step,checkpoint=str(checkpoint),metrics=m)
            history.append(dict(step=step,ce=float(ce.detach()),kl=float(kl.detach()),negative=float(negative.detach()),
                eligible=ok,metrics=m))
            print('FIT',recipe['name'],step,'AQ',m['combined']['aquilaria'],'AR',m['combined']['white'],'best',best['recipe'],best['step'],flush=True)
        torch.save(dict(model_state=cpu_state(heads),step=120,recipe=recipe),run/'last.pt')
        save_json(run/'history.json',history)
        save_json(run/'train_evaluation.json',measure(rows,paired_predict(heads,train,params)))
        verify_backup(backup);unchanged(protocol)
    best.update(sha256=sha256(best['checkpoint']),locked_at=datetime.now(timezone.utc).isoformat(),
        used_test_for_selection=False,used_fresh_for_selection=False,deployed=False)
    save_json(out/'selection_lock.json',best)
    print('LOCKED',best['recipe'],best['step'],flush=True)


def evaluate(out):
    assert not (out/'regression_results.json').exists()
    protocol=read_json(out/'protocol.json');lock=read_json(out/'selection_lock.json');unchanged(protocol)
    assert sha256(lock['checkpoint'])==lock['sha256']
    test=load_pair(out,'test');train=load_pair(out,'train');verify_images(test[0])
    with np.load(out/'train_features.npz') as f:
        params=dict(groups=read_json(out/'classes.json')['groups'],temp=protocol['temperature'],extra=tensor(f['extra_bank']),scale=float(f['scale']))
    base=heads_from_deployment(protocol);candidate=heads_from_deployment(protocol)
    candidate.load_state_dict(torch.load(lock['checkpoint'],map_location='cpu',weights_only=False)['model_state'])
    bp=predict(base,test[:3],params);bd=paired_predict(base,test,params);cp=paired_predict(candidate,test,params)
    trainp=paired_predict(candidate,train,params)
    y=np.array([r['target'] for r in test[0]])
    result=dict(baseline_single=measure(test[0],bp,False),baseline_dual=measure(test[0],bd,False),
        candidate_dual=measure(test[0],cp,False),corrected=int(((bp.argmax(1)!=y)&(cp.argmax(1)==y)).sum()),
        regressed=int(((bp.argmax(1)==y)&(cp.argmax(1)!=y)).sum()),
        interpretation='Previously inspected historical regression, NOT fresh or unbiased evidence.',
        bootstrap=bootstrap_gain(y,bd,cp,[r['group_id'] for r in test[0]],seed=SEED,repeats=2000))
    np.savez_compressed(out/'regression_predictions.npz',baseline=bp,baseline_dual=bd,candidate=cp,targets=y)
    np.savez_compressed(out/'selected_train_predictions.npz',probabilities=trainp,rows_hash=digest(train[0]))
    save_json(out/'regression_results.json',result)
    unchanged(protocol)
    save_json(out/'COMPLETE.json',dict(trained=True,optimizer_steps=360,deployed=False,production_unchanged=True,
        selected=lock,backups=[str(out/r['name']/'pre_training_backup') for r in RECIPES]))
    print('COMPLETE', {k:{s:result[k][s] for s in ('aquilaria','white')} for k in ('baseline_single','baseline_dual','candidate_dual')},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','features','fit','evaluate']);p.add_argument('--out',type=Path,default=OUT)
    a=p.parse_args();globals()[a.phase](a.out.resolve())
