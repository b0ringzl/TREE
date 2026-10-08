"""Three preregistered conservation recipes, one final validation lock.

Starts from the previous selected MLP ensemble; two zero-initialized novel
residual outputs complete a fair 150-class baseline. No production mutation.
"""
import argparse
import gc
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'tools')]
import numpy as np
import torch
import torch.nn.functional as F
from bioclip_probe import ResidualHead
from distillation_core import read_json, save_json, digest, collapse_logits, metrics, bootstrap_gain
from training_safety import sha256, backup_before_training, verify_backup
import train_artocarpus_user_20260918 as previous
import evaluate_artocarpus_aq_regression_20260918 as aq_audit

OUT = ROOT/'outputs/SST_Conservation_Iterative_20260918_v1'
PREV = previous.OUT
TARGETS = [11, 16, 148, 149]
RECIPES = [
    dict(name='A_balanced_continuation', seed=20260919, epochs=120, lr=3e-5, novel_lr=2e-4, focus_weight=1., kl=1., augmentation=False, margin=0.),
    dict(name='B_four_species_focus', seed=20260920, epochs=120, lr=3e-5, novel_lr=2e-4, focus_weight=3., kl=2., augmentation=False, margin=0.),
    dict(name='C_multiview_and_margin', seed=20260921, epochs=120, lr=3e-5, novel_lr=2e-4, focus_weight=3., kl=2., augmentation=True, margin=.1),
]


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


class Conservation150(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = torch.nn.ModuleList([ResidualHead('mlp') for _ in range(3)])
        self.novel = torch.nn.ModuleList([torch.nn.Linear(1024, 2) for _ in range(3)])
        for n in self.novel:
            torch.nn.init.zeros_(n.weight); torch.nn.init.zeros_(n.bias)

    def member_logits(self, x, z, params, native=False):
        tail = x @ params['extra_bank'].T * params['scale']
        novel = x @ params['extension_bank'].T * params['scale']
        members = []
        for h, n in zip(self.heads, self.novel):
            original = torch.cat([collapse_logits(h(x, z), params['groups']), tail], -1)
            full = original if native else torch.cat([original, novel+n(x)], -1)
            members.append(full/params['temperature'])
        return torch.stack(members)

    def logprob(self, x, z, params, native=False):
        return torch.logsumexp(self.member_logits(x,z,params,native).log_softmax(-1),dim=0)-math.log(3.)


def initial_model():
    lock = read_json(PREV/'selection_lock.json')
    require(sha256(lock['selected_checkpoint']) == lock['sha256'], 'Previous candidate changed')
    require(lock['epoch'] == 58, 'Previous selected epoch changed')
    state = torch.load(lock['selected_checkpoint'], map_location='cpu', weights_only=False)
    model = Conservation150().cuda()
    model.heads.load_state_dict(state['model_state'])
    return model


def parameters(out):
    with np.load(out/'train_features.npz', allow_pickle=False) as f:
        extension = torch.tensor(f['extension_bank'], dtype=torch.float32, device='cuda')
        require(torch.allclose(extension.norm(dim=-1),torch.ones(2,device='cuda'),atol=1e-6),'Novel text vectors must be normalized')
        return dict(extra_bank=torch.tensor(f['extra_bank'],device='cuda'), extension_bank=extension,
            scale=float(f['scale']), groups=read_json(out/'classes.json')['groups'],
            temperature=read_json(out/'teacher_choice.json')['calibration_temperature'])


def load_data(out, phase):
    rows=read_json(out/f'{phase}_rows.json')
    with np.load(out/f'{phase}_features.npz',allow_pickle=False) as f:
        require(str(f['rows_hash'])==digest(rows),'Feature rows mismatch '+phase)
        return rows,torch.tensor(f['x'].astype(np.float32),device='cuda'),torch.tensor(f['z'],device='cuda')


def predict(model,data,params,native=False):
    model.eval()
    with torch.no_grad():
        return torch.cat([model.logprob(data[1][i:i+256],data[2][i:i+256],params,native).exp()
                          for i in range(0,len(data[0]),256)]).cpu().numpy()


def score(rows,p,full=False):
    y=np.array([r['target'] for r in rows]); supported=y<p.shape[1]
    m=metrics(y[supported],p[supported])
    if not full: m={k:m[k] for k in ('n','top1','macro_f1','balanced_accuracy')}
    m['correct']=int((p[supported].argmax(1)==y[supported]).sum())
    focus={str(k):previous.binary(y,p,k) for k in TARGETS if k<p.shape[1]}
    return dict(all=m,focus=focus,native_supported_count=int(supported.sum()),input_count=len(rows))


def joint_score(valrows,newrows,pv,pn):
    return score(valrows+newrows,np.concatenate([pv,pn]))


def eligible(v,n,j,base):
    # Guard old categories against the native148 candidate AND fair150 start.
    floor_correct=max(base['native148_old']['all']['correct'],base['fair150_old']['all']['correct'])
    floor_macro=max(base['native148_old']['all']['macro_f1'],base['fair150_old']['all']['macro_f1'])
    if v['all']['correct']<floor_correct or v['all']['macro_f1']<floor_macro-1e-12: return False
    for key in ('11','16'):
        ref=base['native148_old']['focus'][key]; a=v['focus'][key]
        if a['tp']<ref['tp'] or a['fp']>ref['fp']: return False
    for key in map(str,TARGETS):
        a=j['focus'][key]; ref=base['fair150_joint']['focus'][key]
        if a['tp']<ref['tp'] or a['fp']>ref['fp'] or a['f1']<ref['f1']-1e-12: return False
        a=n['focus'][key]; ref=base['fair150_new']['focus'][key]
        if a['tp']<ref['tp'] or a['fp']>ref['fp']: return False
    return True


def rank(v,j):
    return (sum(j['focus'][str(k)]['f1'] for k in TARGETS)/4.,v['all']['correct'],v['all']['macro_f1'])


def unchanged(out):
    protocol=read_json(out/'protocol.json')
    for path,h in protocol['protected_sha256'].items():
        require(sha256(path)==h,'Protected input changed '+path)
    for name,h in protocol['study_inputs'].items():
        require(sha256(out/name)==h,'Study input changed '+name)


def protected_sources(out):
    contract=read_json(out/'contract.json')
    artifacts=contract['artifacts']
    return [Path(a['path']) for a in artifacts]


def freeze(out):
    require(not (out/'protocol.json').exists(),'Protocol already frozen')
    require((out/'augmentation/train_aug_features.npz').is_file(),'Augmentations must be finalized before recipe selection')
    for artifact in read_json(out/'contract.json')['artifacts']:
        require(sha256(artifact['path'])==artifact['sha256'],'Preparation source changed before freeze: '+artifact['path'])
    from artocarpus_iteration_features_20260918 import verify_lock
    augmentation_lock,_,_=verify_lock(out/'augmentation')
    completed=read_json(out/'augmentation/AUGMENTATION_COMPLETE.json')
    require(completed['features_sha256']==sha256(out/'augmentation/train_aug_features.npz'),'Augmentation completion feature hash mismatch')
    require(completed['augmentation_lock_sha256']==sha256(out/'augmentation/augmentation_lock.json'),'Augmentation completion lock mismatch')
    for path,h in aq_audit.FIXED_FILES.items():
        require(sha256(path)==h,'Historical AQ evidence changed: '+str(path))
    model=initial_model(); params=parameters(out)
    train,val,new=[load_data(out,p) for p in ('train','planning','newdev')]
    require(len(train[0])>=4198 and len(val[0])==2640 and len(new[0])==15,'Unexpected datasets')
    pv,pn,pt=[predict(model,d,params) for d in (val,new,train)]
    native=predict(model,val,params,True)
    with np.load(PREV/'fit/selected_predictions.npz') as f:
        require(np.max(np.abs(native-f['planning']))<5e-5,'Native initialization no longer matches best58')
    base=dict(native148_old=score(val[0],native), fair150_old=score(val[0],pv),
              fair150_new=score(new[0],pn), fair150_joint=joint_score(val[0],new[0],pv,pn))
    save_json(out/'baseline_development.json',base)
    np.savez_compressed(out/'baseline_development_predictions.npz',native148=native,planning=pv,newdev=pn,train=pt)
    torch.save(dict(model_state=previous.cpu_state(model),epoch=0,architecture='Conservation150',
        labels=read_json(out/'labels.json'),classes=read_json(out/'classes.json'),
        encoder_sha256=read_json(PREV/'protocol.json')['encoder_sha256'],parent=str(PREV/'fit/best.pt')),
        out/'initial150.pt')
    # Fixed hard negatives selected only using current initial TRAIN probabilities.
    hard={}
    for k in TARGETS:
        ids=[i for i,r in enumerate(train[0]) if r['target']!=k]
        ids=sorted(ids,key=lambda i:(-float(pt[i,k]),train[0][i]['sample_id']))[:256]
        hard[str(k)]=[train[0][i]['sample_id'] for i in ids]
    save_json(out/'margin_training_negative_ids.json',hard)
    study_inputs={str(p.relative_to(out)):sha256(p) for p in sorted(out.glob('*.json'))}
    study_inputs.update({str(p.relative_to(out)):sha256(p) for p in sorted(out.glob('*.npz'))})
    study_inputs['initial150.pt']=sha256(out/'initial150.pt')
    for path in sorted((out/'augmentation').glob('*.json')):
        study_inputs[str(path.relative_to(out))]=sha256(path)
    study_inputs['augmentation/train_aug_features.npz']=sha256(out/'augmentation/train_aug_features.npz')
    sources=list(dict.fromkeys([*previous.protected(),*protected_sources(out),
        PREV/'fit/best.pt',PREV/'selection_lock.json',Path(__file__),
        ROOT/'tools/train_artocarpus_user_20260918.py',ROOT/'tools/evaluate_artocarpus_aq_regression_20260918.py',
        *aq_audit.FIXED_FILES,
        aq_audit.CHALLENGE/'human_review.sqlite3',aq_audit.CHALLENGE/'source_manifest.json',aq_audit.CHALLENGE/'source_group_audit.json',
        aq_audit.CHALLENGE/'challenge_rows.json',PREV/'test_rows.json',PREV/'test_features.npz',
        ROOT/'tools/prepare_conservation_iteration_20260918.py',
        ROOT/'tools/artocarpus_iteration_features_20260918.py']))
    protocol=dict(created_at=datetime.now(timezone.utc).isoformat(),recipes=RECIPES,target_indices=TARGETS,
        target_species=[read_json(out/'labels.json')[k] for k in TARGETS],
        initialization='Each recipe independently starts from previous best58 MLPs + zero novel residuals. No recipe chaining.',
        architecture='Original raw148 MLP -> collapse147 + unchanged foxtail1 + two fixed normalized text logits and per-member Linear(1024,2) residuals -> temperature -> mean member probabilities.',
        trainable='Existing three MLPs and three novel two-output linear residuals. Encoder, label order, old collapse groups, text vectors, temperature frozen.',
        loss='class/source-group weighted parent-mean CE + replay-only KL to fair150 initial + optional 4-class bounded signed margin (0.2 log-probability units)',
        augmentation='Full-view flip, gamma0.85, gamma1.15. For selected parents average original+3 view CE before source/class weights; not independent samples/trees.',
        replay_anchor='Only old 4189 existing_replay original photos; excludes user9 and supplemental38; full150 normalized probability anchor.',
        selection='Per-run development epochs only, and one final comparison after all three. Four species joint F1/TP cannot drop, FP cannot increase vs fair150; source-development TP/FP also guarded. Old white/AQ guarded against native148, old overall correct and macroF1 at least max(native148,fair150). Rank mean4 F1, then old correct, then macro; earlier epoch/run wins tie.',
        valid_improvement='Must pass all guards AND strictly improve mean4 F1 over fair150 initial, not merely improve global accuracy.',
        stop='Exactly these 3 bounded recipes; do not append recipes after viewing regression. No gain means retain previous native148 candidate and report limitation.',
        evaluation='After one final lock only: original148 Demo, previous58 native148, fair expanded150 initial, selected150; all same cached currentDemo-pad inputs. Previously inspected regression, NOT fresh unbiased test.',
        new_species_evidence='Rhodoleia14 train/4 development and Keteleeria16 train/1 development, one source gallery each. No independent new positive test evidence for these two species.',
        deployed=False, protected_sha256={str(p.resolve()):sha256(p) for p in sources},study_inputs=study_inputs)
    save_json(out/'protocol.json',protocol)
    print('FROZEN THREE RECIPES',base,flush=True)


def weighted_groups(rows,multiplier=1.):
    groups=Counter((r['target'],r['group_id']) for r in rows); byclass=defaultdict(set)
    for r in rows: byclass[r['target']].add(r['group_id'])
    w=np.array([(multiplier if r['target'] in TARGETS else 1.)/(len(byclass[r['target']])*groups[(r['target'],r['group_id'])]) for r in rows],dtype=np.float32)
    return torch.tensor(w/w.sum(),device='cuda')


def group_weights(rows):
    counts=Counter(r['group_id'] for r in rows)
    w=np.array([1/counts[r['group_id']] for r in rows],dtype=np.float32)
    return torch.tensor(w/w.sum(),device='cuda')


def make_margin_sets(rows,out):
    hard=read_json(out/'margin_training_negative_ids.json'); sets=[]
    for k in TARGETS:
        ids=set(hard[str(k)]); pos=[i for i,r in enumerate(rows) if r['target']==k]
        neg=[i for i,r in enumerate(rows) if r['sample_id'] in ids]
        sets.append((k,torch.tensor(pos,device='cuda'),torch.tensor(neg,device='cuda'),
                     group_weights([rows[i] for i in pos]),group_weights([rows[i] for i in neg])))
    return sets


def margin_loss(lp,sets):
    losses=[]
    for k,pos,neg,pw,nw in sets:
        indices=torch.cat([pos,neg]); logits=lp[indices]; others=logits.clone(); others[:,k]=-torch.inf
        margin=logits[:,k]-others.max(-1).values
        losses.append(.5*(F.relu(.2-margin[:len(pos)])*pw).sum()+.5*(F.relu(.2+margin[len(pos):])*nw).sum())
    return torch.stack(losses).mean()


def fit(out,recipe_index):
    unchanged(out); protocol=read_json(out/'protocol.json'); recipe=protocol['recipes'][recipe_index]
    run=out/'runs'/recipe['name']; run.mkdir(parents=True,exist_ok=False)
    params=parameters(out); train,val,new=[load_data(out,p) for p in ('train','planning','newdev')]
    for row in train[0]:
        require(sha256(row['source_image'])==row['sha256'],'Training source changed before fit')
    model=Conservation150().cuda(); state=torch.load(out/'initial150.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(state['model_state']); torch.manual_seed(recipe['seed']);torch.cuda.manual_seed_all(recipe['seed'])
    baseline=read_json(out/'baseline_development.json')
    with np.load(out/'baseline_development_predictions.npz') as f:
        anchor=torch.tensor(f['train'],device='cuda')
        require(np.max(np.abs(predict(model,val,params)-f['planning']))<5e-5,'Recipe start differs')
    save_json(run/'config.json',dict(recipe,weight_decay=.01,clip_norm=5.,protocol_sha256=sha256(out/'protocol.json')))
    save_json(run/'baseline_evaluation.json',baseline)
    torch.save(state,run/'initial.pt')
    inputs=[*[Path(p) for p in protocol['protected_sha256']],*[out/p for p in protocol['study_inputs']],
        out/'protocol.json',run/'config.json',run/'baseline_evaluation.json',run/'initial.pt']
    backup=backup_before_training(run,inputs,dict(recipe=recipe['name'],seed=recipe['seed'],no_deployment=True))
    print('BACKUP VERIFIED',recipe['name'],backup,flush=True)
    unchanged(out)
    y=torch.tensor([r['target'] for r in train[0]],device='cuda'); weights=weighted_groups(train[0],recipe['focus_weight'])
    replay=torch.tensor([r['data_role']=='existing_replay' for r in train[0]],device='cuda')
    require(int(replay.sum())==4189,'Replay anchor includes new/extra data')
    margin_sets=make_margin_sets(train[0],out)
    aug=None
    if recipe['augmentation']:
        ar=read_json(out/'augmentation/train_aug_rows.json'); byid={r['sample_id']:i for i,r in enumerate(train[0])}
        ids=[byid[r['parent_sample_id']] for r in ar]
        for r,i in zip(ar,ids):
            require(r['parent_sha256']==train[0][i]['sha256'] and r['target']==train[0][i]['target'],'Augmentation identity changed')
        with np.load(out/'augmentation/train_aug_features.npz') as f:
            require(str(f['rows_hash'])==digest(ar),'Augmentation cache ordering')
            aug=(torch.tensor(f['x'].astype(np.float32),device='cuda'),torch.tensor(f['z'],device='cuda'),torch.tensor(ids,device='cuda'))
        view_count=torch.ones(len(train[0]),device='cuda').index_add_(0,aug[2],torch.ones(len(ar),device='cuda'))
    optimizer=torch.optim.AdamW([dict(params=model.heads.parameters(),lr=recipe['lr']),dict(params=model.novel.parameters(),lr=recipe['novel_lr'])],weight_decay=.01)
    best_state=previous.cpu_state(model); best_epoch=0; best_rank=rank(baseline['fair150_old'],baseline['fair150_joint']); history=[]
    model.requires_grad_(True)
    for epoch in range(1,recipe['epochs']+1):
        model.train();optimizer.zero_grad(set_to_none=True)
        lp=model.logprob(train[1],train[2],params)
        ce=F.nll_loss(lp,y,reduction='none')
        if aug is not None:
            al=model.logprob(aug[0],aug[1],params)
            ce=(ce.index_add(0,aug[2],F.nll_loss(al,y[aug[2]],reduction='none')))/view_count
        ce=(ce*weights).sum(); kl=F.kl_div(lp[replay],anchor[replay],reduction='batchmean')
        margin=margin_loss(lp,margin_sets) if recipe['margin'] else lp.sum()*0.
        loss=ce+recipe['kl']*kl+recipe['margin']*margin
        require(bool(torch.isfinite(loss)),'Nonfinite fit loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);optimizer.step()
        pv=predict(model,val,params);pn=predict(model,new,params)
        v,n=score(val[0],pv),score(new[0],pn);j=joint_score(val[0],new[0],pv,pn)
        ok=eligible(v,n,j,baseline);rk=rank(v,j)
        if ok and rk>best_rank:
            best_state,best_epoch,best_rank=previous.cpu_state(model),epoch,rk
        history.append(dict(epoch=epoch,loss=float(loss.detach()),ce=float(ce.detach()),kl=float(kl.detach()),margin=float(margin.detach()),
                            old=v,new=n,joint=j,eligible=bool(ok),selected_so_far=best_epoch))
        if epoch%20==0:print('FIT',recipe['name'],epoch,'selected',best_epoch,'mean4F1',rk[0],'correct',v['all']['correct'],flush=True)
    torch.save(dict(model_state=previous.cpu_state(model),optimizer_state=optimizer.state_dict(),epoch=recipe['epochs'],config=recipe),run/'last.pt')
    save_json(run/'history.json',history)
    torch.save(dict(model_state=best_state,epoch=best_epoch,config=recipe,labels=read_json(out/'labels.json'),
        classes=read_json(out/'classes.json'),encoder_sha256=state['encoder_sha256'],architecture='Conservation150'),run/'best.pt')
    model.load_state_dict(best_state);pv=predict(model,val,params);pn=predict(model,new,params)
    v,n=score(val[0],pv),score(new[0],pn);j=joint_score(val[0],new[0],pv,pn)
    valid=eligible(v,n,j,baseline) and rank(v,j)[0]>rank(baseline['fair150_old'],baseline['fair150_joint'])[0]+1e-12
    save_json(run/'result.json',dict(recipe=recipe['name'],epoch=best_epoch,old=v,new=n,joint=j,
        valid_focus_gain=bool(valid),selected_checkpoint=str(run/'best.pt'),sha256=sha256(run/'best.pt'),
        backup=str(backup),optimizer_steps=recipe['epochs'],deployed=False))
    np.savez_compressed(run/'development_predictions.npz',planning=pv,newdev=pn)
    verify_backup(backup);unchanged(out)
    print('RUN COMPLETE',recipe['name'],best_epoch,'valid focus improvement',valid,flush=True)


def lock(out):
    unchanged(out); require(not (out/'selection_lock.json').exists(),'Final lock exists')
    baseline=read_json(out/'baseline_development.json');best_rank=rank(baseline['fair150_old'],baseline['fair150_joint'])
    winner=None;runs=[]
    for recipe in RECIPES:
        result=read_json(out/'runs'/recipe['name']/'result.json');runs.append(result)
        require(sha256(result['selected_checkpoint'])==result['sha256'],'Run checkpoint changed')
        valid=eligible(result['old'],result['new'],result['joint'],baseline) and rank(result['old'],result['joint'])[0]>best_rank[0]+1e-12
        # Compare full rank to a previously selected valid model; mean-focus must exceed INITIAL, not each earlier recipe.
        gain=rank(result['old'],result['joint'])[0]>rank(baseline['fair150_old'],baseline['fair150_joint'])[0]+1e-12
        if eligible(result['old'],result['new'],result['joint'],baseline) and gain and rank(result['old'],result['joint'])>best_rank:
            winner=result;best_rank=rank(result['old'],result['joint'])
    selected=Path(winner['selected_checkpoint']) if winner else out/'initial150.pt'
    save_json(out/'selection_lock.json',dict(locked_at=datetime.now(timezone.utc).isoformat(),protocol_sha256=sha256(out/'protocol.json'),
        selected=str(selected),sha256=sha256(selected),winner=winner['recipe'] if winner else 'no-qualified-improvement',
        epoch=winner['epoch'] if winner else 0,valid_focus_improvement=winner is not None,
        fallback='Keep previous native148 candidate; do not deploy an unqualified expanded150 model' if winner is None else None,
        runs=runs,all_three_complete=True,no_current_study_test_predictions_used=True,deployed=False))
    print('FINAL LOCK',winner['recipe'] if winner else 'baseline retained',flush=True)


def compare(rows,baseline,candidate):
    y=np.array([r['target'] for r in rows]); a=baseline.argmax(1);b=candidate.argmax(1)
    require(np.all(y<baseline.shape[1]) and np.all(y<candidate.shape[1]),'Unsupported ground truth in comparison')
    return dict(baseline=score(rows,baseline,True),candidate=score(rows,candidate,True),
        corrected=int(((a!=y)&(b==y)).sum()),regressed=int(((a==y)&(b!=y)).sum()),
        corrected_ids=[r['sample_id'] for r,t in zip(rows,(a!=y)&(b==y)) if t],
        regressed_ids=[r['sample_id'] for r,t in zip(rows,(a==y)&(b!=y)) if t],
        paired_bootstrap=bootstrap_gain(y,baseline,candidate,[r['group_id'] for r in rows],seed=20260918,repeats=2000))


def evaluate(out):
    unchanged(out); sl=read_json(out/'selection_lock.json')
    require(sha256(sl['selected'])==sl['sha256'] and sl['all_three_complete'],'Invalid final lock')
    ev=out/'final_regression';ev.mkdir(exist_ok=False)
    initial=initial_model();candidate=Conservation150().cuda()
    candidate.load_state_dict(torch.load(sl['selected'],map_location='cpu',weights_only=False)['model_state'])
    params=parameters(out)
    # Only here is the historical test cache opened, after the study-wide final lock.
    test=previous.load_data(PREV,'test')
    oldrows=read_json(PREV/'test_rows.json')
    require(digest(test[0])==digest(oldrows),'Regression cohort mismatch')
    p150=predict(initial,test,params); native=predict(initial,test,params,True); pc=predict(candidate,test,params)
    prod=previous.heads_from_deployment(read_json(PREV/'protocol.json'))
    production=previous.predict(prod,test,dict(groups=params['groups'],temp=params['temperature'],extra=params['extra_bank'],scale=params['scale']))
    np.savez_compressed(ev/'test_predictions.npz',original_demo148=production,previous_best148=native,fair_initial150=p150,candidate150=pc,targets=np.array([r['target'] for r in test[0]]),rows_hash=digest(test[0]))
    results=dict(fair150=compare(test[0],p150,pc),previous_native148=compare(test[0],native,pc),original_demo148=compare(test[0],production,pc))
    save_json(ev/'test_results.json',results)
    aqbase=ROOT/'outputs/SST_Conserve_Browser_20260917_v2/post_training_human_regression'
    aqrows=read_json(aqbase/'challenge_rows.json')
    require(len(aqrows)==92 and aqrows==read_json(aq_audit.CHALLENGE/'challenge_rows.json'),'Human regression snapshot changed')
    require(sum(r['binary_target'] for r in aqrows)==33,'Human binary cohort changed')
    train_tokens={t for r in read_json(out/'train_rows.json') for t in aq_audit.identity_tokens(r)}
    require(not any(aq_audit.identity_tokens(r)&train_tokens for r in aqrows),'AQ regression overlaps training')
    with np.load(aqbase/'features.npz') as f:
        require(str(f['rows_hash'])==digest(aqrows),'AQ cache rows mismatch')
        require(str(f['encoder_sha256'])==read_json(PREV/'protocol.json')['encoder_sha256'],'AQ cache encoder mismatch')
        aq=(aqrows,torch.tensor(f['x'].astype(np.float32),device='cuda'),torch.tensor(f['z'],device='cuda'))
    y=[]
    for r in aqrows:
        require(sha256(r['source_image'])==(r.get('sha256') or r.get('source_sha256')),'AQ source changed')
        require('binary_target' in r,'Missing human binary target')
        h=r['human_review']
        require(h['reviewer_identity']=='human' and h['decision'] in ('aquilaria','not_aquilaria'),'Unconfirmed human binary label')
        require(h['sha256']==r['sha256'] and int(h['decision']=='aquilaria')==r['binary_target'],'Human decision does not match content/target')
        y.append(11 if int(r['binary_target'])==1 else -1)
    pb=predict(initial,aq,params);pn=predict(initial,aq,params,True);pq=predict(candidate,aq,params)
    aqresult=dict(n=len(aqrows),ground_truth='Previously human-reviewed AQ/nonAQ only; no source-species truth for negatives',
        previous_native148=previous.binary(y,pn,11),fair_initial150=previous.binary(y,pb,11),candidate150=previous.binary(y,pq,11))
    save_json(ev/'aq92_results.json',aqresult)
    np.savez_compressed(ev/'aq92_predictions.npz',previous_native148=pn,fair_initial150=pb,candidate150=pq,binary_y=np.array(y))
    unchanged(out)
    for run in sl['runs']:verify_backup(run['backup'])
    save_json(out/'COMPLETE.json',dict(completed_at=datetime.now(timezone.utc).isoformat(),fit_attempts=3,optimizer_steps=360,
        selected=sl['selected'],winner=sl['winner'],valid_focus_improvement=sl['valid_focus_improvement'],
        historical_test_scored_only_after_final_lock=True,deployed=False,production_unchanged=True))
    print('EVALUATION COMPLETE',out,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','fit','lock','evaluate']);parser.add_argument('--recipe',type=int,choices=[0,1,2]);parser.add_argument('--out',type=Path,default=OUT)
    args=parser.parse_args();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    require(args.out.resolve().is_relative_to(ROOT/'outputs'),'Output outside workspace outputs')
    if args.stage=='fit':require(args.recipe is not None,'Choose a preregistered recipe');fit(args.out,args.recipe)
    else:globals()[args.stage](args.out)
