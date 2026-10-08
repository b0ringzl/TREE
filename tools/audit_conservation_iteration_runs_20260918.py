"""Independent CPU-only completed-run audit. Never reads test predictions."""
import json
import math
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from training_safety import sha256

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/'outputs/SST_Conservation_Iterative_20260918_v1'
OUT = ROOT/'outputs/conservation_iteration_independent_audit_20260918_v1'
FOCUS = ('11','16','148','149')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def gates(v,n,j,base):
    errors=[]
    floor=max(base['native148_old']['all']['correct'],base['fair150_old']['all']['correct'])
    macro=max(base['native148_old']['all']['macro_f1'],base['fair150_old']['all']['macro_f1'])
    if v['all']['correct']<floor: errors.append('old_correct')
    if v['all']['macro_f1']<macro-1e-12: errors.append('old_macro_f1')
    for k in ('11','16'):
        a,b=v['focus'][k],base['native148_old']['focus'][k]
        if a['tp']<b['tp']:errors.append('native_'+k+'_tp')
        if a['fp']>b['fp']:errors.append('native_'+k+'_fp')
    for k in FOCUS:
        a,b=j['focus'][k],base['fair150_joint']['focus'][k]
        if a['tp']<b['tp']:errors.append('joint_'+k+'_tp')
        if a['fp']>b['fp']:errors.append('joint_'+k+'_fp')
        if a['f1']<b['f1']-1e-12:errors.append('joint_'+k+'_f1')
        a,b=n['focus'][k],base['fair150_new']['focus'][k]
        if a['tp']<b['tp']:errors.append('new_'+k+'_tp')
        if a['fp']>b['fp']:errors.append('new_'+k+'_fp')
    return errors


def ranking(v,j):
    return (sum(j['focus'][k]['f1'] for k in FOCUS)/4,v['all']['correct'],v['all']['macro_f1'])


def binary(y,p,k):
    a,b=y==k,p.argmax(1)==k
    tp,fp,fn,tn=[int(v.sum()) for v in (a&b,~a&b,a&~b,~a&~b)]
    return dict(tp=tp,fp=fp,fn=fn,tn=tn,precision=tp/(tp+fp) if tp+fp else 0.,
                recall=tp/(tp+fn) if tp+fn else 0.,f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)


def scores(rows,p):
    y=np.array([r['target'] for r in rows]);pred=p.argmax(1);c=p.shape[1]
    require(p.shape==(len(y),150),'Development probabilities shape differs')
    require(np.isfinite(p).all() and np.allclose(p.sum(1),1.,atol=1e-5),'Invalid probabilities')
    cm=np.zeros((c,c),dtype=np.int64);np.add.at(cm,(y,pred),1)
    tp=np.diag(cm);support=cm.sum(1);den=support+cm.sum(0);present=support>0
    f1=np.divide(2*tp,den,out=np.zeros(c),where=den>0)
    recall=np.divide(tp,support,out=np.zeros(c),where=support>0)
    return dict(all=dict(n=len(y),top1=float((y==pred).mean()),macro_f1=float(f1[present].mean()),
        balanced_accuracy=float(recall[present].mean()),correct=int((y==pred).sum())),
        focus={str(k):binary(y,p,k) for k in (11,16,148,149)},native_supported_count=len(y),input_count=len(y))


def equal_metrics(a,b):
    if isinstance(a,dict):
        return set(a)==set(b) and all(equal_metrics(a[k],b[k]) for k in a)
    if isinstance(a,float):
        return abs(a-b)<1e-10
    return a==b


def main():
    torch.set_num_threads(2)
    protocol=read(RUN/'protocol.json');base=read(RUN/'baseline_development.json')
    recipes=protocol['recipes']
    require(all((RUN/'runs'/r['name']/'result.json').exists() for r in recipes),'Runs not all complete')
    OUT.mkdir(parents=True,exist_ok=False)
    for name,h in protocol['study_inputs'].items():require(sha256(RUN/name)==h,'Study changed: '+name)
    for path,h in protocol['protected_sha256'].items():require(sha256(path)==h,'Protected source changed: '+path)
    original=torch.load(RUN/'initial150.pt',map_location='cpu',weights_only=False)
    parent=torch.load(ROOT/'outputs/SSTv1_Artocarpus_User_20260918_v1/fit/best.pt',map_location='cpu',weights_only=False)
    require(all(torch.equal(original['model_state']['heads.'+k],v) for k,v in parent['model_state'].items()),'Not a true warm start')
    require(all(torch.count_nonzero(v)==0 for k,v in original['model_state'].items() if k.startswith('novel.')),'Novel branch not zero initialized')
    vr,nr=read(RUN/'planning_rows.json'),read(RUN/'newdev_rows.json')
    results=[];allfiles=0;allbytes=0;source_hash_cache={}
    for recipe in recipes:
        path=RUN/'runs'/recipe['name'];result=read(path/'result.json');history=read(path/'history.json')
        require(len(history)==120 and [r['epoch'] for r in history]==list(range(1,121)),'Incomplete history')
        require(read(path/'baseline_evaluation.json')==base,'Per-run baseline differs')
        initial=torch.load(path/'initial.pt',map_location='cpu',weights_only=False)
        last=torch.load(path/'last.pt',map_location='cpu',weights_only=False)
        best=torch.load(path/'best.pt',map_location='cpu',weights_only=False)
        require(set(initial['model_state'])==set(original['model_state']) and all(torch.equal(v,original['model_state'][k]) for k,v in initial['model_state'].items()),'Recipe initialization differs')
        require(last['epoch']==120 and result['optimizer_steps']==120,'Not120 optimizer steps')
        states=last['optimizer_state']['state'];require(len(states)==24,'Expected24 Adam parameter states')
        steps={int(v['step']) for v in states.values()};require(steps=={120},'Adam state steps differ')
        deltas={k:float((last['model_state'][k]-v).abs().max()) for k,v in initial['model_state'].items()}
        require(all(d>0 for d in deltas.values()),'Some trainable parameters never changed')
        shared={name:max(v for k,v in deltas.items() if k.startswith(name+'.')) for name in ['heads.0','heads.1','heads.2','novel.0','novel.1','novel.2']}
        best_rank=ranking(base['fair150_old'],base['fair150_joint']);epoch=0;chosen=None;fail_counts=Counter();eligible_count=0
        for row in history:
            require(all(math.isfinite(row[k]) for k in ('loss','ce','kl','margin')),'Nonfinite loss')
            bad=gates(row['old'],row['new'],row['joint'],base)
            require(row['eligible']==(not bad),'Recorded eligibility differs')
            fail_counts.update(bad);eligible_count+=not bad
            rank=ranking(row['old'],row['joint'])
            if not bad and rank>best_rank:best_rank,epoch,chosen=rank,row['epoch'],row
            require(row['selected_so_far']==epoch,'Best-so-far selection differs')
        require(result['epoch']==epoch==best['epoch'],'Selected epoch differs')
        require(sha256(path/'best.pt')==result['sha256'],'Selected SHA differs')
        with np.load(path/'development_predictions.npz',allow_pickle=False) as f:
            require(set(f.files)=={'planning','newdev'},'Unexpected prediction keys')
            v,n=scores(vr,f['planning']),scores(nr,f['newdev']);j=scores(vr+nr,np.concatenate([f['planning'],f['newdev']]))
        require(equal_metrics(v,result['old']) and equal_metrics(n,result['new']) and equal_metrics(j,result['joint']),'Selected prediction metrics differ')
        if chosen is not None:require(equal_metrics(v,chosen['old']) and equal_metrics(n,chosen['new']) and equal_metrics(j,chosen['joint']),'Selected epoch metrics differ')
        else:require(equal_metrics(v,base['fair150_old']) and equal_metrics(n,base['fair150_new']),'Epoch0 not baseline')
        valid=not gates(v,n,j,base) and ranking(v,j)[0]>ranking(base['fair150_old'],base['fair150_joint'])[0]+1e-12
        require(result['valid_focus_gain']==valid,'Focus gain flag differs')
        backup=Path(result['backup']);manifest=read(backup/'backup_manifest.json');bytes_=0
        require(backup.parent==path,'Backup not run-specific')
        for entry in manifest['files']:
            src=Path(entry['source']);dest=backup/entry['backup']
            require(dest.is_file() and dest.stat().st_size==entry['bytes'],'Backup size differs')
            require(not os.path.samefile(src,dest),'Backup is same hardlinked file')
            require(dest.stat().st_nlink==1,'Backup has hard links')
            require(sha256(dest)==entry['sha256'],'Corrupted backup copy: '+str(dest))
            if str(src) not in source_hash_cache:source_hash_cache[str(src)]=sha256(src)
            require(source_hash_cache[str(src)]==entry['sha256'],'Backup source changed')
            bytes_+=entry['bytes']
        required=set(protocol['protected_sha256'])|{str((RUN/p).resolve()) for p in protocol['study_inputs']}|{str((RUN/'protocol.json').resolve())}
        copied={str(Path(e['source']).resolve()) for e in manifest['files']}
        require(required<=copied,'Backup missing frozen inputs')
        require(datetime.fromisoformat(manifest['created_at']).timestamp()<=(path/'last.pt').stat().st_mtime,'Backup not before completed fitting')
        allfiles+=len(manifest['files']);allbytes+=bytes_
        summary=dict(recipe=recipe['name'],seed=recipe['seed'],history_epochs=len(history),optimizer_parameter_states=len(states),optimizer_steps=sorted(steps),
            initialized_identically=True,all_24_parameters_updated=True,max_abs_parameter_change=max(deltas.values()),per_branch_parameter_change=shared,
            selected_epoch=epoch,selected_joint_mean4_f1=ranking(v,j)[0],initial_joint_mean4_f1=ranking(base['fair150_old'],base['fair150_joint'])[0],
            valid_focus_gain=valid,eligible_epochs=eligible_count,gate_failure_counts=dict(fail_counts),selected_joint_focus=j['focus'],
            selected_old_correct=v['all']['correct'],selected_new_correct=n['all']['correct'],
            max_raw_joint_mean4_f1=max(ranking(r['old'],r['joint'])[0] for r in history),
            backup_files=len(manifest['files']),backup_bytes=bytes_,physical_copy_sha_and_source_verified=True,all_backup_link_counts_one=True)
        results.append(summary);print('AUDITED',json.dumps(summary,ensure_ascii=False),flush=True)
    winner=None;rank_=ranking(base['fair150_old'],base['fair150_joint'])
    for r in results:
        rr=read(RUN/'runs'/r['recipe']/'result.json');rank_now=ranking(rr['old'],rr['joint'])
        if r['valid_focus_gain'] and rank_now>rank_:winner=r['recipe'];rank_=rank_now
    final=dict(audited_at=datetime.now(timezone.utc).isoformat(),scope='CPU-only weights, train/development histories, selected development predictions, immutable source and physical backup files; no test/regression predictions loaded.',
        protocol_sha256=sha256(RUN/'protocol.json'),initial_best58_warm_start_verified=True,novel_zero_initialization_verified=True,
        runs=results,expected_final_winner=winner or 'no-qualified-improvement',backup_files=allfiles,backup_bytes=allbytes,passed=True,no_fit=True,no_gpu=True)
    if (RUN/'selection_lock.json').exists():
        lock=read(RUN/'selection_lock.json');require(lock['winner']==final['expected_final_winner'],'Final study winner differs');final['final_lock_verified']=True
    with (OUT/'AUDIT.json').open('x',encoding='utf-8') as f:json.dump(final,f,ensure_ascii=False,indent=2)
    print('ALL AUDITS PASS',final['expected_final_winner'],OUT,flush=True)


if __name__=='__main__':main()
