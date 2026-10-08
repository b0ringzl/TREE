"""SSTv1: continue deployed adapted heads with reviewed new photos + replay.

Frozen encoder, unchanged output space and temperature, three fixed members.
No deployment. Every fit has its own physically copied verified backup.
"""
import argparse
import gc
import io
import os
import sqlite3
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

os.environ['HF_HUB_OFFLINE']='1'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image,ImageOps
from bioclip_probe import ResidualHead,SOURCE,TEACHER
from distillation_core import read_json,save_json,digest,collapse_logits,metrics,bootstrap_gain
from training_safety import sha256,backup_before_training,verify_backup,deployed_artifacts

DATA=ROOT/'external_datasets/sstv1_incremental_20260914_v1'
SEED=20260914

def mapping(name,labels):
    if name in labels:return labels.index(name),'exact'
    if name in ('Ficus microcarpa','Ficus benjamina'):
        return next(i for i,s in enumerate(labels) if s.startswith('Ficus microcarpa / Ficus benjamina')),'existing pooled Ficus output; original species retained'
    if name=='Araucaria columnaris':
        return next(i for i,s in enumerate(labels) if s.startswith('Araucaria columnaris(')),'existing pooled Araucaria output; NOT a synonym assertion'
    if name=='Melaleuca cajuputi subsp. cumingiana':return labels.index('Melaleuca cajuputi'),'subspecies mapped to existing parent-species output'
    return None,'No compatible output; do not relabel as a related species'

def snapshot(out):
    out.mkdir(parents=True,exist_ok=False)
    contract=read_json(DATA/'continuation_contract.json')
    for r in contract['initial_heads']+[contract['backbone']]+contract['protected_artifacts']:
        if sha256(r['path'])!=r['sha256']:raise ValueError('Pinned base changed: '+r['path'])
    with sqlite3.connect((DATA/'human_review.sqlite3').as_uri()+'?mode=ro',uri=True) as db:
        db.row_factory=sqlite3.Row;db.execute('BEGIN')
        decisions={r['id']:dict(r) for r in db.execute('SELECT * FROM decisions')}
    save_json(out/'human_review_snapshot.json',decisions)
    classes=read_json(SOURCE/'classes.json');labels=classes['labels']+['Wodyetia bifurcata']
    accepted=[];excluded=[]
    for row in read_json(DATA/'curated_manifest.json'):
        decision=decisions.get(row['id'],{})
        if decision.get('status')!='accepted':continue
        if decision['sha256']!=row['sha256'] or sha256(DATA/row['file'])!=row['sha256']:raise ValueError('Reviewed photo changed')
        if row.get('quality_flags'):raise ValueError('Quarantined image included')
        target,note=mapping(row['source_species'],classes['labels'])
        record={**row,'human_decision':decision,'target':target,'mapping_note':note,
            'source_image':str(DATA/row['file']),'group_id':row['source_group'],'sample_id':row['id']}
        (excluded if target is None else accepted).append(record)
    rng=np.random.default_rng(SEED)
    for species in sorted({r['source_species'] for r in accepted}):
        rows=[r for r in accepted if r['source_species']==species];rng.shuffle(rows)
        n=max(1,int(len(rows)*.2)) if len(rows)>=5 else 0
        for i,row in enumerate(rows):row['phase']='test' if i<n else 'planning' if i<2*n else 'train'
    obs=[str(r['inat_observation_id']) for r in accepted]
    if len(obs)!=len(set(obs)):raise ValueError('Observation repeated across new data')
    source_hashes={};oldgroups=set();oldhashes=set()
    for phase in ('train','planning','selection','test','field','case'):
        path=SOURCE/f'{phase}_rows.json';rows=read_json(path)
        save_json(out/f'old_{phase}_rows.json',rows)
        source_hashes[str(path)]=sha256(path)
        cp=SOURCE/f'teacher_cache/bioclip25/{phase}.npz';source_hashes[str(cp)]=sha256(cp)
        oldgroups.update(r['group_id'].split('observation:')[-1] for r in rows if 'observation:' in r.get('group_id',''))
        oldhashes.update(r.get('source_sha256') for r in rows)
    if set(obs)&oldgroups or any(r['sha256'] in oldhashes for r in accepted):raise ValueError('New/old dataset overlap')
    # Recheck old auxiliary decisions; never replay rejected/pending herbarium images.
    dbpath=ROOT/'external_datasets/herbarium_training_candidates_20260907/human_review.sqlite3'
    with sqlite3.connect(dbpath.as_uri()+'?mode=ro',uri=True) as db:
        db.row_factory=sqlite3.Row
        olddec={r['id']:dict(r) for r in db.execute('SELECT * FROM decisions')}
    save_json(out/'old_herbarium_review_snapshot.json',olddec)
    oldtrain=read_json(out/'old_train_rows.json');keep=[]
    for i,r in enumerate(oldtrain):
        d=olddec.get(r['sample_id'].removeprefix('herbarium_'),{})
        if r.get('auxiliary') and (d.get('status')!='accepted' or d.get('sha256')!=r['source_sha256']):continue
        keep.append(i)
    # Existing exact observation groups/hashes may not cross training/heldout partitions.
    held=[r for p in ('planning','selection','test','field','case') for r in read_json(out/f'old_{p}_rows.json')]
    heldhash={r.get('source_sha256') for r in held if r.get('source_sha256')}
    heldgroups={r['group_id'] for r in held}
    removed=[i for i in keep if oldtrain[i].get('source_sha256') in heldhash or oldtrain[i]['group_id'] in heldgroups]
    keep=[i for i in keep if i not in set(removed)]
    save_json(out/'old_train_indices.json',keep)
    save_json(out/'accepted_replay_manifest.json',[oldtrain[i] for i in keep])
    save_json(out/'accepted_new_manifest.json',accepted);save_json(out/'excluded_accepted_taxa.json',excluded)
    save_json(out/'classes.json',classes);save_json(out/'labels.json',labels);save_json(out/'base_contract.json',contract)
    for name in ('teacher_choice.json',):save_json(out/name,read_json(SOURCE/name))
    save_json(out/'cache_metadata.json',read_json(SOURCE/'teacher_cache/bioclip25/metadata.json'))
    config={'model_name':'SSTv1','seed':SEED,'epochs':60,'lr':.0001,'weight_decay':.01,
        'backbone_trainable':False,'head_initialization':'corresponding existing adapted MLP state; no zero/random restart',
        'loss':'0.75 class-balanced old replay CE + 0.25 class-balanced accepted-new CE + 0.1 replay KL to starting head',
        'temperature':read_json(out/'teacher_choice.json')['calibration_temperature'],
        'output_labels':148,'new_source_taxa':len({r['source_species'] for r in accepted}),
        'focus_targets':sorted({r['target'] for r in accepted}),
        'new_counts':dict(Counter(r['phase'] for r in accepted)),'accepted_compatible':len(accepted),'excluded_unmapped':len(excluded),
        'replay_count':len(keep),'replay_removed_overlap':len(removed),
        'selection':'Per member maximize mean(old planning focus Macro-F1,new planning Macro-F1), subject to full old planning Macro-F1 >= starting member - 0.005. Initial epoch allowed. All 3 members averaged; fixed validation-only recommendation before heldout metrics.',
        'input_new':'Exact Demo prepare (gray letterbox <=1024,JPEG95) then teacher decode224; original encoder frozen.',
        'input_old':'Original verified feature cache center-crop; test/field results are historical regression, not fresh independent street accuracy.',
        'sparse_classes':'Fewer than five accepted images: train only, no per-species heldout improvement claim.',
        'taxonomy':'Preserve existing pooled Ficus/Araucaria outputs and source species. Does not establish microcarpa-vs-benjamina separation. Tabebuia rosea deferred.',
        'deployed':False,'lcsd_images':0,'source_hashes':source_hashes}
    save_json(out/'config.json',config)
    print('PREPARED',config['new_counts'],'replay',len(keep),'deferred',len(excluded),flush=True)
    return config

def prepare_pixels(path):
    from tree_species_teacher_demo.engine import decode
    with Image.open(path) as im:
        im=ImageOps.exif_transpose(im).convert('RGB');edge=min(1024,max(im.size))
        im=ImageOps.pad(im,(edge,edge),method=Image.Resampling.BICUBIC,color=(128,128,128))
        output=io.BytesIO();im.save(output,format='JPEG',quality=95)
    return decode(output.getvalue())[0]

def extract(out):
    from tree_species_teacher_demo.engine import Engine
    engine=Engine();engine.load()
    rows=read_json(out/'accepted_new_manifest.json');xx=[];zz=[]
    with torch.inference_mode():
        for start in range(0,len(rows),4):
            pixels=torch.stack([prepare_pixels(r['source_image']) for r in rows[start:start+4]]).cuda()
            e=engine.model.encode_image(((pixels-engine.mean)/engine.std).to(engine.dtype),normalize=True).float()
            z=e@engine.bank.T*engine.model.logit_scale.exp().float()
            xx.append(e.half().cpu().numpy());zz.append(z.cpu().numpy())
            if start%40==0:print('FEATURES',min(start+4,len(rows)),len(rows),flush=True)
        extra=engine.extra_bank.detach().cpu().numpy();scale=float(engine.model.logit_scale.exp())
    x=np.concatenate(xx);z=np.concatenate(zz)
    if not np.isfinite(x).all() or not np.isfinite(z).all():raise ValueError('Nonfinite features')
    np.savez_compressed(out/'new_features.npz',x=x,z=z,extra_bank=extra,logit_scale=scale,rows_hash=digest(rows))
    del engine;gc.collect();torch.cuda.empty_cache()

def load_old(out,phase):
    cfg=read_json(out/'config.json')
    if phase in ('test','field','case') and not (out/'selection_lock.json').exists():raise ValueError('No selection lock')
    path=SOURCE/f'{phase}_rows.json';cp=SOURCE/f'teacher_cache/bioclip25/{phase}.npz'
    for p in (path,cp):
        if sha256(p)!=cfg['source_hashes'][str(p)]:raise ValueError('Old cache changed')
    rows=read_json(out/f'old_{phase}_rows.json')
    with np.load(cp,allow_pickle=False) as cache:
        if str(cache['rows_hash'])!=digest(rows):raise ValueError('Cache ordering mismatch')
        x=cache['embeddings'].astype(np.float32)
        z=cache['similarities'][read_json(out/'teacher_choice.json')['prompt']].astype(np.float32)*read_json(out/'cache_metadata.json')['logit_scale']
    if phase=='train':
        ids=read_json(out/'old_train_indices.json');rows=[rows[i] for i in ids];x=x[ids];z=z[ids]
    return rows,torch.tensor(x,device='cuda'),torch.tensor(z,device='cuda')

def load_new(out,phase):
    rows=read_json(out/'accepted_new_manifest.json')
    with np.load(out/'new_features.npz') as f:
        if str(f['rows_hash'])!=digest(rows):raise ValueError('New feature order changed')
        ids=[i for i,r in enumerate(rows) if r['phase']==phase]
        return [rows[i] for i in ids],torch.tensor(f['x'][ids].astype(np.float32),device='cuda'),torch.tensor(f['z'][ids],device='cuda')

def logits(head,x,z,groups,temp,extra,scale):
    return torch.cat([collapse_logits(head(x,z),groups),x@extra.T*scale],-1)/temp

def probs(head,data,groups,temp,extra,scale):
    head.eval()
    with torch.no_grad():return logits(head,data[1],data[2],groups,temp,extra,scale).softmax(-1).cpu().numpy()

def score(data,p,focus=None):
    y=np.array([r['target'] for r in data[0]])
    mask=np.isin(y,focus) if focus is not None else np.ones(len(y),dtype=bool)
    return metrics(y[mask],p[mask]) if mask.any() else None

def compact(m):return {k:m[k] for k in ('n','top1','macro_f1','balanced_accuracy')} if m else None

def train(out):
    cfg=read_json(out/'config.json');contract=read_json(out/'base_contract.json');classes=read_json(out/'classes.json')
    groups=classes['groups'];temp=cfg['temperature'];focus=cfg['focus_targets']
    with np.load(out/'new_features.npz') as cache:
        extra=torch.tensor(cache['extra_bank'],device='cuda');scale=float(cache['logit_scale'])
    replay=load_old(out,'train');newtrain=load_new(out,'train');oldval=load_old(out,'planning');newval=load_new(out,'planning')
    ys=[torch.tensor([r['target'] for r in d[0]],device='cuda') for d in (replay,newtrain)]
    ws=[]
    for data in (replay,newtrain):
        counts=Counter(r['target'] for r in data[0]);w=torch.tensor([1/counts[r['target']] for r in data[0]],device='cuda');ws.append(w/w.mean())
    baselines=[];candidates=[];runs=[]
    for member,base in enumerate(contract['initial_heads']):
        seed=SEED+member;torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        run=out/f'member_{member+1}_seed_{seed}';run.mkdir(exist_ok=False)
        initial=torch.load(base['path'],map_location='cpu',weights_only=False)
        head=ResidualHead(initial['config']['architecture']).cuda();head.load_state_dict(initial['model_state'])
        p0=probs(head,oldval,groups,temp,extra,scale);pn0=probs(head,newval,groups,temp,extra,scale)
        baseval=score(oldval,p0);basefocus=score(oldval,p0,focus);basenew=score(newval,pn0)
        save_json(run/'baseline_planning.json',{'all':baseval,'focus':basefocus,'new':basenew})
        torch.save({'model_state':head.state_dict(),'base_source':base},run/'initial.pt')
        save_json(run/'config.json',{**cfg,'seed':seed,'base_checkpoint':base})
        protected=[TEACHER,*[Path(h['path']) for h in contract['initial_heads']],*deployed_artifacts(),
            *[Path(p['path']) for p in contract['protected_artifacts']]]
        inputs=[*protected,run/'initial.pt',run/'config.json',run/'baseline_planning.json',Path(__file__),
            ROOT/'tools/bioclip_probe.py',ROOT/'tools/distillation_core.py',ROOT/'tools/training_safety.py',
            out/'new_features.npz',*[p for p in out.glob('*.json')]]
        backup=backup_before_training(run,inputs,{'model':'SSTv1','member':member+1,'seed':seed})
        print('BACKUP VERIFIED',backup,flush=True)
        head.eval()
        with torch.no_grad():anchor=logits(head,replay[1],replay[2],groups,temp,extra,scale).softmax(-1).detach()
        best=.5*(basefocus['macro_f1']+basenew['macro_f1']);best_epoch=0
        best_state={k:v.detach().cpu().clone() for k,v in head.state_dict().items()};history=[]
        optimizer=torch.optim.AdamW(head.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
        for epoch in range(1,cfg['epochs']+1):
            head.train();optimizer.zero_grad(set_to_none=True)
            oldlog=logits(head,replay[1],replay[2],groups,temp,extra,scale)
            newlog=logits(head,newtrain[1],newtrain[2],groups,temp,extra,scale)
            loss=.75*(F.cross_entropy(oldlog,ys[0],reduction='none')*ws[0]).mean()+.25*(F.cross_entropy(newlog,ys[1],reduction='none')*ws[1]).mean()+.1*F.kl_div(oldlog.log_softmax(-1),anchor,reduction='batchmean')
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),5,error_if_nonfinite=True);optimizer.step()
            pv=probs(head,oldval,groups,temp,extra,scale);pn=probs(head,newval,groups,temp,extra,scale)
            m=score(oldval,pv);mf=score(oldval,pv,focus);mn=score(newval,pn)
            objective=.5*(mf['macro_f1']+mn['macro_f1'])
            if m['macro_f1']>=baseval['macro_f1']-.005 and objective>best:
                best=objective;best_epoch=epoch;best_state={k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
            history.append({'epoch':epoch,'loss':float(loss),'all':compact(m),'focus':compact(mf),'new':compact(mn)})
            if epoch%15==0:print('FIT',member+1,epoch,'best',best_epoch,'new',round(mn['top1'],4),flush=True)
        torch.save({'model_state':head.state_dict(),'optimizer':optimizer.state_dict(),'epoch':cfg['epochs'],'config':{**cfg,'architecture':'mlp'},'classes':classes,'encoder_sha256':contract['backbone']['sha256']},run/'last.pt')
        torch.save({'model_state':best_state,'epoch':best_epoch,'config':{**cfg,'architecture':'mlp'},'classes':classes,'encoder_sha256':contract['backbone']['sha256'],'base_checkpoint':base},run/'best.pt')
        save_json(run/'history.json',history);head.load_state_dict(best_state)
        candidates.append(head.eval());baselines.append(ResidualHead('mlp').cuda().eval())
        baselines[-1].load_state_dict(initial['model_state'])
        runs.append({'member':member+1,'seed':seed,'epoch':best_epoch,'path':str(run/'best.pt'),'sha256':sha256(run/'best.pt'),'backup':str(backup)})
        verify_backup(backup);del optimizer,anchor;gc.collect();torch.cuda.empty_cache()
    def ensemble(heads,data):return np.mean([probs(h,data,groups,temp,extra,scale) for h in heads],axis=0)
    # Validation-only recommendation locked before any heldout probabilities.
    selection=load_old(out,'selection');b=ensemble(baselines,selection);c=ensemble(candidates,selection)
    bn=ensemble(baselines,newval);cn=ensemble(candidates,newval)
    validation={'old_all':{'baseline':score(selection,b),'candidate':score(selection,c)},
        'old_focus':{'baseline':score(selection,b,focus),'candidate':score(selection,c,focus)},
        'new_planning':{'baseline':score(newval,bn),'candidate':score(newval,cn)}}
    save_json(out/'validation_results.json',validation)
    qualified=validation['old_all']['candidate']['macro_f1']>=validation['old_all']['baseline']['macro_f1']-.005
    improved=.5*(validation['old_focus']['candidate']['macro_f1']+validation['new_planning']['candidate']['macro_f1'])>.5*(validation['old_focus']['baseline']['macro_f1']+validation['new_planning']['baseline']['macro_f1'])
    save_json(out/'selection_lock.json',{'model_name':'SSTv1','members':runs,'ensemble':'mean of all three member probabilities',
        'validation_recommends_candidate':qualified and improved,'deployed':False,'heldout_not_yet_scored':True})
    results={}
    for phase,data in [('old_test',load_old(out,'test')),('new_test',load_new(out,'test')),('field_regression',load_old(out,'field'))]:
        bp=ensemble(baselines,data);cp=ensemble(candidates,data);y=np.array([r['target'] for r in data[0]])
        result={'baseline':score(data,bp),'candidate':score(data,cp),'focus_baseline':score(data,bp,focus),'focus_candidate':score(data,cp,focus),
            'paired_accuracy_change':bootstrap_gain(y,bp,cp,[r['group_id'] for r in data[0]],seed=SEED,repeats=1000),
            'improved_examples':[r['sample_id'] for r,t,a,z in zip(data[0],y,bp.argmax(1),cp.argmax(1)) if a!=t and z==t],
            'regressed_examples':[r['sample_id'] for r,t,a,z in zip(data[0],y,bp.argmax(1),cp.argmax(1)) if a==t and z!=t]}
        save_json(out/f'{phase}_results.json',result);save_json(out/f'{phase}_evaluation_rows.json',data[0])
        np.savez_compressed(out/f'{phase}_predictions.npz',baseline=bp,candidate=cp,targets=y)
        results[phase]=result
        print('RESULT',phase,compact(result['baseline']),compact(result['candidate']),flush=True)
    for run in runs:verify_backup(run['backup'])
    for r in contract['initial_heads']+[contract['backbone']]+contract['protected_artifacts']:
        if sha256(r['path'])!=r['sha256']:raise ValueError('Production changed during run')
    save_json(out/'COMPLETE.json',{'model':'SSTv1','training_steps':cfg['epochs']*3,'members':runs,'new_counts':cfg['new_counts'],
        'validation_recommends_candidate':qualified and improved,'deployed':False,'production_verified_unchanged':True})
    report(out,cfg,results,validation,runs,qualified and improved)

def report(out,cfg,results,validation,runs,recommend):
    labels=read_json(out/'labels.json')
    lines=['# SSTv1 — incremental adapted teacher','',
        f"Continued all three current adapted MLP heads for {cfg['epochs']} epochs each; frozen BioCLIP 2.5 encoder. No restart, no LCSD photos, no deployment.",
        '',f"Human-accepted compatible new photos: {cfg['accepted_compatible']}; splits {cfg['new_counts']}; old replay {cfg['replay_count']}. Deferred accepted Tabebuia rosea: {cfg['excluded_unmapped']}.",
        '', 'All comparisons keep the same 148 output candidates. Probabilities are not measured accuracy. Ficus/Araucaria pooled outputs are preserved, not claimed to be pure species. Rare classes with <5 photos have no new heldout evidence.',
        '',f'Validation-only recommendation: {recommend}. Actual deployment remains unchanged.',
        '', '| Evaluation | N | Baseline accuracy | SSTv1 accuracy | Change | Baseline Macro-F1 | SSTv1 Macro-F1 |', '|---|---:|---:|---:|---:|---:|---:|']
    entries={**{k:{'baseline':v['baseline'],'candidate':v['candidate']} for k,v in validation.items()},**results}
    for name,r in entries.items():
        b,c=r['baseline'],r['candidate']
        lines.append(f"| {name} | {b['n']} | {b['top1']:.2%} | {c['top1']:.2%} | {(c['top1']-b['top1'])*100:+.2f} pp | {b['macro_f1']:.4f} | {c['macro_f1']:.4f} |")
    for name in ('old_test','new_test','field_regression'):
        r=results[name];ci=r['paired_accuracy_change']['ci95']
        lines.extend(['',f"{name}: paired cluster bootstrap 95% interval for accuracy change [{ci[0]*100:+.2f}, {ci[1]*100:+.2f}] pp. Corrected {len(r['improved_examples'])}, newly wrong {len(r['regressed_examples'])}."])
    lines.extend(['','## Limits','',
        'Historical network and 13-image field splits have previously been examined: regression evidence only. New holdout is prospectively isolated from fitting/epoch selection but still shares a web source and may overlap encoder pretraining; not an independent city-street benchmark.',
        'Old feature caches use center-crop preprocessing; new images use current Demo letterboxing. Comparisons are paired within each dataset; do not compare their absolute accuracy as one identical distribution.',
        'The holdout has no sufficient evidence for the rare one-image white-paperbark or alexandra-palm classes. No claim that all 27 official entries improved.',
        '', '## New heldout per output class','', '| Label | N | Baseline recall | SSTv1 recall | Baseline F1 | SSTv1 F1 |','|---|---:|---:|---:|---:|---:|'])
    b,c=results['new_test']['baseline'],results['new_test']['candidate']
    for i,n in enumerate(b['support']):
        if n:lines.append(f"| {labels[i]} | {n} | {b['per_class_recall'][i]:.2%} | {c['per_class_recall'][i]:.2%} | {b['per_class_f1'][i]:.4f} | {c['per_class_f1'][i]:.4f} |")
    lines+=['','## Verified physical pre-training backups','']+[f"- {r['backup']} (selected epoch {r['epoch']})" for r in runs]
    lines+=['','Existing deployed image, point and fusion files verified unchanged. The candidate checkpoints can be loaded only with the pinned encoder, original label mapping/temperature, and unchanged foxtail extra text vector.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();out=a.out.resolve()
    snapshot(out);extract(out);train(out)
