"""Versioned, inference-frozen palm extension. Explicit backups before fitting.

Only a two-row residual layer is fitted. Existing BioCLIP and three MLP heads
remain frozen; every class still participates in the expanded softmax.
"""
import argparse
import gc
import json
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tree_species_teacher_demo.engine import Engine, decode, EXPERIMENT
from bioclip_probe import load_phase
from distillation_core import read_json, save_json, collapse_logits, metrics
from training_safety import backup_before_training, verify_backup, sha256, deployed_artifacts
from collect_herbarium_candidates import perceptual_hash

DATA = ROOT / 'external_datasets/palm_confusion_20260908'
SEED = 20260908


def main(out):
    out.mkdir(parents=True, exist_ok=False)
    db = sqlite3.connect(f'file:{(DATA / "human_review.sqlite3").as_posix()}?mode=ro', uri=True)
    decisions = {r[0]: {'sha256': r[1], 'status': r[2], 'revision': r[3]} for r in db.execute('SELECT id,sha256,status,revision FROM decisions')}
    db.close()
    manifest = read_json(DATA / 'curated_manifest.json')
    accepted = []
    for row in manifest:
        decision = decisions.get(row['id'], {})
        if decision.get('status') != 'accepted': continue
        if decision['sha256'] != row['sha256'] or sha256(DATA / row['file']) != row['sha256']:
            raise ValueError('Reviewed image changed')
        if row['exact_duplicate_matches']: raise ValueError('Duplicate incorrectly included')
        accepted.append({**row, 'human_decision': decision})
    save_json(out / 'review_snapshot.json', {'decisions': decisions, 'accepted': accepted})
    # All old observation identities stay in their original partition.
    oldrows = {p: read_json(EXPERIMENT / f'{p}_rows.json') for p in ('train', 'planning', 'selection', 'test')}
    known, pathphase = {}, {}
    for phase, rows in oldrows.items():
        for r in rows:
            for key in ('source_image', 'materialized_path'):
                if r.get(key): pathphase[str(Path(r[key]).resolve()).casefold()] = phase
            if 'observation:' in r['group_id']:
                key = r['group_id'].split('observation:')[-1]
                if key in known and known[key] != phase: raise ValueError('Existing group crosses splits')
                known[key] = phase
    # Conservative near-duplicate audit. Quarantine cross-observation matches;
    # do not claim a perceptual hash proves identity.
    from PIL import Image
    index = [json.loads(line) for line in (ROOT / 'external_datasets/herbarium_training_candidates_20260907/existing_image_index.jsonl').read_text(encoding='utf-8').splitlines()]
    index = [(r, int(r['phash'], 16)) for r in index if r.get('phash') and str(Path(r['path']).resolve()).casefold() in pathphase]
    for row in accepted:
        with Image.open(DATA / row['file']) as im: row['phash'] = perceptual_hash(im)
        row['phase'] = known.get(str(row['inat_observation_id']))
        near = [r['path'] for r, h in index if (int(row['phash'], 16) ^ h).bit_count() <= 4
                and pathphase[str(Path(r['path']).resolve()).casefold()] != row['phase']]
        row['near_cross_partition_matches'] = near
        if near: row['phase'] = 'quarantine'
    for i, a in enumerate(accepted):
        for b in accepted[i+1:]:
            if a['source_group'] != b['source_group'] and (int(a['phash'],16)^int(b['phash'],16)).bit_count() <= 4:
                a['phase'] = b['phase'] = 'quarantine'
    rng = np.random.default_rng(SEED)
    for species in sorted({r['source_species'] for r in accepted}):
        groups = sorted({r['source_group'] for r in accepted if r['source_species'] == species and r['phase'] is None})
        rng.shuffle(groups)
        # New-species groups: at least one validation and two diagnostic groups.
        assignment = {g: ('test' if i < 2 else 'planning' if i == 2 else 'train') for i,g in enumerate(groups)} if species == 'Wodyetia bifurcata' else {g:'train' for g in groups}
        for row in accepted:
            if row['source_species'] == species and row['phase'] is None: row['phase'] = assignment[row['source_group']]
    save_json(out / 'accepted_split_manifest.json', accepted)
    counts = Counter((r['source_species'], r['phase']) for r in accepted)
    print('Accepted split:', dict(counts), flush=True)
    if any(counts[('Wodyetia bifurcata',p)] == 0 for p in ('train','planning','test')):
        raise ValueError('Not enough independent foxtail groups after leakage audit; no training')
    engine = Engine(); engine.load()
    import open_clip
    with torch.inference_mode():
        extra = engine.model.encode_text(open_clip.get_tokenizer('ViT-H-14')(['a photo of Wodyetia bifurcata']).cuda(), normalize=True).float()
        extra = F.normalize(extra, dim=-1).detach().clone()
    labels = engine.classes['labels'] + ['Wodyetia bifurcata']
    palm_ids = [labels.index('Roystonea regia'), labels.index('Wodyetia bifurcata')]
    temperature = engine.choice['calibration_temperature']
    save_json(out / 'labels.json', {'labels':labels, 'palm_indices':palm_ids})

    def base(x, raw):
        # Preserve three-head probability ensembling, adding the same new
        # zero-shot logit to each frozen head before normalization.
        added = x @ extra.T * engine.model.logit_scale.exp().float()
        with torch.inference_mode():
            p = torch.stack([torch.cat([collapse_logits(h(x,raw),engine.classes['groups']),added],-1).div(temperature).softmax(-1) for h in engine.heads]).mean(0)
        return p.clamp_min(1e-20).log().detach().clone()

    feature_rows = []
    with torch.inference_mode():
        for row in accepted:
            if row['phase'] == 'quarantine': continue
            pixels,_,_ = decode((DATA/row['file']).read_bytes())
            e = engine.model.encode_image(((pixels[None].cuda()-engine.mean)/engine.std).to(engine.dtype),normalize=True).float()
            raw = e @ engine.bank.T * engine.model.logit_scale.exp().float()
            x = e.half().float()
            feature_rows.append((row, x.detach().clone(), base(x,raw)))
    torch.save({'extra_text_embedding':extra.cpu(), 'samples':[(r,x.cpu(),z.cpu()) for r,x,z in feature_rows]}, out/'accepted_features.pt')

    def phase_data(phase):
        rows,x,raw = load_phase(EXPERIMENT,phase)
        # Original web replay only: no stale auxiliary human-review decisions.
        if phase=='train':
            keep=[i for i,r in enumerate(rows) if not r.get('auxiliary',False)]
            rows,x,raw=[rows[i] for i in keep],x[keep],raw[keep]
        z=base(x,raw)
        selected=[(r,a,b) for r,a,b in feature_rows if r['phase']==phase]
        y=torch.tensor([r['target'] for r in rows],device='cuda')
        n=len(rows)
        if selected:
            x=torch.cat([x]+[a for _,a,_ in selected]);z=torch.cat([z]+[b for _,_,b in selected])
            y=torch.cat([y,torch.tensor([labels.index(r['source_species']) for r,_,_ in selected],device='cuda')])
        save_json(out/f'{phase}_rows.json',rows+[dict(r,target=labels.index(r['source_species'])) for r,_,_ in selected])
        return x.detach().clone(),z.detach().clone(),y,n

    train=phase_data('train'); val=phase_data('planning')
    config={'seed':SEED,'epochs':80,'lr':.01,'weight_decay':.1,'new_sample_weight':20,
        'trainable':'zero-initialized linear 1024 -> 2 residual on log probabilities',
        'frozen':'BioCLIP and all three existing residual MLPs', 'candidate_classes':148,
        'selection':'maximize mean web Macro-F1 and palm balanced recall on planning only; web F1 may fall at most .005 vs expanded baseline; initial epoch allowed',
        'test_scope':'Historical regression and already-inspected small palm diagnosis, NOT fresh blind accuracy'}
    save_json(out/'config.json',config)
    torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    head=torch.nn.Linear(1024,2).cuda()
    torch.nn.init.zeros_(head.weight);torch.nn.init.zeros_(head.bias)
    torch.save(head.state_dict(),out/'initial.pt')
    def scored(data):
        x,z,y,n=data
        with torch.no_grad():
            logits=z.clone();logits[:,palm_ids]+=head(x)
            p=logits.softmax(-1).cpu().numpy()
        truth=y.cpu().numpy()
        web=metrics(truth[:n],p[:n])
        palm_mask=np.isin(truth,palm_ids)
        palm=metrics(truth[palm_mask],p[palm_mask])
        new=metrics(truth[n:],p[n:]) if len(truth)>n else None
        return {'web':web,'palms_including_old_web':palm,'accepted_new':new},p
    baseline,_=scored(val);save_json(out/'baseline_planning.json',baseline)
    selected=read_json(EXPERIMENT/'selection_lock.json')
    protected=deployed_artifacts()+[Path(r['run'])/'best.pt' for r in selected['selected_runs']]
    protected += [ROOT/'tree_species_teacher_demo/engine.py',ROOT/'tree_species_teacher_demo/server.py',EXPERIMENT/'classes.json',EXPERIMENT/'teacher_choice.json',EXPERIMENT/'selection_lock.json']
    teacher=Path(read_json(EXPERIMENT/'protocol.json')['teacher_weight'])
    inputs=[out/'initial.pt',teacher,*protected,DATA/'human_review.sqlite3',out/'review_snapshot.json',out/'accepted_split_manifest.json',out/'labels.json',out/'config.json',out/'baseline_planning.json',out/'train_rows.json',out/'planning_rows.json',out/'accepted_features.pt',Path(__file__)]
    backup=backup_before_training(out,inputs,metadata={'training_purpose':'palm extension; no production switch'})
    print('Verified pre-training backup:',backup,flush=True)
    def objective(m):return (m['web']['macro_f1']+m['palms_including_old_web']['balanced_accuracy'])/2
    best=objective(baseline); best_epoch=0;best_state={k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
    optimizer=torch.optim.AdamW(head.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
    x,z,y,n=train
    weight=torch.ones(len(y),device='cuda');weight[n:]=config['new_sample_weight']
    # Balance foxtail/royal contribution within reviewed training images.
    for label in palm_ids:
        ids=(y==label)&(torch.arange(len(y),device='cuda')>=n)
        if ids.any():weight[ids]*=(len(y)-n)/(2*int(ids.sum()))
    history=[]
    for epoch in range(1,config['epochs']+1):
        head.train();optimizer.zero_grad(set_to_none=True)
        logits=z.clone();logits[:,palm_ids]+=head(x)
        loss=(F.cross_entropy(logits,y,reduction='none')*weight).sum()/weight.sum()
        if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),5,error_if_nonfinite=True);optimizer.step()
        head.eval();m,_=scored(val)
        if m['web']['macro_f1']>=baseline['web']['macro_f1']-.005 and objective(m)>best:
            best=objective(m);best_epoch=epoch;best_state={k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
        history.append({'epoch':epoch,'loss':loss.item(),'web_macro_f1':m['web']['macro_f1'],'palm_balanced_accuracy':m['palms_including_old_web']['balanced_accuracy']})
    torch.save({'state_dict':best_state,'config':config,'labels':labels,'palm_indices':palm_ids,'teacher_sha256':sha256(teacher),'original_head_hashes':selected['weights_sha256'],'extra_text_embedding':extra.cpu()},out/'best.pt')
    save_json(out/'selection_lock.json',{'epoch':best_epoch,'weight_sha256':sha256(out/'best.pt'),'selected_on':'planning only','time':datetime.now(timezone.utc).isoformat()})
    save_json(out/'history.json',history)
    head.load_state_dict(best_state)
    results={}
    for phase in ('selection','test'):
        data=phase_data(phase)
        candidate,p=scored(data)
        base_probs=data[1].softmax(-1).cpu().numpy();truth=data[2].cpu().numpy();n=data[3]
        results[phase]={'candidate':candidate,'expanded_baseline_web':metrics(truth[:n],base_probs[:n]),
            'expanded_baseline_new':metrics(truth[n:],base_probs[n:]) if len(truth)>n else None}
        np.savez(out/f'{phase}_predictions.npz',baseline=base_probs,candidate=p,targets=truth)
        save_json(out/f'{phase}_results.json',results[phase])
    verify_backup(backup)
    save_json(out/'COMPLETE.json',{'selected_epoch':best_epoch,'backup':str(backup),'production_unchanged':True,'training_steps':config['epochs'],'new_reviewed_samples':len(feature_rows)})
    print('COMPLETE',out,'epoch',best_epoch,flush=True)
    for phase,r in results.items():
        print(phase,'web:',r['expanded_baseline_web']['top1'],r['candidate']['web']['top1'],'new:',r['expanded_baseline_new']['top1'] if r['expanded_baseline_new'] else None,r['candidate']['accepted_new']['top1'] if r['candidate']['accepted_new'] else None,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    main(p.parse_args().out.resolve())
