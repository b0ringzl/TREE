"""Nested catalog-size trials on frozen YOLO features; never deploys automatically."""
import argparse
import csv
import hashlib
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

os.environ.setdefault('MPLBACKEND', 'Agg')
import numpy as np
import torch
from PIL import Image, ImageOps
from ultralytics import YOLO
from ultralytics.data.augment import classify_transforms

from inventory_herbarium_images import ROOT, write_json
from training_safety import sha256, backup_before_training, deployed_artifacts, verify_backup

BASE = ROOT / 'full_image_species_baseline/experiments/full_web_yolo11s_cls_20260906_seed1'
INITIAL = BASE / 'pretrain/weights/selected_by_val_group_macro_f1.pt'
AUX = ROOT / 'external_datasets/herbarium_training_candidates_20260907/experiments/yolo_aux_20260907_224616'
FICUS = 'Ficus microcarpa / Ficus benjamina (榕樹)'
PRIORITY = [FICUS, 'Livistona chinensis', 'Delonix regia', 'Aleurites moluccana', 'Albizia lebbeck', 'Araucaria columnaris(Araucaria heterophylla)']


def grouped_name(name):
    return FICUS if name in ('Ficus microcarpa', 'Ficus benjamina') else name


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def source_rows():
    with (BASE / 'datasets/full/inventory.csv').open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r['class_index'] = int(r['class_index']); r['label'] = grouped_name(r['species']); r['auxiliary'] = False
    # Use the already reviewed, frozen training snapshot, never pending/rejected photos.
    for r in load_json(AUX / 'included_herbarium.json'):
        if r['human_review']['status'] != 'accepted':
            raise RuntimeError('Unaccepted auxiliary sample')
        path = AUX.parents[1] / r['file']
        if sha256(path) != r['sha256']:
            raise RuntimeError('Auxiliary original changed')
        name = load_json(BASE / 'datasets/full/dataset_summary.json')['class_names'][r['class_index']]
        rows.append({'sample_id': 'herbarium_' + r['id'], 'split': 'train', 'class_index': r['class_index'],
                     'species': name, 'label': grouped_name(name), 'group_id': 'herbarium_taxon_' + r['source_species_id'],
                     'source_image': str(path), 'materialized_path': str(path), 'source_sha256': r['sha256'], 'auxiliary': True})
    return rows


def partition_validation(rows):
    """Disjoint groups: planning ranks labels/tunes heads; selection chooses K."""
    assignments = {}
    for name in sorted({r['label'] for r in rows if r['split'] == 'val'}):
        groups = {r['group_id'] for r in rows if r['split'] == 'val' and r['label'] == name}
        ordered = sorted(groups, key=lambda g: hashlib.sha256(('scale-v1|' + g).encode()).hexdigest())
        for i, group in enumerate(ordered):
            assignments[group] = 'planning' if i % 2 == 0 else 'selection'
    for r in rows:
        r['phase'] = assignments[r['group_id']] if r['split'] == 'val' else r['split']
    sets = {s: {r['group_id'] for r in rows if r['phase'] == s} for s in ('train','planning','selection','test')}
    for a in sets:
        for b in sets:
            if a < b and sets[a] & sets[b]:
                raise RuntimeError('Group leakage: ' + a + '/' + b)


def extract(out, phase, rows):
    paths = [r['materialized_path'] for r in rows]
    model = YOLO(str(INITIAL)).model.to('cuda').eval().float()
    head = model.model[-1]
    if not (out / 'original_linear.pt').exists():
        torch.save({'weight': head.linear.weight.detach().cpu(), 'bias': head.linear.bias.detach().cpu(),
                    'names': model.names, 'encoder_sha256': sha256(INITIAL)}, out / 'original_linear.pt')
    transform = classify_transforms(size=224)
    def read_image(path):
        with Image.open(path) as im:
            return transform(ImageOps.exif_transpose(im).convert('RGB'))
    captured = []
    handle = head.drop.register_forward_hook(lambda m, i, o: captured.append(o.detach()))
    features = []
    try:
        with ThreadPoolExecutor(max_workers=8) as pool, torch.inference_mode():
            for start in range(0, len(paths), 128):
                x = torch.stack(list(pool.map(read_image, paths[start:start+128]))).to('cuda')
                captured.clear(); model(x)
                features.append(captured[0].flatten(1).cpu().numpy().astype(np.float32))
                if start % 2048 == 0:
                    print(f'Extract {phase}: {min(start+128,len(paths))}/{len(paths)}', flush=True)
    finally:
        handle.remove()
    result = np.concatenate(features)
    np.save(out / (phase + '_features.npy'), result)
    write_json(out / (phase + '_rows.json'), rows)
    del model; torch.cuda.empty_cache()
    return result


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    rows = source_rows(); partition_validation(rows)
    write_json(out / 'locked_rows.json', rows)
    protocol = {
        'stage': 'feature_extraction', 'scales': [20,30,40,60], 'seeds': [20260908,20260909,20260910],
        'known_labels_plus_one_other': True, 'priority': PRIORITY, 'target_known_accuracy': .85,
        'target_macro_f1': .80, 'min_known_coverage': .60, 'max_other_false_accept_rate': .10,
        'selection_policy': 'Largest K meeting all targets on selection validation; otherwise report unmet target and Pareto frontier, do not invent optimum.',
        'feature_encoder': str(INITIAL), 'feature_encoder_sha256': sha256(INITIAL),
        'phases': dict(Counter(r['phase'] for r in rows)),
        'test_policy': 'No test features or metrics until K and seed/threshold selection is frozen. Existing test is a historical regression set, not fresh field evidence.',
        'scope': 'Image classification only. Existing 148 outputs become 147 labels after the user-required Ficus merge. Wodyetia/Leucaena are not outputs of this encoder classifier; their separate demo branches are not changed.',
        'training': 'Frozen YOLO11s-cls pooled features; genuinely fit a new K+1 linear classifier, 80 full-batch epochs, dropout=0.1, AdamW lr=0.001. Not end-to-end fine-tuning.',
        'other_policy': 'Remaining catalog taxa train a pooled Other class. Tests only measure held-out images from these taxa, not arbitrary unseen species.',
        'deployment': False,
    }
    write_json(out / 'protocol.json', protocol)
    for phase in ('train','planning','selection'):
        selected = [r for r in rows if r['phase']==phase]
        extract(out, phase, selected)
    write_json(out / 'status.json', {'stage':'features_ready'})
    print('FEATURES_READY=' + str(out), flush=True)


def softmax(x):
    x=x-x.max(1,keepdims=True); p=np.exp(x); return p/p.sum(1,keepdims=True)


def aggregate(rows, probabilities, labels):
    groups={}
    for i,r in enumerate(rows): groups.setdefault(r['group_id'],[]).append(i)
    y=[];p=[]
    for ids in groups.values():
        truth={int(labels[i]) for i in ids}
        if len(truth)!=1:raise ValueError('Conflicting group labels')
        y.append(truth.pop());p.append(probabilities[ids].mean(0))
    return np.asarray(y),np.asarray(p)


def macro_f1(y,pred,n):
    confusion=np.zeros((n,n),dtype=np.int64); np.add.at(confusion,(y,pred),1)
    tp=np.diag(confusion);den=confusion.sum(0)+confusion.sum(1)
    values=np.divide(2*tp,den,out=np.zeros(n),where=den>0)
    return float(values.mean())


def metrics(y,p,k,threshold=0,margin=0):
    known=y<k;other=~known; guess=p[:,:k].argmax(1); scores=p[:,:k].max(1)
    second=np.partition(p[:,:k],-2,axis=1)[:,-2]
    accepted=(scores>=threshold)&(scores-second>=margin)&(scores>p[:,k])
    correct=known&(guess==y)
    return {'known_count':int(known.sum()),'other_count':int(other.sum()),
            'known_closed_accuracy':float(correct[known].mean()),
            'known_closed_macro_f1':macro_f1(y[known],guess[known],k),
            'joint_macro_f1':macro_f1(y,p.argmax(1),k+1),
            'known_coverage':float(accepted[known].mean()),
            'known_correct_and_accepted_rate':float((accepted&correct)[known].mean()),
            'accepted_precision_all_sources':float(correct[accepted].mean()) if accepted.any() else 0.,
            'other_false_accept_rate':float(accepted[other].mean()) if other.any() else 0.,
            'threshold':threshold,'margin':margin}


def tune_gate(y,p,k,protocol):
    rows=[metrics(y,p,k,float(t),float(m)) for t in np.arange(0,1,.05) for m in [0,.05,.1,.2]]
    valid=[r for r in rows if r['accepted_precision_all_sources']>=protocol['target_known_accuracy'] and r['other_false_accept_rate']<=protocol['max_other_false_accept_rate'] and r['known_coverage']>0]
    if valid:return max(valid,key=lambda r:(r['known_coverage'],r['accepted_precision_all_sources']))
    return max(rows,key=lambda r:r['known_correct_and_accepted_rate']-r['other_false_accept_rate'])


def targets_met(m,protocol):
    return (m['known_closed_accuracy']>=protocol['target_known_accuracy'] and m['known_closed_macro_f1']>=protocol['target_macro_f1']
            and m['accepted_precision_all_sources']>=protocol['target_known_accuracy']
            and m['known_coverage']>=protocol['min_known_coverage'] and m['other_false_accept_rate']<=protocol['max_other_false_accept_rate'])


def original_probabilities(x,original):
    with torch.inference_mode():
        return torch.softmax(torch.tensor(x,device='cuda')@original['weight'].to('cuda').T+original['bias'].to('cuda'),1).cpu().numpy()


def collapsed_baseline(p,names,selected):
    cols=[];used=[]
    for label in selected:
        ids=[i for i,n in enumerate(names) if grouped_name(n)==label];used+=ids;cols.append(p[:,ids].sum(1))
    remaining=sorted(set(range(len(names)))-set(used));cols.append(p[:,remaining].sum(1))
    return np.stack(cols,axis=1)


def train_scales(out):
    if (out/'head_validation_results.json').exists():
        raise FileExistsError('Head-fit results already exist; preserve them and use a fresh experiment for another search')
    protocol=load_json(out/'protocol.json')
    if sha256(INITIAL)!=protocol['feature_encoder_sha256']:raise RuntimeError('Encoder changed since extraction')
    original=torch.load(out/'original_linear.pt',map_location='cpu',weights_only=False)
    names=load_json(BASE/'datasets/full/dataset_summary.json')['class_names']
    data={phase:(load_json(out/(phase+'_rows.json')),np.load(out/(phase+'_features.npy'))) for phase in ('train','planning','selection')}
    plan_rows,plan_x=data['planning'];baseline_plan=original_probabilities(plan_x,original)
    globals_=list(dict.fromkeys(grouped_name(n) for n in names))
    global_probs=collapsed_baseline(baseline_plan,names,globals_)[:,:len(globals_)]
    gy,gp=aggregate(plan_rows,global_probs,np.array([globals_.index(r['label']) for r in plan_rows]));gpred=gp.argmax(1)
    ranking=[]
    for i,label in enumerate(globals_):
        trgroups={r['group_id'] for r in data['train'][0] if r['label']==label}
        support=int((gy==i).sum());tp=int(((gy==i)&(gpred==i)).sum());fp=int(((gy!=i)&(gpred==i)).sum())
        f1=2*tp/(support+tp+fp) if support+tp+fp else 0
        ranking.append({'label':label,'planning_group_support':support,'training_group_support':len(trgroups),'planning_group_f1':f1})
    ranked=sorted([r for r in ranking if r['training_group_support']>=20 and r['planning_group_support']>=5 and r['label'] not in PRIORITY], key=lambda r:(-r['planning_group_f1'],-r['planning_group_support'],r['label']))
    ordered=PRIORITY+[r['label'] for r in ranked]
    if len(ordered)<max(protocol['scales']):raise RuntimeError('Insufficient supported labels for requested scales')
    write_json(out/'class_ranking_planning_only.json',ranking)
    selected_sets={str(k):ordered[:k] for k in protocol['scales']};write_json(out/'nested_class_sets.json',selected_sets)
    all_results=[]
    for k in protocol['scales']:
        selected=selected_sets[str(k)]; mapping={n:i for i,n in enumerate(selected)}
        y={phase:np.array([mapping.get(r['label'],k) for r in data[phase][0]]) for phase in data}
        tx=torch.tensor(data['train'][1],device='cuda');ty=torch.tensor(y['train'],device='cuda',dtype=torch.long)
        px=torch.tensor(plan_x,device='cuda');sx=torch.tensor(data['selection'][1],device='cuda')
        # Equal mass per retained label; total Other mass 0.2, balanced across its source taxa.
        train_rows=data['train'][0];counts=Counter(r['label'] for r in train_rows);other_labels=set(counts)-set(selected)
        weights=np.array([(0.8/k if r['label'] in mapping else 0.2/len(other_labels))/counts[r['label']]*(3 if r['auxiliary'] else 1) for r in train_rows])
        tw=torch.tensor(weights/weights.sum(),device='cuda',dtype=torch.float32)
        for seed in protocol['seeds']:
            run=out/f'head_k{k}_seed{seed}'
            if run.exists():run=out/f'head_k{k}_seed{seed}_attempt_{datetime.now().strftime("%H%M%S")}'
            run.mkdir(exist_ok=False)
            torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            head=torch.nn.Linear(tx.shape[1],k+1).to('cuda')
            with torch.no_grad():
                for j,label in enumerate(selected):
                    ids=[i for i,n in enumerate(names) if grouped_name(n)==label]
                    head.weight[j].copy_(original['weight'][ids].mean(0));head.bias[j].copy_(original['bias'][ids].mean()+np.log(len(ids)))
                head.weight[k].normal_(0,.002);head.bias[k].zero_()
            torch.save({'state_dict':{n:t.detach().cpu().clone() for n,t in head.state_dict().items()},'classes':selected+['Other'],'encoder_sha256':protocol['feature_encoder_sha256']},run/'initial_head.pt')
            write_json(run/'config.json',{'k':k,'seed':seed,'protocol':protocol,'selected_classes':selected,'source_manifest_sha256':sha256(out/'locked_rows.json')})
            backup_before_training(run,[run/'initial_head.pt',*deployed_artifacts()],{'stage':'frozen_YOLO_head_fit','k':k,'seed':seed,'epochs':80})
            optimizer=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.0002)
            best_score=-1;history=[]
            for epoch in range(1,81):
                head.train();optimizer.zero_grad(set_to_none=True)
                logits=head(torch.nn.functional.dropout(tx,p=.1,training=True));loss=(torch.nn.functional.cross_entropy(logits,ty,reduction='none')*tw).sum()
                loss.backward();optimizer.step()
                if epoch%5==0:
                    head.eval()
                    with torch.inference_mode(): pp=head(px).softmax(1).cpu().numpy()
                    yy,gg=aggregate(plan_rows,pp,y['planning']);score=macro_f1(yy,gg.argmax(1),k+1)
                    history.append({'epoch':epoch,'loss':float(loss.detach()),'planning_joint_group_macro_f1':score})
                    if score>best_score:
                        best_score=score;best_epoch=epoch
                        torch.save({'state_dict':{n:t.detach().cpu().clone() for n,t in head.state_dict().items()},'classes':selected+['Other'],'encoder_sha256':protocol['feature_encoder_sha256'],'epoch':epoch},run/'best_head.pt')
            head.load_state_dict(torch.load(run/'best_head.pt',weights_only=False)['state_dict']);head.eval()
            with torch.inference_mode(): pp=head(px).softmax(1).cpu().numpy();sp=head(sx).softmax(1).cpu().numpy()
            gate=tune_gate(y['planning'],pp,k,protocol)
            image_metrics=metrics(y['selection'],sp,k,gate['threshold'],gate['margin'])
            yg,pg=aggregate(data['selection'][0],sp,y['selection']);group_metrics=metrics(yg,pg,k,gate['threshold'],gate['margin'])
            core=np.array([r['label'] in PRIORITY for r in data['selection'][0]])
            result={'k':k,'seed':seed,'best_epoch':best_epoch,'run':str(run),'planning_gate':gate,
                    'selection':image_metrics,'selection_group':group_metrics,
                    'common_priority_closed_accuracy':float((sp[core,:k].argmax(1)==y['selection'][core]).mean()),
                    'target_met':targets_met(image_metrics,protocol)}
            write_json(run/'history.json',history);write_json(run/'validation_result.json',result)
            np.save(run/'selection_probabilities.npy',sp)
            verify_backup(run/'pre_training_backup');all_results.append(result)
            write_json(out/'head_validation_results.json',all_results)
            print(json.dumps({'k':k,'seed':seed,'closed_accuracy':image_metrics['known_closed_accuracy'],'coverage':image_metrics['known_coverage'],'other_fpr':image_metrics['other_false_accept_rate'],'met':result['target_met']}),flush=True)
            del head,optimizer
        del tx,ty,px,sx;torch.cuda.empty_cache()
    summaries=[]
    for k in protocol['scales']:
        runs=[r for r in all_results if r['k']==k]
        med={key:float(np.median([r['selection'][key] for r in runs])) for key in runs[0]['selection']}
        representative=sorted(runs,key=lambda r:r['selection']['known_closed_macro_f1'])[len(runs)//2]
        summaries.append({'k':k,'median_selection':med,'min_accuracy':min(r['selection']['known_closed_accuracy'] for r in runs),
                          'max_accuracy':max(r['selection']['known_closed_accuracy'] for r in runs),'all_seeds_met_target':all(r['target_met'] for r in runs),'representative_run':representative['run']})
    passing=[r for r in summaries if r['all_seeds_met_target']]
    if passing:
        preferred=max(passing,key=lambda r:r['k'])
    else:
        preferred=max(summaries,key=lambda r:r['median_selection']['known_closed_macro_f1'])
    # Two finalists: preferred and adjacent size; no test data used to decide these.
    adjacent=min([r for r in summaries if r['k']!=preferred['k']],key=lambda r:abs(r['k']-preferred['k']))
    result={'scales':summaries,'target_met':bool(passing),'preferred_preliminary_k':preferred['k'],
            'finalists':[preferred,adjacent],'test_accessed':False,'status':'requires_end_to_end_confirmation'}
    write_json(out/'scale_selection.json',result);write_json(out/'status.json',{'stage':'head_search_complete','finalists':[r['k'] for r in result['finalists']]})
    print('HEAD_SEARCH_COMPLETE='+json.dumps(result,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    torch.set_num_threads(4)
    parser=argparse.ArgumentParser(); parser.add_argument('stage',choices=['prepare','train']);parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    output=args.output or ROOT/'external_datasets/species_scale_trials'/datetime.now().strftime('%Y%m%d_%H%M%S')
    print('EXPERIMENT='+str(output),flush=True)
    prepare(output) if args.stage=='prepare' else train_scales(output)
