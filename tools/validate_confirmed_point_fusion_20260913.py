"""Fixed inference-only fusion audit. Never fits, selects weights or deploys."""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'tools'))
from training_safety import sha256
from tree_species_teacher_demo.engine import Engine, decode, collapse_logits, EXPERIMENT
from spacescan_trees_demo_v1.server import prepare

BASE=ROOT/'outputs/point_assist_case_selection_20260913_v1'
OUT=ROOT/'outputs/confirmed_point_fusion_20260913_v1'
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8')

def main():
    OUT.mkdir(exist_ok=False)
    rows=read(BASE/'manifest.json')
    confirmation=dict(user_text='确认配对',recorded_at=datetime.now(timezone.utc).isoformat(),
        scope='Same-individual image/cloud pairing for the 16 displayed candidate records; species labels unchanged; not height or segmentation certification',
        ids=[r['id'] for r in rows],manifest_sha256=sha256(BASE/'manifest.json'),source='explicit_user_chat')
    save(OUT/'pair_confirmation.json',confirmation)
    save(BASE/'chat_pair_confirmation.json',confirmation)
    reviews=read(BASE/'human_case_reviews.json') if (BASE/'human_case_reviews.json').exists() else {}
    # No silent override of any explicit per-record rejection.
    eligible=[r for r in rows if r['point_results'] and reviews.get(r['id'],{}).get('decision') not in ['wrong_species','needs_repair']]
    classes=read(ROOT/'outputs/bauhinia_ptv3_fewshot_20260912_v1/config.json')['classes']
    assert len(eligible)>1
    protocol=dict(created_at=datetime.now(timezone.utc).isoformat(),training=False,classes=classes,
        inclusion='All confirmed candidate records with existing held-out PTv2 predictions; no image-correctness filter',
        ids=[r['id'] for r in eligible],excluded=[r['id'] for r in rows if r not in eligible],
        point_fold='For Bauhinia use its own leave-one-tree-out fold; for other species always fold_1, not best fold',
        image='Unchanged Demo padding and teacher prompts; six-class probabilities renormalized from all logits',
        fusion='Fixed 0.5 image + 0.5 point probabilities; no threshold search, no new fitting',
        negative_control='Fixed cyclic shift of point predictions in eligible manifest order; preserve every result',
        primary_metric='Tree-level mean-view probabilities, Top-1 and macro-F1 over supported species',
        limitations=['Selected previously inspected regression cases, not unbiased validation',
            'Only three target species have truth labels; remaining three are output candidates, not evaluated negative cases',
            'Existing point model excludes these individual trees but not all same-site context',
            'Legacy teacher Ficus microcarpa / Ficus benjamina merged label mapped to project Ficus microcarpa display label',
            'No absolute-height feature in this trial; no claim of height causality',
            'Uncalibrated point probabilities; fixed exploratory fusion only'])
    save(OUT/'protocol.json',protocol)
    save(OUT/'manifest_snapshot.json',rows)
    sources={str(p):sha256(p) for p in [BASE/'manifest.json',BASE/'ptv2_results_snapshot.json',Path(__file__)]}
    for r in eligible:
        for v in r['views']:sources[v['image']]=sha256(v['image'])
        for p in r['point_results']:
            if p['fold']=='fold_1' or r['case']=='bauhinia':
                path=Path(p['model'])/'model.joblib';sources[str(path)]=sha256(path)
    save(OUT/'source_hashes.json',sources)
    engine=Engine();engine.load()
    indices=[]
    for c in classes:
        matches=[i for i,n in enumerate(engine.labels) if n==c or (c=='Ficus microcarpa' and n.startswith('Ficus microcarpa /'))]
        assert len(matches)==1,(c,matches)
        indices.append(matches[0])
    save(OUT/'teacher_provenance.json',dict(encoder_sha256=engine.encoder_sha256,labels=engine.labels,
        selected_indices=indices,choice=engine.choice,protocol=read(EXPERIMENT/'protocol.json')))
    point=[]
    for r in eligible:
        pr=r['point_results'][0] if r['case']=='bauhinia' else next(p for p in r['point_results'] if p['fold']=='fold_1')
        point.append(np.array([pr['scores'][c] for c in classes]))
    records=[]
    for i,r in enumerate(eligible):
        for v in r['views']:
            data,_,_=prepare(Path(v['image']).read_bytes());pixels,_,_=decode(data)
            with torch.inference_mode():
                emb=engine.model.encode_image(((pixels[None].cuda()-engine.mean)/engine.std).to(engine.dtype),normalize=True).float()
                logits=emb@engine.bank.T*engine.model.logit_scale.exp().float()
                extra=emb@engine.extra_bank.T*engine.model.logit_scale.exp().float()
                logits=torch.cat([collapse_logits(logits,engine.classes['groups']),extra],-1)/engine.choice['calibration_temperature']
                full=logits.softmax(-1)[0].cpu().numpy()
                probs=logits[:,indices].softmax(-1)[0].cpu().numpy()
            branches=dict(image=probs,point=point[i],fusion=.5*probs+.5*point[i],wrong_cloud=.5*probs+.5*point[(i+1)%len(point)])
            rec=dict(id=r['id'],view=v['index'],truth=r['species'],image=v['image'],
                full148_prediction=engine.labels[int(full.argmax())],full148_probabilities=full.tolist(),
                wrong_cloud_id=eligible[(i+1)%len(point)]['id'],
                branches={k:dict(prediction=classes[int(p.argmax())],scores=p.tolist(),correct=classes[int(p.argmax())]==r['species']) for k,p in branches.items()})
            records.append(rec)
            print(r['id'],v['index'],{k:x['prediction'] for k,x in rec['branches'].items()},flush=True)
    save(OUT/'predictions.json',records)
    trees=[]
    for r in eligible:
        views=[v for v in records if v['id']==r['id']]
        b={k:np.mean([v['branches'][k]['scores'] for v in views],axis=0) for k in ['image','point','fusion','wrong_cloud']}
        trees.append(dict(id=r['id'],truth=r['species'],views=len(views),branches={k:dict(scores=p.tolist(),prediction=classes[int(p.argmax())]) for k,p in b.items()}))
    truth=[r['truth'] for r in trees]
    metrics={}
    for k in ['image','point','fusion','wrong_cloud']:
        pred=[r['branches'][k]['prediction'] for r in trees]
        report=classification_report(truth,pred,labels=classes,output_dict=True,zero_division=0)
        metrics[k]=dict(correct=sum(a==b for a,b in zip(truth,pred)),n=len(truth),accuracy=accuracy_score(truth,pred),
            macro_f1_supported=float(np.mean([report[c]['f1-score'] for c in sorted(set(truth))])),per_class=report)
    metrics['fusion_changes']=dict(corrected=[r['id'] for r in trees if r['branches']['image']['prediction']!=r['truth'] and r['branches']['fusion']['prediction']==r['truth']],
        regressed=[r['id'] for r in trees if r['branches']['image']['prediction']==r['truth'] and r['branches']['fusion']['prediction']!=r['truth']])
    save(OUT/'tree_results.json',trees);save(OUT/'metrics.json',metrics)
    for p,h in sources.items():assert sha256(p)==h,p
    save(OUT/'verification.json',dict(sources_unchanged=True,source_count=len(sources),training_performed=False,production_modified=False))
    print(json.dumps({k:{a:b for a,b in v.items() if a!='per_class'} for k,v in metrics.items()},ensure_ascii=False))

if __name__=='__main__':main()
