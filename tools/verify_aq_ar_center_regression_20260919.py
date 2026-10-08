"""Batch-one re-encoding after checkpoint lock, eliminating old batch-cache drift."""
import gc
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import numpy as np
import torch
from distillation_core import canonical_image, read_json, save_json, digest, bootstrap_gain
from train_artocarpus_user_20260918 import Engine, heads_from_deployment, logprob, measure
from train_aq_ar_diagnostic_20260919 import unchanged, verify_images
from training_safety import sha256

OUT=ROOT/'outputs/SSTv1_AQ_AR_Diagnostic_20260919_v1'
PREV=ROOT/'outputs/SSTv1_Artocarpus_User_20260918_v1'


def main():
    target=OUT/'exact_regression'; target.mkdir(exist_ok=False)
    lock=read_json(OUT/'selection_lock.json'); protocol=read_json(OUT/'protocol.json')
    assert sha256(lock['checkpoint'])==lock['sha256']
    unchanged(protocol)
    rows=read_json(PREV/'test_rows.json'); verify_images(rows)
    torch.set_num_threads(4)
    engine=Engine(); engine.load()
    assert engine.sstv1_version==protocol['deployment_manifest']['version']
    candidate=heads_from_deployment(protocol)
    candidate.load_state_dict(torch.load(lock['checkpoint'],map_location='cpu',weights_only=False)['model_state'])
    candidate.eval()
    with np.load(PREV/'train_features.npz') as f:
        params=dict(groups=engine.classes['groups'],temp=protocol['temperature'],
                    extra=torch.tensor(f['extra_bank'],device='cuda'),scale=float(f['scale']))
    xx=[];zz=[];base=[];cand=[]; started=time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as pool,torch.inference_mode():
        for start in range(0,len(rows),64):
            pixels=list(pool.map(lambda r:canonical_image(r['source_image']),rows[start:start+64]))
            for pixel in pixels:
                e=engine.model.encode_image(((pixel[None].cuda()-engine.mean)/engine.std).to(engine.dtype),normalize=True).float()
                z=e@engine.bank.T*engine.model.logit_scale.exp().float(); x=e.half().float()
                xx.append(e.half().cpu().numpy());zz.append(z.cpu().numpy())
                base.append(logprob(engine.sstv1_heads,x,z,**params).exp().cpu().numpy())
                cand.append(logprob(candidate,x,z,**params).exp().cpu().numpy())
            print('EXACT_CENTER',min(start+64,len(rows)),'/',len(rows),'seconds',round(time.perf_counter()-started),flush=True)
    np.savez_compressed(target/'center_features.npz',x=np.concatenate(xx),z=np.concatenate(zz),rows_hash=digest(rows))
    bcenter=np.concatenate(base);ccenter=np.concatenate(cand)
    with np.load(OUT/'regression_predictions.npz') as f:
        before=f['baseline']; single=f['candidate_pad']; legacy=f['baseline_center']
    final=(single+ccenter)*.5 if lock['view']=='dual' else single
    y=np.array([r['target'] for r in rows]); a=before.argmax(1);b=final.argmax(1)
    result=dict(baseline=measure(rows,before,False),baseline_dual=measure(rows,(before+bcenter)*.5,False),
        candidate_single=measure(rows,single,False),candidate=measure(rows,final,False),
        corrected=int(((a!=y)&(b==y)).sum()),regressed=int(((a==y)&(b!=y)).sum()),
        legacy_cache_parity=dict(max_probability_error=float(abs(legacy-bcenter).max()),
                                changed_center_top1=int((legacy.argmax(1)!=bcenter.argmax(1)).sum())),
        bootstrap=bootstrap_gain(y,before,final,[r['group_id'] for r in rows],seed=20260919,repeats=2000),
        batch_size=1,selection_lock_sha256=sha256(OUT/'selection_lock.json'),
        note='Final batch-one same-Demo encoder regression. Replaces approximate legacy-center scores, no reselection or training.')
    save_json(target/'results.json',result)
    np.savez_compressed(target/'predictions.npz',baseline=before,baseline_center=bcenter,candidate_pad=single,candidate_center=ccenter,selected=final,targets=y)
    labels=read_json(OUT/'labels.json'); details=[]
    for i,r in enumerate(rows):
        if r['target'] in (11,16) or a[i] in (11,16) or b[i] in (11,16):
            details.append(dict(sample_id=r['sample_id'],source_image=r['source_image'],group_id=r['group_id'],
                truth=labels[r['target']],baseline=labels[int(a[i])],candidate=labels[int(b[i])],
                baseline_true_score=float(before[i,r['target']]),candidate_true_score=float(final[i,r['target']])))
    save_json(target/'focal_cases.json',details)
    unchanged(protocol)
    save_json(target/'COMPLETE.json',dict(no_training=True,no_deployment=True,production_unchanged=True,n=len(rows),
        selection_unchanged=sha256(OUT/'selection_lock.json')==result['selection_lock_sha256']))
    print('EXACT REGRESSION COMPLETE', {k:{'all_top1':result[k]['all']['top1'],'aquilaria':result[k]['aquilaria'],'white':result[k]['white']} for k in ('baseline','baseline_dual','candidate')},flush=True)


if __name__=='__main__':
    main()
