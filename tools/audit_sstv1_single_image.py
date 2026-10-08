"""Post-lock heldout audit using batch-size-one, matching deployed image path."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from bioclip_probe import ResidualHead
from train_sstv1_incremental import prepare_pixels,logits
from distillation_core import metrics
from tree_species_teacher_demo.engine import Engine
from training_safety import sha256
RUN=ROOT/'outputs/SSTv1_incremental_20260914_v1'
def read(p):return json.loads(p.read_text(encoding='utf-8'))

def main():
    dest=RUN/'single_image_holdout.json'
    if dest.exists():raise FileExistsError(dest)
    lock=read(RUN/'selection_lock.json');engine=Engine();engine.load();heads=[]
    for member in lock['members']:
        assert sha256(member['path'])==member['sha256']
        state=torch.load(member['path'],map_location='cpu',weights_only=False)
        head=ResidualHead('mlp').cuda().eval();head.load_state_dict(state['model_state']);heads.append(head)
    rows=read(RUN/'new_test_evaluation_rows.json');bp=[];cp=[]
    with torch.inference_mode():
        for i,r in enumerate(rows):
            pixel=prepare_pixels(r['source_image'])[None].cuda()
            e=engine.model.encode_image(((pixel-engine.mean)/engine.std).to(engine.dtype),normalize=True).float()
            z=e@engine.bank.T*engine.model.logit_scale.exp().float();x=e.half().float()
            def score(members):return torch.stack([logits(h,x,z,engine.classes['groups'],engine.choice['calibration_temperature'],engine.extra_bank,engine.model.logit_scale.exp().float()).softmax(-1) for h in members]).mean(0)[0].cpu().numpy()
            bp.append(score(engine.heads));cp.append(score(heads))
            if (i+1)%16==0:print('SINGLE IMAGE AUDIT',i+1,len(rows),flush=True)
    b=np.stack(bp);c=np.stack(cp);y=np.array([r['target'] for r in rows])
    with np.load(RUN/'new_test_predictions.npz') as cached:
        delta={'baseline_max_probability_difference':float(np.max(np.abs(b-cached['baseline']))),
            'candidate_max_probability_difference':float(np.max(np.abs(c-cached['candidate']))),
            'baseline_top1_changes':int(np.sum(b.argmax(1)!=cached['baseline'].argmax(1))),
            'candidate_top1_changes':int(np.sum(c.argmax(1)!=cached['candidate'].argmax(1)))}
    result={'scope':'post-selection-lock additional inference audit; not used for tuning or checkpoint selection',
        'baseline':metrics(y,b),'candidate':metrics(y,c),'batch_vs_single':delta,
        'improved_examples':[r['sample_id'] for r,t,a,z in zip(rows,y,b.argmax(1),c.argmax(1)) if a!=t and z==t],
        'regressed_examples':[r['sample_id'] for r,t,a,z in zip(rows,y,b.argmax(1),c.argmax(1)) if a==t and z!=t]}
    dest.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    np.savez_compressed(RUN/'single_image_holdout_predictions.npz',baseline=b,candidate=c,targets=y)
    print(json.dumps({'baseline':result['baseline']['top1'],'candidate':result['candidate']['top1'],**delta}),flush=True)

if __name__=='__main__':main()
