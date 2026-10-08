"""Standalone specialist and optional user-confirmed same-tree multi-view experiment.
Never changes the production Demo, never queries example names for labels.
"""
import argparse
import io
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from distillation_core import read_json
from training_safety import sha256
from train_sstv1_incremental import prepare_pixels
from tree_species_teacher_demo.engine import Engine
from conservation_pilot import odds,AQ

def main(run,images,same_tree,output):
    if not 1<=len(images)<=4:raise ValueError('Use 1–4 views')
    if len(images)>1 and not same_tree:raise ValueError('Confirm the images show the SAME physical tree with --same-tree; do not fuse unrelated examples')
    lock=read_json(run/'selection_lock.json');selected=lock['chosen'];assert sha256(selected['path'])==selected['sha256']
    saved=torch.load(selected['path'],map_location='cpu',weights_only=False)
    engine=Engine();engine.load();assert engine.sstv1_version==saved['baseline_deployment']['version']
    assert engine.encoder_sha256==saved['encoder_sha256']
    head=torch.nn.Linear(1024,1).cuda().eval();head.load_state_dict(saved['model_state'])
    from train_sstv1_incremental import logits
    records=[];base=[];adjusted=[]
    with torch.inference_mode():
        for path in images:
            pixels=prepare_pixels(path)[None].cuda();e=engine.model.encode_image(((pixels-engine.mean)/engine.std).to(engine.dtype),normalize=True).float()
            z=e@engine.bank.T*engine.model.logit_scale.exp().float();x=e.half().float()
            p=torch.stack([logits(h,x,z,engine.classes['groups'],engine.choice['calibration_temperature'],engine.extra_bank,engine.model.logit_scale.exp().float()).softmax(-1) for h in engine.sstv1_heads]).mean(0)[0]
            b=float(p[AQ]);s=float(torch.sigmoid(torch.tensor(float(odds(np.array(b))),device='cuda')+head(x).squeeze()))
            top=p.topk(3);records.append({'source_image':str(path.resolve()),'sha256':sha256(path),'general_top3':[{'species':engine.labels[j],'score':v} for j,v in zip(top.indices.cpu().tolist(),top.values.cpu().tolist())],'general_aquilaria':b,'specialist_aquilaria':s})
            base.append(b);adjusted.append(s)
    b=float(np.mean(base));s=float(np.mean(adjusted))
    result={'model':'SST-Conserve-v1-pilot','deployed':False,'candidate':selected['sha256'], 'same_tree_confirmed':bool(same_tree),
        'records':records,'general_aquilaria_score':b,'specialist_score':s,'specialist_threshold':saved['threshold'],
        'specialist_flag':s>=saved['threshold'],'baseline_flag':b>=saved['baseline_threshold'],
        'multi_view_method':'fixed equal probability mean' if len(images)>1 else 'single view',
        'multi_view_calibration_validated':False if len(images)>1 else None,
        'notice':'Experimental source-labelled regression model. Scores are not calibrated accuracy or protected status. Multi-view is a functional preview, not an independently validated gain. No point model used.'}
    if output:
        if output.exists():raise FileExistsError(output)
        output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--image',type=Path,action='append',required=True);p.add_argument('--same-tree',action='store_true');p.add_argument('--output',type=Path);a=p.parse_args();main(a.run,a.image,a.same_tree,a.output)
