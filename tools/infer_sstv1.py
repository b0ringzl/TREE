"""Standalone SSTv1 inference; does not change the deployed Demo or its weights."""
import argparse
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from bioclip_probe import ResidualHead
from train_sstv1_incremental import prepare_pixels,logits
from training_safety import sha256
from tree_species_teacher_demo.engine import Engine

def main(run,image,output):
    lock=json.loads((run/'selection_lock.json').read_text(encoding='utf-8'))
    config=json.loads((run/'config.json').read_text(encoding='utf-8'))
    labels=json.loads((run/'labels.json').read_text(encoding='utf-8'))
    engine=Engine();engine.load();heads=[]
    for member in lock['members']:
        path=Path(member['path'])
        if sha256(path)!=member['sha256']:raise ValueError('Candidate checkpoint changed')
        state=torch.load(path,map_location='cpu',weights_only=False)
        if state['classes']!=engine.classes or state['encoder_sha256']!=engine.encoder_sha256:raise ValueError('Incompatible candidate')
        head=ResidualHead('mlp').cuda().eval();head.load_state_dict(state['model_state']);heads.append(head)
    with torch.inference_mode():
        pixels=prepare_pixels(image)[None].cuda()
        e=engine.model.encode_image(((pixels-engine.mean)/engine.std).to(engine.dtype),normalize=True).float()
        raw=e@engine.bank.T*engine.model.logit_scale.exp().float();x=e.half().float()
        def predict(members):
            return torch.stack([logits(h,x,raw,engine.classes['groups'],config['temperature'],engine.extra_bank,engine.model.logit_scale.exp().float()).softmax(-1) for h in members]).mean(0)[0].cpu().numpy()
        b=predict(engine.heads);c=predict(heads)
    def top(p):return [{'species':labels[i],'score':float(p[i])} for i in np.argsort(-p)[:3]]
    result={'image':str(image.resolve()),'model':'SSTv1','baseline':'BioCLIP 2.5 + MLP Adapted teacher',
        'baseline_top3':top(b),'SSTv1_top3':top(c),'baseline_distribution':b.tolist(),'SSTv1_distribution':c.tolist(),
        'labels':labels,'input_preprocessing':'Demo gray letterbox then teacher decode224','deployed':False,
        'notice':'Scores are uncalibrated model confidence, not measured accuracy.'}
    if output:
        if output.exists():raise FileExistsError(output)
        output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('labels','baseline_distribution','SSTv1_distribution')},ensure_ascii=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--image',type=Path,required=True);p.add_argument('--output',type=Path)
    a=p.parse_args();main(a.run.resolve(),a.image.resolve(),a.output)
