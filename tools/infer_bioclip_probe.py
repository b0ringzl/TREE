"""Offline image inference for the experimental frozen BioCLIP + residual heads.

Does not register a Demo model or change production. Runtime input is only the
image; no ground truth or nearest cached sample is used to produce predictions.
"""
import argparse
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from bioclip_probe import ROOT, SOURCE, ResidualHead
from distillation_core import read_json, save_json, canonical_image, collapse_logits
from training_safety import sha256

sys.path.insert(0, str(ROOT/'external_datasets/distillation_runtime'))


def infer(experiment, image_path):
    import open_clip
    started=time.perf_counter()
    protocol=read_json(experiment/'protocol.json')
    classes=read_json(experiment/'classes.json')
    choice=read_json(experiment/'teacher_choice.json')
    meta=read_json(experiment/'cache_metadata.json')
    lock=read_json(experiment/'selection_lock.json')
    weight=Path(protocol['teacher_weight'])
    if sha256(weight)!=protocol['teacher_sha256']:
        raise ValueError('Teacher checkpoint changed')
    alias_path=SOURCE/'aliases.json'
    if sha256(alias_path)!=meta['aliases_sha256']:
        raise ValueError('Taxonomy prompt mapping changed')
    aliases=read_json(alias_path)
    model,_,_=open_clip.create_model_and_transforms('local-dir:'+str(weight.parent),precision=meta['precision'],device='cuda')
    model.eval().requires_grad_(False)
    tokenizer=open_clip.get_tokenizer('ViT-H-14')
    templates=['{}','a photo of {}','a photo of the tree species {}']
    bank=[]
    with torch.inference_mode():
        for row in aliases:
            if choice['prompt']<3:
                texts=[templates[choice['prompt']].format(name) for name in row['prompt_terms']]
            else:
                texts=[t.format(name) for name in row['prompt_terms'] for t in
                    ['{}','a photo of {}','a street tree of {}','a photo of the tree species {}']]
            encoded=model.encode_text(tokenizer(texts).cuda(),normalize=True).float()
            bank.append(F.normalize(encoded.mean(0),dim=0))
        bank=torch.stack(bank)
    heads=[]
    for run in lock['selected_runs']:
        path=Path(run['run'])/'best.pt'
        if sha256(path)!=lock['weights_sha256'][run['run']]:
            raise ValueError('Adaptation head changed after selection')
        state=torch.load(path,map_location='cpu',weights_only=False)
        if state['encoder_sha256']!=protocol['teacher_sha256'] or state['classes']!=classes:
            raise ValueError('Encoder or class compatibility mismatch')
        head=ResidualHead(state['config']['architecture']).cuda().eval()
        head.load_state_dict(state['model_state'])
        heads.append(head)
    torch.cuda.synchronize(); loaded=time.perf_counter()
    with torch.inference_mode():
        pixels=canonical_image(image_path)[None].cuda()
        mean=torch.tensor([.48145466,.4578275,.40821073],device='cuda')[None,:,None,None]
        std=torch.tensor([.26862954,.26130258,.27577711],device='cuda')[None,:,None,None]
        dtype=torch.bfloat16 if meta['precision']=='bf16' else torch.float16
        embedding=model.encode_image(((pixels-mean)/std).to(dtype),normalize=True).float()
        raw=embedding@bank.T*model.logit_scale.exp().float()
        # Reproduce the stored feature precision on which the correction was trained.
        feature=embedding.half().float()
        p=torch.stack([(collapse_logits(head(feature,raw),classes['groups'])/choice['calibration_temperature']).softmax(-1)
                       for head in heads]).mean(0)[0].cpu().numpy()
    torch.cuda.synchronize(); finished=time.perf_counter()
    order=np.argsort(-p)
    return {'image':str(image_path),'image_sha256':sha256(image_path),'model':'BioCLIP 2.5 + residual classifier ensemble',
        'experimental':True,'production_deployed':False,'num_labels':len(classes['labels']),
        'top3':[{'species':classes['labels'][int(i)],'score':float(p[i])} for i in order[:3]],
        'score_notice':'Adapted model scores, not validated open-world confidence or field accuracy.',
        'load_seconds':loaded-started,'single_cold_inference_seconds':finished-loaded,
        'timing_notice':'One cold run, not an end-to-end deployment benchmark.',
        'probabilities':p.tolist(),'encoder_sha256':protocol['teacher_sha256']}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--experiment',type=Path,required=True)
    parser.add_argument('--image',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    result=infer(args.experiment.resolve(),args.image.resolve());save_json(args.output,result)
    print({k:result[k] for k in ['top3','load_seconds','single_cold_inference_seconds']})
