"""Second, independent object check on crops for which background wins in both CLIP views."""
import argparse
import inspect
import json
from pathlib import Path
import sys
import torch
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'outputs/tree_detector_benchmark_20260909_v1/runtime'))
from training_safety import sha256
from vmms_object_gate import save

def main(source,out):
    out.mkdir(exist_ok=False,parents=True)
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    folder=ROOT/'outputs/tree_detector_benchmark_20260909_v1/assets/grounding_dino_base'
    processor=AutoProcessor.from_pretrained(folder,local_files_only=True,trust_remote_code=False)
    model=AutoModelForZeroShotObjectDetection.from_pretrained(folder,local_files_only=True,trust_remote_code=False).cuda().eval().requires_grad_(False)
    torch.set_num_threads(6)
    rows=json.loads((source/'decisions.json').read_text(encoding='utf8'))
    save(out/'policy.json',{'first_pass':str(source),'model':'Grounding DINO base, frozen local',
        'weight_sha256':sha256(folder/'model.safetensors'), 'query':'tree. palm tree.',
        'trigger':'CLIP background similarity >= tree similarity in both whole and center views',
        'support':'Grounding DINO score >= .20 and box area >= .08 of crop',
        'veto':'CLIP background wins in both views AND no Grounding DINO tree support',
        'training':False,'threshold_calibrated':False,
        'diagnostic':'Designed after inspecting the reported ceiling false positives and v1 failures; not unbiased accuracy evidence.'})
    candidates=[r for r in rows if all(v['background_minus_tree']>=0 for v in r['views'])]
    for i,row in enumerate(candidates):
        with Image.open(row['crop']) as f:im=f.convert('RGB')
        inputs=processor(images=im,text='tree. palm tree.',return_tensors='pt').to('cuda')
        with torch.inference_mode():result=model(**inputs)
        method=processor.post_process_grounded_object_detection
        parameter='threshold' if 'threshold' in inspect.signature(method).parameters else 'box_threshold'
        predictions=method(result,inputs.input_ids,**{parameter:.20,'text_threshold':.20,'target_sizes':[(im.height,im.width)]})[0]
        support=[]
        for box,score in zip(predictions['boxes'].cpu().tolist(),predictions['scores'].cpu().tolist()):
            x1,y1,x2,y2=box
            fraction=max(0,min(im.width,x2)-max(0,x1))*max(0,min(im.height,y2)-max(0,y1))/(im.width*im.height)
            if fraction>=.08:support.append({'box':box,'score':score,'area_fraction':fraction})
        row['grounding_support']=support
        row['decision']='retain_unverified' if support else 'suppress_non_tree'
        print(f'{i+1}/{len(candidates)} {row["id"]}: {row["decision"]}',flush=True)
    save(out/'decisions.json',rows)
    from collections import Counter
    save(out/'summary.json',{group:dict(Counter(r['decision'] for r in rows if r['group']==group)) for group in ['survey','historical_positive_regression']})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();main(args.source.resolve(),args.out.resolve())
