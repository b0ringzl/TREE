"""Frozen generic CLIP object/background veto; never fit or overwrite labels."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import numpy as np
import torch
from PIL import Image, ImageOps
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'outputs/tree_detector_benchmark_20260909_v1/runtime'))
from training_safety import sha256

TREE=['a tree','a palm tree','a tree canopy with leaves','a tree trunk and branches',
      'a dark tree in shadow','a leafless tree','a street tree in front of a building',
      'a blurry photograph of a tree']
BACKGROUND=['the underside of a bridge','a concrete ceiling','a building roof',
            'an architectural canopy','a building facade with windows','a concrete wall',
            'a metal fence','a road with cars','a bus','a street lamp',
            'the sky','a construction site','a building column']
MARGIN=.035

def save(path,data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf8')

class ObjectGate:
    def __init__(self):
        import clip
        self.weight=ROOT/'outputs/tree_detector_benchmark_20260909_v1/assets/clip/ViT-B-32.pt'
        self.model,self.preprocess=clip.load(str(self.weight),device='cuda',jit=False)
        self.model.eval().requires_grad_(False)
        self.prompts=['a photo of '+t.removeprefix('a ') for t in TREE+BACKGROUND]
        with torch.inference_mode():
            self.text=self.model.encode_text(clip.tokenize(self.prompts).cuda()).float()
            self.text/=self.text.norm(dim=-1,keepdim=True)

    def predict(self,paths):
        tensors=[]
        for path in paths:
            with Image.open(path) as source:im=source.convert('RGB')
            tensors.extend([self.preprocess(ImageOps.pad(im,(224,224),color=(124,116,104))),self.preprocess(im)])
        with torch.inference_mode():
            x=self.model.encode_image(torch.stack(tensors).cuda()).float()
            x/=x.norm(dim=-1,keepdim=True)
            scores=(x@self.text.T).cpu().numpy().reshape(len(paths),2,-1)
        results=[]
        for values in scores:
            views=[]
            for kind,s in zip(['whole_letterbox','center_crop'],values):
                ti=int(s[:len(TREE)].argmax());ni=int(s[len(TREE):].argmax())+len(TREE)
                views.append({'view':kind,'tree_similarity':float(s[ti]),'background_similarity':float(s[ni]),
                    'background_minus_tree':float(s[ni]-s[ti]),'tree_prompt':self.prompts[ti],
                    'background_prompt':self.prompts[ni]})
            veto=all(v['background_minus_tree']>=MARGIN and v['background_similarity']>=.24 for v in views)
            results.append({'decision':'suppress_non_tree' if veto else 'retain_unverified',
                'views':views,'calibrated_probability':False,'human_verified':False})
        return results

def main(source,out):
    out.mkdir(parents=True,exist_ok=False)
    gate=ObjectGate();torch.set_num_threads(6)
    save(out/'policy.json',{'model':'OpenAI CLIP ViT-B/32, frozen local checkpoint','weight':str(gate.weight),
        'weight_sha256':sha256(gate.weight),'tree_prompts':TREE,'background_prompts':BACKGROUND,
        'rule':'Both whole and center views: background cosine minus tree cosine >= .035 and background cosine >= .24',
        'fixed_before_first_inference':True,'training':False,'threshold_calibrated':False,
        'warning':'A conservative veto, not proof of a tree, species, single-instance box, or calibrated accuracy.'})
    entries=[]
    for p in sorted((source/'frames').glob('*/result.json')):
        for r in json.loads(p.read_text(encoding='utf8'))['instances']:
            entries.append({'id':r['id'],'crop':r['crop'],'group':'survey'})
    for p in sorted((ROOT/'outputs/focus8_20260909_v1/vmms_positive_audit/crops').glob('*.jpg')):
        entries.append({'id':'historical_positive_'+p.stem,'crop':str(p),'group':'historical_positive_regression'})
    rows=[]
    for start in range(0,len(entries),24):
        batch=entries[start:start+24]
        for entry,result in zip(batch,gate.predict([e['crop'] for e in batch])):
            rows.append({**entry,**result})
        print(f'{min(start+24,len(entries))}/{len(entries)}',flush=True)
    save(out/'decisions.json',rows)
    summary={group:dict(Counter(r['decision'] for r in rows if r['group']==group)) for group in ['survey','historical_positive_regression']}
    summary['reported_frame']=[r for r in rows if r['id'].startswith('hewentian_pano__002357__')]
    save(out/'summary.json',summary);print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();main(args.source.resolve(),args.out.resolve())
