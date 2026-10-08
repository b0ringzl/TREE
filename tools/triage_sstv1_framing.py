"""Inference-only review prioritization. Never accepts labels or fits weights."""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'external_datasets/distillation_runtime'))
os.environ['HF_HUB_OFFLINE']='1'
from spacescan_trees_demo_v1.score_reference_framing import PROMPTS
import open_clip
import torch
from PIL import Image, ImageOps, ImageDraw
from training_safety import sha256

def main(data):
    rows=json.loads((data/'curated_manifest.json').read_text(encoding='utf-8'))
    target=data/'framing_triage.json'
    if target.exists(): raise FileExistsError(target)
    weight=ROOT/'external_datasets/reference_framing_model/models--timm--vit_base_patch32_clip_224.openai/snapshots/a6f597a30f7b82c51704746581f9a4e41421e878/open_clip_model.safetensors'
    model,_,preprocess=open_clip.create_model_and_transforms('ViT-B-32',pretrained=str(weight),force_quick_gelu=True,device='cuda',precision='fp32')
    model.eval().requires_grad_(False); tokenizer=open_clip.get_tokenizer('ViT-B-32')
    scores={}
    with torch.inference_mode():
        bank=model.encode_text(tokenizer(PROMPTS).cuda(),normalize=True).float()
        for start in range(0,len(rows),32):
            batch=rows[start:start+32]; ims=[]
            for r in batch:
                with Image.open(data/r['file']) as im:
                    im=ImageOps.pad(ImageOps.exif_transpose(im).convert('RGB'),(224,224),color=(128,128,128))
                    ims.append(preprocess(im))
            feature=model.encode_image(torch.stack(ims).cuda(),normalize=True).float()
            sim=(feature@bank.T).cpu()
            if not torch.isfinite(sim).all(): raise ValueError('Non-finite framing scores')
            for r,s in zip(batch,sim):
                idx=int(s.argmax()); margin=float(s[:3].max()-s[3:].max())
                scores[r['id']]={'prompt_index':idx,'prompt':PROMPTS[idx],'whole_crown_margin':margin,'species_verified':False}
            print('FRAMING',min(start+32,len(rows)),len(rows),flush=True)
    target.write_text(json.dumps({'model':'Frozen CLIP ViT-B/32; framing only','weight_sha256':sha256(weight),
        'note':'Review ordering, not species verification or human acceptance. Model scores are not probabilities.',
        'scores':scores},ensure_ascii=False,indent=2),encoding='utf-8')
    tags={0:'tree_or_crown_context',1:'tree_or_crown_context',2:'tree_or_crown_context',3:'flower_or_inflorescence',
        4:'fruit_seed_or_cone',5:'bark_trunk_or_roots',6:'foliage_or_leafy_branch',7:'macro_or_staged_detail',
        8:'macro_or_staged_detail',9:'macro_or_staged_detail',10:'macro_or_staged_detail'}
    for r in rows:
        score=scores[r['id']]
        r['organ_review']={'organ_tags':[tags[score['prompt_index']]],'status':'automatic_framing_suggestion_only'}
        r['framing_triage']=score
        r['candidate_lane']='tree_context_candidate' if score['whole_crown_margin']>0 else 'organ_auxiliary_candidate'
    # Put likely whole-tree/crown photos first without discarding organ details.
    rows.sort(key=lambda r:(r['framing_triage']['whole_crown_margin']<=0,r['source_species'],-r['framing_triage']['whole_crown_margin']))
    (data/'curated_manifest.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    counts={'candidate_count':len(rows),'whole_crown_priority':sum(r['framing_triage']['whole_crown_margin']>0 for r in rows),
        'human_accepted_by_this_script':0,'training_performed':False}
    (data/'framing_summary.json').write_text(json.dumps(counts,indent=2),encoding='utf-8')
    sheet=Image.new('RGB',(1440,1440),'white');draw=ImageDraw.Draw(sheet)
    for i,r in enumerate(rows[:24]):
        im=Image.open(data/r['file']).convert('RGB'); im.thumbnail((230,300));x=(i%6)*240;y=(i//6)*360
        sheet.paste(im,(x+(230-im.width)//2,y));draw.text((x+3,y+305),r['source_species'][:28],fill='black')
        draw.text((x+3,y+325),r['id'].replace('sstv1_',''),fill='black')
    sheet.save(data/'priority_preview.jpg',quality=90)
    print(json.dumps(counts),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);a=p.parse_args();main(a.data)
