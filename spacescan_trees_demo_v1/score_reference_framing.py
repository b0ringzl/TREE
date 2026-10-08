"""Inference-only framing triage. No fitting, label changes or model deployment.

Scores only propose which images to visually inspect; they are not botanical truth.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'external_datasets/distillation_runtime'))
if '--general' not in sys.argv:
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
import open_clip
import torch
from PIL import Image, ImageOps

PROMPTS = [
    'a photograph of a whole tree showing its trunk and entire leafy crown',
    'a photograph of the full leafy crown of a tree showing its overall canopy shape',
    'a photograph of a whole palm tree showing its trunk and crown',
    'a photograph of only flowers in close-up',
    'a photograph of only fruits, seeds or nuts in close-up',
    'a photograph of only tree bark or trunk without a crown',
    'a close-up photograph of individual leaves or a single small leafy twig',
    'a photograph of a small seedling on the ground',
    'a landscape photograph of distant trees without a clear target tree',
    'a dried herbarium specimen on a sheet of paper',
    'a photograph of a person or an animal rather than a tree',
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--library', type=Path, required=True)
    p.add_argument('--watch', action='store_true')
    p.add_argument('--general', action='store_true')
    args = p.parse_args()
    lib = args.library.resolve()
    protocol = json.loads((ROOT/'outputs/distillation/bioclip_probe_20260908_0227/protocol.json').read_text(encoding='utf-8'))
    weight = Path(protocol['teacher_weight'])
    assert hashlib.sha256(weight.read_bytes()).hexdigest() == protocol['teacher_sha256']
    if args.general:
        # Separate generic visual-language encoder for framing only, never species deployment.
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai', force_quick_gelu=True, precision='bf16', device='cuda', cache_dir=str(ROOT/'external_datasets/reference_framing_model'))
    else:
        model, _, preprocess = open_clip.create_model_and_transforms('local-dir:' + str(weight.parent), precision='bf16', device='cuda')
    model.eval().requires_grad_(False)
    tokenizer = open_clip.get_tokenizer('ViT-B-32' if args.general else 'ViT-H-14')
    with torch.inference_mode():
        bank = model.encode_text(tokenizer(PROMPTS).cuda(), normalize=True).float()
    dest = lib/('framing_scores_general_qgelu.json' if args.general else 'framing_scores.json')
    scores = json.loads(dest.read_text(encoding='utf-8')) if dest.exists() else {}
    idle = 0
    while True:
        try:
            rows = json.loads((lib/'manifest.json').read_text(encoding='utf-8'))
        except (json.JSONDecodeError, FileNotFoundError):
            time.sleep(2)
            continue
        todo = [r for r in rows if r['id'] not in scores]
        for start in range(0,len(todo),16):
            batch = todo[start:start+16]
            tensors = []
            for row in batch:
                with Image.open(lib/row['file']) as im:
                    im = ImageOps.pad(ImageOps.exif_transpose(im).convert('RGB'),(224,224),color=(128,128,128))
                    tensors.append(preprocess(im))
            with torch.inference_mode():
                features = model.encode_image(torch.stack(tensors).cuda().bfloat16(), normalize=True).float()
                sim = (features @ bank.T).cpu()
            for row, s in zip(batch,sim):
                good, bad = float(s[:3].max()), float(s[3:].max())
                scores[row['id']] = {'whole_crown_margin': good-bad, 'best_prompt': PROMPTS[int(s.argmax())],
                                    'similarities': s.tolist(), 'inference_only': True}
            temp = dest.with_suffix('.tmp')
            temp.write_text(json.dumps(scores,ensure_ascii=False,indent=2),encoding='utf-8')
            temp.replace(dest)
        if todo:
            print('Framing scored: ' + str(len(scores)),flush=True)
            idle=0
        else:
            idle+=1
        if not args.watch or (lib/'collection_complete.json').exists() or idle>360:
            break
        time.sleep(5)


if __name__=='__main__':
    main()
