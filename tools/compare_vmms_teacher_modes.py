"""Read-only comparison of frozen original vs previously adapted teacher.

Uses the SAME original whole-crop pixels and broad class list for both. Historical
two-class positives remain diagnostics, not independent validation evidence.
"""
import argparse
from pathlib import Path
import torch
from PIL import Image
from run_vmms20m_teacher import Survey,Engine,ROOT,write
from distillation_core import read_json,metrics,collapse_logits


def main(out):
    out.mkdir(parents=True,exist_ok=False)
    survey=Survey.__new__(Survey);survey.teacher=Engine();survey.teacher.load()
    survey.labels=survey.teacher.classes['original_names']+['Wodyetia bifurcata']
    survey.bank=torch.cat([survey.teacher.bank,survey.teacher.extra_bank],0)
    rows=read_json(ROOT/'outputs/focus8_20260909_v1/vmms_positive_audit/manifest.json')
    images=[]
    for row in rows:
        with Image.open(row['crop']) as im:images.append(im.convert('RGB'))
    features,original=survey.encode(images)
    with torch.inference_mode():
        x=torch.from_numpy(features).cuda()
        raw=x@survey.teacher.bank.T*survey.teacher.model.logit_scale.exp().float()
        xx=x.half().float()
        extra=xx@survey.teacher.extra_bank.T*survey.teacher.model.logit_scale.exp().float()
        adapted=torch.stack([torch.cat([head(xx,raw),extra],-1).div(survey.teacher.choice['calibration_temperature']).softmax(-1)
            for head in survey.teacher.heads]).mean(0).cpu().numpy()
        groups=survey.teacher.classes['groups'];temperature=survey.teacher.choice['calibration_temperature']
        original_extra=x@survey.teacher.extra_bank.T*survey.teacher.model.logit_scale.exp().float()
        grouped_original=torch.cat([collapse_logits(raw,groups),original_extra],-1).div(temperature).softmax(-1).cpu().numpy()
        grouped_adapted=torch.stack([torch.cat([collapse_logits(head(xx,raw),groups),extra],-1).div(temperature).softmax(-1)
            for head in survey.teacher.heads]).mean(0).cpu().numpy()
    truth=[survey.labels.index(row['species']) for row in rows]
    results={'original':metrics(truth,original),'adapted':metrics(truth,adapted)}
    grouped_truth=[next(i for i,g in enumerate(groups) if y in g) for y in truth]
    production_group_results={'original':metrics(grouped_truth,grouped_original),'adapted':metrics(grouped_truth,grouped_adapted)}
    write(out/'results.json',{'results':results,'labels':survey.labels,'n':len(rows),
        'scope':'Historical positive-only crops, strict raw species outputs; not full-catalog independent accuracy',
        'adapted_mapping_limitation':'Existing heads trained with merged Ficus loss; uncollapsed outputs are a strict-species diagnostic, not evidence of a separately trained microcarpa head',
        'preprocessing':'Identical native whole-crop letterbox224, no enhancement',
        'production_group_results':production_group_results,
        'production_group_labels':survey.teacher.classes['labels']+['Wodyetia bifurcata'],
        'selected_for_scene_diagnostic':max(production_group_results,key=lambda mode:production_group_results[mode]['top1']),
        'decision_basis':'Inspected two-class regression only. No guarantee for six unverified focus taxa. Production unchanged.'})
    import numpy as np
    np.savez(out/'predictions.npz',original=original,adapted=adapted,embeddings=features,grouped_original=grouped_original,grouped_adapted=grouped_adapted)
    print('Strict', {k:round(v['top1'],4) for k,v in results.items()},flush=True)
    print('Production-group', {k:round(v['top1'],4) for k,v in production_group_results.items()},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out.resolve())
