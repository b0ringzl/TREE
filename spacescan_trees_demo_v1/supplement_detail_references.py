"""Versioned reference-only selection: retain whole trees, propose flower/leaf gaps."""
import json
import argparse
from collections import Counter
from .collect_references import ROOT, sheets, write_json

DATA = ROOT/'external_datasets/spacescan_whole_tree_references_20260910_v2'
OUT = DATA/'mixed_v3'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--round2',action='store_true');args=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    existing=json.loads((DATA/'accepted_manifest.json').read_text(encoding='utf-8'))
    pool=json.loads((DATA/'manifest.json').read_text(encoding='utf-8'))
    scores=json.loads((DATA/'framing_scores_general_qgelu.json').read_text(encoding='utf-8'))
    old_review=json.loads((ROOT/'external_datasets/spacescan_references_20260910_v1/visual_review.json').read_text(encoding='utf-8'))
    ids={r['id'] for r in existing}; counts=Counter(r['species'] for r in existing)
    decisions={}
    if args.round2:
        decisions=json.loads((OUT/'visual_review.json').read_text(encoding='utf-8'))
        for r in pool:
            if decisions.get(r['id'],{}).get('status')=='accepted':
                counts[r['species']]+=1
        ids.update(decisions)
    classes=json.loads((ROOT/'outputs/distillation/bioclip_probe_20260908_0227/classes.json').read_text(encoding='utf-8'))['labels']+['Wodyetia bifurcata']
    def rank(r):
        sims=scores.get(r['id'],{}).get('similarities',[0]*11)
        framing=max(sims[i] for i in (0,1,2,3,6))-max(sims[i] for i in (4,5,7,8,9,10))
        prior=old_review.get(r['id'],{}).get('status')
        return framing + (0.025 if prior=='accepted_for_reference' else -.1 if prior=='rejected_for_reference' else 0)
    proposals=[]
    for species in classes:
        choices=sorted([r for r in pool if r['species']==species and r['id'] not in ids],key=rank,reverse=True)
        proposals.extend(choices[:max(0,2-counts[species])])
    output=OUT/'round2' if args.round2 else OUT
    output.mkdir(exist_ok=True)
    write_json(output/'proposals.json',proposals)
    sheets(output,[{**r,'file':str((DATA/r['file']).resolve())} for r in proposals])
    print(json.dumps({'preserved':len(existing),'proposed':len(proposals),'missing_classes':[s for s in classes if not any(r['species']==s for r in pool)]}))

if __name__=='__main__':main()
