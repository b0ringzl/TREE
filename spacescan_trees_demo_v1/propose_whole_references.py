"""Render shortlisted whole-tree references for explicit visual acceptance."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from .collect_references import ROOT, GROUPS, sheets, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--upto', type=int, default=148)
    p.add_argument('--pending-only', action='store_true')
    args = p.parse_args()
    root=ROOT/'external_datasets/spacescan_whole_tree_references_20260910_v2'
    rows=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    scorefile = root/'framing_scores_general_qgelu.json'
    if not scorefile.exists():
        scorefile=root/'framing_scores.json'
    scores=json.loads(scorefile.read_text(encoding='utf-8'))
    decisions=json.loads((root/'visual_review.json').read_text(encoding='utf-8')) if (root/'visual_review.json').exists() else {}
    labels=json.loads((ROOT/'outputs/distillation/bioclip_probe_20260908_0227/classes.json').read_text(encoding='utf-8'))['labels']+['Wodyetia bifurcata']
    byspecies=defaultdict(list)
    for r in rows:
        if r['id'] in scores and decisions.get(r['id'],{}).get('status') != 'rejected_for_reference':
            byspecies[r['species']].append({**r, **scores[r['id']]})
    selected=[]
    for species in labels[:args.upto]:
        choices=sorted(byspecies[species],key=lambda r:(decisions.get(r['id'],{}).get('status')=='accepted_for_reference',r['whole_crown_margin']),reverse=True)
        if species in GROUPS:
            for query in GROUPS[species]:
                selected.extend([r for r in choices if r['query']==query][:1])
        else:
            selected.extend(choices[:2])
    output = root
    if args.pending_only:
        selected = [r for r in selected if r['id'] not in decisions]
        output = root/'pending_review'
        output.mkdir(exist_ok=True)
    write_json(output/'proposals.json',selected)
    # Contact sheets resolve image paths against the collection root.
    if args.pending_only:
        selected = [{**r, 'file': str((root/r['file']).resolve())} for r in selected]
    sheets(output,selected)
    print(json.dumps({'selected':len(selected),'nonnegative_margin':sum(r['whole_crown_margin']>=0 for r in selected),'species':len({r['species'] for r in selected})}))


if __name__=='__main__':main()
