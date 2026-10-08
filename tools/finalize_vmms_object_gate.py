"""Combine background veto and independent tree-support check; preserve all attempts."""
import argparse
from collections import Counter
from pathlib import Path
from vmms_object_gate import save
from distillation_core import read_json

def decide(row):
    strong=all(v['background_minus_tree']>=.035 and v['background_similarity']>=.24 for v in row['views'])
    conflict=all(v['background_minus_tree']>=0 for v in row['views'])
    support=max([r['score'] for r in row.get('grounding_support',[])],default=0)
    blocked=strong or (conflict and support<.30)
    return ('suppress_non_tree' if blocked else 'retain_unverified',
        'consistent_strong_background' if strong else 'background_conflict_without_independent_tree_support' if blocked else 'not_vetoed_not_verified')

def main(source,out):
    out.mkdir(parents=True,exist_ok=False)
    rows=read_json(source/'decisions.json')
    for row in rows:
        row['previous_decision']=row['decision']
        row['decision'],row['reason']=decide(row)
    save(out/'decisions.json',rows)
    save(out/'policy.json',{'source':str(source),'training':False,
        'rule':'Strong CLIP background in both views vetoes; otherwise CLIP background conflict requires Grounding DINO tree score >= .30 with area >= .08 to pass.',
        'scope':'All cached anchor and neighboring crops; not a new species model or trained detector.',
        'threshold_selection':'Adjusted on the reported ceiling failure and checked against 166 already-inspected historical positive crops. NOT a fresh held-out accuracy evaluation.',
        'passing_does_not_confirm_tree_or_species':True})
    summary={group:dict(Counter(r['decision'] for r in rows if r['group']==group)) for group in ['survey','historical_positive_regression']}
    summary['reported_case']={r['id']:r['decision'] for r in rows if r['id'].startswith('hewentian_pano__002357__')}
    save(out/'summary.json',summary)
    save(out/'suppressed.json',[r for r in rows if r['decision']=='suppress_non_tree'])
    print(summary)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();main(args.source.resolve(),args.out.resolve())
