"""Apply explicit recorded visual decisions to automatic veto proposals, no training."""
import argparse
from collections import Counter
from pathlib import Path
from distillation_core import read_json
from vmms_object_gate import save
from training_safety import sha256

def main(source,out):
    out.mkdir(exist_ok=False,parents=True)
    rows=read_json(source/'decisions.json');review=read_json(source/'visual_review.json')
    for row in rows:
        if row['group']=='survey' and row['decision']=='suppress_non_tree':
            evidence=review['decisions'].get(row['id'])
            if evidence is None:raise ValueError('Unreviewed automatic suppression: '+row['id'])
            row['automatic_decision']=row['decision'];row['decision']=evidence['decision']
            row['assistant_visual_review']=evidence
    save(out/'decisions.json',rows);save(out/'visual_review.json',review)
    save(out/'policy.json',{'training':False,'automatic_source':str(source),
        'automatic_decisions_sha256':sha256(source/'decisions.json'),
        'review_sha256':sha256(source/'visual_review.json'),
        'automatic_suppression_requires_visual_review':True,
        'three_real_tree_crops_rescued':True,'scope':'Current batch correction, not a validated universal automatic gate'})
    summary={g:dict(Counter(r['decision'] for r in rows if r['group']==g)) for g in ['survey','historical_positive_regression']}
    save(out/'summary.json',summary);print(summary)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();main(args.source.resolve(),args.out.resolve())
