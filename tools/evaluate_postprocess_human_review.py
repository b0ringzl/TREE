"""Evaluate reviewed postprocessing diagnostics; preserve source and deployment."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
from distillation_core import read_json
from run_vmms20m_teacher import write
from training_safety import sha256

def canonical(value):
    return {'狐尾椰':'Wodyetia bifurcata','Ficus microcarpa L. f.':'Ficus microcarpa'}.get(value,value.strip())

def grouped(value):
    return 'Ficus microcarpa / Ficus benjamina' if value in ('Ficus microcarpa','Ficus benjamina') else value

def metrics(rows,field,merge=False):
    norm=grouped if merge else lambda x:x
    before=[norm(r[field])==norm(r['reference']) for r in rows]
    after=[norm(r['fused'])==norm(r['reference']) for r in rows]
    return {'n':len(rows),'before_correct':sum(before),'after_correct':sum(after),
        'wrong_to_right':sum(not a and b for a,b in zip(before,after)),
        'right_to_wrong':sum(a and not b for a,b in zip(before,after)),
        'not_accuracy_on_a_fresh_random_test_set':True}

def main(root,review,out):
    out.mkdir(parents=True,exist_ok=False)
    shutil.copy2(review,out/'human_review_original.json')
    panel=read_json(root/'postprocess_audit_20260910_v1/manifest.json');answers=read_json(review)['answers']
    rows=[];dark=[];restore=[]
    for case in panel['cases']:
        answer=answers[case['id']];reference=canonical(answer.get('species',''))
        row={'number':case['number'],'id':case['id'],'reference':reference,'certainty':answer.get('certainty'),
            'tree':answer.get('tree'),'same_tree':answer.get('same_tree'),
            'evidence':answer.get('evidence',''),'single_composite':case['original'][0]['species'],
            'fused':case['processed'][0]['species'],
            'answer_revealed_before_reference':bool(answer.get('model_revealed') and not answer.get('species_before_reveal')),
            'training_eligible':False}
        if case['id'].startswith('restoration_'):
            variants=[r for r in read_json(root/'sr_quality_trial/results.json') if 'restoration_'+r['crop_id']==case['id']]
            row['variants']=[{'variant':v['variant'],'top1':v['top3'][0],
                'correct_against_user_reference':v['top3'][0]['species']==reference if answer['certainty']=='sure' else None} for v in variants]
            restore.append(row);continue
        key=case['id'].split('__tree_')[0]
        raw=next(r for r in read_json(root/'anchors'/f'{key}.json')['instances'] if r['id']==case['id'])
        row['whole_original']=raw['original_top3'][0]['species']
        rows.append(row)
        lift=next((v for v in raw['augmentation_predictions'] if v['variant']=='bounded_shadow_lift'),None)
        if lift:dark.append({**row,'before':row['whole_original'],'after':lift['top3'][0]['species']})
    sure=[r for r in rows if r['certainty']=='sure']
    less_bias=[r for r in sure if not r['answer_revealed_before_reference']]
    subsets={'sure':sure,'sure_excluding_known_answer_exposure':less_bias,'including_probable_sensitivity_only':rows}
    results={name:{'single_composite_to_multiview':metrics(rs,'single_composite'),
        'whole_original_to_final_combination':metrics(rs,'whole_original'),
        'merged_ficus_single_composite_to_multiview':metrics(rs,'single_composite',True)} for name,rs in subsets.items()}
    def darkmetrics(rs):
        a=[r['before']==r['reference'] for r in rs];b=[r['after']==r['reference'] for r in rs]
        return {'n':len(rs),'original_correct':sum(a),'lifted_correct':sum(b),
            'wrong_to_right':sum(not x and y for x,y in zip(a,b)),
            'right_to_wrong':sum(x and not y for x,y in zip(a,b))}
    google=[r for r in rows+restore if 'Google' in r['evidence'] or '谷歌' in r['evidence']]
    summary={'source':str(review),'source_sha256':sha256(review),'copied_sha256':sha256(out/'human_review_original.json'),
        'answers_complete':len(answers)==len(panel['cases']),'certainty_counts':dict(Counter(a['certainty'] for a in answers.values())),
        'multiview':results,'dark_lift':{'sure_only':darkmetrics([r for r in dark if r['certainty']=='sure']),
            'including_probable_sensitivity_only':darkmetrics(dark)},
        'google_user_reported_use':len(google),'google_user_certainty_counts':dict(Counter(r['certainty'] for r in google)),
        'google_no_controlled_before_after_human_judgment':True,
        'google_links_capture_dates_not_in_user_export':True,'training_performed':False,'deployed_files_modified':False,
        'limitations':['14 selected diagnostic cases, not random sample or eight-class balanced test',
            'Known model-answer exposure at case 1; exclusion is sensitivity analysis, not guaranteed blind review',
            'Probable species are not gold labels','Same-tree responses for restoration variants are not independent multiframe validation',
            'Manual Google-assisted reference is not model improvement or automatically verified taxonomy']}
    write(out/'evaluation.json',summary);write(out/'per_case.json',{'multiview':rows,'dark_lift':dark,'restoration':restore})
    write(out/'label_conflicts.json',{'restoration_0003':{'old_reference':'Delonix regia',
        'old_provenance':'explicit_refined_pair_accept','latest_user_decision':'uncertain',
        'latest_user_note':answers['restoration_0003']['evidence'],
        'recommendation':'Exclude from confirmed-correct SR demonstrations and future hard-label training/evaluation until resolved',
        'source_files_modified':False,'training_eligible':False}})
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--review',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();main(a.root.resolve(),a.review.resolve(),a.out.resolve())
