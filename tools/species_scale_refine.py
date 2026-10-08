"""End-to-end confirmation of validation-selected catalog sizes; test last."""
import csv
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault('MPLBACKEND','Agg')
import numpy as np
import torch
from ultralytics import YOLO
from species_scale_experiment import (BASE, ROOT, INITIAL, PRIORITY, load_json, write_json, metrics,
                                     aggregate, macro_f1, tune_gate, targets_met, collapsed_baseline)
from training_safety import sha256, verify_backup

SCRIPTS=ROOT/'full_image_species_baseline/scripts'
sys.path.insert(0,str(SCRIPTS))
from evaluate_yolo11_cls import predict
from materialize_yolo_cls import safe_class


def prepare_finalist(out, finalist):
    k=finalist['k']; branch=out/f'full_k{k}';branch.mkdir(exist_ok=False)
    selected=load_json(out/'nested_class_sets.json')[str(k)];classes=selected+['Other'];mapping={n:i for i,n in enumerate(selected)}
    rows=load_json(out/'locked_rows.json');train=[r for r in rows if r['phase']=='train'];counts=Counter(r['label'] for r in train)
    outside=set(counts)-set(selected);weights={};dataset=branch/'dataset'
    manifest=[]
    for r in rows:
        row={**r,'scale_class_index':mapping.get(r['label'],k)}
        if r['phase'] in ('train','planning'):
            split='train' if r['phase']=='train' else 'val'
            destination=dataset/'training_view'/split/safe_class(row['scale_class_index'],classes[row['scale_class_index']])/(r['sample_id']+Path(r['materialized_path']).suffix)
            destination.parent.mkdir(parents=True,exist_ok=True)
            os.link(r['materialized_path'],destination);row['materialized_path']=str(destination)
            if split=='train':weights[destination.name]=(0.8/k if r['label'] in mapping else 0.2/len(outside))/counts[r['label']]*(3 if r['auxiliary'] else 1)
        manifest.append(row)
    write_json(branch/'mapped_rows.json',manifest);write_json(branch/'sampling_weights.json',weights)
    write_json(dataset/'dataset_summary.json',{'class_names':classes,'test_hidden_from_training_view':True,'validation_scope':'planning groups only'})
    wrapper=YOLO(str(INITIAL));trained_head=torch.load(Path(finalist['representative_run'])/'best_head.pt',map_location='cpu',weights_only=False)
    if trained_head['classes']!=classes:raise RuntimeError('Head class order mismatch')
    layer=torch.nn.Linear(1280,k+1);layer.load_state_dict(trained_head['state_dict'])
    wrapper.model.model[-1].linear=layer;wrapper.model.yaml['nc']=k+1
    wrapper.model.names={i:safe_class(i,n) for i,n in enumerate(classes)}
    wrapper.save(branch/'initial_full_model.pt')
    write_json(branch/'config.json',{'k':k,'classes':classes,'head_source':finalist['representative_run'],'epochs':3,'learning_rate':0.00005,'seed':20260908,'selection_policy':'Planning group macro-F1 picks checkpoint; selection partition compares K. No test during fit.'})
    return branch,manifest,classes


def run(out):
    out=Path(out);protocol=load_json(out/'protocol.json');shortlist=load_json(out/'scale_selection.json')
    if (out/'final_choice_locked.json').exists():raise FileExistsError('Selection already locked')
    full_results=[]
    for finalist in shortlist['finalists']:
        branch,rows,classes=prepare_finalist(out,finalist);k=finalist['k']
        command=[sys.executable,str(SCRIPTS/'train_yolo11_cls.py'),'--dataset-root',str(branch/'dataset'),'--run-dir',str(branch/'candidate'),
                 '--model',str(branch/'initial_full_model.pt'),'--epochs','3','--batch-size','96','--image-size','224','--seed','20260908',
                 '--learning-rate','0.00005','--warmup-epochs','.5','--warmup-bias-lr','.00005']
        env=dict(os.environ,MPLBACKEND='Agg',PYTHONUTF8='1',TREE_SAMPLE_WEIGHTS_FILE=str(branch/'sampling_weights.json'),TREE_HERBARIUM_SAMPLE_WEIGHT='1')
        write_json(out/'status.json',{'stage':'end_to_end_training','k':k});print('FINE_TUNE_START='+str(k),flush=True)
        with (branch/'training.log').open('w',encoding='utf-8') as log:
            subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        verify_backup(branch/'candidate/pre_training_backup')
        plan=[r for r in rows if r['phase']=='planning'];y=np.array([r['scale_class_index'] for r in plan])
        choices=[]
        for checkpoint in [branch/'initial_full_model.pt',*(branch/'candidate/weights').glob('epoch*.pt')]:
            p=predict(checkpoint,plan,224,192);yg,pg=aggregate(plan,p,y)
            score=macro_f1(yg,pg.argmax(1),k+1)
            choices.append({'checkpoint':str(checkpoint),'planning_joint_group_macro_f1':score})
            np.save(branch/(checkpoint.stem+'_planning_probabilities.npy'),p)
        chosen=max(choices,key=lambda r:r['planning_joint_group_macro_f1']);checkpoint=Path(chosen['checkpoint'])
        pp=np.load(branch/(checkpoint.stem+'_planning_probabilities.npy'));gate=tune_gate(y,pp,k,protocol)
        selection=[r for r in rows if r['phase']=='selection'];sy=np.array([r['scale_class_index'] for r in selection])
        sp=predict(checkpoint,selection,224,192);sm=metrics(sy,sp,k,gate['threshold'],gate['margin']);sgy,sgp=aggregate(selection,sp,sy)
        result={'k':k,'branch':str(branch),'checkpoint':str(checkpoint),'checkpoint_sha256':sha256(checkpoint),'planning_checkpoints':choices,'planning_gate':gate,
                'selection':sm,'selection_group':metrics(sgy,sgp,k,gate['threshold'],gate['margin']),'target_met':targets_met(sm,protocol),
                'full_fine_tune_selected':checkpoint.name!='initial_full_model.pt'}
        np.save(branch/'selection_probabilities.npy',sp);write_json(branch/'validation_result.json',result)
        full_results.append(result);write_json(out/'full_validation_results.json',full_results)
        print('FULL_VALIDATION='+json.dumps(result),flush=True)
    passing=[r for r in full_results if r['target_met']]
    chosen=max(passing,key=lambda r:r['k']) if passing else max(full_results,key=lambda r:r['selection']['known_closed_macro_f1'])
    lock={'selected':chosen,'target_met':bool(passing),'reason':'Largest validated K meeting all targets' if passing else 'No tested finalist meets every target; highest validation closed Macro-F1 is only a provisional choice',
          'test_accessed':False,'deployment':False}
    write_json(out/'final_choice_locked.json',lock)
    evaluate_test(out)


def evaluate_test(out):
    out=Path(out)
    if (out/'final_test.json').exists():raise FileExistsError('Test results already recorded; no repeated selection on test')
    locked=load_json(out/'final_choice_locked.json');selected=locked['selected'];k=selected['k'];branch=Path(selected['branch'])
    rows=load_json(branch/'mapped_rows.json');classes=load_json(branch/'config.json')['classes'];protocol=load_json(out/'protocol.json')
    test=[r for r in rows if r['phase']=='test'];y=np.array([r['scale_class_index'] for r in test])
    checkpoint=Path(selected['checkpoint'])
    if sha256(checkpoint)!=selected['checkpoint_sha256']:raise RuntimeError('Selected checkpoint changed')
    write_json(out/'status.json',{'stage':'locked_test_evaluation','k':k})
    p=predict(checkpoint,test,224,192)
    # Same catalog, same test pictures: distinguish reduced competition from actual retraining gains.
    original_names=load_json(BASE/'datasets/full/dataset_summary.json')['class_names']
    baseline=collapsed_baseline(predict(INITIAL,test,224,192),original_names,classes[:-1])
    plan=[r for r in rows if r['phase']=='planning'];py=np.array([r['scale_class_index'] for r in plan])
    baseline_plan=collapsed_baseline(predict(INITIAL,plan,224,192),original_names,classes[:-1])
    baseline_gate=tune_gate(py,baseline_plan,k,protocol)
    gate=selected['planning_gate'];yg,pg=aggregate(test,p,y);byg,bpg=aggregate(test,baseline,y)
    result={'k':k,'class_labels':classes,'candidate':metrics(y,p,k,gate['threshold'],gate['margin']),
            'candidate_group':metrics(yg,pg,k,gate['threshold'],gate['margin']),
            'baseline_same_catalog':metrics(y,baseline,k,baseline_gate['threshold'],baseline_gate['margin']),
            'baseline_group':metrics(byg,bpg,k,baseline_gate['threshold'],baseline_gate['margin']),
            'target_met_on_regression_test':targets_met(metrics(y,p,k,gate['threshold'],gate['margin']),protocol),
            'deployed':False,'test_selection_frozen_before_evaluation':True}
    np.save(out/'final_test_probabilities.npy',p);np.save(out/'baseline_same_catalog_test_probabilities.npy',baseline)
    write_json(out/'final_test.json',result)
    for r in load_json(out/'head_validation_results.json'):verify_backup(Path(r['run'])/'pre_training_backup')
    for r in load_json(out/'full_validation_results.json'):verify_backup(Path(r['branch'])/'candidate/pre_training_backup')
    write_json(out/'status.json',{'stage':'complete','production_unchanged':True,'selected_k':k,'deployment':False})
    print('FINAL_TEST='+json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    torch.set_num_threads(4)
    run(Path(sys.argv[1]))
