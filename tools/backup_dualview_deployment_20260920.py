"""Verify selection gates and physically snapshot the live stack before edits."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from distillation_core import read_json,save_json
from training_safety import backup_before_training,sha256
from train_artocarpus_user_20260918 import protected

OUT=ROOT/'outputs/SSTv1_Dualview_Deployment_20260920_v1'
EXPERIMENT=ROOT/'outputs/SSTv1_Inference_Optimization_20260920_v2'
if __name__=='__main__':
    reg=read_json(EXPERIMENT/'regression_results.json')
    assert reg['deployment_eligible'] and not reg['failed_deployment_gates']
    assert reg['recipe']==dict(name='views50_anchor0',center_weight=.5,anchor_weight=0.,preserve_confident=None)
    paths=list(protected())+[ROOT/'AGENTS.md',Path(__file__)]
    for directory in ('tree_species_teacher_demo','spacescan_trees_demo_v1'):
        paths += [p for p in (ROOT/directory).rglob('*') if p.is_file() and
                  p.suffix in {'.py','.js','.html','.css','.json'} and '__pycache__' not in p.parts]
    paths += list(EXPERIMENT.glob('*.json'))
    backup=backup_before_training(OUT,paths,dict(
        purpose='Before live pipeline update; no optimizer, no change to classifier/point weights',
        authorization='User 2026-09-20 explicitly waived per-plan approval; backup required',
        selected=reg['recipe'],validation_and_regression_gates_passed=True))
    save_json(OUT/'deployment_plan.json',dict(backup=str(backup),training=False,
        classifier='sstv1-aquilaria-user-crops-20260919-h120',
        pipeline='sstv1-confirmed-target-whole-center-mean-v2',
        inference='Exact same H120 heads; two serial batch1 encodes, arithmetic mean of probabilities',
        compatibility='Only image-only SSTv1 path; legacy teacher API remains single, point-fusion unchanged',
        known_limitations='Historical web-image classification not blind end-to-end HK benchmark. Same image two crops, not multiple angles.',
        prior_delivery='Existing Windows H120 ZIP and Android files retained unchanged; do not mislabel as new pipeline',
        criteria='Live probabilities must match selected cached implementation within 1e-5; API selected-tree and point regression; rollback on failure',
        experiment_sha256=sha256(EXPERIMENT/'regression_results.json')))
    print('DEPLOYMENT_BACKUP_VERIFIED',backup,flush=True)
