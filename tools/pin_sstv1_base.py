"""Pin the deployed adaptation as SSTv1's continuation base; NOT a training run."""
import argparse
import json
from pathlib import Path
from datetime import datetime, timezone
from training_safety import sha256

ROOT=Path(__file__).resolve().parents[1]

def main(data):
    out=data/'continuation_contract.json'
    if out.exists(): raise FileExistsError(out)
    experiment=ROOT/'outputs/distillation/bioclip_probe_20260908_0227'
    selection=json.loads((experiment/'selection_lock.json').read_text(encoding='utf-8'))
    heads=[]
    for row in selection['selected_runs']:
        path=Path(row['run'])/'best.pt'; actual=sha256(path)
        if actual!=selection['weights_sha256'][row['run']]: raise ValueError('Adapted base head changed')
        heads.append({'path':str(path.resolve()),'sha256':actual,'seed':row['seed'],'architecture':row['architecture']})
    teacher=ROOT/'external_datasets/distillation_models/bioclip25/open_clip_model.safetensors'
    source=ROOT/'outputs/distillation/20260908_0045'
    artifacts=[experiment/'selection_lock.json',source/'classes.json',source/'teacher_choice.json',
        ROOT/'tree_species_teacher_demo/engine.py', ROOT/'spacescan_trees_demo_v1/server.py',
        ROOT/'spacescan_trees_demo_v1/models/ptv2_seven_20260913_v1/model.joblib']
    split_files=[source/f'{s}_rows.json' for s in ('train','planning','selection','test','field','case')]
    contract={'model_name':'SSTv1','created_at':datetime.now(timezone.utc).isoformat(),
        'base_model':'BioCLIP 2.5 + MLP · Adapted teacher','initial_heads':heads,
        'backbone':{'path':str(teacher),'sha256':sha256(teacher),'trainable':False},
        'initialization_rule':'Load EACH corresponding adapted head state_dict; do not restart heads from zero/random initialization.',
        'training_recipe':{'method':'Incremental fine-tuning of existing adapted MLP heads',
            'new_images':'Human-accepted new image IDs with SHA-256-matched SQLite decisions, compatible exact taxon mapping and clean duplicate audit only.',
            'replay':'Include accepted original training examples from non-target classes. Recheck current herbarium decisions.',
            'learning_rate_policy':'Low learning rate relative to initial MLP fitting; select only on fixed validation.',
            'unknown_taxa':'No guessed or nearest-species relabels. Defer missing/ambiguous labels pending taxonomy decision.',
            'loss':'Supervised source labels; model predictions must not replace labels.',
            'ensemble':'Continue all three existing members; retain fixed ensemble convention.',
            'pre_fit_gate':'NEW run-specific physical copies of every initialization weight and potentially affected deployed weight, accepted manifest, label map, all splits, config, seeds and baseline results. Verify hashes BEFORE any optimizer update.',
            'evaluation':'Same original fixed validation/test for base and candidate; previously inspected test/field data are regression diagnostics, not a fresh unbiased test.',
            'deployment':'Requires separate user approval after comparison; current Demo remains unchanged.'},
        'protected_artifacts':[{'path':str(p),'sha256':sha256(p)} for p in artifacts],
        'fixed_split_sources':[{'path':str(p),'sha256':sha256(p)} for p in split_files],
        'training_performed':False,'this_is_not_a_pre_training_weight_backup':True}
    out.write_text(json.dumps(contract,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Pinned three existing adapted heads; no fitting or deployment.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);a=p.parse_args();main(a.data)
