"""Snapshot the explicitly authorized single-image deployment. No model fitting."""
import json
from pathlib import Path
from training_safety import backup_before_training
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/adapted_teacher_deployment_20260914_v1'
if __name__=='__main__':
    base=ROOT/'outputs/distillation/bioclip_probe_20260908_0227'
    selection=json.loads((base/'selection_lock.json').read_text(encoding='utf-8'))
    paths=[ROOT/p for p in ['spacescan_trees_demo_v1/server.py','tree_species_teacher_demo/engine.py','spacescan_trees_demo_v1/static/index.html']]
    paths.extend([base/'selection_lock.json',base/'classes.json',base/'protocol.json',base/'teacher_choice.json'])
    paths.extend(Path(r['run'])/'best.pt' for r in selection['selected_runs'])
    dest=backup_before_training(OUT,paths,{'purpose':'pre-deployment backup, no training','single_image_mode':'adapted','dual_source_mode':'teacher unchanged','user_authorized':True})
    print(dest)
