"""Stage user-authorized model copies and rollback snapshot; no fitting."""
import json
import shutil
from pathlib import Path
from training_safety import backup_before_training, sha256

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'outputs/SSTv1_Aquilaria_User_20260914_v1'
OUT = ROOT / 'outputs/SSTv1_Aquilaria_deployment_20260915_v1'
OLD = ROOT / 'tree_species_teacher_demo/models/SSTv1_20260914_v1'
NEW = ROOT / 'tree_species_teacher_demo/models/SSTv1_Aquilaria_User_20260915_v1'

def read(p): return json.loads(p.read_text(encoding='utf-8'))
def save(p, data): p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')

def main():
    done = read(RUN/'COMPLETE.json')
    assert read(RUN/'VERIFIED.json')['passed']
    assert len(done['members']) == 3
    for member in done['members']:
        assert sha256(member['path']) == member['sha256']
    OUT.mkdir(exist_ok=False)
    protected = [ROOT/p for p in ['tree_species_teacher_demo/engine.py', 'tree_species_teacher_demo/server.py',
        'spacescan_trees_demo_v1/server.py', 'spacescan_trees_demo_v1/static/index.html',
        'spacescan_trees_demo_v1/models/ptv2_seven_20260913_v1/deployment.json',
        'spacescan_trees_demo_v1/models/ptv2_seven_20260913_v1/model.joblib']]
    backup = backup_before_training(OUT, [*protected, *OLD.glob('*'), RUN/'COMPLETE.json', RUN/'VERIFIED.json'],
        {'purpose':'User-authorized deployment rollback, no training', 'prior_model_retained':True})
    NEW.mkdir(exist_ok=False)
    manifest = read(OLD/'deployment.json')
    manifest.update(version='sstv1-aquilaria-user-20260915-v1', source_run=str(RUN), members=[],
        pre_deployment_backup=str(backup), user_authorized=True, training_performed_during_deployment=False)
    for i, member in enumerate(done['members'], 1):
        target = NEW/f'member_{i}.pt'
        shutil.copy2(member['path'], target)
        assert sha256(target) == member['sha256']
        manifest['members'].append({'file':target.name, 'sha256':member['sha256'], 'source':member['path'], 'epoch':member['epoch']})
    save(NEW/'deployment.json', manifest)
    save(OUT/'STAGED.json', {'manifest':str(NEW/'deployment.json'), 'backup':str(backup), 'old_manifest':str(OLD/'deployment.json')})
    print(NEW)

if __name__ == '__main__': main()
