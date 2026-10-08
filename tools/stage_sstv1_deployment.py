"""User-authorized deployment staging; physical rollback copies, no fitting."""
import json
import shutil
from pathlib import Path
from training_safety import backup_before_training,sha256
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'outputs/SSTv1_incremental_20260914_v1'
OUT=ROOT/'outputs/SSTv1_deployment_20260914_v1'
MODEL=ROOT/'tree_species_teacher_demo/models/SSTv1_20260914_v1'

def main():
    if OUT.exists() or MODEL.exists():raise FileExistsError('Use new versioned deployment directory')
    done=json.loads((RUN/'VERIFIED.json').read_text(encoding='utf-8'))
    assert done['all_checks_passed']
    lock=json.loads((RUN/'selection_lock.json').read_text(encoding='utf-8'))
    base=json.loads((RUN/'base_contract.json').read_text(encoding='utf-8'))
    for item in lock['members']+base['initial_heads']:
        assert sha256(item['path'])==item['sha256']
    files=[ROOT/p for p in ('tree_species_teacher_demo/engine.py','tree_species_teacher_demo/server.py',
        'spacescan_trees_demo_v1/server.py','spacescan_trees_demo_v1/static/index.html')]
    files += [Path(h['path']) for h in base['initial_heads']]+[Path(h['path']) for h in lock['members']]
    files += [RUN/'selection_lock.json',RUN/'base_contract.json',RUN/'VERIFIED.json',RUN/'REPORT_ZH.md']
    backup=backup_before_training(OUT,files,{'purpose':'Pre-deployment rollback copy; no training',
        'user_authorized':True,'single_image':'SSTv1','dual_image':'teacher unchanged','point':'unchanged'})
    MODEL.mkdir(parents=True,exist_ok=False);members=[]
    for row in lock['members']:
        path=MODEL/f"member_{row['member']}.pt";shutil.copy2(row['path'],path)
        assert sha256(path)==row['sha256']
        members.append({'file':path.name,'sha256':row['sha256'],'source':row['path'],'epoch':row['epoch']})
    manifest={'model':'SSTv1','version':'sstv1-20260914-v1','members':members,'output_count':148,
        'encoder_sha256':base['backbone']['sha256'],'temperature':json.loads((RUN/'config.json').read_text(encoding='utf-8'))['temperature'],
        'ensemble':'three member probability mean; unchanged foxtail zero-shot extension',
        'source_run':str(RUN),'pre_deployment_backup':str(backup),'user_authorized':True,'training_performed_during_deployment':False}
    (MODEL/'deployment.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'staging.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print('STAGED',MODEL,'BACKUP',backup,flush=True)

if __name__=='__main__':main()
