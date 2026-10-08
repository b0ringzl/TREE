"""Verify inference outputs without fitting models or changing source labels."""
import argparse
from collections import Counter
from pathlib import Path
from distillation_core import read_json
from run_vmms20m_teacher import write, stamp
from training_safety import sha256

def main(out):
    plan=read_json(out/'plan.json')
    missing=[]; errors=[]; rows=[]
    for key in plan['anchors']:
        path=out/'anchors'/f'{key}.json'
        if not path.exists():
            missing.append(str(path));continue
        frame=read_json(path);rows.extend(frame['instances'])
        for row in frame['instances']:
            for evidence in row['evidence']:
                if not Path(evidence['crop']).is_file():missing.append(evidence['crop'])
        if not (out/'frames'/key/'overview.jpg').is_file():missing.append(key+'/overview.jpg')
    cameras=list((out/'frames').glob('*/result.json'))
    for path in cameras:
        result=read_json(path)
        errors.extend(result.get('errors',[]))
    protected=read_json(out/'protected_artifacts.json')
    checks={path:sha256(path)==expected for path,expected in protected.items()}
    result={'checked_at':stamp(),'planned_anchors':len(plan['anchors']),
        'completed_anchors':len(list((out/'anchors').glob('*.json'))),
        'processed_camera_frames':len(cameras),'candidate_observations':len(rows),
        'missing_files':missing,'frame_errors':errors,'protected_artifacts_unchanged':checks,
        'higher_score_focus_counts':dict(Counter(r['fused_top3'][0]['species'] for r in rows if r['showcase_candidate'])),
        'training_performed':False,'production_unchanged':all(checks.values()),
        'code_sha256':{str(p.resolve()):sha256(p) for p in Path('tools').glob('*vmms*20m*.py')}}
    write(out/'verification.json',result)
    print(result)
    if missing or errors or not all(checks.values()):raise RuntimeError('Verification failed')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    main(p.parse_args().out.resolve())
