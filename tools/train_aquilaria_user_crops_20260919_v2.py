"""Fresh run after v1's post-fit report-path error; never resumes old optimizer state."""
import argparse
from pathlib import Path
import train_aquilaria_user_crops_20260919 as impl
from distillation_core import read_json,save_json
from training_safety import sha256,verify_backup

OUT=impl.ROOT/'outputs/SSTv1_Aquilaria_UserCrops_20260919_v2'


def replay_fixed(out,baseline,after,rows,name):
    label_path=out/'labels.json'
    if not label_path.exists():label_path=out.parent/'labels.json'
    labels=read_json(label_path);details=[]
    for i,r in enumerate(rows):
        def summary(p):return dict(top1=labels[int(p.argmax())],true_score=float(p[r['target']]),correct=int(p.argmax())==r['target'])
        details.append(dict(filename=r['filename'],source_image=r['source_image'],baseline=summary(baseline[i]),candidate=summary(after[i])))
    save_json(out/name,dict(scope='TRAIN replay only; not held-out accuracy',images=details,
        baseline_correct=sum(r['baseline']['correct'] for r in details),candidate_correct=sum(r['candidate']['correct'] for r in details)))


impl.crop_replay=replay_fixed


def prepare(out):
    prior=impl.OUT
    verify_backup(prior/'G_crops_group_balanced/pre_training_backup')
    impl.prepare(out)
    protocol=read_json(out/'protocol.json')
    protocol['source_sha256'][str(Path(__file__).resolve())]=sha256(__file__)
    protocol['previous_attempt']=dict(directory=str(prior),completed_optimizer_steps=120,
        status='G fit completed, failed in post-fit crop report because labels path used run subdirectory; H never began',
        reused_optimizer_state=False,correction='Resolve labels from experiment root in report helper; all recipes start fresh in v2')
    save_json(out/'protocol.json',protocol)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','features','fit','evaluate']);a=p.parse_args()
    if a.phase=='prepare':prepare(OUT)
    else:getattr(impl,a.phase)(OUT)
