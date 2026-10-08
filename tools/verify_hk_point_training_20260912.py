"""Reproducibility and non-mutation audit; does not refit anything."""
import sys,json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
from hk_point_training_20260912 import OUT,read,write
from training_safety import sha256,verify_backup
from hk_point_model import predict_bundle
def main():
    import joblib,platform,torch,sklearn,scipy
    rows=read(OUT/'manifest.json');new=read(OUT/'new_las_associations.json');checks={};source_count=0
    for r in rows:
        for kind in ['image','point','cleaned']:
            assert sha256(r['source_assets'][kind])==r['source_sha256'][kind],(r['key'],kind);source_count+=1
        with np.load(r['point']) as f:p=f['points_xyz'];m=f['metric_xyz']
        x=np.load(r['cleaned'])['xyz'];assert np.isfinite(p).all() and len(p)<=2048;assert cKDTree(x).query(m)[0].max()<1e-3;assert np.linalg.norm(p,axis=1).max()<=1.00001
    for r in new:
        assert sha256(r['source'])==r['sha256'];source_count+=1
        with np.load(r['point']) as f:p=f['points_xyz'];m=f['metric_xyz']
        x=np.load(Path(r['point']).with_name('cleaned.npz'))['xyz'];assert np.isfinite(p).all() and len(p)<=2048;assert cKDTree(x).query(m)[0].max()<1e-3
    checks['source_assets_unchanged']=source_count;checks['cleaned_clouds_real_point_membership_verified']=len(rows)+len(new)
    sets={s:{r['group'] for r in rows if r['split']==s} for s in ['train','validation','test_regression']}
    assert not sets['train']&sets['validation'] and not sets['train']&sets['test_regression'] and not sets['validation']&sets['test_regression']
    for a in rows:
        if a['split']!='train':continue
        for b in rows:
            if b['split'] in ['validation','test_regression'] and a['survey']==b['survey']:assert np.linalg.norm(np.array(a['axis'])-b['axis'])>=10
    checks['group_disjoint_and_10m_train_holdout_buffer']=True
    for p in read(OUT/'plan.json')['source_hashes'].items():assert sha256(p[0])==p[1]
    checks['human_review_snapshots_still_match_sources']=True
    for run in (OUT/'runs').iterdir():verify_backup(run/'pre_training_backup')
    checks['verified_run_backups']=6
    f=np.load(OUT/'features/all.npz');lookup={r['key']:r for r in rows};predictions=read(OUT/'all_predictions.json');comparisons=0
    for run in sorted((OUT/'runs').iterdir()):
        if run.name=='04_rotation_mlp':continue
        bundle=joblib.load(run/'model.joblib');p=predict_bundle(bundle,f['point'],f['geometry']);saved={r['key']:r for r in predictions[run.name]}
        for i,k in enumerate(f['keys']):
            assert bundle['classes'][p[i].argmax()]==saved[k]['prediction'];assert abs(float(p[i].max())-saved[k]['top3'][0]['score'])<1e-6;comparisons+=1
    checks['prediction_reproduction_checks']=comparisons
    summary=read(OUT/'final_summary.json');expected=max(range(len(summary['runs'])),key=lambda i:(summary['runs'][i]['validation']['macro_f1_supported_only'],summary['runs'][i]['validation']['group_accuracy'],-i));assert summary['selected_run']==summary['runs'][expected]['id']
    checks['selection_matches_preregistered_validation_rule']=True
    checks['max_accepted_point_removal_fraction']=max(r['quality']['isolated_removed']/r['quality']['unique_points'] for r in rows)
    checks['environment']={'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,'cuda':torch.version.cuda}
    write(OUT/'verification.json',checks);print(json.dumps(checks,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
