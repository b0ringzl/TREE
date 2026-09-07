"""Read-back reproducibility and leakage checks, no retraining or model changes."""
from pathlib import Path
import numpy as np
import joblib
from train_reviewed_five_species import OUT,CLASSES,ensure_sources,sha
from build_hk_lidar_pairs import read,write

def norm(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-9)
def main():
    ensure_sources();rows=read(OUT/'split.json');accepted=read(OUT/'accepted_manifest.json')
    assert len(accepted)==len({r['id'] for r in accepted})
    byid={r['id']:r for r in accepted}
    resolved=['hewentian_pano__004270__c93c341996','hewentian_pano__004276__479647ba27','hewentian_pano__004287__9515e6f1a6']
    assert {byid[i]['species'] for i in resolved}=={'Albizia lebbeck'}
    assert len({byid[i]['identity_group'] for i in resolved})==1
    assert 'hewentian_pano__004496__f9261fbcdc' not in byid
    silver=[r for r in rows if r['species']=='Leucaena leucocephala']
    assert len({r['leakage_group'] for r in silver})==1 and all(r['split']=='train' for r in silver)
    train=[r for r in rows if r['split']=='train'];test=[r for r in rows if r['split']=='heldout']
    assert not ({r['leakage_group'] for r in train}&{r['leakage_group'] for r in test})
    distances=[float(np.linalg.norm(np.array(a['quality']['axis_local_xy'])-b['quality']['axis_local_xy'])) for a in train for b in test if a['survey']==b['survey']]
    assert not distances or min(distances)>=25
    for r in accepted:
        assert sha(r['point'])==r['point_sha256'] and sha(r['image'])==r['image_sha256']
    for p,h in read(OUT/'encoder_hashes.json').items():assert sha(p)==h
    bundle=joblib.load(OUT/'model.joblib');assert bundle['classes']==CLASSES and not bundle['deployed']
    saved={(r['id'],r['mode']):r for r in read(OUT/'heldout_predictions.json')};checked=0
    for r in test:
        files=list((OUT/'features').glob(r['id']+'_*.npz'));assert len(files)==1
        with np.load(files[0],allow_pickle=False) as f:iv=norm(f['image'][None]);pv=norm(f['point'][None])
        for mode,x in {'image':iv,'point':pv,'fusion':norm(np.c_[iv,pv])}.items():
            head=bundle['heads'][mode];pred=int(head.predict(x)[0]);score=float(head.predict_proba(x).max())
            expected=saved[(r['id'],mode)];assert CLASSES[pred]==expected['prediction'];assert abs(score-expected['score'])<1e-7;checked+=1
    result={'accepted_assets_checked':len(accepted),'heldout_predictions_reproduced':checked,
        'minimum_within_survey_train_holdout_axis_distance_m':min(distances) if distances else None,
        'shared_leakage_groups':0,'three_adjudicated_views_share_one_identity':True,
        'silver_leadtree_not_split_across_train_test':True,'rejected_4496_pair_excluded':True,
        'source_reviews_and_encoder_weights_unchanged':True,'production_deployment':False}
    write(OUT/'verification.json',result);print(result)

if __name__=='__main__':main()
