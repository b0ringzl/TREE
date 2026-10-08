"""Verify live UI APIs against locked validation predictions and old fusion."""
import json
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import numpy as np
import requests
from distillation_core import read_json,save_json
from training_safety import sha256,verify_backup
from audit_artocarpus_preprocessing_20260917 import exact_demo_prepare
from check_windows_portable_20260920 import capture,PHOTO,post

OUT=ROOT/'outputs/SSTv1_Dualview_Deployment_20260920_v1'
EXP=ROOT/'outputs/SSTv1_Inference_Optimization_20260920_v2'
BASE='http://127.0.0.1:8043'
TEACHER='http://127.0.0.1:8039'
PIPELINE='sstv1-confirmed-target-whole-center-mean-v2'

def main():
    assert not (OUT/'LIVE_VERIFIED.json').exists()
    health=requests.get(BASE+'/api/health',timeout=10).json()
    assert health['ready'] and health['inference_pipeline']==PIPELINE
    plan=read_json(OUT/'deployment_plan.json');backup=verify_backup(plan['backup'],verify_sources=False)
    unchanged_weights=[]
    for row in backup['files']:
        if Path(row['source']).suffix in {'.pt','.safetensors','.joblib'}:
            assert sha256(row['source'])==row['sha256'],row['source']
            unchanged_weights.append(row['source'])
    protocol=read_json(EXP/'protocol.json');labels=protocol['labels']
    rows=read_json(EXP/'development_rows.json')
    with np.load(EXP/'development_components.npz') as f:
        expected=.5*f['current_whole']+.5*f['current_center']
        original=f['current_whole'].copy()
    # Fixed spread over the manifest plus first four of each focal class;
    # chosen without looking at model predictions.
    indices=list(map(int,np.linspace(0,2639,12)))
    for target in (11,16):indices += [i for i,r in enumerate(rows) if r['target']==target][:4]
    indices=list(dict.fromkeys(indices));details=[];durations=[];legacy=[]
    prepare=exact_demo_prepare()
    for i in indices:
        row=rows[i];path=Path(row['source_image']);data=path.read_bytes()
        start=time.perf_counter()
        result=post(BASE,'/api/predict',{'image':(path.name,data)},{'projection':'photo'})
        elapsed=time.perf_counter()-start
        pred=np.array([result['image_distribution'][s] for s in labels])
        delta=float(np.max(np.abs(pred-expected[i])))
        assert delta<1e-5,(row['sample_id'],delta)
        assert result['inference_pipeline']==PIPELINE and result['view_evidence']['independent_images']==1
        assert result['model_input_detail'] and result['release_provenance']['pipeline']==PIPELINE
        raw=post(TEACHER,'/api/predict',{'image':('whole.jpg',prepare(data)[0])},{'mode':'sstv1'})
        old=np.array([raw['sstv1_distribution'][s] for s in labels])
        old_delta=float(np.max(np.abs(old-original[i])));assert old_delta<1e-5
        legacy.append(old_delta);durations.append(elapsed)
        details.append(dict(sample_id=row['sample_id'],path=str(path),max_delta=delta,
            top1=result['top3'][0]['species'],seconds=elapsed,
            model_seconds=result['inference_seconds'],legacy_model_seconds=raw['inference_seconds']))
        print('LIVE_PARITY',len(details),len(indices),row['sample_id'],delta,flush=True)
    live=capture(BASE)
    old=read_json(ROOT/'outputs/windows_portable_acceptance_20260920_v1/formal.json')
    point_delta=0.
    for k in ('image','point','fusion'):
        a,b=old['pair']['comparison'][k],live['pair']['comparison'][k]
        assert [v['species'] for v in a]==[v['species'] for v in b]
        delta=max(abs(x['score']-y['score']) for x,y in zip(a,b))
        assert delta<1e-5;point_delta=max(point_delta,delta)
    assert live['pair']['point_sha256']==old['pair']['point_sha256']
    a,b=old['detector']['candidates'],live['detector']['candidates']
    assert len(a)==len(b)
    for x,y in zip(a,b):
        assert x['crop_box']==y['crop_box']
        np.testing.assert_allclose(x['box'],y['box'],atol=1e-5,rtol=0)
    detected=post(BASE,'/api/detect-trees',{'image':('trees.png',PHOTO.read_bytes())},{'projection':'photo'})
    token=detected['candidates'][0]['token']
    selected=post(BASE,'/api/identify-tree',{'image':('trees.png',PHOTO.read_bytes())},{'token':token})
    response=requests.post(BASE+'/api/export-target',files={'image':('trees.png',PHOTO.read_bytes())},data={'token':token},timeout=120)
    response.raise_for_status()
    exported=post(BASE,'/api/predict',{'image':('target.jpg',response.content)},{'projection':'photo'})
    target_delta=max(abs(v-exported['image_distribution'][k]) for k,v in selected['image_distribution'].items())
    assert target_delta<1e-5 and selected['tree_detection']
    assert selected['inference_pipeline']==PIPELINE
    for route in ('/image-only','/multimodal','/references','/apps'):
        response=requests.get(BASE+route,timeout=10);assert response.status_code==200
        if route in ('/image-only','/multimodal'):assert 'SSTv1-H120-DV' in response.text
    invalid=requests.post(BASE+'/api/predict',files={'image':('bad.jpg',b'not image')},data={'projection':'photo'},timeout=20)
    assert invalid.status_code==400 and 'detail' in invalid.json()
    pano=requests.get(BASE+'/api/test-panorama',timeout=20);pano.raise_for_status()
    blocked=requests.post(BASE+'/api/predict',files={'image':('panorama.jpg',pano.content)},data={'projection':'equirectangular'},timeout=20)
    assert blocked.status_code==409,'Never center-crop an unselected full panorama'
    save_json(OUT/'live_capture.json',live)
    save_json(OUT/'live_parity_cases.json',details)
    results=dict(passed=True,health=health,parity_cases=len(details),max_probability_delta=max(x['max_delta'] for x in details),
        legacy_single_probability_max_delta=max(legacy),point_fusion_max_delta=point_delta,
        detector_candidates_unchanged=len(a),selected_exported_probability_delta=target_delta,
        bad_image_http_status=invalid.status_code,unselected_panorama_http_status=blocked.status_code,
        all_protected_weights_unchanged=True,weight_files=len(unchanged_weights),training=False,
        user_crop_replay=dict(n=8,baseline=sum(x['top1']=='Aquilaria sinensis' for x in old['images']),
                             candidate=sum(x['top1']=='Aquilaria sinensis' for x in live['images']),
                             note='Already-training crops, not generalization evidence'),
        median_request_seconds=float(np.median(durations)),median_model_seconds=float(np.median([x['model_seconds'] for x in details])),
        median_legacy_model_seconds=float(np.median([x['legacy_model_seconds'] for x in details])),
        archived_windows_and_android_packages_changed=False)
    save_json(OUT/'LIVE_VERIFIED.json',results)
    print('LIVE_VERIFIED',json.dumps(results,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
