"""Live upload/crop/fusion checks for authorized SSTv1 deployment; no fitting."""
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
import requests
from fastapi.testclient import TestClient
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from spacescan_trees_demo_v1 import server,tree_detection as td
from training_safety import verify_backup,sha256
OUT=ROOT/'outputs/SSTv1_deployment_20260914_v1'
RUN=ROOT/'outputs/SSTv1_incremental_20260914_v1'
BASE=ROOT/'outputs/SpaceScan_Trees_Demo_Files_20260913_56b49420/02_cloud_assist_gain'
URL='http://127.0.0.1:8043'

def save(name,value):(OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
    for attempt in range(20):
        try:
            health=requests.get(URL+'/api/health',timeout=5).json()
            if health.get('ready'):break
        except requests.RequestException:pass
        time.sleep(2)
    else:raise RuntimeError('SSTv1 health did not become ready')
    assert health['single_image_mode']=='sstv1' and health['dual_image_mode']=='teacher'
    html=requests.get(URL,timeout=15).text;assert 'SSTv1 · BioCLIP' in html
    smoke=read(RUN/'inference_smoke.json');data=Path(smoke['image']).read_bytes()
    for scenario in ('street','conservation'):
        r=requests.post(URL+'/api/predict',files={'image':('tree.jpg',data,'image/jpeg')},data={'projection':'photo','scenario':scenario},timeout=90)
        assert r.status_code==200,r.text
        actual=r.json();assert actual['mode']=='sstv1' and actual['image_model_version']=='sstv1-20260914-v1'
        assert len(actual['image_distribution'])==148
        np.testing.assert_allclose([actual['image_distribution'][s] for s in smoke['labels']],smoke['SSTv1_distribution'],atol=1e-5)
        assert all(r['scientific_name'] and r['chinese_name'] for r in actual['top3'])
        save(f'single_{scenario}.json',actual)
    # Original adapted mode remains loadable for rollback/comparison.
    prepared,_,_=server.prepare(data)
    r=requests.post(server.UPSTREAM+'/api/predict',files={'image':('tree.jpg',prepared)},data={'mode':'adapted'},timeout=90)
    assert r.status_code==200,r.text;adapted=r.json()
    np.testing.assert_allclose([adapted['adapted_distribution'][s] for s in smoke['labels']],smoke['baseline_distribution'],atol=1e-5)
    save('original_adapted_retained.json',adapted)
    fused=[]
    for folder in sorted(BASE.iterdir()):
        if not folder.is_dir():continue
        image=(folder/'image.jpg').read_bytes();cp=next(folder.glob('cloud_for_replay.*'))
        r=requests.post(URL+'/api/predict-with-cloud',files={'image':('input.jpg',image),'cloud':(cp.name,cp.read_bytes())},
            data={'projection':'photo','paired':'true','units':'metres','z_up':'true'},timeout=120)
        assert r.status_code==200,r.text;dual=r.json();assert dual['mode']=='teacher' and dual['fusion_used']
        before=read(ROOT/'outputs/point_live_deployment_20260913_v1'/(folder.name+'_live.json'))
        for branch in ('image','point','fusion'):
            assert [x['species'] for x in dual['comparison'][branch]]==[x['species'] for x in before['comparison'][branch]]
            np.testing.assert_allclose([x['score'] for x in dual['comparison'][branch]],[x['score'] for x in before['comparison'][branch]],atol=1e-5)
        save(folder.name+'_dual_unchanged.json',dual);fused.append(folder.name)
        print('DUAL UNCHANGED',folder.name,flush=True)
    client=TestClient(server.app)
    width,height=td.decode(data).size
    meta={'expires':time.time()+300,'sha256':hashlib.sha256(data).hexdigest(),'crop_box':[0,0,width,height]}
    r=client.post('/api/identify-tree',files={'image':('tree.jpg',data)},data={'token':td.sign(meta)})
    assert r.status_code==200,r.text;assert r.json()['mode']=='sstv1' and r.json()['tree_detection']
    save('selected_crop.json',r.json())
    pano=requests.get(URL+'/api/test-panorama',timeout=30).content
    spec=td.panorama.view_spec(td.decode(pano).size,0,30)
    meta={'expires':time.time()+300,'sha256':hashlib.sha256(pano).hexdigest(),'crop_box':[0,0,600,600],
        'projection_view':spec,'projection':{'inferred_from_ratio':False}}
    r=client.post('/api/identify-tree',files={'image':('pano.jpg',pano)},data={'token':td.sign(meta)})
    assert r.status_code==200,r.text;assert r.json()['mode']=='sstv1' and 'perspective' in r.json()['preprocessing']
    save('projected_crop.json',r.json())
    backup=verify_backup(OUT/'pre_training_backup',verify_sources=False)
    for row in backup['files']:
        if Path(row['source']).suffix=='.pt':assert sha256(row['source'])==row['sha256']
    contract=read(RUN/'base_contract.json')
    assert sha256(contract['backbone']['path'])==contract['backbone']['sha256']
    for row in contract['protected_artifacts']:
        if Path(row['path']).suffix=='.joblib':assert sha256(row['path'])==row['sha256']
    result={'deployed':True,'single_image_model':'SSTv1','health':health,'single_image_matches_verified_candidate':True,
        'both_scenarios_checked':True,'old_adapted_available_unchanged':True,'dual_cases_unchanged':fused,
        'selected_crop_sstv1':True,'panorama_perspective_crop_sstv1':True,'pre_deployment_backup_verified':True,
        'old_weights_unchanged':True,'teacher_encoder_unchanged':True,'training_performed':False}
    save('VERIFIED.json',result)
    (OUT/'DEPLOYMENT.md').write_text('# SSTv1 Demo deployment\n\nUser-authorized single-image switch completed.\n\n'
        '- Demo: http://127.0.0.1:8043/?v=sstv1\n- Both image scenarios, selected crops, and perspective-corrected panorama crops use SSTv1.\n'
        '- Image + point fusion retains original teacher and PTv2 seven-class outputs; two packaged regressions unchanged.\n'
        '- Original adapted mode remains available on the upstream API (mode=adapted); no original weight overwritten.\n'
        '- Physical rollback files: '+str(OUT/'pre_training_backup')+'\n'
        '- To roll back the entire UI switch, restore the four Python/HTML source files from that verified backup, then restart only ports 8039 and 8043. Do not restore over later edits without comparing them. The separately staged SSTv1 checkpoint files need not be deleted.\n'
        '- No training was performed during deployment. Prior evaluation limitations remain unchanged.\n',encoding='utf-8')
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
