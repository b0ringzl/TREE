"""Live version/distribution/dual-source regression checks, no training."""
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
import requests
from fastapi.testclient import TestClient
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from spacescan_trees_demo_v1 import server,tree_detection as td
from training_safety import verify_backup,sha256
OUT=ROOT/'outputs/SSTv1_Aquilaria_deployment_20260915_v1'
URL='http://127.0.0.1:8043'
VERSION='sstv1-aquilaria-user-20260915-v1'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(name,obj):(OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
    health=requests.get(URL+'/api/health',timeout=5).json()
    assert health['ready'] and health['single_image_version']==VERSION
    smoke=read(OUT/'expected.json');data=Path(smoke['image']).read_bytes()
    for scenario in ('street','conservation'):
        r=requests.post(URL+'/api/predict',files={'image':('tree.jpg',data)},data={'scenario':scenario,'projection':'photo'},timeout=90)
        r.raise_for_status();actual=r.json()
        assert actual['image_model_version']==VERSION
        np.testing.assert_allclose([actual['image_distribution'][s] for s in smoke['labels']],smoke['candidate_distribution'],atol=1e-5)
        save('single_'+scenario+'.json',actual)
    client=TestClient(server.app)
    w,h=td.decode(data).size
    meta={'expires':time.time()+300,'sha256':hashlib.sha256(data).hexdigest(),'crop_box':[0,0,w,h]}
    r=client.post('/api/identify-tree',files={'image':('tree.jpg',data)},data={'token':td.sign(meta)})
    assert r.status_code==200,r.text
    assert r.json()['image_model_version']==VERSION and r.json()['tree_detection']
    save('selected_crop.json',r.json())
    pano=requests.get(URL+'/api/test-panorama',timeout=30).content
    meta={'expires':time.time()+300,'sha256':hashlib.sha256(pano).hexdigest(),'crop_box':[0,0,600,600],
          'projection_view':td.panorama.view_spec(td.decode(pano).size,0,30),'projection':{'inferred_from_ratio':False}}
    r=client.post('/api/identify-tree',files={'image':('pano.jpg',pano)},data={'token':td.sign(meta)})
    assert r.status_code==200,r.text
    assert r.json()['image_model_version']==VERSION and 'perspective' in r.json()['preprocessing']
    save('perspective_crop.json',r.json())
    base=ROOT/'outputs/SpaceScan_Trees_Demo_Files_20260913_56b49420/02_cloud_assist_gain'
    checked=[]
    for folder in sorted(base.iterdir()):
        if not folder.is_dir():continue
        cp=next(folder.glob('cloud_for_replay.*'))
        r=requests.post(URL+'/api/predict-with-cloud',files={'image':('input.jpg',(folder/'image.jpg').read_bytes()),'cloud':(cp.name,cp.read_bytes())},data={'projection':'photo','paired':'true','units':'metres','z_up':'true'},timeout=120)
        r.raise_for_status();dual=r.json()
        assert dual['mode']=='teacher' and dual['fusion_used']
        before=read(ROOT/'outputs/point_live_deployment_20260913_v1'/(folder.name+'_live.json'))
        for branch in ('image','point','fusion'):
            assert [x['species'] for x in dual['comparison'][branch]]==[x['species'] for x in before['comparison'][branch]]
            np.testing.assert_allclose([x['score'] for x in dual['comparison'][branch]],[x['score'] for x in before['comparison'][branch]],atol=1e-5)
        save(folder.name+'_dual.json',dual);checked.append(folder.name)
    backup=verify_backup(OUT/'pre_training_backup',verify_sources=False)
    for row in backup['files']:
        if Path(row['source']).suffix in ('.pt','.joblib'):assert sha256(row['source'])==row['sha256']
    result={'deployed':True,'version':VERSION,'health':health,'live_matches_candidate':True,'both_scenarios':True,
            'selected_crop':True,'perspective_crop':True,'dual_unchanged':checked,'old_weights_unchanged':True,
            'backup_verified':True,'training_performed':False}
    save('VERIFIED.json',result)
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
