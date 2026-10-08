"""Real upstream and upload smoke/regression checks; no fitting."""
import hashlib,json,sys,time
from pathlib import Path
import requests,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from spacescan_trees_demo_v1 import server,tree_detection as td
from fastapi.testclient import TestClient
OUT=ROOT/'outputs/adapted_teacher_deployment_20260914_v1'
BASE=ROOT/'outputs/SpaceScan_Trees_Demo_Files_20260913_56b49420/02_cloud_assist_gain'
URL='http://127.0.0.1:8043'
def main():
    assert requests.get(URL+'/api/health').json()['ready']
    results=[]
    for folder in sorted(BASE.iterdir()):
        if not folder.is_dir():continue
        image=(folder/'image.jpg').read_bytes();prepared,_,_=server.prepare(image)
        expected=requests.post(server.UPSTREAM+'/api/predict',files={'image':('p.jpg',prepared)},data={'mode':'adapted'},timeout=90).json()
        r=requests.post(URL+'/api/predict',files={'image':('input.jpg',image)},data={'projection':'photo'},timeout=90);assert r.status_code==200,r.text
        single=r.json();assert single['mode']=='adapted' and len(single['image_distribution'])==148
        assert [x['species'] for x in single['top3']]==[x['species'] for x in expected['comparison']['adapted']]
        np.testing.assert_allclose([x['score'] for x in single['top3']],[x['score'] for x in expected['comparison']['adapted']],atol=1e-5)
        cp=next(folder.glob('cloud_for_replay.*'))
        r=requests.post(URL+'/api/predict-with-cloud',files={'image':('input.jpg',image),'cloud':(cp.name,cp.read_bytes())},data={'projection':'photo','paired':'true','units':'metres','z_up':'true'},timeout=120)
        assert r.status_code==200,r.text
        dual=r.json();assert dual['mode']=='teacher' and dual['fusion_used']
        before=json.loads((ROOT/'outputs/point_live_deployment_20260913_v1'/(folder.name+'_live.json')).read_text(encoding='utf-8'))
        for branch in ('image','point','fusion'):
            assert [x['species'] for x in dual['comparison'][branch]]==[x['species'] for x in before['comparison'][branch]]
            np.testing.assert_allclose([x['score'] for x in dual['comparison'][branch]],[x['score'] for x in before['comparison'][branch]],atol=1e-5)
        results.append(dict(case=folder.name,adapted=single['top3'],original=expected['comparison']['teacher'],distribution_count=len(single['image_distribution']),dual_unchanged=True))
        print(folder.name,'adapted',single['top3'][0]['species'],single['top3'][0]['score'],'dual unchanged',flush=True)
    # Exercise the actual selected-crop endpoint with a signed selection. Detection is not changed/tested here.
    meta=dict(expires=time.time()+300,sha256=hashlib.sha256(image).hexdigest(),crop_box=[0,0,600,600])
    client=TestClient(server.app)
    r=client.post('/api/identify-tree',files={'image':('image.jpg',image)},data={'token':td.sign(meta)})
    assert r.status_code==200,r.text
    assert r.json()['mode']=='adapted' and r.json()['tree_detection']
    # Same endpoint following spherical-to-perspective projection.
    pano=requests.get(URL+'/api/test-panorama',timeout=30).content
    spec=td.panorama.view_spec(td.decode(pano).size,0,30)
    meta=dict(expires=time.time()+300,sha256=hashlib.sha256(pano).hexdigest(),crop_box=[0,0,600,600],projection_view=spec,projection={'inferred_from_ratio':False})
    r=client.post('/api/identify-tree',files={'image':('panorama.jpg',pano)},data={'token':td.sign(meta)})
    assert r.status_code==200,r.text
    assert r.json()['mode']=='adapted' and 'perspective' in r.json()['preprocessing']
    (OUT/'verification.json').write_text(json.dumps(dict(cases=results,selected_crop_adapted=True,projected_crop_adapted=True,detector_retrained=False),ensure_ascii=False,indent=2),encoding='utf-8')
    print('ALL DEPLOYMENT CHECKS PASSED',flush=True)
if __name__=='__main__':main()
