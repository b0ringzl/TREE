"""HTTP upload regression tests against the live Demo; no fitting."""
import io,json,time
from pathlib import Path
import numpy as np,requests
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/point_live_deployment_20260913_v1'
BASE=ROOT/'outputs/SpaceScan_Trees_Demo_Files_20260913_56b49420/02_cloud_assist_gain'
URL='http://127.0.0.1:8043'
def post(image,cloud,name,**options):
    return requests.post(URL+'/api/predict-with-cloud',files={'image':('input.jpg',image,'image/jpeg'),'cloud':(name,cloud,'application/octet-stream')},data=dict(paired='true',z_up='true',units='metres',scenario='street',projection='photo',**options),timeout=180)
def main():
    assert requests.get(URL+'/api/health').json()['ready']
    checks=[]
    for folder in sorted(BASE.iterdir()):
        if not folder.is_dir():continue
        image=(folder/'image.jpg').read_bytes();cp=next(folder.glob('cloud_for_replay.*'));cloud=cp.read_bytes()
        response=post(image,cloud,cp.name)
        assert response.status_code==200,response.text[:600]
        result=response.json();(OUT/(folder.name+'_live.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        assert result['fusion_used'],result['point_evidence']['warnings']
        assert len(result['class_scope'])==7
        single=requests.post(URL+'/api/predict',files={'image':('input.jpg',image,'image/jpeg')},data={'source':'image','projection':'photo'},timeout=60).json()
        assert len(single['image_distribution'])==148
        assert [r['species'] for r in single['top3']]==[r['species'] for r in result['image_full_catalog_top3']]
        assert max(abs(a['score']-b['score']) for a,b in zip(single['top3'],result['image_full_catalog_top3']))<1e-5
        summary={k:[(r['species'],round(r['score']*100,2)) for r in v] for k,v in result['comparison'].items()}
        print(folder.name,json.dumps(summary),flush=True)
        renamed=post(image,cloud,'unrelated_name'+cp.suffix).json()
        diffs=[]
        for branch in result['comparison']:
            a=result['comparison'][branch];b=renamed['comparison'][branch]
            assert [v['species'] for v in a]==[v['species'] for v in b]
            diffs.extend(abs(v['score']-w['score']) for v,w in zip(a,b))
        print('Repeat maximum score difference',max(diffs),flush=True)
        assert max(diffs)<1e-5,'filename must not affect predictions (GPU numerical tolerance 1e-5)'
        assert renamed['point_model']['model_sha256']==result['point_model']['model_sha256']
        checks.append(dict(case=folder.name,comparison=summary,filename_invariant=True,repeat_max_abs_difference=max(diffs),cloud_sha256=result['point_evidence']['input_sha256']))
    # Check omission of orientation and pair confirmation without expensive point inference.
    files={'image':('input.jpg',image,'image/jpeg'),'cloud':(cp.name,cloud)}
    r=requests.post(URL+'/api/predict-with-cloud',files=files,data={'paired':'false'},timeout=30);assert r.status_code==400
    r=requests.post(URL+'/api/predict-with-cloud',files=files,data={'paired':'true','z_up':'false','projection':'photo'},timeout=60);assert r.status_code==200 and not r.json()['fusion_used']
    # Synthetic invalid inputs must never receive fabricated point classifications.
    for kind in ['coincident','planar']:
        x=np.zeros((300,3))
        if kind=='planar':x[:,:2]=np.random.default_rng(1).normal(size=(300,2))
        b=io.BytesIO();np.save(b,x)
        r=post(image,b.getvalue(),'shape.npy')
        if kind=='coincident':assert r.status_code==400
        else:assert r.status_code==200 and not r.json()['fusion_used']
    (OUT/'http_checks.json').write_text(json.dumps(dict(cases=checks,unpaired_rejected=True,z_axis_fallback=True,coincident_rejected=True,planar_fallback=True),ensure_ascii=False,indent=2),encoding='utf-8')
    print('HTTP CHECKS PASSED',flush=True)
if __name__=='__main__':main()
