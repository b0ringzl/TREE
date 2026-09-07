"""Exercise real HTTP inference, every packaged sample, and invalid uploads."""
import io
import json
import time
from collections import Counter
from pathlib import Path
import numpy as np
import requests

ROOT=Path(__file__).resolve().parent
BASE='http://127.0.0.1:8037'

def main():
    samples=requests.get(BASE+'/api/samples',timeout=10).json()
    details=[]; checks=[]
    for idx,s in enumerate(samples):
        response=requests.post(BASE+'/api/sample/'+s['id'],data={'mode':'both'},timeout=120)
        response.raise_for_status(); r=response.json()
        assert all(np.isfinite(p['score']) for m in r['results'].values() for p in m.get('top3',[]))
        details.append({'id':s['id'],'domain':s['domain'],'truth':s['truth'],'prediction':r['decision']['label'],'correct':r['reference']['correct'],'quality_caution':s.get('quality_caution',False),'mode_outputs':list(r['results']),'elapsed_seconds':r['elapsed_seconds']})
        if idx%10==0: print(f'{idx+1}/{len(samples)} tested',flush=True)
    w=next(s for s in samples if s['domain']=='wuhan' and s['truth']!='Unknown')
    for mode in ('image','point'):
        r=requests.post(BASE+'/api/sample/'+w['id'],data={'mode':mode},timeout=60)
        r.raise_for_status(); assert list(r.json()['results'])==[mode]; checks.append(mode+'_only_passed')
    files={'image':('input.jpg',(ROOT/w['image']).read_bytes(),'image/jpeg'),'point':('cloud.npz',(ROOT/w['point']).read_bytes(),'application/octet-stream')}
    for paired in (False,True):
        r=requests.post(BASE+'/api/predict',data={'domain':'wuhan','paired':str(paired).lower()},files=files,timeout=60)
        r.raise_for_status(); assert ('fusion' in r.json()['results'])==paired
        checks.append('multipart_paired_'+str(paired)+'_passed')
    r=requests.post(BASE+'/api/predict',data={'domain':'hongkong'},files={'point':files['point']},timeout=60)
    r.raise_for_status(); assert r.json()['decision']['label']=='Unknown'; checks.append('unsupported_hk_point_unknown_passed')
    r=requests.post(BASE+'/api/predict',data={'domain':'wuhan'},timeout=10); assert r.status_code==400; checks.append('empty_input_rejected')
    bad=io.BytesIO(); np.save(bad,np.full((64,3),np.nan))
    r=requests.post(BASE+'/api/predict',data={'domain':'wuhan'},files={'point':('bad.npy',bad.getvalue())},timeout=10); assert r.status_code==400; checks.append('nonfinite_points_rejected')
    # Upload with a deliberately wrong class_index metadata: result must not change.
    with np.load(ROOT/w['point'],allow_pickle=False) as f: xyz=f['points_xyz']; centroid=f['centroid_xyz']; scale=f['scale']
    buf=io.BytesIO(); np.savez(buf,points_xyz=xyz,centroid_xyz=centroid,scale=scale,class_index=999)
    base=requests.post(BASE+'/api/predict',data={'domain':'wuhan'},files={'point':files['point']},timeout=60).json()
    changed=requests.post(BASE+'/api/predict',data={'domain':'wuhan'},files={'point':('wrong_label.npz',buf.getvalue())},timeout=60).json()
    assert base['decision']['top3']==changed['decision']['top3']; checks.append('npz_label_metadata_ignored')
    for kind in ('quality_dark','quality_blur'):
        item=next(d for d in details if d['id']==kind); assert item['prediction']=='Unknown'; checks.append(kind+'_rejected')
    c=requests.get(BASE+'/api/catalog',timeout=10).json(); assert len(c['hongkong'])==19 and len(c['wuhan'])==17
    counts=Counter((s['domain'],s['truth']) for s in samples)
    assert all(counts[(domain,item['scientific_name'])]>=2 for domain in ('hongkong','wuhan') for item in c[domain]); checks.append('all_36_known_classes_have_two_samples')
    summary={domain:{'total':sum(x['domain']==domain and x['truth']!='Unknown' for x in details),'correct':sum(x['domain']==domain and x['truth']!='Unknown' and x['correct'] for x in details),'unknown':sum(x['domain']==domain and x['truth']!='Unknown' and x['prediction']=='Unknown' for x in details)} for domain in ('hongkong','wuhan')}
    result={'status':'passed','scope':'quality-selected demo subset, NOT generalization benchmark','checks':checks,'sample_count':len(details),'known_sample_results':summary,'details':details}
    (ROOT/'验收测试结果.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='details'},ensure_ascii=False,indent=2))

if __name__=='__main__': main()
