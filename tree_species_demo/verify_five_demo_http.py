"""Real HTTP/model smoke tests. Does not submit reviews or modify weights."""
import io,json,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
import requests
from hk_five_species import DATA,MODEL,DOMAIN,NAMES
ROOT=Path(__file__).resolve().parent;URL='http://127.0.0.1:8037'
def get(url):
    r=requests.get(URL+url,timeout=30);r.raise_for_status();return r
def post(url,**kwargs):
    r=requests.post(URL+url,timeout=120,**kwargs);r.raise_for_status();return r.json()
def main():
    protected=json.loads((DATA/'protected_sources.json').read_text(encoding='utf-8'))
    protected.update(json.loads((DATA/'encoder_hashes.json').read_text(encoding='utf-8')))
    protected[str(MODEL)]=hashlib.sha256(MODEL.read_bytes()).hexdigest()
    samples=[s for s in get('/api/samples').json() if s['domain']==DOMAIN];assert len(samples)==5
    assert len(get('/api/catalog').json()[DOMAIN])==5
    checks=[]
    for s in samples:
        for mode in ('image','point','both'):
            d=post('/api/sample/'+s['id'],data={'mode':mode})
            assert d['domain']==DOMAIN and d['experimental']
            assert d['reference']['split']==s['split']
            assert d['decision']['label'] in NAMES+['Unknown']
            if mode!='both':assert list(d['results'])==[mode]
            else:assert {'image','point'}<=d['results'].keys()
            assert all(len(v['top3'])==3 for v in d['results'].values())
            if mode!='image':assert d['quality']['point']['inference_points']<=2048
            checks.append({'id':s['id'],'mode':mode,'decision':d['decision']['label'],'branches':list(d['results']),'reference_split':s['split']})
            print(s['truth'],mode,'OK',flush=True)
    s=samples[1];ib=get('/'+s['image']).content;pb=get('/'+s['point']).content
    def upload(image,point,paired,image_name='input.jpg'):
        files={}
        if image is not None:files['image']=(image_name,image)
        if point is not None:files['point']=('point.npz',point)
        return post('/api/predict',data={'domain':DOMAIN,'paired':str(paired).lower()},files=files)
    unpaired=upload(ib,pb,False);assert 'fusion' not in unpaired['results']
    normal=upload(ib,pb,True)
    if not any(q['severe'] for q in normal['quality'].values()):assert 'fusion' in normal['results']
    renamed=upload(ib,pb,True,'Leucaena_leucocephala_GROUND_TRUTH.jpg')
    assert normal['results']==renamed['results']
    dark=io.BytesIO();Image.new('RGB',(256,256),'black').save(dark,format='PNG')
    d=upload(dark.getvalue(),pb,True);assert d['quality']['image']['severe'] and 'fusion' not in d['results']
    assert d['decision_source_mode']=='point'
    a,z=np.meshgrid(np.linspace(-1,1,32),np.linspace(-1,1,32));plane=np.c_[a.ravel(),np.zeros(a.size),z.ravel()];buf=io.BytesIO();np.savez(buf,points_xyz=plane)
    d=upload(ib,buf.getvalue(),True);assert d['quality']['point']['severe'] and 'fusion' not in d['results'];assert d['decision_source_mode']=='image'
    sparse=DATA.parent/'leucaena_recovery_v1_20260907/instances/hewentian_pano__003656__bbb62cc708/point.npz'
    d=upload(None,sparse.read_bytes(),False);assert d['quality']['point']['inference_points']==1081
    invalid=requests.post(URL+'/api/predict',data={'domain':DOMAIN},timeout=30);assert invalid.status_code==400
    malformed=requests.post(URL+'/api/predict',data={'domain':DOMAIN},files={'point':('bad.npz',b'not a point cloud')},timeout=30);assert malformed.status_code==400
    all_samples=get('/api/samples').json();regressions=[]
    for domain in ['wuhan','hongkong','hongkong_vmms']:
        sample=next(s for s in all_samples if s['domain']==domain)
        d=post('/api/sample/'+sample['id'],data={'mode':'image' if domain=='hongkong' else 'both'})
        assert d['domain']==domain;regressions.append({'domain':domain,'id':sample['id'],'branches':list(d['results'])})
    for p,h in protected.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
    result={'new_sample_calls':checks,'negative_cases':['unpaired','dark_image','planar_cloud','missing_input','malformed_cloud'],
        'filename_not_used_for_classification':True,'sparse_1081_points_not_padded':True,'old_domain_regressions':regressions,
        'protected_sources_and_models_unchanged':True,'real_review_posts':0,'scope':'Integration smoke test, not accuracy validation'}
    out=ROOT/'diagnostics/hk_five_demo_integration';out.mkdir(parents=True,exist_ok=True)
    (out/'http_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print('PASS: 15 sample calls, 5 negative cases, 3 prior domains; no review/model changes.',flush=True)

if __name__=='__main__':main()
