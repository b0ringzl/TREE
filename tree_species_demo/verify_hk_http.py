"""Exercise actual demo endpoints without submitting any real review decision."""
import io,json
from pathlib import Path
import numpy as np
from PIL import Image
import requests

URL='http://127.0.0.1:8037';ROOT=Path(__file__).resolve().parent.parent/'hk-tree-vmms/derived/hk_dualsource_three_species_20260907'
def main():
    samples=requests.get(URL+'/api/samples',timeout=10).json();samples=[s for s in samples if s['domain']=='hongkong_vmms'];assert len(samples)==6
    results=[]
    for sample in samples:
        for mode in ('image','point','both'):
            r=requests.post(URL+'/api/sample/'+sample['id'],data={'mode':mode},timeout=120);r.raise_for_status();d=r.json()
            assert d['domain']=='hongkong_vmms' and d['experimental'] is True
            assert d['reference']['split']=='experimental_group_holdout_weak_labels'
            if mode!='both':assert list(d['results'])==[mode]
            results.append({'id':sample['id'],'mode':mode,'decision':d['decision']['label'],'reference':d['reference']['truth'],'modalities':list(d['results'])})
    folder=ROOT/'instances'/samples[0]['pair_id'];ib=(folder/'image.jpg').read_bytes();pb=(folder/'point.npz').read_bytes()
    def upload(image,point,paired):
        r=requests.post(URL+'/api/predict',data={'domain':'hongkong_vmms','paired':str(paired).lower()},files={'image':('input.png',image),'point':('point.npz',point)},timeout=120);r.raise_for_status();return r.json()
    unpaired=upload(ib,pb,False);assert 'fusion' not in unpaired['results']
    dark=io.BytesIO();Image.new('RGB',(256,256),'black').save(dark,format='PNG');d=upload(dark.getvalue(),pb,True);assert d['quality']['image']['severe'] and 'fusion' not in d['results']
    x,z=np.meshgrid(np.linspace(-1,1,32),np.linspace(-1,1,32));plane=np.c_[x.ravel(),np.zeros(x.size),z.ravel()];buf=io.BytesIO();np.savez(buf,points_xyz=plane);d=upload(ib,buf.getvalue(),True);assert d['quality']['point']['severe'] and 'fusion' not in d['results']
    old=requests.post(URL+'/api/sample/wuhan_041',data={'mode':'both'},timeout=120);old.raise_for_status();assert 'fusion' in old.json()['results']
    report={'hk_sample_calls':results,'unpaired_fusion_disabled':True,'dark_image_fusion_disabled':True,'planar_cloud_fusion_disabled':True,'wuhan_fusion_regression':True,'real_review_decisions_submitted':0}
    (ROOT/'http_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'hk_calls':len(results),'negative_cases':3,'wuhan_regression':'passed','review_writes':0}))

if __name__=='__main__':main()
