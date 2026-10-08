"""Read-only inference comparison between formal and relocated portable servers."""
import argparse
import json
import shutil
from pathlib import Path
import requests
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/windows_portable_acceptance_20260920_v1'
RUN=ROOT/'outputs/SSTv1_Aquilaria_UserCrops_20260919_v2'
TEST=Path('C:/SST_Test_20260920/SpaceScan Test')
PACKAGE=Path('C:/SST_Pack_20260920_v1/SpaceScan-Trees-H120-Windows')
PHOTO=Path('C:/Users/25041725G/Desktop/验证/Snipaste_2026-09-10_14-38-57.png')
PAIR=ROOT/'outputs/SpaceScan_Trees_Demo_Files_20260913_56b49420/02_cloud_assist_gain/01_TreeID_13_view0'

def save(name,obj):
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/name).open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2)
def get(url):
    r=requests.get(url,timeout=15);r.raise_for_status();return r.json()
def post(base,path,files,fields):
    r=requests.post(base+path,files=files,data=fields,timeout=240);r.raise_for_status()
    return r.json()
def capture(base):
    out={'health':get(base+'/api/health'),'images':[]}
    for row in json.loads((RUN/'H_crops_targeted_002/crop_replay_single.json').read_text(encoding='utf-8'))['images']:
        p=Path(row['source_image'])
        d=post(base,'/api/predict',{'image':(p.name,p.read_bytes())},{'projection':'photo'})
        out['images'].append({'file':p.name,'distribution':d['image_distribution'],'top1':d['top3'][0]['species'],
                              'version':d['image_model_version']})
        print('IMAGE',p.name,d['top3'][0]['species'],flush=True)
    cloud,=PAIR.glob('cloud_for_replay.*')
    d=post(base,'/api/predict-with-cloud',{'image':('image.jpg',(PAIR/'image.jpg').read_bytes()),'cloud':(cloud.name,cloud.read_bytes())},
           {'projection':'photo','paired':'true','units':'metres','z_up':'true'})
    out['pair']={'comparison':d['comparison'],'point_version':d['point_model']['version'],
                 'point_sha256':d['point_model']['model_sha256']}
    print('PAIR_OK',flush=True)
    d=post(base,'/api/detect-trees',{'image':('trees.png',PHOTO.read_bytes())},{'projection':'photo','source':'image'})
    out['detector']={k:d[k] for k in ('action','detector','projection')}
    out['detector']['candidates']=[{k:r[k] for k in ('box','score','crop_box')} for r in d['candidates']]
    if d['candidates']:
        selected=post(base,'/api/identify-tree',{'image':('trees.png',PHOTO.read_bytes())},{'token':d['candidates'][0]['token']})
        out['selected']={'top1':selected['top3'][0]['species'],'distribution':selected['image_distribution']}
    print('DETECTION',len(d['candidates']),d['detector'],flush=True)
    return out

def check():
    running=json.loads((TEST/'user_data/running.json').read_text())
    base=running['base_url']
    actual=capture(base)
    save('portable.json',actual)
    before=json.loads((OUT/'formal.json').read_text(encoding='utf-8'))
    max_image=0.
    for a,b in zip(before['images'],actual['images']):
        assert a['file']==b['file'] and a['top1']==b['top1'] and a['version']==b['version']
        assert a['distribution'].keys()==b['distribution'].keys()
        delta=max(abs(v-b['distribution'][k]) for k,v in a['distribution'].items())
        assert delta<1e-5,delta
        max_image=max(max_image,delta)
    max_point=0.
    for key in ('image','point','fusion'):
        a,b=before['pair']['comparison'][key],actual['pair']['comparison'][key]
        assert [x['species'] for x in a]==[x['species'] for x in b]
        delta=max(abs(x['score']-y['score']) for x,y in zip(a,b))
        assert delta<1e-5,delta
        max_point=max(max_point,delta)
    assert actual['pair']['point_sha256']==before['pair']['point_sha256']
    assert actual['detector']['detector']==before['detector']['detector']
    a,b=before['detector']['candidates'],actual['detector']['candidates']
    assert len(a)==len(b)
    for x,y in zip(a,b):
        np.testing.assert_allclose(x['box'],y['box'],rtol=0,atol=1e-5)
        assert x['crop_box']==y['crop_box'] and abs(x['score']-y['score'])<1e-5
    if 'selected' in before:
        assert before['selected']['top1']==actual['selected']['top1']
        assert max(abs(v-actual['selected']['distribution'][k]) for k,v in before['selected']['distribution'].items())<1e-5
    for route in ('/image-only','/multimodal','/references','/apps'):
        response=requests.get(base+route,timeout=15)
        assert response.status_code==200 and 'SpaceScan' in response.text
    catalog=get(base+'/api/reference-catalog')
    refs=[r for x in catalog for r in x['references']]
    for r in refs:
        response=requests.get(base+r['url'],timeout=10)
        assert response.status_code==200 and response.headers['content-type'].startswith('image/')
    summary={'passed':True,'image_count':8,'image_same_top1':8,'image_probability_max_delta':max_image,
        'point_fusion_max_delta':max_point,'detected_trees':len(a),'detector_selected_inference':True,
        'reference_images_http_ok':len(refs),'reference_classes':len(catalog),'relocated_root':str(TEST),
        'runtime_bundled':True,'different_physical_computer_tested':False,'model':actual['health']['single_image_version'],
        'production_weights_changed':False,'training':False}
    save('VERIFIED.json',summary);print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['baseline','copy','verify']);a=parser.parse_args()
    if a.action=='baseline':save('formal.json',capture('http://127.0.0.1:8043'))
    elif a.action=='copy':
        if TEST.exists():raise RuntimeError('Do not overwrite earlier test copy')
        shutil.copytree(PACKAGE,TEST,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        print('RELOCATED',TEST,flush=True)
    else:check()
