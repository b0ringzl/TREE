"""Read-only inference and localization checks; never submits reviews."""
import json,re
from pathlib import Path
import requests
URL='http://127.0.0.1:8037'
BANNED=re.compile(r'香港|武汉|武漢|hong\s*kong|wuhan',re.I)
def check(v):
    encoded=json.dumps(v,ensure_ascii=False)
    assert not BANNED.search(encoded),encoded[:400]
def main():
    catalog=requests.get(URL+'/api/ui/catalog',timeout=30).json()
    for m in catalog['models']:check({k:v for k,v in m.items() if k!='key'})
    samples=requests.get(URL+'/api/ui/samples',timeout=30).json()
    for s in samples:check(s['display_name'])
    records=[]
    for m in catalog['models']:
        s=next(s for s in samples if s['domain']==m['key'])
        for mode in ('image','point','both'):
            if mode=='point' and not s.get('point'):continue
            r=requests.post(URL+'/api/sample/'+s['id'],data={'mode':mode},timeout=180);r.raise_for_status();d=r.json();p=d['presentation'];check(p)
            assert p['decision']['score']==d['decision']['score']
            records.append({'model':m['name'],'mode':mode,'presentation':p})
            print(m['name'],mode,'PASS',flush=True)
    for data in ({'domain':'invalid'},{'domain':'hongkong_vmms_five'}):
        r=requests.post(URL+'/api/predict',data=data,timeout=30);assert r.status_code==400;check(r.json())
        assert re.search('[\u4e00-\u9fff]',r.json()['detail']) and re.search('[A-Za-z]',r.json()['detail'])
    for file in ('demo.html','demo.js','demo.css'):
        r=requests.get(URL+'/static/'+file,timeout=30);r.raise_for_status()
        if file=='demo.html':check(r.text);assert 'zh-Hant' in r.text
    out=Path(__file__).parent/'diagnostics/demo_locale';out.mkdir(parents=True,exist_ok=True)
    (out/'http_verification.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'PASS: {len(records)} actual inference presentations, all catalog/sample display labels, bilingual errors and static assets.',flush=True)
if __name__=='__main__':main()
