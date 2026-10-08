import json
from pathlib import Path
import requests
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    path=Path('C:/Users/25041725G/Desktop/验证/Snipaste_2026-09-10_15-54-52.png')
    files={'image':(path.name,path.read_bytes(),'image/png')}
    records=[]
    for i in range(3):
        r=requests.post('http://127.0.0.1:8043/api/detect-trees',files=files,timeout=120)
        assert r.status_code==200,r.text
        d=r.json();json.dumps(d,allow_nan=False)
        assert d['candidates'],'Expected tree candidate'
        target=next((x for x in d['candidates'] if x['id']==d.get('auto_id')),d['candidates'][0])
        p=requests.post('http://127.0.0.1:8043/api/identify-tree',files=files,data={'token':target['token']},timeout=120)
        assert p.status_code==200,p.text
        result=p.json();assert result['mode']=='adapted';json.dumps(result,allow_nan=False)
        records.append(dict(attempt=i+1,action=d['action'],candidate_count=len(d['candidates']),checks=[x['background_check'] for x in d['candidates']],mode=result['mode'],top3=result['top3']))
        print('PASS',i+1,d['action'],len(d['candidates']),result['top3'][0]['species'],flush=True)
    (ROOT/'outputs/detection_json_fix_20260914_v1/verification.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
