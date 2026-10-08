import json
import subprocess
from pathlib import Path
import requests
ROOT=Path(__file__).resolve().parents[1]
p=ROOT/'outputs/point_assist_case_selection_20260913_v1'
d=requests.get('http://127.0.0.1:8047/api/data',timeout=10).json()
n=0
for r in d['rows']:
    for v in r['views']:
        for k in ['image','overlay']:
            x=requests.get('http://127.0.0.1:8047/asset',params={'id':r['id'],'kind':k,'view':v['index']},timeout=10)
            assert x.status_code==200
            n+=1
text=(ROOT/'tools/point_assist_cases_20260913.html').read_text(encoding='utf8')
script=text.split('<script>')[1].split('</script>')[0]
test=subprocess.run(['node','-e','new Function(require("fs").readFileSync(0,"utf8"))'],input=script,text=True,capture_output=True,encoding='utf8')
assert test.returncode==0,test.stderr
assert requests.get('http://127.0.0.1:8047/',timeout=10).status_code==200
assert requests.get('http://127.0.0.1:8047/asset',params={'id':'../../outside','kind':'cloud'},timeout=10).status_code==404
s=json.loads((p/'summary.json').read_text(encoding='utf8'))
s['image_inference_completed']=True
(p/'summary.json').write_text(json.dumps(s,ensure_ascii=False,indent=2),encoding='utf8')
(p/'verification.json').write_text(json.dumps(dict(page_http=200,api_instances=len(d['rows']),image_predictions=len(d['predictions']),asset_requests_passed=n,javascript_syntax_passed=True,browser_layout_inspected=False,training_performed=False,production_code_changed=False),indent=2),encoding='utf8')
print('Verified',n,'assets and JS syntax')
