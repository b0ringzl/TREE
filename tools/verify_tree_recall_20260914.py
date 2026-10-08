import base64,io,json
from pathlib import Path
import requests
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/tree_recall_fix_20260914_v1'
if __name__=='__main__':
 path=Path('C:/Users/25041725G/Desktop/验证/Snipaste_2026-09-10_14-38-57.png');data=path.read_bytes()
 r=requests.post('http://127.0.0.1:8043/api/detect-trees',files={'image':(path.name,data)},timeout=150)
 assert r.status_code==200,r.text
 d=r.json();assert len(d['candidates'])==3 and d['action']=='choose';assert d['partial_boxes_removed']==1
 canvas=Image.new('RGB',(1080,650),'#edf2e8');draw=ImageDraw.Draw(canvas);summary=[]
 for i,c in enumerate(d['candidates']):
  im=Image.open(io.BytesIO(base64.b64decode(c['thumbnail'].split(',')[1])));im.thumbnail((330,560));canvas.paste(im,(i*360+(360-im.width)//2,15))
  draw.text((i*360+15,595),f'Tree {i+1} | {c["score"]:.1%} | Grounding DINO',fill='#224d3c')
  assert c['native_size'][1]>550
  p=requests.post('http://127.0.0.1:8043/api/identify-tree',files={'image':(path.name,data)},data={'token':c['token']},timeout=120)
  assert p.status_code==200,p.text
  assert p.json()['mode']=='adapted'
  summary.append({k:c[k] for k in ('id','score','box','crop_box','native_size','detector')})
  print('PASS',i+1,c['score'],c['native_size'],flush=True)
 canvas.save(OUT/'three_candidates.png')
 (OUT/'verification.json').write_text(json.dumps(dict(candidates=summary,partial_boxes_removed=d['partial_boxes_removed'],action=d['action'],all_identify_routes_passed=True),indent=2),encoding='utf-8')
