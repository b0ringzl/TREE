import sys,json
from pathlib import Path
from PIL import ImageDraw
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from spacescan_trees_demo_v1 import tree_detection as td
OUT=ROOT/'outputs/tree_recall_fix_20260914_v1'
if __name__=='__main__':
 OUT.mkdir(exist_ok=True)
 im=td.decode(Path('C:/Users/25041725G/Desktop/验证/Snipaste_2026-09-10_14-38-57.png').read_bytes())
 e=td.Engine();result={}
 for name,rows in [('yolo',e.boxes(im)),('dino',e.supplemental_boxes(im))]:
  rows=td.clean_boxes(sorted(rows,key=lambda r:-r['score']),*im.size)
  checks=e.background_checks([im.crop(td.crop_rect(r['box'],im.size)) for r in rows])
  pic=im.copy();draw=ImageDraw.Draw(pic)
  for i,(r,c) in enumerate(zip(rows,checks)):
   r['background_check']=c;box=[round(v*s) for v,s in zip(r['box'],[*im.size,*im.size])];draw.rectangle(box,outline='red',width=3);draw.text((box[0]+3,box[1]+3),f'{i}: {r["score"]:.3f}',fill='red')
  pic.save(OUT/(name+'.png'));result[name]=rows;print(name,json.dumps(rows),flush=True)
 (OUT/'raw_comparison.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
