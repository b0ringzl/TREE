"""Actual image + full-cloud inference; filenames/labels never determine predictions."""
import argparse,json,sys
from pathlib import Path
from datetime import datetime
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
from package_point_image_demo import Engine,image_scores,point_scores,canon

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--image',required=True);parser.add_argument('--cloud',required=True);parser.add_argument('--point-model',required=True);parser.add_argument('--output-dir',required=True);a=parser.parse_args()
 torch.set_num_threads(4);torch.manual_seed(20260913);engine=Engine();engine.load()
 logits,quality=image_scores(engine,a.image);pp,bundle,_=point_scores(a.cloud,a.point_model);classes=bundle['classes']
 idx=[next(i for i,n in enumerate(engine.labels) if canon(n)==c) for c in classes];q=logits[idx];ip=np.exp(q-q.max());ip/=ip.sum();full=np.exp(logits-logits.max());full/=full.sum()
 result=dict(image=str(Path(a.image).resolve()),cloud=str(Path(a.cloud).resolve()),point_model=str(Path(a.point_model).resolve()),class_scope=classes,formula='0.5 image + 0.5 cloud',main_demo_modified=False,quality=quality,full148_image_top3=[dict(species=engine.labels[i],score=float(full[i])) for i in np.argsort(-full)[:3]],branches={k:[dict(species=classes[i],score=float(p[i])) for i in np.argsort(-p)] for k,p in [('image_six',ip),('point_six',pp),('fusion_six',.5*ip+.5*pp)]},warning='Experimental fixed six-class comparison; uncalibrated scores; not a 148-class online fusion feature')
 out=Path(a.output_dir);out.mkdir(exist_ok=True,parents=True);path=out/('replay_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.json')
 with path.open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
 print(json.dumps(result,ensure_ascii=False,indent=2));print('Saved:',path)

if __name__=='__main__':main()
