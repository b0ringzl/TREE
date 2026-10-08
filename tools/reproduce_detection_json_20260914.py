import sys,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from spacescan_trees_demo_v1 import tree_detection as td
def bad(x,path='root'):
    if isinstance(x,float) and not math.isfinite(x):print(path,repr(x),flush=True)
    elif isinstance(x,dict):
        for k,v in x.items():bad(v,path+'.'+k)
    elif isinstance(x,list):
        for k,v in enumerate(x):bad(v,path+'.'+str(k))
if __name__=='__main__':
    d=td.detect(Path('C:/Users/25041725G/Desktop/验证/Snipaste_2026-09-10_15-54-52.png').read_bytes())
    bad(d);print('ACTION',d['action'],flush=True);json.dumps(d,allow_nan=False);print('STRICT JSON OK',flush=True)
