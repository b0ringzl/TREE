"""Replay fixed point validation/regression plus three training trees through upload preprocessing."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from spacescan_trees_demo_v1.point_recognition import infer_upload
BASE=ROOT/'outputs/aquilaria_ptv2_fewshot_20260913_44056021'
if __name__=='__main__':
    records=[]
    for r in json.loads((BASE/'manifest.json').read_text(encoding='utf-8')):
        if r['split']=='train' and r['species']!='Aquilaria sinensis':continue
        path=BASE/r['derived'] if r['species']=='Aquilaria sinensis' else Path(r['source'])
        evidence,p=infer_upload(path.read_bytes(),path.name,'metres',True)
        pred=p['classes'][max(range(7),key=lambda i:p['scores'][i])] if p else None
        row=dict(key=r['key'],split=r['split'],truth=r['species'],prediction=pred,correct=pred==r['species'],quality_gate_passed=p is not None,path=str(path),scores=p['scores'] if p else None,warnings=evidence['warnings'])
        records.append(row);print(r['key'],r['split'],r['species'],pred,flush=True)
    out=ROOT/'outputs/point_live_deployment_20260913_v1/point_upload_regression.json'
    out.write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
