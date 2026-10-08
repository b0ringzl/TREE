"""Downsample real full-scene points for location recognition; no model operations."""
import sys,json,base64
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
DEST=Path('C:/Users/25041725G/.codex/visualizations/2026/09/05/01a07269-4abc-7743-8aba-4e5138eb11c9')
def main():
    records=[]
    for name in ['tree991','tree993','tree9111','tree9112','tree9113']:
        folder=ROOT/'outputs/hk_fullscene_extract_20260912_v1/maps'/('recent_'+name);x=np.load(folder/'coarse.npy');meta=json.loads((folder/'coarse.json').read_text(encoding='utf-8'));rng=np.random.default_rng(20260912);q=x[rng.choice(len(x),min(19500,len(x)),replace=False)].astype(np.float64);lo=x.min(0).astype(float);hi=x.max(0).astype(float);step=max(hi-lo)/65534.;values=np.rint((q-lo)/step).astype('<u2');records.append(dict(id=name,extent=(hi-lo).tolist(),min=lo.tolist(),center=((hi+lo)/2).tolist(),span=float(max(hi-lo)),step=step,encoded=base64.b64encode(values.tobytes()).decode(),original_points=meta['scanned_points'],preview_points=len(q)))
    template=(ROOT/'tools/capture_locations_fragment.html').read_text(encoding='utf-8');out=template.replace('__SCENE_DATA__',json.dumps(records,separators=(',',':')));DEST.mkdir(parents=True,exist_ok=True);path=DEST/'capture-locations.html';path.write_text(out,encoding='utf-8');assert path.stat().st_size<1000000
    print(json.dumps(dict(path=str(path),bytes=path.stat().st_size,scenes=[{k:v for k,v in r.items() if k!='encoded'} for r in records]),ensure_ascii=False))
if __name__=='__main__':main()
