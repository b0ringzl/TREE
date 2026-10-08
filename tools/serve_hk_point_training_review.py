"""Local read-only training result viewer. Does not deploy or load any model."""
import json,sys,mimetypes
from pathlib import Path
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
from urllib.parse import urlparse,parse_qs
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/hk_point_training_20260912_v1'
def read(path):return json.loads(path.read_text(encoding='utf-8'))
class Handler(BaseHTTPRequestHandler):
    def send(self,data,typ='application/json; charset=utf-8',status=200):
        if not isinstance(data,bytes):data=json.dumps(data,ensure_ascii=False,allow_nan=False).encode('utf-8')
        self.send_response(status);self.send_header('Content-Type',typ);self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.end_headers();self.wfile.write(data)
    def do_GET(self):
        u=urlparse(self.path);query=parse_qs(u.query);key=query.get('key',[''])[0]
        if u.path=='/':return self.send((ROOT/'tools/hk_point_training_review.html').read_bytes(),'text/html; charset=utf-8')
        if u.path=='/api/summary':return self.send(read(OUT/'final_summary.json'))
        if u.path=='/api/catalog':return self.send([dict(key=r['key'],species=r['species'],split=r['split'],id=r['id']) for r in read(OUT/'viewer_manifest.json')])
        if u.path=='/api/new':return self.send(read(OUT/'new_las_associations.json'))
        if u.path=='/report':return self.send((OUT/'REPORT.md').read_bytes(),'text/plain; charset=utf-8')
        rows={r['key']:r for r in read(OUT/'viewer_manifest.json')};new={r['id']:r for r in read(OUT/'new_las_associations.json')};evidence={r['id']:r for r in read(OUT/'new_las_evidence/manifest.json')};r=rows.get(key) or new.get(key)
        if r is None:return self.send({'error':'not found'},status=404)
        if u.path=='/api/sample':
            predictions={run:next((p for p in ps if p['key']==key),None) for run,ps in read(OUT/'all_predictions.json').items()}
            return self.send(dict(record=r,predictions=predictions,evidence=evidence.get(key)))
        if u.path=='/api/points':
            with np.load(r['point'],allow_pickle=False) as f:p=f['points_xyz'].astype(float)
            return self.send(dict(points=p.round(6).tolist(),quality=r['quality']))
        if u.path=='/asset':
            kind=query.get('kind',[''])[0];assets=dict(point=r['point'],views=r['views'])
            if 'source_assets' in r:assets.update({k:v for k,v in r['source_assets'].items() if k in ['image','projection']})
            if key in evidence:assets.update({k:evidence[key][k] for k in ['projection','image']})
            if kind not in assets:return self.send({'error':'invalid asset'},status=404)
            path=Path(assets[kind]);return self.send(path.read_bytes(),mimetypes.guess_type(path.name)[0] or 'application/octet-stream')
        return self.send({'error':'not found'},status=404)
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8044);a=p.parse_args();print(f'Point training result viewer: http://127.0.0.1:{a.port}/',flush=True);ThreadingHTTPServer(('127.0.0.1',a.port),Handler).serve_forever()
