"""Read-only full-scene extraction audit, independent of deployed inference."""
import sys,json,mimetypes
from pathlib import Path
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
from urllib.parse import urlparse,parse_qs
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
from hk_fullscene_extract_20260912 import OUT,read
from hk_fullscene_refine_20260912 import DEST

def catalog():
    rows=[]
    for p in (DEST/'maps').glob('*/candidates.json'):rows+=read(p)
    evidence={p.parent.name:read(p) for p in (DEST/'image_evidence').glob('*/record.json')}
    dense={r['id']:r for p in (DEST/'dense_recovery').glob('*/manifest.json') for r in read(p)}
    audit={r['id']:r for r in read(DEST/'agent_visual_audit.json')['records']} if (DEST/'agent_visual_audit.json').exists() else {}
    for r in rows:
        r['evidence']=evidence.get(r['id']);r['dense']=dense.get(r['id']);r['image_status']=r['evidence']['status'] if r['evidence'] else 'not_checked'
        r['visual_audit']=audit.get(r['id'])
    return rows

class Handler(BaseHTTPRequestHandler):
    def send(self,x,typ='application/json; charset=utf-8',status=200):
        if not isinstance(x,bytes):x=json.dumps(x,ensure_ascii=False,allow_nan=False).encode()
        self.send_response(status);self.send_header('Content-Type',typ);self.send_header('Content-Length',str(len(x)));self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.end_headers();self.wfile.write(x)
    def do_GET(self):
        try:self.get()
        except (KeyError,ValueError,FileNotFoundError) as e:self.send({'error':str(e)},status=404)
    def get(self):
        u=urlparse(self.path);q=parse_qs(u.query)
        if u.path=='/':return self.send((ROOT/'tools/hk_fullscene_review.html').read_bytes(),'text/html; charset=utf-8')
        if u.path=='/report':return self.send((DEST/'REPORT.md').read_bytes(),'text/plain; charset=utf-8')
        if u.path=='/api/summary':
            sources=read(OUT/'sources.json');scanned=[read(p) for p in (OUT/'maps').glob('*/coarse.json')];summary=read(DEST/'summary.json') if (DEST/'summary.json').exists() else {}
            return self.send(dict(sources=sources,scanned_maps=len(scanned),scanned_points=sum(s['scanned_points'] for s in scanned),refined=summary))
        rows=catalog()
        if u.path=='/api/catalog':return self.send([{k:r[k] for k in ['id','map_id','image_status']}|{'dense':bool(r['dense'])} for r in rows])
        key=q.get('key',[''])[0];r=next((r for r in rows if r['id']==key),None)
        if r is None:return self.send({'error':'not found'},status=404)
        if u.path=='/api/sample':return self.send(r)
        selected=r['dense'] or r
        if u.path=='/api/points':
            with np.load(selected['point'],allow_pickle=False) as f:points=f['points_xyz'].astype(float)
            return self.send({'points':points.round(6).tolist()})
        if u.path=='/asset':
            kind=q.get('kind',[''])[0];assets={'point':selected['point'],'views':selected['views'],'coarse':r['views']}
            for i,v in enumerate((r.get('evidence') or {}).get('views',[])):
                if 'crop' in v:assets['image'+str(i)]=v['crop'];assets['panorama'+str(i)]=v['panorama']
            if kind not in assets:return self.send({'error':'asset not available'},status=404)
            path=Path(assets[kind]);return self.send(path.read_bytes(),mimetypes.guess_type(path.name)[0] or 'application/octet-stream')
        return self.send({'error':'not found'},status=404)
if __name__=='__main__':
    print('Full scene review http://127.0.0.1:8045/',flush=True);ThreadingHTTPServer(('127.0.0.1',8045),Handler).serve_forever()
