"""Local allowlisted screening review; decisions stay separate from training labels."""
import json
import mimetypes
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/point_assist_case_selection_20260913_v1'
ROWS=json.loads((OUT/'manifest.json').read_text(encoding='utf8'))
BY={r['id']:r for r in ROWS}
LOCK=threading.Lock()

class Handler(BaseHTTPRequestHandler):
    def data(self,obj):
        b=json.dumps(obj,ensure_ascii=False).encode('utf8')
        self.send_response(200);self.send_header('Content-Type','application/json; charset=utf-8');self.send_header('Content-Length',str(len(b)));self.end_headers();self.wfile.write(b)
    def file(self,p):
        p=Path(p)
        if not p.is_file():return self.send_error(404)
        self.send_response(200);self.send_header('Content-Type',mimetypes.guess_type(str(p))[0] or 'application/octet-stream');self.send_header('Content-Length',str(p.stat().st_size));self.send_header('Cache-Control','no-store');self.end_headers()
        with p.open('rb') as f:
            while b:=f.read(1024*1024):self.wfile.write(b)
    def do_GET(self):
        q=urlparse(self.path);args=parse_qs(q.query)
        if q.path=='/':return self.file(ROOT/'tools/point_assist_cases_20260913.html')
        if q.path=='/api/data':return self.data(dict(rows=ROWS,predictions=json.loads((OUT/'image_predictions.json').read_text(encoding='utf8')),reviews=self.reviews()))
        if q.path=='/asset':
            try:
                r=BY[args['id'][0]];kind=args['kind'][0]
                if kind in ['preview','cloud']:return self.file(r[kind])
                view=next(v for v in r['views'] if v['index']==int(args['view'][0]))
                if kind in ['image','overlay','panorama']:return self.file(view[kind])
            except (KeyError,ValueError,StopIteration,TypeError):pass
        self.send_error(404)
    def reviews(self):
        p=OUT/'human_case_reviews.json'
        return json.loads(p.read_text(encoding='utf8')) if p.exists() else {}
    def do_POST(self):
        if self.path!='/api/review':return self.send_error(404)
        if self.headers.get('Origin') not in [None,'http://127.0.0.1:8047']:return self.send_error(403)
        try:
            n=int(self.headers.get('Content-Length','0'))
            if not 0<n<20000:return self.send_error(400)
            r=json.loads(self.rfile.read(n));key=r['id'];assert key in BY
            assert r['decision'] in ['accept_pair','needs_repair','wrong_species']
            r={k:r[k] for k in ['id','decision','note']};assert isinstance(r['note'],str)
            r['updated_at']=datetime.now(timezone.utc).isoformat()
            r['scope']='Case-screening feedback only; not training acceptance or measured-height certification'
            with LOCK:
                reviews=self.reviews();reviews[key]=r
                with (OUT/'human_case_review_history.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(r,ensure_ascii=False)+'\n')
                tmp=OUT/'human_case_reviews.tmp';tmp.write_text(json.dumps(reviews,ensure_ascii=False,indent=2),encoding='utf8');tmp.replace(OUT/'human_case_reviews.json')
            self.data(r)
        except (KeyError,ValueError,AssertionError):self.send_error(400)

if __name__=='__main__':
    ThreadingHTTPServer(('127.0.0.1',8047),Handler).serve_forever()
