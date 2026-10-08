import json,mimetypes,threading
from pathlib import Path
from datetime import datetime,timezone
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlparse,parse_qs
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/tree991_single_tree_review_20260913_v1';LOCK=threading.Lock()
ROWS=json.loads((OUT/'manifest.json').read_text(encoding='utf8'));BY={r['id']:r for r in ROWS}
class Handler(BaseHTTPRequestHandler):
 def file(self,p):
  if not p.is_file():return self.send_error(404)
  self.send_response(200);self.send_header('Content-Type',mimetypes.guess_type(str(p))[0] or 'application/octet-stream');self.send_header('Content-Length',str(p.stat().st_size));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(p.read_bytes())
 def do_GET(self):
  p=urlparse(self.path);q=parse_qs(p.query)
  if p.path=='/':return self.file(ROOT/'tools/tree991_review.html')
  if p.path=='/manifest.json':return self.file(OUT/'manifest.json')
  if p.path=='/overview.png':return self.file(OUT/'overview.png')
  if p.path=='/asset':
   try:
    r=BY[q['id'][0]];kind=q['kind'][0];mapping={'preview':r['preview'],'cloud':r['cloud'],'point':r['point'],'display':str(OUT/r['id']/'display.json')}
    return self.file(Path(mapping[kind]))
   except (KeyError,ValueError):pass
  self.send_error(404)
 def do_POST(self):
  if self.path!='/review':return self.send_error(404)
  if self.headers.get('Origin') not in [None,'http://127.0.0.1:8048']:return self.send_error(403)
  try:
   n=int(self.headers.get('Content-Length',0));assert 0<n<10000
   r=json.loads(self.rfile.read(n));assert r['id'] in BY;assert r['status'] in ['single_tree','mixed','not_tree','unsure'];assert isinstance(r.get('note',''),str)
   r={k:r.get(k,'') for k in ['id','status','note']};r['time']=datetime.now(timezone.utc).isoformat();r['scope']='Geometry review only, no species or projection confirmation'
   with LOCK:
    with (OUT/'reviews.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(r,ensure_ascii=False)+'\n')
   self.send_response(200);self.send_header('Content-Type','application/json');self.end_headers();self.wfile.write(b'{"saved":true}')
  except (ValueError,KeyError,AssertionError):self.send_error(400)
if __name__=='__main__':ThreadingHTTPServer(('127.0.0.1',8048),Handler).serve_forever()
