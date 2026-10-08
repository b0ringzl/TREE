"""Local read-only, allowlisted evidence viewer; no arbitrary file serving."""
import json,mimetypes
from pathlib import Path
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
from urllib.parse import urlparse,parse_qs
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/single_tree_panorama_selection_v1'
ROWS=json.loads((OUT/'screened_manifest.json').read_text(encoding='utf8'));BY={r['id']:r for r in ROWS}
class Handler(BaseHTTPRequestHandler):
 def do_GET(self):
  p=urlparse(self.path);q=parse_qs(p.query)
  try:
   if p.path=='/':return self.file(ROOT/'tools/single_tree_pairs.html')
   if p.path=='/api/manifest':return self.data(ROWS)
   if p.path=='/api/summary':return self.file(OUT/'screening_summary.json')
   if p.path=='/api/priority':return self.file(OUT/'priority_pairs.json')
   r=BY[q.get('id',[''])[0]];kind=q.get('kind',[''])[0]
   if p.path=='/asset':
    if kind=='geometry':return self.file(Path(r['geometry_preview']))
    if kind=='overview':return self.file(Path(r['panorama_overview']))
    if kind=='points':return self.file(OUT/r['id']/'points.json')
    if kind=='las':return self.file(Path(r['path']))
    v=next(v for v in r['views'] if v['index']==int(q.get('view',['0'])[0]))
    if kind in ['image','overlay','panorama']:return self.file(Path(v[kind]))
   self.send_error(404)
  except (KeyError,ValueError,StopIteration,FileNotFoundError):self.send_error(404)
 def data(self,x):
  b=json.dumps(x,ensure_ascii=False).encode();self.send_response(200);self.send_header('Content-Type','application/json; charset=utf-8');self.send_header('Content-Length',str(len(b)));self.end_headers();self.wfile.write(b)
 def file(self,path):
  self.send_response(200);self.send_header('Content-Type',mimetypes.guess_type(str(path))[0] or 'application/octet-stream');self.send_header('Content-Length',str(path.stat().st_size));self.send_header('Cache-Control','no-cache');self.end_headers()
  with path.open('rb') as f:
   while b:=f.read(1024*1024):self.wfile.write(b)
if __name__=='__main__':
 print('Read-only pair review http://127.0.0.1:8046/',flush=True);ThreadingHTTPServer(('127.0.0.1',8046),Handler).serve_forever()
