"""Local 991 orthographic box selection; export original measured XYZ only."""
import json, mimetypes, re, secrets, shutil, sys, threading, uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
from hk_fullscene_extract_20260912 import header,chunks
from rebuild_banyan_coverage_preview import sample_ids
from training_safety import sha256
DATA=ROOT/'outputs/tree991_manual_boxes_20260913_v1'
STATIC=ROOT/'tools/tree991_box_ui'
SOURCE=ROOT/'lidar data/hk/tree9/tree991/map_dense.pcd'
COARSE=ROOT/'outputs/hk_fullscene_extract_20260912_v1/maps/recent_tree991/coarse.npy'
REFERENCE=Path('C:/Users/25041725G/AppData/Local/Temp/codex-clipboard-ae9af62f-0ef7-4aa6-adc1-da9638ff18fd.png')
LOCK=threading.Lock();EXPORT_LOCK=threading.Lock();TOKEN=secrets.token_urlsafe(24)
def read(p):return json.loads(p.read_text(encoding='utf8'))
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def empty():return dict(revision=0,slots={k:dict(ops=[],zmin=-10,zmax=80,confirmed=False,note='') for k in 'ABC'})
def validate_slot(s):
 if not isinstance(s,dict):raise ValueError('Invalid selection')
 ops=s.get('ops');assert isinstance(ops,list) and len(ops)<=40,'Too many selection operations'
 lo,hi=float(s['zmin']),float(s['zmax']);assert np.isfinite([lo,hi]).all() and -1000<=lo<hi<=1000,'Invalid Z range'
 clean=[]
 for j,o in enumerate(ops):
  assert o['mode'] in (['replace'] if j==0 else ['replace','intersect','subtract']),'First operation must replace'
  r=np.array(o['right'],float);u=np.array(o['up'],float);b=np.array(o['bounds'],float)
  assert r.shape==(3,) and u.shape==(3,) and b.shape==(4,) and np.isfinite(np.r_[r,u,b]).all(),'Invalid coordinates'
  assert abs(np.linalg.norm(r)-1)<1e-5 and abs(np.linalg.norm(u)-1)<1e-5 and abs(r@u)<1e-5,'Invalid projection basis'
  assert b[0]<b[1] and b[2]<b[3] and abs(b).max()<10000,'Invalid box'
  item=dict(mode=o['mode'],right=r.tolist(),up=u.tolist(),bounds=b.tolist())
  if 'polygon' in o:
   p=np.asarray(o['polygon'],float)
   assert p.ndim==2 and p.shape[1]==2 and 3<=len(p)<=128 and np.isfinite(p).all() and abs(p).max()<10000,'Invalid polygon (3–128 vertices)'
   assert (np.linalg.norm(p-np.roll(p,1,axis=0),axis=1)>1e-8).all(),'Duplicate polygon vertices'
   assert abs(np.sum(p[:,0]*np.roll(p[:,1],-1)-p[:,1]*np.roll(p[:,0],-1)))>1e-8,'Polygon has no area'
   item['polygon']=p.tolist()
   item['bounds']=[float(p[:,0].min()),float(p[:,0].max()),float(p[:,1].min()),float(p[:,1].max())]
  clean.append(item)
 note=s.get('note','');assert isinstance(note,str) and len(note)<=3000,'Note too long'
 assert isinstance(s.get('confirmed'),bool),'Confirmation must be boolean'
 return dict(ops=clean,zmin=lo,zmax=hi,confirmed=s['confirmed'],note=note)
def polygon_mask(a,b,p):
 inside=np.zeros(len(a),bool);edge=np.zeros(len(a),bool)
 for i in range(len(p)):
  x1,y1=p[i-1];x2,y2=p[i];dx=x2-x1;dy=y2-y1
  edge |= (abs((a-x1)*dy-(b-y1)*dx)<=1e-8*max(1,float(np.hypot(dx,dy)))) & (a>=min(x1,x2)-1e-8)&(a<=max(x1,x2)+1e-8)&(b>=min(y1,y2)-1e-8)&(b<=max(y1,y2)+1e-8)
  if dy!=0:inside ^= ((y1>b)!=(y2>b)) & (a<(x2-x1)*(b-y1)/dy+x1)
 return inside|edge

def selection_mask(x,s):
 result=np.zeros(len(x),bool)
 for o in s['ops']:
  a=x@np.array(o['right']);b=x@np.array(o['up']);v=o['bounds']
  inside=(a>=v[0])&(a<=v[1])&(b>=v[2])&(b<=v[3])
  if 'polygon' in o:
   ids=np.flatnonzero(inside);inside[ids]=polygon_mask(a[ids],b[ids],o['polygon'])
  if o['mode']=='replace':result=inside
  elif o['mode']=='intersect':result &= inside
  else:result &= ~inside
 return result & (x[:,2]>=s['zmin']) & (x[:,2]<=s['zmax']) & np.isfinite(x).all(1)
def init():
 DATA.mkdir(exist_ok=True);(DATA/'exports').mkdir(exist_ok=True);(DATA/'history').mkdir(exist_ok=True)
 if not (DATA/'source.json').exists():
  shutil.copy2(REFERENCE,DATA/'reference.png')
  q=np.load(COARSE);q.astype('<f4').tofile(DATA/'scene.bin')
  write(DATA/'source.json',dict(source=str(SOURCE),source_sha256=sha256(SOURCE),bytes=SOURCE.stat().st_size,mtime_ns=SOURCE.stat().st_mtime_ns,
   preview_source=str(COARSE),preview_sha256=sha256(COARSE),preview_count=len(q),source_count=int(header(SOURCE)['POINTS'][0]),
   bounds=[q.min(0).tolist(),q.max(0).tolist()],reference_sha256=sha256(REFERENCE),
   reference_scope='User confirms three circled image trees as Aquilaria sinensis; 3D correspondence awaits user selection',
   photo_labels={'A':'left circled tree','B':'middle circled tree','C':'right circled tree'},cloud_units='metres',training=False))
  write(DATA/'state.json',empty())
 return read(DATA/'source.json')
def export_cloud(slot,s,directory,source=SOURCE,meta=None):
 s=validate_slot(s);assert s['ops'],'Draw a selection first'
 if meta:
  assert source.stat().st_size==meta['bytes'] and source.stat().st_mtime_ns==meta['mtime_ns'],'Source changed'
  assert sha256(source)==meta['source_sha256'],'Source hash changed'
 source_h=sha256(source);before=source.stat();parts=[];ids=[];offset=0;total=0
 for x in chunks(source,header(source)):
  keep=np.flatnonzero(selection_mask(x,s));total+=len(keep)
  if total>2000000:raise ValueError('Selection exceeds 2 million points; tighten the box / 請縮小框選範圍')
  if len(keep):parts.append(x[keep]);ids.append(keep.astype(np.int64)+offset)
  offset+=len(x)
 assert total>=10,'Fewer than 10 points selected / 框選點數不足'
 xyz=np.concatenate(parts).astype('<f4');indices=np.concatenate(ids)
 assert (before.st_size,before.st_mtime_ns)==(source.stat().st_size,source.stat().st_mtime_ns),'Source changed during export'
 assert sha256(source)==source_h,'Source hash changed during export'
 directory.mkdir(parents=True,exist_ok=False)
 np.savez_compressed(directory/'full.npz',xyz=xyz,source_indices=indices)
 with (directory/'full.ply').open('wb') as f:
  f.write(f'ply\nformat binary_little_endian 1.0\ncomment original measured local metric XYZ; ROI not certified segmentation\nelement vertex {len(xyz)}\nproperty float x\nproperty float y\nproperty float z\nend_header\n'.encode('ascii'));f.write(xyz.tobytes())
 chosen=sample_ids(xyz,2048);np.savez_compressed(directory/'sample2048.npz',metric_xyz=xyz[chosen],source_indices=indices[chosen])
 record=dict(slot=slot,species='Aquilaria sinensis' if s['confirmed'] else None,chinese='土沉香' if s['confirmed'] else None,
  species_evidence='User-confirmed image and explicit 3D correspondence checkbox' if s['confirmed'] else '3D correspondence not confirmed',
  selection=s,source=str(source),source_sha256=source_h,point_count=len(xyz),sample_count=len(chosen),
  extent_m=np.ptp(xyz,axis=0).tolist(),coordinate_system='original local metric XYZ; Z up',
  geometry_quality='manual ROI only; ground, adjacent trees and crown completeness require later review',
  image_projection_verified=False,training_eligible=False,created_at=datetime.now(timezone.utc).isoformat(),
  files={name:dict(bytes=(directory/name).stat().st_size,sha256=sha256(directory/name)) for name in ['full.npz','full.ply','sample2048.npz']})
 write(directory/'record.json',record);return record

class Handler(BaseHTTPRequestHandler):
 def data(self,obj,status=200):
  b=json.dumps(obj,ensure_ascii=False,allow_nan=False).encode('utf8');self.send_response(status);self.send_header('Content-Type','application/json; charset=utf-8');self.send_header('Content-Length',str(len(b)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(b)
 def file(self,p,download=False):
  if not p.is_file():return self.send_error(404)
  self.send_response(200);self.send_header('Content-Type',mimetypes.guess_type(str(p))[0] or 'application/octet-stream');self.send_header('Content-Length',str(p.stat().st_size));self.send_header('Cache-Control','no-store')
  if download:self.send_header('Content-Disposition',f'attachment; filename="{p.parent.name}_{p.name}"')
  self.end_headers()
  with p.open('rb') as f:
   while b:=f.read(1024*1024):self.wfile.write(b)
 def do_GET(self):
  path=self.path.split('?')[0]
  if path=='/':return self.file(STATIC/'index.html')
  if path in ['/app.js','/polygon.js','/style.css']:return self.file(STATIC/path[1:])
  if path=='/api/init':return self.data(dict(meta=META,state=read(DATA/'state.json'),token=TOKEN))
  if path=='/scene.bin':return self.file(DATA/'scene.bin')
  if path=='/reference.png':return self.file(DATA/'reference.png')
  if path=='/api/exports':return self.data([dict(id=p.parent.name,**read(p)) for p in sorted((DATA/'exports').glob('*/record.json'))])
  m=re.fullmatch(r'/exports/([ABC]_[a-f0-9]{12})/(full.npz|full.ply|sample2048.npz|record.json)',path)
  if m:return self.file(DATA/'exports'/m[1]/m[2],True)
  self.send_error(404)
 def do_POST(self):
  if self.headers.get('Origin') not in [None,'http://127.0.0.1:8049'] or self.headers.get('X-Local-Token')!=TOKEN:return self.send_error(403)
  try:
   n=int(self.headers.get('Content-Length',0));assert 0<n<100000,'Invalid request size'
   body=json.loads(self.rfile.read(n))
   if self.path=='/api/save':
    slots=body['slots'];assert set(slots)==set('ABC'),'Invalid slots';clean={k:validate_slot(s) for k,s in slots.items()}
    with LOCK:
     old=read(DATA/'state.json')
     if body.get('revision')!=old['revision']:return self.data(dict(error='A newer save exists; reload before saving / 已有較新記錄，請重新載入'),409)
     record=dict(revision=old['revision']+1,slots=clean,updated_at=datetime.now(timezone.utc).isoformat())
     write(DATA/'history'/f'{record["revision"]:05d}_{uuid.uuid4().hex[:8]}.json',record)
     tmp=DATA/'state.tmp';write(tmp,record);tmp.replace(DATA/'state.json')
    return self.data(record)
   if self.path=='/api/export':
    slot=body['slot'];assert slot in 'ABC' and len(slot)==1,'Invalid slot';s=validate_slot(body['selection'])
    if not EXPORT_LOCK.acquire(blocking=False):return self.data(dict(error='Another export is running / 另一匯出正在執行'),409)
    try:
     key=slot+'_'+uuid.uuid4().hex[:12];record=export_cloud(slot,s,DATA/'exports'/key,meta=META)
    finally:EXPORT_LOCK.release()
    return self.data(dict(id=key,**record))
   self.send_error(404)
  except (AssertionError,ValueError,KeyError,TypeError) as e:self.data(dict(error=str(e)),400)
  except Exception as e:self.log_error('%s',e);self.data(dict(error='Export failed; original data unchanged / 匯出失敗，原始資料未修改'),500)
if __name__=='__main__':
 META=init();print('991 box tool http://127.0.0.1:8049/',flush=True);ThreadingHTTPServer(('127.0.0.1',8049),Handler).serve_forever()
