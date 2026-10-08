"""Download official public checkpoint, verify its HF LFS SHA-256."""
import requests,hashlib,json,concurrent.futures
from urllib3.util.retry import Retry
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'models/sonata_official_v1';D.mkdir(exist_ok=True)
info=requests.get('https://huggingface.co/api/models/facebook/sonata?blobs=true',timeout=30).json();entry=next(s for s in info['siblings'] if s['rfilename']=='sonata.pth');expected=entry['lfs']['sha256'];p=D/'sonata.pth';tmp=D/'sonata.direct-download.part'
if p.exists():
 assert hashlib.sha256(p.read_bytes()).hexdigest()==expected
else:
 url='https://huggingface.co/facebook/sonata/resolve/'+info['sha']+'/sonata.pth?download=true';size=entry['lfs']['size'];chunk=8*1024*1024;parts=D/'download_parts';parts.mkdir(exist_ok=True)
 def fetch(i):
  a=i*chunk;b=min(size,(i+1)*chunk)-1;piece=parts/f'{i:03d}.part'
  if piece.exists() and piece.stat().st_size==b-a+1:return piece
  session=requests.Session();session.mount('https://',requests.adapters.HTTPAdapter(max_retries=Retry(total=5,backoff_factor=1,status_forcelist=[429,500,502,503,504])))
  r=session.get(url,headers={'Range':f'bytes={a}-{b}'},stream=True,timeout=(20,90));r.raise_for_status();assert r.status_code==206
  with piece.open('wb') as f:
   for block in r.iter_content(1024*1024):f.write(block)
  assert piece.stat().st_size==b-a+1
  print('downloaded part',i,flush=True);return piece
 with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:ordered=list(pool.map(fetch,range((size+chunk-1)//chunk)))
 h=hashlib.sha256()
 with tmp.open('wb') as f:
  for piece in ordered:
   with piece.open('rb') as src:
    while b:=src.read(4*1024*1024):f.write(b);h.update(b)
 assert h.hexdigest()==expected,(h.hexdigest(),expected)
 tmp.rename(p)
(D/'provenance.json').write_text(json.dumps(dict(repository='facebook/sonata',revision=info['sha'],filename='sonata.pth',sha256=expected,size=p.stat().st_size,source='https://huggingface.co/facebook/sonata',license='Apache-2.0 (official repository); no redistribution in this experiment'),indent=2),encoding='utf8');print('VERIFIED',p,flush=True)
