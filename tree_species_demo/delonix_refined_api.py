"""Review versioned refined Delonix pairs without touching scene or older pair reviews."""
import json,os,threading
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from fastapi import APIRouter,HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

ROOT=Path(__file__).resolve().parent.parent/'hk-tree-vmms/derived/delonix_refined_v1_20260907'
router=APIRouter();LOCK=threading.Lock()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def records():
    for name in ('manifest.json','manifest.progress.json'):
        if (ROOT/name).exists():return read(ROOT/name)
    return []
def resolve(ident):
    r=next((x for x in records() if x['id']==ident),None)
    if r is None:raise HTTPException(404,'实例不存在')
    return r
@router.get('/hk-delonix-refined')
def page():return FileResponse(Path(__file__).resolve().parent/'static/hk-delonix-refined.html')
@router.get('/api/hk-delonix-refined')
def listing():
    reviews=read(ROOT/'reviews.json') if (ROOT/'reviews.json').exists() else {}
    qa=read(ROOT/'agent_visual_qa.json') if (ROOT/'agent_visual_qa.json').exists() else {}
    return {'items':[{**r,'pair_review':reviews.get(r['id']),'agent_qa':qa.get(r['id'])} for r in records()]}
@router.get('/hk-delonix-refined-file/{ident}/{name}')
def file(ident:str,name:str):
    resolve(ident)
    if name not in {'projection.jpg','panorama.jpg','point_views.jpg','before_views.jpg','image.jpg','point.npz','cleaned_local.npz','record.json'}:raise HTTPException(404)
    path=ROOT/'instances'/ident/name
    if not path.exists():raise HTTPException(404)
    return FileResponse(path)
@router.get('/api/hk-delonix-refined/{ident}/preview')
def preview(ident:str):
    if resolve(ident)['status']!='candidate_pending_review':raise HTTPException(404)
    with np.load(ROOT/'instances'/ident/'point.npz',allow_pickle=False) as p:return {'points':p['points_xyz'].round(5).tolist()}
class Review(BaseModel):
    decision:str
    note:str=''
@router.post('/api/hk-delonix-refined/{ident}/review')
def review(ident:str,value:Review):
    r=resolve(ident)
    if r['status']!='candidate_pending_review':raise HTTPException(400,'未生成可验收点云')
    if value.decision not in ('accept','reject','pending') or len(value.note)>2000:raise HTTPException(400,'无效审核值')
    with LOCK:
        path=ROOT/'reviews.json';data=read(path) if path.exists() else {}
        entry={**value.model_dump(),'species':'Delonix regia','version':ROOT.name,
               'updated_at':datetime.now(timezone.utc).isoformat(),'source':'explicit_refined_delonix_pair_review'}
        data[ident]=entry;tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(tmp,path)
    return {'saved':True,'review':entry}
