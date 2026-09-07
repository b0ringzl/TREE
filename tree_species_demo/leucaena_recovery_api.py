"""Versioned recovery review. No writes to accepted sources or models."""
import hashlib,json,os,threading
from datetime import datetime,timezone
from pathlib import Path
from typing import Literal
import numpy as np
from fastapi import APIRouter,HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel,Field
ROOT=Path(__file__).resolve().parent.parent/'hk-tree-vmms/derived/leucaena_recovery_v1_20260907'
router=APIRouter();LOCK=threading.Lock()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def records():return read(ROOT/'manifest.json')
def resolve(ident):
    item=next((r for r in records() if r['id']==ident),None)
    if not item:raise HTTPException(404)
    return item
def stale():return any(hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h for p,h in read(ROOT/'protected_hashes.json').items())
@router.get('/hk-leucaena-recovery')
def page():return FileResponse(Path(__file__).resolve().parent/'static/hk-leucaena-recovery.html')
@router.get('/api/hk-leucaena-recovery')
def listing():
    reviews=read(ROOT/'reviews.json') if (ROOT/'reviews.json').exists() else {};qa=read(ROOT/'agent_visual_qa.json')
    return {'items':[{**r,'review':reviews.get(r['id']),'agent_qa':qa.get(r['id'])} for r in records()],'source_changed':stale(),'summary':read(ROOT/'summary.json')}
@router.get('/hk-leucaena-recovery-file/{ident}/{name}')
def file(ident:str,name:str):
    resolve(ident)
    if name not in {'panorama.jpg','label.jpg','image.jpg','projection.jpg','point_views.jpg','before_views.jpg','point.npz','cleaned_local.npz','record.json'}:raise HTTPException(404)
    p=ROOT/'instances'/ident/name
    if not p.is_file():raise HTTPException(404)
    return FileResponse(p)
@router.get('/api/hk-leucaena-recovery/{ident}/preview')
def preview(ident:str):
    if resolve(ident)['status']!='candidate_pending_review':raise HTTPException(404)
    with np.load(ROOT/'instances'/ident/'point.npz',allow_pickle=False) as p:return {'points':p['points_xyz'].round(5).tolist()}
class Review(BaseModel):
    decision:Literal['accept','reject','pending']
    species:Literal['Leucaena leucocephala','Albizia lebbeck','Unknown']
    reference_identity:Literal['unrelated','same_adjudicated_tree','uncertain']='uncertain'
    note:str=Field(default='',max_length=2000)
@router.post('/api/hk-leucaena-recovery/{ident}/review')
def review(ident:str,value:Review):
    r=resolve(ident)
    if value.decision=='accept' and r['status']!='candidate_pending_review':raise HTTPException(400,'未生成点云，不能接受配对')
    if value.reference_identity=='same_adjudicated_tree' and not r.get('conflicting_reference_links'):raise HTTPException(400,'该目标没有关联的已裁决参考树')
    if value.reference_identity=='same_adjudicated_tree' and value.species!='Albizia lebbeck':raise HTTPException(400,'已确认参考树为大叶合欢；若仍不确定请保持待定')
    if value.decision=='accept' and (value.species=='Unknown' or (r.get('conflicting_reference_links') and value.reference_identity=='uncertain')):raise HTTPException(400,'接受前请确认树种及关联目标身份；也可保存待定')
    with LOCK:
        if stale():raise HTTPException(409,'原验收或同树裁决已变更，请先刷新证据版本')
        p=ROOT/'reviews.json';data=read(p) if p.exists() else {}
        entry={**value.model_dump(),'updated_at':datetime.now(timezone.utc).isoformat(),'version':ROOT.name,
            'source':'explicit_leucaena_recovery_review','original_species':r['species'],
            'split_group':'adjudicated_same_tree_4276_4287' if value.reference_identity=='same_adjudicated_tree' else None,
            'training_eligible':False,'coverage_scope':'partial_crown_not_complete_tree'}
        data[ident]=entry;temp=p.with_suffix('.tmp');temp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(temp,p)
    return {'saved':True,'review':entry}
