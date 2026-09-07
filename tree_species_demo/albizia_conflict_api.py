"""Explicit adjudication sidecar; original human reviews and assets stay read-only."""
import hashlib,json,os,threading
from datetime import datetime,timezone
from pathlib import Path
from typing import Literal
from fastapi import APIRouter,HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel,Field

ROOT=Path(__file__).resolve().parent.parent/'hk-tree-vmms/derived/albizia_conflict_review_v1_20260907'
SOURCE=ROOT.parent/'hk_dualsource_five_species_20260907/reviews.json'
router=APIRouter();LOCK=threading.Lock()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def manifest():return read(ROOT/'manifest.json')
def latest():
    history=read(ROOT/'adjudications.json') if (ROOT/'adjudications.json').exists() else []
    entry=history[-1] if history else None
    return entry if entry and entry.get('source_review_sha256')==manifest()['source_review_sha256'] else None
def check_source():
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest()!=manifest()['source_review_sha256']:
        raise HTTPException(409,'原验收已变更，请先重新生成冲突证据，避免覆盖更新后的判断')

@router.get('/hk-label-conflicts')
def page():return FileResponse(Path(__file__).resolve().parent/'static/hk-label-conflicts.html')
@router.get('/api/hk-label-conflicts')
def listing():
    data=manifest();data['adjudication']=latest()
    data['source_changed']=hashlib.sha256(SOURCE.read_bytes()).hexdigest()!=data['source_review_sha256']
    return data
@router.get('/hk-label-conflict-file/{ident}/{name}')
def file(ident:str,name:str):
    if ident not in {x['id'] for x in manifest()['items']} or name not in {'native_crop.jpg','panorama.jpg','label.jpg','cross_projection.jpg'}:raise HTTPException(404)
    p=ROOT/ident/name
    if not p.is_file():raise HTTPException(404)
    return FileResponse(p)

class Decision(BaseModel):
    identity:Literal['same_tree','different_trees','mixed','uncertain']
    species:Literal['Albizia lebbeck','Leucaena leucocephala','other','uncertain']='uncertain'
    other_species:str=Field(default='',max_length=200)
    note:str=Field(default='',max_length=2000)

@router.post('/api/hk-label-conflicts/adjudicate')
def adjudicate(value:Decision):
    if value.species=='other' and not value.other_species.strip():raise HTTPException(400,'请填写其他树种，或选择树种待定')
    if value.species!='other' and value.other_species.strip():raise HTTPException(400,'仅其他树种允许填写学名')
    if value.identity!='same_tree' and value.species!='uncertain':raise HTTPException(400,'不同树或混提时不能统一树种，须重新分离后逐棵复核')
    with LOCK:
        check_source();p=ROOT/'adjudications.json';history=read(p) if p.exists() else []
        entry={**value.model_dump(),'source':'explicit_conflict_adjudication','updated_at':datetime.now(timezone.utc).isoformat(),
            'source_review_sha256':manifest()['source_review_sha256'],
            'status':'identity_and_species_confirmed' if value.identity=='same_tree' and value.species not in ('uncertain','other') else 'requires_followup',
            'training_started':False}
        history.append(entry);temp=p.with_suffix('.tmp');temp.write_text(json.dumps(history,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(temp,p)
    return {'saved':True,'adjudication':entry}

@router.get('/api/hk-label-conflicts/training-candidates')
def candidates():
    check_source();items=read(ROOT/'training_candidates.json');decision=latest()
    # Only explicit same-target AND resolved species can release the label conflict.
    if decision and decision['status']=='identity_and_species_confirmed':
        for item in manifest()['items']:
            if item['role']!='conflict':continue
            items.append({'id':item['id'],'species':decision['species'],'original_species':item['original_species'],
                'tree_group':item['tree_group'],'split_group':'adjudicated_same_tree_4276_4287','image':item.get('image'),'point':item.get('point'),
                'original_pair_decision':item['human_review']['decision'],'reason':'explicit_species_adjudication_requires_grouped_split',
                'training_eligible':False})
    return {'items':items,'count':len(items),'training_started':False,
        'policy':'Candidate export only. All views of each identity group must share one split. Original model/datasets are not changed.'}
