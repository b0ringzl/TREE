"""Scene-level Delonix verification; separate species and point alignment decisions."""
import json,os,threading
from datetime import datetime,timezone
from pathlib import Path
from fastapi import APIRouter,HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

ROOT=Path(__file__).resolve().parent.parent/'hk-tree-vmms/derived/delonix_scene_review_20260907'
router=APIRouter();LOCK=threading.Lock()
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def records():
    for n in ('manifest.json','manifest.progress.json'):
        if (ROOT/n).exists():return read(ROOT/n)
    return []
def resolve(ident):
    r=next((r for r in records() if r['id']==ident),None)
    if r is None:raise HTTPException(404,'场景不存在')
    return r

@router.get('/hk-delonix')
def page():return FileResponse(Path(__file__).resolve().parent/'static/hk-delonix.html')

@router.get('/api/hk-delonix')
def listing():
    reviews=read(ROOT/'scene_reviews.json') if (ROOT/'scene_reviews.json').exists() else {}
    summary=read(ROOT/'summary.json') if (ROOT/'summary.json').exists() else {'processing':True}
    return {'items':records(),'reviews':reviews,'summary':summary}

@router.get('/hk-delonix-file/{ident}/{name}')
def file(ident:str,name:str):
    resolve(ident)
    if name not in {'panorama.jpg','labels.jpg','scene_projection.jpg','target_projection.jpg','previous_cleaned_projection.jpg','record.json'}:raise HTTPException(404)
    path=ROOT/'scenes'/ident/name
    if not path.exists():raise HTTPException(404,'该视图未生成')
    return FileResponse(path)

class SceneReview(BaseModel):
    target_id:str
    identification:str
    alignment:str
    corrected_species:str=''
    note:str=''

@router.post('/api/hk-delonix/{ident}/review')
def review(ident:str,value:SceneReview):
    r=resolve(ident)
    if value.target_id not in {t['id'] for t in r['targets']}:raise HTTPException(400,'目标不属于该场景')
    if value.identification not in {'delonix','not_delonix','uncertain'} or value.alignment not in {'aligned','offset','background','incomplete','uncertain'}:raise HTTPException(400,'无效审核值')
    if value.alignment=='aligned' and r.get('status')!='ready':raise HTTPException(400,'未生成投影，不能确认对应')
    if len(value.note)>2000 or len(value.corrected_species)>150:raise HTTPException(400,'文本过长')
    if value.identification!='not_delonix' and value.corrected_species.strip():raise HTTPException(400,'只有非凤凰木判断可填写替代树种')
    with LOCK:
        path=ROOT/'scene_reviews.json';data=read(path) if path.exists() else {}
        entry={**value.model_dump(),'scene_id':ident,'version':ROOT.name,
               'updated_at':datetime.now(timezone.utc).isoformat(),'source':'explicit_delonix_scene_review_ui'}
        data[value.target_id]=entry;tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(tmp,path)
    return {'saved':True,'review':entry}
