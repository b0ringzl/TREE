"""Local audit endpoints: source labels/maps are never mutated."""
import json,os,threading
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from fastapi import APIRouter,HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

ROOT=Path(__file__).resolve().parent.parent/'hk-tree-vmms/derived/hk_dualsource_three_species_20260907'
router=APIRouter();LOCK=threading.Lock()
from delonix_scenes_api import router as delonix_router
router.include_router(delonix_router)
from delonix_refined_api import router as delonix_refined_router
router.include_router(delonix_refined_router)
from albizia_conflict_api import router as albizia_conflict_router
router.include_router(albizia_conflict_router)
from leucaena_recovery_api import router as leucaena_recovery_router
router.include_router(leucaena_recovery_router)
CLASSES=['榕树','Livistona chinensis 蒲葵','Wodyetia bifurcata 狐尾椰子']
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def records():
    for name in ('manifest.json','manifest.progress.json'):
        p=ROOT/name
        if p.exists():return read(p)
    return []
def resolve(ident):
    r=next((r for r in records() if r['id']==ident),None)
    if r is None:raise HTTPException(404,'实例不存在')
    return r

def experiment_samples():
    curated=ROOT/'experiment/demo_samples.json'
    if curated.exists():return read(curated)
    path=ROOT/'experiment/split.json'
    if not path.exists():return []
    selected=[];seen=set();counts={}
    for r in read(path):
        if r['split']!='heldout' or r['tree_group'] in seen:continue
        species=r['species']
        if counts.get(species,0)>=2:continue
        seen.add(r['tree_group']);counts[species]=counts.get(species,0)+1
        truth=['榕树','Livistona chinensis','Wodyetia bifurcata'][CLASSES.index(species)]
        selected.append({'id':'hk3_'+r['id'],'pair_id':r['id'],'domain':'hongkong_vmms','truth':truth,'image':f'hk-pair-file/{r["id"]}/image.jpg','point':f'hk-pair-file/{r["id"]}/point.npz','split':'experimental_group_holdout_weak_labels','purpose':'待验收配对示例，不是独立人工真值'})
    return selected

@router.get('/api/hk-pairs')
def pairs():
    data=records();review=read(ROOT/'pair_reviews.json') if (ROOT/'pair_reviews.json').exists() else {}
    metric_path=ROOT/'metric_height_v1/measurements.json'
    metrics=read(metric_path) if metric_path.exists() else {}
    repair_path=ROOT/'metric_height_v1/banyan_repairs.json'
    repairs={r['id']:r for r in read(repair_path)} if repair_path.exists() else {}
    items=[]
    for r in data:
        if r['status']!='geometry_candidate_pending_review':continue
        items.append({k:r.get(k) for k in ('id','frame_key','route','species','tree_group','group_label_conflict','quality')})
        items[-1]['review']=review.get(r['id'])
        items[-1]['metric_height']=metrics.get(r['id'])
        items[-1]['stem_repair']=repairs.get(r['id'])
    summary=read(ROOT/'summary.json') if (ROOT/'summary.json').exists() else {'processing':True,'processed':len(data)}
    report=read(ROOT/'experiment/report.json') if (ROOT/'experiment/report.json').exists() else None
    return {'items':items,'summary':summary,'experiment':report,'classes':CLASSES,'policy':'未复核候选不等同于已验收三维标签；审核独立保存，不改原全景标注。'}

@router.get('/hk-pair-download/{name}')
def package_download(name:str):
    if name not in {'香港三类_双源测试包.zip','香港三类双源交付说明.md','verification.json'}:raise HTTPException(404)
    path=ROOT/name
    if not path.exists():raise HTTPException(404,'文件仍在准备')
    return FileResponse(path,filename=name)

@router.get('/api/hk-pairs/{ident}/preview')
def preview(ident:str):
    r=resolve(ident)
    if not r.get('point'):raise HTTPException(404,'该候选无点云')
    with np.load(r['point'],allow_pickle=False) as f:xyz=f['points_xyz']
    return {'points':xyz.round(5).tolist(),'record':r}

@router.get('/hk-pair-file/{ident}/{name}')
def pair_file(ident:str,name:str):
    resolve(ident)
    if name not in {'image.jpg','projection.jpg','point_views.jpg','point.npz','cleaned_local.npz','record.json','height_views.jpg'}:raise HTTPException(404)
    path=ROOT/'metric_height_v1/previews'/(ident+'.jpg') if name=='height_views.jpg' else ROOT/'instances'/ident/name
    if not path.exists():raise HTTPException(404)
    return FileResponse(path)

@router.get('/hk-stem-repair/{ident}/{name}')
def stem_repair_file(ident:str,name:str):
    resolve(ident)
    if name not in {'before.jpg','after.jpg','projection.jpg','point.npz','cleaned_local.npz','report.json'}:raise HTTPException(404)
    path=ROOT/'metric_height_v1/banyan_repairs'/ident/name
    if not path.exists():raise HTTPException(404,'该实例无返工候选')
    return FileResponse(path)

class Review(BaseModel):
    decision:str
    species:str
    note:str=''

def pilot_root(version='pilot'):
    if version not in ('pilot','batch'):raise HTTPException(400,'无效版本')
    return ROOT/('banyan_tracking_batch_v2' if version=='batch' else 'banyan_tracking_pilot_v1')

def pilot_record(ident,version='pilot'):
    path=pilot_root(version)/'results.json'
    r=next((r for r in read(path) if r['id']==ident),None) if path.exists() else None
    if r is None:raise HTTPException(404,'返工样本不存在')
    return r

@router.get('/hk-stem-pilot')
def pilot_page():return FileResponse(Path(__file__).resolve().parent/'static/hk-stem-pilot.html')

@router.get('/api/hk-stem-pilot')
def pilot_list(version='pilot'):
    root=pilot_root(version)
    data=read(root/'results.json') if (root/'results.json').exists() else []
    qa=read(root/'agent_visual_qa.json') if (root/'agent_visual_qa.json').exists() else {}
    reviews=read(root/'reviews.json') if (root/'reviews.json').exists() else {}
    return {'items':[{**r,'agent_qa':qa.get(r['id']),'review':reviews.get(r['id'])} for r in data]}

@router.get('/hk-stem-pilot-file/{ident}/{name}')
def pilot_file(ident:str,name:str,version='pilot'):
    pilot_record(ident,version)
    if name not in {'panorama.jpg','before.jpg','after.jpg','projection.jpg','point.npz','cleaned_local.npz','report.json'}:raise HTTPException(404)
    path=pilot_root(version)/ident/name
    if not path.exists():raise HTTPException(404)
    return FileResponse(path)

@router.post('/api/hk-stem-pilot/{ident}/review')
def pilot_review(ident:str,value:Review,version='pilot'):
    r=pilot_record(ident,version)
    if r['status']!='candidate_requires_review':raise HTTPException(400,'未生成返工点云，不能验收')
    if value.decision not in ('accept','reject','pending') or value.species!='榕树' or len(value.note)>2000:raise HTTPException(400,'无效审核值')
    with LOCK:
        path=pilot_root(version)/'reviews.json';data=read(path) if path.exists() else {}
        entry={'decision':value.decision,'species':value.species,'note':value.note,'version':pilot_root(version).name,
               'updated_at':datetime.now(timezone.utc).isoformat(),'source':'explicit_pilot_review_ui'}
        data[ident]=entry;tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(tmp,path)
    return {'saved':True,'review':entry}


# Five-class expansion is deliberately isolated from all accepted three-class data.
EXPANSION_ROOT=ROOT.parent/'hk_dualsource_five_species_20260907'
EXPANSION_CLASSES=['Delonix regia','Aleurites moluccana','Leucaena leucocephala','Albizia lebbeck',
                   'Araucaria columnaris(Araucaria heterophylla)','Unknown']

def expansion_records():
    for name in ('manifest.json','manifest.progress.json'):
        if (EXPANSION_ROOT/name).exists():return read(EXPANSION_ROOT/name)
    return []

def expansion_resolve(ident):
    r=next((r for r in expansion_records() if r['id']==ident),None)
    if r is None:raise HTTPException(404,'扩展样本不存在')
    return r

@router.get('/hk-expansion')
def expansion_page():return FileResponse(Path(__file__).resolve().parent/'static/hk-expansion.html')

@router.get('/api/hk-expansion')
def expansion_list():
    reviews=read(EXPANSION_ROOT/'reviews.json') if (EXPANSION_ROOT/'reviews.json').exists() else {}
    qa=read(EXPANSION_ROOT/'agent_visual_qa.json') if (EXPANSION_ROOT/'agent_visual_qa.json').exists() else {}
    summary=read(EXPANSION_ROOT/'summary.json') if (EXPANSION_ROOT/'summary.json').exists() else {'processing':True}
    return {'items':[{**r,'review':reviews.get(r['id']),'agent_qa':qa.get(r['id'])} for r in expansion_records()],
            'summary':summary,'classes':EXPANSION_CLASSES}

@router.get('/hk-expansion-file/{ident}/{name}')
def expansion_file(ident:str,name:str):
    expansion_resolve(ident)
    if name not in {'image.jpg','projection.jpg','point_views.jpg','point.npz','cleaned_local.npz','record.json'}:raise HTTPException(404)
    path=EXPANSION_ROOT/'instances'/ident/name
    if not path.exists():raise HTTPException(404)
    return FileResponse(path)

@router.get('/api/hk-expansion/{ident}/preview')
def expansion_preview(ident:str):
    r=expansion_resolve(ident)
    if r['status']!='geometry_candidate_pending_review':raise HTTPException(404,'没有候选点云')
    path=EXPANSION_ROOT/'instances'/ident/'point.npz'
    if not path.exists():raise HTTPException(404)
    with np.load(path,allow_pickle=False) as f:points=f['points_xyz']
    return {'points':points.round(5).tolist()}

@router.post('/api/hk-expansion/{ident}/review')
def expansion_review(ident:str,value:Review):
    r=expansion_resolve(ident)
    if r['status']!='geometry_candidate_pending_review':raise HTTPException(400,'未通过几何检查的样本不能接受')
    if value.decision not in ('accept','reject','pending') or value.species not in EXPANSION_CLASSES or len(value.note)>2000:raise HTTPException(400,'无效审核值')
    with LOCK:
        path=EXPANSION_ROOT/'reviews.json';data=read(path) if path.exists() else {}
        entry={'decision':value.decision,'species':value.species,'note':value.note,
               'original_species':r.get('original_species',r['species']),'version':EXPANSION_ROOT.name,
               'updated_at':datetime.now(timezone.utc).isoformat(),'source':'explicit_expansion_review_ui'}
        data[ident]=entry;tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(tmp,path)
    return {'saved':True,'review':entry}

@router.get('/hk-stem-batch')
def batch_page():return FileResponse(Path(__file__).resolve().parent/'static/hk-stem-batch.html')

@router.get('/api/hk-stem-batch')
def batch_list():return pilot_list('batch')

@router.get('/hk-stem-batch-file/{ident}/{name}')
def batch_file(ident:str,name:str):return pilot_file(ident,name,'batch')

@router.post('/api/hk-stem-batch/{ident}/review')
def batch_review(ident:str,value:Review):return pilot_review(ident,value,'batch')

@router.post('/api/hk-pairs/{ident}/review')
def review(ident:str,value:Review):
    r=resolve(ident)
    if r['status']!='geometry_candidate_pending_review':raise HTTPException(400,'无法验收已拒绝的提取结果')
    if value.decision not in ('accept','reject','pending') or value.species not in CLASSES:raise HTTPException(400,'无效审核值')
    if len(value.note)>2000:raise HTTPException(400,'备注过长')
    path=ROOT/'pair_reviews.json'
    with LOCK:
        data=read(path) if path.exists() else {}
        entry={'decision':value.decision,'species':value.species,'note':value.note,'updated_at':datetime.now(timezone.utc).isoformat(),'source':'explicit_pair_review_ui','original_species':r['species']}
        data[ident]=entry;tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(tmp,path)
    return {'saved':True,'review':entry}
