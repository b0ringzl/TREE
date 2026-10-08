"""Measured geometry and explicit review, not an unvalidated species classifier."""
import hashlib
import io
import json
import os
import threading
import zipfile
from datetime import datetime,timezone
from pathlib import Path

import numpy as np
from fastapi import APIRouter,HTTPException,File,UploadFile,Form,Request
from fastapi.responses import FileResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
from .species_names import bilingual_names as demo_names

def bilingual_names(species):
    # Evidence assets can contain taxa not in the image model's current catalog.
    local={'Leucaena leucocephala':'銀合歡','Albizia lebbeck':'大葉合歡'}
    if species in local:return dict(scientific_name=species,chinese_name=local[species],name=species+' / '+local[species])
    return demo_names(species)

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'outputs/hk_point_assist_20260911_v1'
STATIC=Path(__file__).resolve().parent/'static'
MAX_BYTES=64*1024*1024
LOCK=threading.Lock()
router=APIRouter()
STRESS=ROOT/'outputs/hk_point_robustness_20260911_v1'
COUNTERFACTUAL=ROOT/'outputs/hk_point_counterfactual_20260911_v1'
BANYAN=ROOT/'hk-tree-vmms/derived/banyan_coverage_display_v3_20260912'
def read(p,default=None):return json.loads(p.read_text(encoding='utf-8')) if p.is_file() else default
def rows():return read(DATA/'catalog.json',[])

def coverage(r):
    if r['species']!='Ficus microcarpa':return None
    item=next((v for v in read(BANYAN/'manifest.json',[]) if v['key']==r['key']),None)
    if item and item['original_geometry_sha256']==r['sha256']['cleaned']:return item
    return None

def coverage_links(r):
    item=coverage(r)
    if not item:return {}
    base='/api/point-assist/coverage/'+r['key']+'/'
    return dict(projection=base+'all',archived_projection='/api/point-assist/asset/'+r['key']+'/projection',
        coverage_views={k:base+k for k in item['asset_paths']},display_version=item['display_version'])
def record(key):
    r=next((r for r in rows() if r['key']==key),None)
    if not r:raise HTTPException(404,'Asset not found / 找不到資料')
    return r

def checked_npy(data):
    stream=io.BytesIO(data);version=np.lib.format.read_magic(stream)
    shape,fortran,dtype=(np.lib.format.read_array_header_1_0(stream) if version==(1,0) else np.lib.format.read_array_header_2_0(stream))
    if dtype.hasobject or np.prod(shape,dtype=object)*dtype.itemsize>MAX_BYTES:raise ValueError('Array too large or unsafe / 陣列過大或不安全')
    return np.load(io.BytesIO(data),allow_pickle=False)

def decode(data,name,units='unknown'):
    if not data or len(data)>MAX_BYTES:raise ValueError('Cloud limit: 64 MB / 點雲上限 64 MB')
    if units not in {'unknown','metres'}:raise ValueError('Invalid units / 單位無效')
    ext=Path(name).suffix.lower();normalized=False;restored=False
    if ext=='.npz':
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            if sum(i.file_size for i in z.infolist())>MAX_BYTES or len(z.infolist())>24:raise ValueError('Expanded archive too large / 解壓後資料過大')
            keys={Path(n).stem:n for n in z.namelist()}
            key=next((k for k in ['xyz','metric_xyz','metric','points_xyz','points'] if k in keys),None)
            if key is None:raise ValueError('NPZ requires XYZ array / NPZ 需要 XYZ 陣列')
            xyz=checked_npy(z.read(keys[key]));normalized=key=='points_xyz' and 'centroid_xyz' in keys and 'scale' in keys
            if normalized and units=='metres':
                center=checked_npy(z.read(keys['centroid_xyz']));scale=checked_npy(z.read(keys['scale']))
                if center.size!=3 or scale.size!=1 or not np.isfinite(center).all() or not np.isfinite(scale).all() or float(scale.reshape(-1)[0])<=0:raise ValueError('Invalid normalization metadata / 正規化資訊無效')
                if xyz.ndim!=2 or xyz.shape[1]<3:raise ValueError('Expected N × 3 / 需要 N × 3 陣列')
                xyz=xyz[:,:3]*float(scale.reshape(-1)[0])+center.reshape(1,3);restored=True
    elif ext=='.npy':xyz=checked_npy(data)
    elif ext in {'.xyz','.txt','.csv'}:xyz=np.loadtxt(io.BytesIO(data),delimiter=',' if ext=='.csv' else None,max_rows=2_000_001)
    elif ext in {'.las','.laz'}:
        import laspy
        with laspy.open(io.BytesIO(data)) as f:
            if f.header.point_count>2_000_000:raise ValueError('Extract one tree, at most 2 million points / 請提取單株，最多二百萬點')
            p=f.read();xyz=np.column_stack([p.x,p.y,p.z])
    else:raise ValueError('Use NPZ / NPY / XYZ / TXT / CSV / LAS / LAZ')
    if xyz.ndim!=2 or xyz.shape[1]<3 or not 32<=len(xyz)<=2_000_000:raise ValueError('Use N × 3 XYZ, 32–2,000,000 points / 請提供 32 至二百萬個 XYZ 點')
    xyz=np.asarray(xyz[:,:3],dtype=np.float64)
    if not np.isfinite(xyz).all():raise ValueError('Nonfinite coordinates / 座標包含無效值')
    return xyz,dict(units=units,normalized_archive=normalized,metric_restored_from_declared_metadata=restored)

def analyze(xyz,units='unknown',z_up=False,provenance='user_declared'):
    xyz=np.asarray(xyz,dtype=np.float64);unique=np.unique(xyz,axis=0)
    if len(unique)<2:raise ValueError('Coordinates coincide / 座標全部重合')
    q=np.quantile(unique,[.005,.5,.995],axis=0);span=q[2]-q[0]
    eig=np.linalg.eigvalsh(np.cov(unique.T));fraction=float(eig[0]/max(eig.sum(),1e-12))
    warnings=[]
    if len(unique)<256:warnings.append('Sparse independent points / 獨立點過少')
    if len(unique)/len(xyz)<.6:warnings.append('Many duplicate points / 重複點偏多')
    if fraction<.01:warnings.append('Nearly planar / linear structure; check background or missing crown / 結構接近平面或細線，請檢查背景及缺失樹冠')
    if units!='metres':warnings.append('Metric scale unconfirmed; no metre measurements / 未確認公制尺度，不提供米制量測')
    if not z_up:warnings.append('Vertical axis unconfirmed / 尚未確認垂直軸')
    warnings.append('Root, crown completeness and same-tree alignment need verification / 根部、樹冠完整性與同株對齊仍需核驗')
    metric=units=='metres' and z_up
    hist=np.histogram(unique[:,2],bins=np.linspace(q[0,2],q[2,2]+1e-12,21))[0] if z_up else None
    idx=np.linspace(0,len(unique)-1,min(2048,len(unique)),dtype=int);preview=unique[idx]-q[1]
    scale=max(float(np.linalg.norm(preview,axis=1).max()),1e-9);preview/=scale
    # Width of observed upper points, not a certified crown boundary.
    upper=unique[unique[:,2]>=np.quantile(unique[:,2],.65)]
    width=np.ptp(np.quantile(upper[:,:2],[.05,.95],axis=0),axis=0)
    return dict(version='point-evidence-v1',point_count=len(xyz),unique_count=len(unique),units=units,
        provenance=provenance,z_up_confirmed=z_up,robust_extent_input_units=span.tolist(),
        observed_vertical_span_m=float(span[2]) if metric else None,
        observed_upper_width_m=width.tolist() if metric else None,
        observed_height_width_ratio=float(span[2]/max(width.max(),1e-9)) if z_up else None,
        height_profile=hist.tolist() if hist is not None else [],
        eigen_min_fraction=fraction,preview_points=preview.round(5).tolist(),warnings=warnings,
        total_tree_height_m=None,total_height_verified=False,
        support_state='review_required' if len(unique)>=256 and fraction>=.01 and metric else 'insufficient_evidence',
        model_scores_changed=False,fusion_used=False,validated_accuracy_gain=False,
        reason='Geometry is measurable evidence, not a species decision or proof of accuracy gain. / 幾何量測是核驗依據，不等同樹種判定或準確率增益證明。')

def from_upload(data,name,units,z_up):
    xyz,meta=decode(data,name,units);out=analyze(xyz,units,z_up)
    out.update(input_sha256=hashlib.sha256(data).hexdigest(),input_name=Path(name).name,normalization=meta)
    return out

@router.get('/point-assist')
def page():return FileResponse(STATIC/'point-assist.html',headers={'Cache-Control':'no-store'})

@router.get('/point-stress')
def stress_page():return FileResponse(STATIC/'point-stress.html',headers={'Cache-Control':'no-store'})

@router.get('/api/point-stress')
def stress_results():
    result=read(COUNTERFACTUAL/'results.json',{})
    if not result:raise HTTPException(503,'Experiment not ready / 實驗尚未完成')
    names=[bilingual_names(n) for n in result['classes']]
    records=read(COUNTERFACTUAL/'predictions.json',[])
    items=[]
    for r in records:
        items.append({**r,'image':'/api/point-stress/image/'+r['condition']+'/'+r['id']})
    return dict(**result,names=names,records=items,uncertainty=read(COUNTERFACTUAL/'paired_uncertainty.json',{}),
        views=10,groups=7,recorded_inference=True,production_model_changed=False,
        scope='Five outputs; heldout labels cover three classes. Controlled synthetic degradation on historical regression, not a new natural-image test.')

@router.get('/api/point-stress/image/{condition}/{ident}')
def stress_image(condition:str,ident:str):
    row=next((r for r in read(STRESS/'predictions.json',[]) if r['id']==ident and r['condition']==condition),None)
    if row is None:raise HTTPException(404)
    path=(STRESS/row['image']).resolve()
    if not path.is_relative_to((STRESS/'images').resolve()) or not path.is_file():raise HTTPException(404)
    return FileResponse(path)

@router.get('/api/point-assist/catalog')
def catalog():
    summary=read(DATA/'summary.json',{});new=read(DATA/'geometry_reviews.json',{})
    items=[dict(key=r['key'],id=r['id'],**bilingual_names(r['species']),version=r['version'],
        review_decision=r['review_decision'],survey=r['survey'],tree_group=r['provisional_tree_group'],
        geometry_review=new.get(r['key']),image='/api/point-assist/asset/'+r['key']+'/image',
        projection='/api/point-assist/asset/'+r['key']+'/projection') for r in rows()]
    for item,r in zip(items,rows()):item.update(coverage_links(r))
    return dict(items=items,accepted_views=len(items),classes=[dict(**c,**bilingual_names(c['species'])) for c in summary.get('species',[])],
        validated_accuracy_gain=False,verified_total_heights=0)

def asset(r,kind):
    if kind not in {'image','point','cleaned','projection'}:raise HTTPException(404)
    path=Path(r['asset_paths'][kind]).resolve()
    if not path.is_relative_to((ROOT/'hk-tree-vmms/derived').resolve()) or not path.is_file():raise HTTPException(404)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=r['sha256'][kind]:raise HTTPException(409,'Asset changed: rebuild inventory / 資料已變更，請重新盤點')
    return path

@router.get('/api/point-assist/asset/{key}/{kind}')
def get_asset(key:str,kind:str):return FileResponse(asset(record(key),kind))

@router.get('/api/point-assist/coverage/{key}/{kind}')
def coverage_asset(key:str,kind:str):
    r=record(key);item=coverage(r)
    if not item or kind not in item['asset_paths']:raise HTTPException(404)
    # Fail closed if the original geometry or rendered asset changes.
    asset(r,'cleaned')
    path=Path(item['asset_paths'][kind]).resolve()
    if not path.is_relative_to(BANYAN.resolve()) or not path.is_file():raise HTTPException(404)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256'][kind]:raise HTTPException(409)
    return FileResponse(path)

@router.get('/api/point-assist/record/{key}')
def inspect(key:str):
    r=record(key);p=asset(r,'cleaned')
    with np.load(p,allow_pickle=False) as f:xyz=f['xyz']
    d=analyze(xyz,'metres',True,'versioned_VMMS_local_XYZ')
    cov=coverage(r)
    audit=None
    if cov:
        d.update(preview_points=cov['preview_points'],preview_lower_mask=cov['preview_lower_mask'],
            preview_sampling=cov['preview_sampling'],observed_full_extent_m=cov['full_extent_m'],
            display_version=cov['display_version'],preview_geometry_changed=False)
        audit={k:v for k,v in cov.items() if k not in {'preview_points','preview_lower_mask'}}
    return dict(evidence=d,review=r['review'],historical_ground=r['ground'],tracking=r['tracking'],
        coverage_scope=r['coverage_scope'],asset_version=r['version'],asset_hashes=r['sha256'],
        acquisition=r['survey'],provisional_tree_group=r['provisional_tree_group'],coverage_audit=audit)

@router.post('/api/point-assist/analyze')
async def upload(cloud:UploadFile=File(...),units:str=Form('unknown'),z_up:bool=Form(False)):
    try:data=await cloud.read(MAX_BYTES+1)
    finally:await cloud.close()
    try:return await run_in_threadpool(from_upload,data,cloud.filename or '',units,z_up)
    except Exception as e:
        if isinstance(e,(ValueError,OSError,KeyError,zipfile.BadZipFile,EOFError)):raise HTTPException(400,str(e))
        raise

class Review(BaseModel):
    same_tree:bool=False
    isolated_tree:bool=False
    base_visible:bool=False
    crown_complete:bool=False
    species_verified:bool=False
    corrected_species:str=''
    note:str=''

@router.post('/api/point-assist/review/{key}')
def save_review(key:str,value:Review,request:Request):
    if request.headers.get('X-Point-Review')!='1':raise HTTPException(403)
    origin=request.headers.get('origin')
    if origin and origin.rstrip('/')!=str(request.base_url).rstrip('/'):raise HTTPException(403)
    r=record(key)
    if len(value.note)>2000 or len(value.corrected_species)>150:raise HTTPException(400)
    for k in ['image','point','cleaned']:asset(r,k)
    entry=dict(**value.model_dump(),updated_at=datetime.now(timezone.utc).isoformat(),source='explicit_point_evidence_review_ui',
        asset_hashes=r['sha256'],asset_version=r['version'],original_species=r['species'],
        display_version=(coverage(r) or {}).get('display_version',r['version']),
        applies_to='This exact pair version only; not surveyed height truth or automatic training acceptance')
    with LOCK:
        path=DATA/'geometry_reviews.json';history=read(DATA/'geometry_review_history.json',[]);allreviews=read(path,{})
        history.append(dict(key=key,review=entry));allreviews[key]=entry
        for target,data in [(DATA/'geometry_review_history.json',history),(path,allreviews)]:
            tmp=target.with_suffix('.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(tmp,target)
    return {'saved':True,'review':entry}
