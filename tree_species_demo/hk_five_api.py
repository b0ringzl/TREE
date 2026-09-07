"""Read-only demo catalog and deterministic, non-cherry-picked pilot samples."""
import json
from pathlib import Path
from fastapi import APIRouter,HTTPException
from fastapi.responses import FileResponse
from hk_five_species import DATA,MODEL,NAMES,CHINESE,DOMAIN,UNVALIDATED
router=APIRouter()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def catalog():
    plan=read(DATA/'plan.json')
    return [{'scientific_name':n,'chinese_name':c,'domain':DOMAIN,'modalities':['image','point','fusion'],
       'status':'experimental_weights_available' if MODEL.exists() else 'unavailable','accepted_views':plan['accepted_by_class'].get(n,0),
       'validation_scope':'no_independent_class_evaluation' if n in UNVALIDATED else 'small_selected_holdout_only'} for n,c in zip(NAMES,CHINESE)]
def selected():
    rows=read(DATA/'split.json');result=[]
    for species in NAMES:
        pool=[r for r in rows if r['species']==species and r['split']=='heldout']
        if not pool:pool=[r for r in rows if r['species']==species and r['split']=='train']
        if pool:result.append(min(pool,key=lambda r:r['id']))
    return result
def resolve(sample_id):
    r=next((r for r in selected() if 'hk5_'+r['id']==sample_id),None)
    if r is None:raise HTTPException(404,'五类测试样本不存在')
    return r
def samples():
    result=[]
    for r in selected():
        ident='hk5_'+r['id'];split=r['split'];b='hk-five-sample-file/'+ident+'/'
        result.append({'id':ident,'domain':DOMAIN,'pair_id':r['id'],'truth':r['species'],
          'image':b+'image.jpg','point':b+'point.npz','split':'pilot_heldout' if split=='heldout' else 'training_example_not_test',
          'purpose':'小留出示例，非独立现场测试' if split=='heldout' else '训练示例，仅验证软件流程，不能据此计算精度',
          'selection_policy':'First ID per class, prefer holdout; never selected using prediction correctness.'})
    return result
@router.get('/hk-five-sample-file/{sample_id}/{name}')
def file(sample_id:str,name:str):
    r=resolve(sample_id)
    if name not in {'image.jpg','point.npz','projection.jpg','point_views.jpg'}:raise HTTPException(404)
    p=Path(r['image'] if name=='image.jpg' else r['point'] if name=='point.npz' else Path(r['asset_dir'])/name)
    if not p.is_file():raise HTTPException(404)
    return FileResponse(p)
@router.get('/api/hk-five-demo-info')
def info():return {'classes':catalog(),'samples':samples(),'report':read(DATA/'report.json'),'deployment':'opt_in_research_demo'}
@router.get('/hk-five-report')
def report():return FileResponse(DATA/'五类实验训练报告.md',filename='五类实验训练报告.md')
