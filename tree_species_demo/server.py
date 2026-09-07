import json
import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from engine import Engine,ROOT
from hk_pairs_api import router as pairs_router,experiment_samples,resolve as resolve_pair
from hk_five_api import router as five_router,catalog as five_catalog,samples as five_samples,resolve as resolve_five_sample
from demo_locale import attach,text,catalog_ui,species_name,split_name

app=FastAPI(title='Tree Atlas / 樹識 · Multimodal Demo / 多模態演示')
app.include_router(pairs_router)
app.include_router(five_router)
engine=Engine()
MAX_BYTES=80*1024*1024

@app.get('/')
def home(): return FileResponse(ROOT/'static/demo.html',headers={'Cache-Control':'no-cache'})

@app.get('/demo-guide')
def demo_guide(): return FileResponse(ROOT/'Demo_Guide_EN_zh-Hant.md',filename='Demo_Guide_EN_zh-Hant.md')

@app.get('/api/ui/catalog')
def ui_catalog(): return catalog_ui(catalog())

@app.get('/api/ui/samples')
def ui_samples():
    return [{**s,'display_name':f"Example {i+1:02d} / 示例 {i+1:02d} · {species_name(s['truth'])} · {split_name(s['split'])}"} for i,s in enumerate(samples())]

@app.get('/hk-pairs')
def hk_pairs():return FileResponse(ROOT/'static/hk-pairs.html')

@app.get('/api/catalog')
def catalog():
    from hk_three_species import NAMES,CHINESE,MODEL
    return {**engine.catalog,'hongkong_vmms_five':five_catalog(),'hongkong_vmms':[{'scientific_name':n,'chinese_name':c,'domain':'hongkong_vmms','modalities':['image','point','fusion'],'status':'experimental_weights_available' if MODEL.exists() else 'preparing','validation_scope':'not human-accepted accuracy'} for n,c in zip(NAMES,CHINESE)]}

@app.get('/api/samples')
def samples():
    return json.loads((ROOT/'samples.json').read_text(encoding='utf-8'))+experiment_samples()+five_samples()

@app.get('/api/health')
def health(): return {'status':'ok','device':engine.device,'loaded_models':list(engine.models)}

async def content(upload):
    if upload is None: return None
    data=await upload.read(MAX_BYTES+1)
    if len(data)>MAX_BYTES: raise HTTPException(413,text('单文件限80MB'))
    return data

@app.post('/api/predict')
async def predict(domain:str=Form(...),paired:bool=Form(False),explain:bool=Form(True),image:UploadFile|None=File(None),point:UploadFile|None=File(None)):
    image_bytes=await content(image); point_bytes=await content(point)
    try:
        return attach(await run_in_threadpool(engine.predict,domain,image_bytes,point_bytes,point.filename if point else '',paired,explain))
    except (ValueError,OSError,KeyError) as e: raise HTTPException(400,text(str(e)))

@app.post('/api/sample/{sample_id}')
async def predict_sample(sample_id:str,mode:str=Form('both')):
    sample=next((s for s in samples() if s['id']==sample_id),None)
    if sample is None: raise HTTPException(404,text('样本不存在'))
    if mode not in ('both','image','point'): raise HTTPException(400,text('无效模式'))
    source=resolve_five_sample(sample_id) if sample['domain']=='hongkong_vmms_five' else resolve_pair(sample['pair_id']) if sample['domain']=='hongkong_vmms' else sample
    local_pair=sample['domain'] in ('hongkong_vmms','hongkong_vmms_five')
    image_path=Path(source['image']) if local_pair else ROOT/source['image']
    point_path=Path(source['point']) if local_pair else ROOT/source.get('point','')
    image=image_path.read_bytes() if source.get('image') and mode!='point' else None
    point=point_path.read_bytes() if source.get('point') and mode!='image' else None
    try:
        result=await run_in_threadpool(engine.predict,sample['domain'],image,point,'point.npz',True,True)
        # Truth is attached only AFTER inference; never used by the engine.
        result['reference']={'truth':sample['truth'],'split':sample['split'],'sample_id':sample_id,'correct':result['decision']['label']==sample['truth'],'quality_caution':sample.get('quality_caution',False)}
        if sample.get('quality_caution'):
            result['warnings'].append('该类未找到足够严格质检通过样本：本例为质量相对较好的备选，需人工检查完整性和可见范围。')
        return attach(result)
    except (ValueError,OSError,KeyError) as e: raise HTTPException(400,text(str(e)))

@app.get('/download/{name}')
def download(name:str):
    allowed={'catalog.json','samples.json','树种列表与测试说明.md','测试数据集.zip'}
    if name not in allowed: raise HTTPException(404)
    return FileResponse(ROOT/name,filename=name)

app.mount('/test_data',StaticFiles(directory=ROOT/'test_data'),name='test_data')
app.mount('/static',StaticFiles(directory=ROOT/'static'),name='static')

if __name__=='__main__':
    import uvicorn
    uvicorn.run(app,host='127.0.0.1',port=8037)
