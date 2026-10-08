"""Local UI prototype; only forwards classification to the existing BioCLIP app."""
from pathlib import Path
import io
import warnings
import requests
from PIL import Image, ImageOps, UnidentifiedImageError
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

ROOT=Path(__file__).resolve().parents[1]/'outputs/tree_target_demo_v1'
app=FastAPI(title='Tree Atlas selection preview')


def prepare_selected_tree(data):
    """Keep the complete selected rectangle; never crop or stretch its contents."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error',Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as source:
                if source.format not in ('JPEG','PNG','WEBP') or source.width*source.height>40_000_000:
                    raise ValueError('Unsupported or oversized image')
                image=ImageOps.exif_transpose(source).convert('RGB')
        width,height=image.size
        edge=min(1024,max(width,height))
        square=ImageOps.pad(image,(edge,edge),method=Image.Resampling.BICUBIC,color=(128,128,128),centering=(.5,.5))
        encoded=io.BytesIO();square.save(encoded,format='JPEG',quality=95)
        return encoded.getvalue(),{'policy':'selected_tree_letterbox','original_width':width,'original_height':height,
            'forwarded_width':edge,'forwarded_height':edge,'retained_fraction':1.0}
    except (ValueError,OSError,UnidentifiedImageError,Image.DecompressionBombError,Image.DecompressionBombWarning) as exc:
        raise HTTPException(400,'Invalid image; use JPG/PNG/WEBP up to 40 million pixels. / 影像無效；請使用不超過四千萬像素的 JPG／PNG／WEBP。') from exc

@app.get('/')
def home():return FileResponse(ROOT/'selection.html',headers={'Cache-Control':'no-store'})

@app.get('/api/health')
def health():
    try:
        r=requests.get('http://127.0.0.1:8039/api/health',timeout=5)
        r.raise_for_status();return r.json()
    except requests.RequestException:
        return {'status':'error'}

@app.post('/api/predict')
async def predict(image:UploadFile=File(...),mode:str=Form('adapted'),selected_crop:bool=Form(False)):
    data=await image.read(20*1024*1024+1);await image.close()
    if len(data)>20*1024*1024:raise HTTPException(400,'Maximum 20 MB / 上限 20 MB')
    def forward():
        payload,preprocessing=prepare_selected_tree(data) if selected_crop else (data,None)
        try:
            r=requests.post('http://127.0.0.1:8039/api/predict',files={'image':('crop.jpg',payload,'image/jpeg')},data={'mode':mode},timeout=90)
            result=r.json()
            if r.status_code!=200:raise HTTPException(r.status_code,result.get('detail','Inference unavailable / 推理暫不可用'))
            if preprocessing:
                result['input_preprocessing']=preprocessing
                result['quality']['original_width']=preprocessing['original_width']
                result['quality']['original_height']=preprocessing['original_height']
                if min(preprocessing['original_width'],preprocessing['original_height'])<224:
                    result['quality']['warnings'].append('The original crop is below 224 pixels on its shorter side; padding does not restore detail. / 原始切片短邊不足 224 像素；補邊不會恢復細節。')
            return result
        except requests.RequestException as exc:
            raise HTTPException(503,'Start the teacher Demo on 8039. / 請啟動 8039 教師版服務。') from exc
    return await run_in_threadpool(forward)

app.mount('/assets',StaticFiles(directory=ROOT),name='assets')
