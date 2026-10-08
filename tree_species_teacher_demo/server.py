"""Separate local experimental Demo on 8039. Existing 8037 app is untouched."""
import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from tree_species_teacher_demo.engine import Engine, MAX_BYTES, display_name, CONTEXT_PIPELINE

ROOT = Path(__file__).resolve().parent
engine = Engine()


@asynccontextmanager
async def lifespan(app):
    async def warm():
        try:
            await run_in_threadpool(engine.load)
        except Exception:
            logging.exception('Teacher initialization failed')
    task = asyncio.create_task(warm())
    yield
    await task


app = FastAPI(title='Tree Atlas Teacher', lifespan=lifespan)


@app.get('/')
def index():
    return FileResponse(ROOT / 'static/index.html', headers={'Cache-Control': 'no-store'})


@app.get('/api/health')
def health():
    return {'status': engine.state, 'sstv1_status': engine.sstv1_state, 'sstv1_version': getattr(engine, 'sstv1_version', None), 'class_count': len(engine.labels), 'app': 'tree-atlas-teacher-v2-148', 'palm_correction_loaded': False, 'context_pipeline': CONTEXT_PIPELINE, 'flora_status': engine.flora_state, 'flora_available': (ROOT.parent / 'external_datasets/bioclip25_flora_20261008_v1/manifest.json').is_file()}


@app.post('/api/predict-flora')
async def predict_flora(image: UploadFile = File(...), detail: UploadFile = File(None)):
    try:
        data = await image.read(MAX_BYTES + 1)
        detail_data = await detail.read(MAX_BYTES + 1) if detail else None
    finally:
        await image.close()
        if detail:
            await detail.close()
    try:
        return await run_in_threadpool(engine.predict_flora, data, detail_data)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:
        logging.exception('Broad BioCLIP inference unavailable')
        raise HTTPException(503, 'Broad plant model unavailable; no fallback was substituted.') from exc


@app.get('/api/catalog')
def catalog():
    return [{'species': n, 'name': display_name(n)} for n in engine.labels]


@app.post('/api/predict')
async def predict(image: UploadFile = File(...), mode: str = Form('adapted')):
    data = await image.read(MAX_BYTES + 1)
    await image.close()
    try:
        return await run_in_threadpool(engine.predict, data, mode)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:
        logging.exception('Teacher inference unavailable')
        raise HTTPException(503, 'Model unavailable; check service status and retry. / 模型暫不可用；請檢查服務狀態後重試。') from exc


@app.post('/api/predict-context')
async def predict_context(image: UploadFile = File(...), detail: UploadFile = File(...)):
    try:
        whole_data = await image.read(MAX_BYTES + 1)
        original_data = await detail.read(MAX_BYTES + 1)
    finally:
        await image.close()
        await detail.close()
    try:
        return await run_in_threadpool(engine.predict_context, whole_data, original_data)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:
        logging.exception('Context inference unavailable')
        raise HTTPException(503, 'Context model unavailable; no silent single-view fallback. / 雙取景推理暫不可用；不會靜默改用單取景。') from exc


app.mount('/static', StaticFiles(directory=ROOT / 'static'), name='static')
