"""Loopback-only, separate review UI; leaves existing Demo/review DBs unchanged."""
import os
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'outputs/artocarpus_fresh_review_20260917_v1'
os.environ['TREE_HERBARIUM_REVIEW_DATA'] = str(DATA)
sys.path.insert(0, str(ROOT / 'tree_species_demo'))
import herbarium_review_api as review

app = FastAPI(title='White guimu review / 白桂木補圖核驗')


@app.get('/')
def index():
    return FileResponse(DATA / 'review.html', headers={'Cache-Control': 'no-store'})


@app.get('/herbarium-review.js')
def javascript():
    return FileResponse(DATA / 'review.js', media_type='application/javascript', headers={'Cache-Control': 'no-store'})


app.include_router(review.router)


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8053)
