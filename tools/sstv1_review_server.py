"""Separate SSTv1 review service; never changes the production inference app."""
import os
import sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

ROOT=Path(__file__).resolve().parents[1]
DATA=Path(os.environ['SSTV1_REVIEW_DATA']).resolve()
os.environ['TREE_HERBARIUM_REVIEW_DATA']=str(DATA)
sys.path.insert(0,str(ROOT/'tree_species_demo'))
import herbarium_review_api

app=FastAPI(title='SSTv1 supplemental image review')
@app.get('/')
def page():
    html=(DATA/'review.html').read_text(encoding='utf-8')
    html=html.replace('Image review / 影像驗收','SSTv1 · Supplemental images / 新增影像核驗')
    return HTMLResponse(html,headers={'Cache-Control':'no-store'})
app.include_router(herbarium_review_api.router)
