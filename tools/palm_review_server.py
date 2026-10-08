"""Reuse tested review operations in a separate process and separate SQLite DB."""
import os
import json
import sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import FileResponse

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'external_datasets/palm_confusion_20260908'
os.environ['TREE_HERBARIUM_REVIEW_DATA'] = str(DATA)
sys.path.insert(0, str(ROOT / 'tree_species_demo'))
import herbarium_review_api


def novel_records():
    # Preserve all originals/manifests; only remove exact duplicate candidates
    # from this review queue. No acceptance or rejection decision is fabricated.
    rows = json.loads((DATA / 'curated_manifest.json').read_text(encoding='utf-8'))
    return [row for row in rows if not row['exact_duplicate_matches']]


herbarium_review_api.records = novel_records

app = FastAPI(title='Palm image review')


@app.get('/')
@app.get('/herbarium-review')
def page():
    return FileResponse(DATA / 'review.html', headers={'Cache-Control': 'no-store'})


app.include_router(herbarium_review_api.router)
