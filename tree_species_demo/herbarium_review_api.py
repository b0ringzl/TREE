"""Local, recoverable human review of the isolated herbarium collection.

Run standalone on 8038 to leave an active inference/demo process untouched.
SQLite decisions and append-only events are authoritative; source images and
assistant manifests are never modified. One row version prevents stale writes.
"""
import json
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from datetime import datetime, timezone
from typing import Literal

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get('TREE_HERBARIUM_REVIEW_DATA', str(ROOT.parent / 'external_datasets/herbarium_training_candidates_20260907')))
router = APIRouter()


def records():
    return json.loads((DATA / 'curated_manifest.json').read_text(encoding='utf-8'))


@contextmanager
def connect():
    db = sqlite3.connect(DATA / 'human_review.sqlite3', timeout=10)
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA synchronous=FULL')
    db.execute('CREATE TABLE IF NOT EXISTS decisions (id TEXT PRIMARY KEY, sha256 TEXT NOT NULL, status TEXT NOT NULL, revision INTEGER NOT NULL, updated_at TEXT NOT NULL)')
    db.execute('CREATE TABLE IF NOT EXISTS events (seq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT NOT NULL, sha256 TEXT NOT NULL, previous_status TEXT NOT NULL, status TEXT NOT NULL, revision INTEGER NOT NULL, updated_at TEXT NOT NULL)')
    db.commit()
    try:
        with db:
            yield db
    finally:
        db.close()


def find(ident):
    row = next((r for r in records() if r['id'] == ident), None)
    if row is None:
        raise HTTPException(404, 'Image not found / 找不到影像')
    return row


def reviewed_rows():
    with connect() as db:
        decisions = {r['id']: dict(r) for r in db.execute('SELECT * FROM decisions')}
    result = []
    for r in records():
        d = decisions.get(r['id'])
        if d and d['sha256'] != r['sha256']:
            raise HTTPException(409, 'Source changed; review requires migration / 來源已變更，需核對驗收記錄')
        result.append({**r, 'human_review': d or {'status': 'pending', 'revision': 0},
                       'image_url_local': '/api/herbarium-review/image/' + r['id']})
    return result


@router.get('/herbarium-review')
def page():
    return FileResponse(ROOT / 'static/herbarium-review.html', headers={'Cache-Control': 'no-store'})


@router.get('/herbarium-review.js')
def script():
    return FileResponse(ROOT / 'static/herbarium-review.js', media_type='text/javascript', headers={'Cache-Control': 'no-store'})


@router.get('/api/herbarium-review')
def listing():
    rows = reviewed_rows()
    return JSONResponse({'rows': rows, 'counts': {s: sum(r['human_review']['status'] == s for r in rows) for s in ('pending', 'accepted', 'rejected')}}, headers={'Cache-Control': 'no-store'})


@router.get('/api/herbarium-review/image/{ident}')
def image_file(ident: str):
    row = find(ident)
    p = (DATA / row['file']).resolve()
    if not p.is_relative_to((DATA / 'images').resolve()) or not p.is_file():
        raise HTTPException(404)
    return FileResponse(p)


class Decision(BaseModel):
    status: Literal['pending', 'accepted', 'rejected']
    expected_revision: int = Field(ge=0)


@router.post('/api/herbarium-review/decision/{ident}')
def decide(ident: str, action: Decision, request: Request):
    # JSON + origin check keeps an unrelated website from changing local labels.
    origin = request.headers.get('origin')
    if origin and origin.rstrip('/') != str(request.base_url).rstrip('/'):
        raise HTTPException(403, 'Cross-origin write denied / 不允許跨來源寫入')
    row = find(ident)
    with connect() as db:
        db.execute('BEGIN IMMEDIATE')
        old = db.execute('SELECT * FROM decisions WHERE id=?', (ident,)).fetchone()
        revision = old['revision'] if old else 0
        previous = old['status'] if old else 'pending'
        if (old and old['sha256'] != row['sha256']) or revision != action.expected_revision:
            raise HTTPException(409, 'Review changed in another tab. Reload before retrying. / 其他分頁已修改，請重新載入後再試。')
        now = datetime.now(timezone.utc).isoformat()
        version = revision + 1
        db.execute('INSERT INTO decisions VALUES (?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET sha256=excluded.sha256,status=excluded.status,revision=excluded.revision,updated_at=excluded.updated_at', (ident, row['sha256'], action.status, version, now))
        db.execute('INSERT INTO events(id,sha256,previous_status,status,revision,updated_at) VALUES (?,?,?,?,?,?)', (ident, row['sha256'], previous, action.status, version, now))
    return {'id': ident, 'status': action.status, 'previous_status': previous, 'revision': version, 'updated_at': now, 'original_preserved': True}


@router.get('/api/herbarium-review/export')
def export():
    rows = reviewed_rows()
    accepted = []
    for r in rows:
        if r['human_review']['status'] != 'accepted':
            continue
        accepted.append({**r, 'review_status': 'human_accepted_for_image_quality',
                         'training_eligible': False, 'training_gates': ['intended_use_and_training_rights_confirmation', 'experiment_manifest_acceptance']})
    return JSONResponse({'accepted_count': len(accepted), 'accepted': accepted,
                         'decisions': [{'id': r['id'], **r['human_review']} for r in rows],
                         'note': 'Human image-quality acceptance is not training-rights authorization.'},
                        headers={'Content-Disposition': 'attachment; filename="herbarium-human-review.json"', 'Cache-Control': 'no-store'})


app = FastAPI(title='Image review / 影像驗收')
app.include_router(router)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8038)
