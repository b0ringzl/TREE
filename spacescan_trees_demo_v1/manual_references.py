"""Local, reversible user reference slots. Never feeds a training manifest."""
import hashlib
import io
import json
import sqlite3
import uuid
import warnings
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

from PIL import Image, ImageOps
from .species_names import bilingual_names

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT / 'external_datasets/spacescan_manual_references_v1'
MAX_BYTES = 20 * 1024 * 1024


def labels():
    return json.loads((PROJECT/'outputs/distillation/bioclip_probe_20260908_0227/classes.json').read_text(encoding='utf-8'))['labels'] + ['Wodyetia bifurcata']


@contextmanager
def connect():
    ROOT.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(ROOT/'references.sqlite3', timeout=15)
    db.row_factory = sqlite3.Row
    db.execute('CREATE TABLE IF NOT EXISTS refs (id TEXT PRIMARY KEY, species TEXT, slot INTEGER, active INTEGER, metadata TEXT)')
    db.execute('CREATE UNIQUE INDEX IF NOT EXISTS active_slot ON refs(species, slot) WHERE active=1')
    try:
        with db:
            yield db
    finally:
        db.close()


def rows(species=None, history=False):
    if not (ROOT/'references.sqlite3').exists():
        return []
    with connect() as db:
        query = 'SELECT * FROM refs WHERE 1=1'
        params = []
        if species is not None:
            query += ' AND species=?'; params.append(species)
        if not history:
            query += ' AND active=1'
        query += ' ORDER BY rowid DESC'
        return [{**json.loads(r['metadata']), 'active': bool(r['active'])} for r in db.execute(query, params)]


def save(data, species, slot, author, source_url, rights, confirmed, source_record=None):
    if species not in labels() or slot not in (1, 2):
        raise ValueError('Choose a catalog species and slot 1 or 2. / 請選擇清單樹種及第 1 或 2 張。')
    if not confirmed:
        raise ValueError('Confirm species and tree, flower or leaf visibility. / 請確認樹種及整株、花或葉片的可見性。')
    if rights not in {'own', 'permission', 'cc0', 'cc-by', 'cc-by-sa', 'local-only'}:
        raise ValueError('Select image rights. / 請選擇圖片權利狀態。')
    author = author.strip(); source_url = source_url.strip()
    if not author or len(author)>300 or len(source_url)>2000:
        raise ValueError('Enter the photographer or source name. / 請填寫攝影者或來源名稱。')
    if source_url and (urlsplit(source_url).scheme not in {'http', 'https'} or not urlsplit(source_url).netloc):
        raise ValueError('Source must be an HTTP(S) page URL. / 來源須為 HTTP(S) 網頁網址。')
    if rights != 'own' and not source_url:
        raise ValueError('Enter the original source page URL. / 請填寫原始來源網頁。')
    if not data or len(data)>MAX_BYTES:
        raise ValueError('Image limit: 20 MB. / 圖片上限 20 MB。')
    with warnings.catch_warnings():
        warnings.simplefilter('error', Image.DecompressionBombWarning)
        with Image.open(io.BytesIO(data)) as image:
            if image.format not in {'JPEG','PNG','WEBP'} or image.width*image.height>40_000_000:
                raise ValueError('Use JPG / PNG / WEBP up to 40 MP. / 請使用 JPG／PNG／WEBP，上限四千萬像素。')
            image.load()
            preview = ImageOps.exif_transpose(image).convert('RGB')
    if min(preview.size)<224:
        raise ValueError('Use at least 224 pixels on the short edge. / 請使用短邊至少 224 像素的圖片。')
    digest = hashlib.sha256(data).hexdigest()
    key = uuid.uuid4().hex
    folder = ROOT/'images'/key
    folder.mkdir(parents=True)
    (folder/'original.bin').write_bytes(data)  # Original bytes and watermarks retained.
    preview.thumbnail((1600,1600))
    preview.save(folder/'display.jpg', quality=94)
    license_names = {'own':'Own photograph / 自行拍攝', 'permission':'Permission recorded by user / 使用者確認授權', 'local-only':'Local reference only / 僅限本機參考', 'cc0':'CC0', 'cc-by':'CC BY', 'cc-by-sa':'CC BY-SA'}
    license_urls = {'cc0':'https://creativecommons.org/publicdomain/zero/1.0/', 'cc-by':'https://creativecommons.org/licenses/by/4.0/', 'cc-by-sa':'https://creativecommons.org/licenses/by-sa/4.0/'}
    row = {'id':key, 'species':species, 'slot':slot, **bilingual_names(species),
        'url':'/api/manual-reference/'+key, 'source':'User supplied / 使用者補充', 'source_url':source_url,
        'photo_page':source_url, 'attribution':author, 'author':author, 'rights':rights,
        'license':license_names[rights], 'license_url':license_urls.get(rights,''),
        'date':datetime.now(timezone.utc).isoformat(), 'sha256':digest,
        'reference_taxon':bilingual_names(species)['scientific_name'],
        'note':'User-confirmed reference; not the same individual. / 使用者確認的參考圖，並非同一株樹。',
        'framing':'whole_tree_or_flower_leaf_detail', 'training_eligible':False, 'redistribution_approved':False}
    if source_record:
        row.update({'pool_id':source_record['id'], 'reference_taxon':source_record['observed_taxon'],
            'attribution':source_record['attribution'], 'license_url':source_record['license_url'],
            'license':source_record['license'].upper(), 'source_url':source_record['source_url'],
            'photo_page':source_record['photo_page']})
    with connect() as db:
        db.execute('BEGIN IMMEDIATE')
        if any(r['sha256']==digest and r['slot']!=slot for r in rows(species)):
            raise ValueError('This image is already in the other slot. / 另一位置已有相同圖片。')
        db.execute('UPDATE refs SET active=0 WHERE species=? AND slot=?', (species,slot))
        db.execute('INSERT INTO refs VALUES (?,?,?,?,?)', (key,species,slot,1,json.dumps(row,ensure_ascii=False)))
    return row


def set_active(key, active):
    with connect() as db:
        db.execute('BEGIN IMMEDIATE')
        r = db.execute('SELECT species,slot FROM refs WHERE id=?',(key,)).fetchone()
        if r is None:
            raise KeyError(key)
        if active:
            db.execute('UPDATE refs SET active=0 WHERE species=? AND slot=?',(r['species'],r['slot']))
        db.execute('UPDATE refs SET active=? WHERE id=?',(int(active),key))


def asset_path(key):
    if len(key)!=32 or any(c not in '0123456789abcdef' for c in key):
        raise KeyError(key)
    if not any(r['id']==key for r in rows(history=True)):
        raise KeyError(key)
    path = ROOT/'images'/key/'display.jpg'
    if not path.is_file():
        raise KeyError(key)
    return path
