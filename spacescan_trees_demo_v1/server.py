"""Local interface preview. Reuses the deployed teacher; never trains or writes weights."""
import io
import hashlib
import json
import math
import os
import re
import warnings
from pathlib import Path

import requests
from PIL import Image, ImageOps
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape
from starlette.concurrency import run_in_threadpool
from .species_names import bilingual_names
from . import reference_library
from . import manual_references
from . import tree_detection
from . import sam_assist
from . import point_assist
from . import point_recognition
from . import official_release
from . import presentation_groups
from . import video_pipeline
from . import vmms_corridor
from . import flora_scope

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
DATA = PROJECT / 'tree_species_demo'
UPSTREAM = os.getenv('SPACESCAN_TEACHER_URL', 'http://127.0.0.1:8039')
MAX_BYTES = 20 * 1024 * 1024
SAMPLES = json.loads((DATA / 'samples.json').read_text(encoding='utf-8'))
ASSETS = {str(i): row for i, row in enumerate(SAMPLES) if row.get('image')}
app = FastAPI(title='SpaceScan Trees')
app.include_router(point_assist.router)
VIDEO_JOBS = video_pipeline.VideoJobManager(ROOT / 'runtime' / 'video_jobs', tree_detection.WEIGHT)
CORRIDOR_JOBS = vmms_corridor.VmmsCorridorJobManager(ROOT / 'runtime' / 'corridor_jobs')
PAGES = Environment(loader=FileSystemLoader(ROOT / 'templates'),
                    autoescape=select_autoescape(['html']), undefined=StrictUndefined)


@app.exception_handler(Exception)
async def unexpected_error(request, exc):
    import logging
    logging.getLogger(__name__).error('Unhandled request error: %s', request.url.path,
                                     exc_info=(type(exc),exc,exc.__traceback__))
    return JSONResponse(status_code=500,content={'detail':'Local service error. Please retry; if it persists, check the service log. / 本地服務發生錯誤，請重試；若持續發生，請檢查服務日誌。'})


def asset_path(key):
    row = ASSETS.get(key)
    if not row:
        raise HTTPException(404)
    p = (DATA / row['image']).resolve()
    if not p.is_relative_to((DATA / 'test_data').resolve()) or not p.is_file():
        raise HTTPException(404)
    return p


def references(species, exclude_hash=None):
    # Never fall back to old organ-only test photos when the reviewed library lacks a class.
    return reference_library.references(species, exclude_hash)


def display_candidates(distribution, exclude_hash=None):
    rows = []
    for species, score in presentation_groups.top_grouped(distribution):
        names = bilingual_names(species)
        if species == presentation_groups.BAUHINIA_GROUP:
            members = sorted(presentation_groups.BAUHINIA_MEMBERS,
                             key=lambda member: -float(distribution[member]))
            breakdown = [dict(species=member, score=float(distribution[member]),
                              **bilingual_names(member),
                              common_name_en=('Hong Kong Orchid Tree' if member == 'Bauhinia x blakeana' else None))
                         for member in members]
            member_references = [reference for member in members
                                 for reference in references(member, exclude_hash)[:1]]
            rows.append(dict(species=species, score=score, **names, references=member_references,
                             group_members=breakdown, group_confirmed=False,
                             score_policy='Sum of three original image-class probabilities; not a species-level identification.'))
        else:
            rows.append(dict(species=species, score=score, **names,
                             references=references(species, exclude_hash)))
    return rows


def prepare(data):
    if not data or len(data) > MAX_BYTES:
        raise ValueError('Use an image up to 20 MB. / 請使用不超過 20 MB 的影像。')
    with warnings.catch_warnings():
        warnings.simplefilter('error', Image.DecompressionBombWarning)
        with Image.open(io.BytesIO(data)) as source:
            if source.format not in ('JPEG', 'PNG', 'WEBP') or source.width * source.height > 40_000_000:
                raise ValueError('Use JPG, PNG or WEBP, up to 40 MP. / 請使用 JPG、PNG 或 WEBP，上限四千萬像素。')
            im = ImageOps.exif_transpose(source).convert('RGB')
    width, height = im.size
    edge = min(1024, max(im.size))
    im = ImageOps.pad(im, (edge, edge), method=Image.Resampling.BICUBIC, color=(128, 128, 128))
    output = io.BytesIO()
    im.save(output, format='JPEG', quality=95)
    return output.getvalue(), width, height


def infer(data, scenario, mode='sstv1', view_policy='context', recognition_scope='hong_kong', check_scope=False):
    if recognition_scope not in flora_scope.SCOPES:
        raise HTTPException(400, 'Unsupported recognition scope')
    if mode not in {'sstv1','adapted','teacher'}:
        raise ValueError('Unsupported image model')
    if view_policy not in {'context','single'} or (mode != 'sstv1' and view_policy != 'context'):
        raise ValueError('Unsupported image view policy')
    try:
        pixels, width, height = prepare(data)
    except (ValueError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise HTTPException(400, 'Invalid image; use JPG / PNG / WEBP up to 20 MB and 40 MP. / 影像無效；上限 20 MB、四千萬像素。')
    if recognition_scope == 'broad_plants':
        try:
            broad = flora_scope.query(UPSTREAM, pixels, data, view_policy)
            return flora_scope.broad_result(broad, width, height, references, hashlib.sha256(data).hexdigest(),
                                           hashlib.sha256(data).hexdigest(), hashlib.sha256(pixels).hexdigest())
        except (requests.RequestException, ValueError, KeyError, TypeError):
            raise HTTPException(503, '世界模型暫不可用；沒有用香港模型代替。')
    try:
        files = {'image': ('image.jpg', pixels, 'image/jpeg')}
        if mode == 'sstv1' and view_policy == 'context':
            files['detail'] = ('target', data)
        endpoint = '/api/predict-context' if mode == 'sstv1' and view_policy == 'context' else '/api/predict'
        r = requests.post(UPSTREAM + endpoint, files=files, data={'mode': mode}, timeout=120)
        r.raise_for_status()
        raw = r.json()
        if mode == 'sstv1':
            if view_policy == 'context' and raw.get('inference_pipeline') != official_release.RELEASE['web_pipeline']:
                raise ValueError('Inference pipeline mismatch; no silent fallback')
            if view_policy == 'single' and raw.get('deployment') != official_release.RELEASE['classifier_version']:
                raise ValueError('Classifier version mismatch; no silent fallback')
        top = raw['comparison'][mode]
        if len(top) < 3 or any(not math.isfinite(float(x['score'])) or not 0 <= float(x['score']) <= 1 for x in top):
            raise ValueError('Malformed scores')
        distribution = raw[mode+'_distribution']
        digest = hashlib.sha256(data).hexdigest()
        rows = display_candidates(distribution, digest)
    except (requests.RequestException, ValueError, KeyError, TypeError):
        raise HTTPException(503, 'Teacher unavailable. Please retry after the local service is ready. / 教師模型暫不可用，請待本地服務就緒後重試。')
    notices = list(raw.get('quality', {}).get('warnings', []))
    comparison_review = None
    if mode == 'sstv1' and any('Dark input' in note for note in notices):
        notices.insert(0, 'Low-light evidence is insufficient for a confirmed species; the first ranked candidate may be wrong. / 低照度影像不足以確認樹種，首位候選可能有誤。')
        teacher_rows = raw.get('comparison', {}).get('teacher') or []
        teacher_distribution = raw.get('teacher_distribution')
        teacher_top = (presentation_groups.top_grouped(teacher_distribution)[0]
                       if teacher_distribution else None)
        if teacher_top and teacher_top[0] != rows[0]['species']:
            alternative_names = bilingual_names(teacher_top[0])
            comparison_review = {'model': 'BioCLIP 2.5 comparison',
                                 'species': teacher_top[0],
                                 'scientific_name': alternative_names['scientific_name'],
                                 'chinese_name': alternative_names['chinese_name'],
                                 'score': float(teacher_top[1])}
            notices.append('A comparison model ranks ' + alternative_names['scientific_name']
                           + ' first at ' + format(100 * comparison_review['score'], '.1f')
                           + '%. The models disagree; neither output is a confirmed identification. / 對照模型的首位候選為 '
                           + alternative_names['chinese_name'] + '；兩模型意見不一致，均非已確認鑑定。')
    if min(width, height) < 224:
        notices.append('Small native image: add a sharper close-up. / 原生影像偏小，請補充清晰近景。')
    if rows[0]['score'] < .5 or rows[0]['score'] - rows[1]['score'] < .15:
        notices.append('Add another view or a clear leaf / flower detail. / 請補充另一視角或清晰葉片／花部細節。')
    if any(row['species'] == presentation_groups.BAUHINIA_GROUP for row in rows):
        notices.append('Bauhinia group combines three image classes. Its score is their sum, not a confirmed Hong Kong Orchid Tree identification. / 洋紫荊為三類影像分數合併；並未確認為香港市花 Bauhinia × blakeana。')
    actual_version = raw.get('deployment', 'sstv1-version-unavailable') if mode == 'sstv1' else {'adapted':'bioclip25-mlp-3seed-148','teacher':'bioclip25-original-148'}[mode]
    actual_pipeline = ('sstv1-single-view-crop-v1' if mode == 'sstv1' and view_policy == 'single'
                       else raw.get('inference_pipeline'))
    result = {'brand': 'SpaceScan Trees', 'scenario': scenario, 'source': 'image',
            'model': {'sstv1':'SSTv1 · Adapted teacher / 增量適配教師','adapted':'BioCLIP 2.5 + MLP · Adapted teacher / 適配教師','teacher':'BioCLIP 2.5'}[mode], 'mode': mode, 'top3': rows,
            'image_distribution': raw.get(mode+'_distribution', {}),
            'image_display_distribution': presentation_groups.grouped_distribution(distribution),
            'raw_model_top3': top[:3], 'display_class_count': 146,
            'image_model_version': actual_version,
            'release_provenance': official_release.result_provenance(mode, actual_version, data, pixels,
                                                                     pipeline=actual_pipeline),
            'warnings': notices, 'width': width, 'height': height,
            'inference_seconds': raw.get('inference_seconds'), 'model_input': raw.get('model_input'),
            'model_input_detail': raw.get('model_input_detail'), 'view_evidence': raw.get('view_evidence'),
            'inference_pipeline': actual_pipeline,
            'preprocessing': ('One pass on the full selected crop, with no second center-detail vote; same H120 weights. / 僅對完整所選裁片推理一次，不再額外進行中心細節投票；H120 權重不變。' if mode == 'sstv1' and view_policy == 'single' else 'Whole selected image + center detail, equal probability mean; same H120 weights. / 完整所選影像＋中心細節，等權綜合；H120 權重不變。' if mode == 'sstv1' else 'Aspect-preserving padding; whole selected image retained. / 等比例補邊，保留完整所選影像。'),
            'single_view': mode == 'sstv1' and view_policy == 'single',
            'multiview_fusion': False, 'specialist_model': False, 'tree_detection': False,
            'comparison_review': comparison_review}
    if check_scope and mode == 'sstv1':
        result['recognition_scope'] = 'hong_kong'
        try:
            broad = flora_scope.query(UPSTREAM, pixels, data, view_policy)
            result['flora_check'] = {k: v for k, v in broad.items() if k not in {'model_input', 'model_input_detail', 'quality'}}
            result['scope_advisory'] = flora_scope.advisory(broad, rows[0], notices)
            result['flora_alternative'] = flora_scope.broad_result(broad, width, height, references, digest, digest,
                                                                   hashlib.sha256(pixels).hexdigest())
            result['inference_seconds'] = float(result['inference_seconds'] or 0) + broad['inference_seconds']
        except (requests.RequestException, ValueError, KeyError, TypeError):
            result['flora_check'] = {'available': False}
            result['scope_advisory'] = {'action': 'unavailable', 'message': '世界模型範圍檢查暫不可用；本次保留原 SST 結果。', 'auto_switched': False}
    return result


def demo_page(edition):
    # Presentation editions share the same inference APIs and pinned weights.
    client_id = os.getenv('SPACESCAN_ONEDRIVE_CLIENT_ID', '').strip()
    if not re.fullmatch(r'[0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}', client_id):
        client_id = ''
    dropbox_app_key = os.getenv('SPACESCAN_DROPBOX_APP_KEY', '').strip()
    if not re.fullmatch(r'[A-Za-z0-9_-]{6,128}', dropbox_app_key):
        dropbox_app_key = ''
    return HTMLResponse(PAGES.get_template('index.html').render(
        edition=edition, supports_cloud=edition == 'multimodal',
        onedrive_client_id=client_id if edition == 'image-only' else '',
        dropbox_app_key=dropbox_app_key if edition == 'image-only' else ''),
        headers={'Cache-Control': 'no-store'})


@app.get('/')
@app.get('/multimodal')
def index():
    return demo_page('multimodal')


@app.get('/image-only')
def image_only():
    return demo_page('image-only')


@app.get('/video')
def video_demo():
    return FileResponse(ROOT / 'static/video.html', headers={'Cache-Control': 'no-store'})


@app.get('/corridor')
def corridor_demo():
    return FileResponse(ROOT / 'static/corridor.html', headers={'Cache-Control': 'no-store'})


@app.get('/static/index.html', include_in_schema=False)
def legacy_index():
    return demo_page('multimodal')


@app.get('/apps')
def official_apps():
    return FileResponse(ROOT / 'static/apps.html', headers={'Cache-Control': 'no-store'})


@app.get('/api/release')
def release_info():
    current = health()
    return {**official_release.public_release(current['single_image_version']), 'ready': current['ready']}


@app.get('/api/official-download/{key}')
def official_download(key: str):
    try:
        path = official_release.download_path(key)
    except KeyError:
        raise HTTPException(404, 'No such official download / 找不到此正式檔案')
    return FileResponse(path, filename=path.name, headers={'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff'})


@app.get('/api/health')
def health():
    upstream = {}
    try:
        r = requests.get(UPSTREAM + '/api/health', timeout=3)
        r.raise_for_status()
        upstream = r.json()
        ready = upstream.get('status') == 'ready' and upstream.get('sstv1_status') == 'ready'
    except (requests.RequestException, ValueError):
        ready = False
    alignment = official_release.version_state(upstream.get('sstv1_version'))
    pipeline_aligned = upstream.get('context_pipeline') == official_release.RELEASE['web_pipeline']
    return {'ready': ready and alignment == 'aligned' and pipeline_aligned, 'release_id': official_release.RELEASE['release_id'],
            'inference_pipeline': upstream.get('context_pipeline'), 'pipeline_alignment': 'aligned' if pipeline_aligned else 'mismatch',
            'classifier_alignment': alignment, 'expected_single_image_version': official_release.RELEASE['classifier_version'],
            'fusion': point_recognition.available(), 'point_class_count': 7, 'point_evidence': True, 'multiview_fusion': False,
            'single_image_mode':'sstv1', 'single_image_model':'SSTv1', 'single_image_version':upstream.get('sstv1_version'), 'dual_image_mode':'teacher',
            'detector': tree_detection.WEIGHT.is_file() and tree_detection.CLIP_WEIGHT.is_file()
                        and (tree_detection.DINO_DIR/'model.safetensors').is_file(),
            'video_import': True, 'video_tracking': 'BoT-SORT + geometric fallback',
            'video_multiview': True, 'video_max_bytes': video_pipeline.MAX_VIDEO_BYTES,
            'video_max_seconds': video_pipeline.MAX_DURATION_SECONDS}


@app.get('/api/flora-scopes')
def flora_scopes():
    path = PROJECT / 'external_datasets/bioclip25_flora_20261008_v1/manifest.json'
    manifest = json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}
    return {'default': 'hong_kong', 'automatic_switch': False,
            'scopes': [{'id': 'hong_kong', 'name': '香港常見樹木 · SST', 'count': 146},
                       {'id': 'broad_plants', 'name': '世界模型',
                        'count': manifest.get('plant_species_count'), 'available': bool(manifest),
                        'version': manifest.get('version'), 'experimental': True}]}


@app.get('/api/reference/{key}')
def reference(key: str):
    return FileResponse(asset_path(key))


@app.get('/api/web-reference/{key}')
def web_reference(key: str):
    try:
        path = reference_library.asset_path(key)
    except KeyError:
        raise HTTPException(404)
    return FileResponse(path)


@app.get('/api/test-photo')
def test_photo():
    return FileResponse(DATA / 'test_data/hongkong_025/image.jpg')


@app.get('/api/test-panorama')
def test_panorama():
    # Fixed local regression photograph, no fixture boxes or fixture species results.
    return FileResponse(PROJECT / 'hk-tree-vmms/derived/tree_detector_iteration2_full_20260906/images/val/base_000187_jianshazui__jianshazui_pano_1__002971.jpg')


@app.get('/references')
def reference_gallery():
    return FileResponse(ROOT / 'static/reference-gallery.html', headers={'Cache-Control': 'no-store'})


@app.get('/api/reference-catalog')
def reference_catalog():
    labels = json.loads((PROJECT / 'outputs/distillation/bioclip_probe_20260908_0227/classes.json').read_text(encoding='utf-8'))['labels'] + ['Wodyetia bifurcata']
    return [{**bilingual_names(s), 'species': s, 'references': reference_library.references(s)} for s in labels]


@app.get('/references/add')
def reference_editor():
    return FileResponse(ROOT/'static/reference-editor.html', headers={'Cache-Control':'no-store'})


def pool_rows():
    path = reference_library.ROOT/'mixed_v3/pool_manifest.json'
    if not path.exists():
        path = reference_library.ROOT/'manifest.json'
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else []


@app.get('/api/reference-pool')
def reference_pool(species: str):
    if species not in manual_references.labels():
        raise HTTPException(400, 'Invalid species / 樹種無效')
    review_path = reference_library.ROOT/'visual_review.json'
    decisions = json.loads(review_path.read_text(encoding='utf-8')) if review_path.exists() else {}
    revised=reference_library.ROOT/'mixed_v3/visual_review.json'
    if revised.exists():
        for key,decision in json.loads(revised.read_text(encoding='utf-8')).items():
            decisions[key]={**decision,'status':'accepted_for_reference' if decision['status']=='accepted' else 'rejected_for_reference'}
    return [{**bilingual_names(species), 'id':r['id'], 'url':'/api/reference-pool/'+r['id'],
        'author':r['author'], 'source_url':r['source_url'], 'license':r['license'],
        'status':decisions.get(r['id'],{}).get('status','pending'), 'note':decisions.get(r['id'],{}).get('note',''),
        'reference_taxon':r['observed_taxon']} for r in pool_rows() if r['species']==species]


@app.get('/api/reference-pool/{key}')
def reference_pool_photo(key: str):
    row = next((r for r in pool_rows() if r['id']==key), None)
    if row is None:
        raise HTTPException(404)
    path = (reference_library.ROOT/row['file']).resolve()
    if not path.is_relative_to((reference_library.ROOT/'images').resolve()) or not path.is_file():
        raise HTTPException(404)
    return FileResponse(path)


def local_write(request):
    # Reject cross-origin browser submissions, including opaque/file origins.
    if request.headers.get('X-Reference-Editor') != '1':
        raise HTTPException(403, 'Use the local editor / 請使用本機補圖介面')
    origin = request.headers.get('origin')
    if origin and origin.rstrip('/') != str(request.base_url).rstrip('/'):
        raise HTTPException(403, 'Cross-origin write rejected / 不接受跨來源寫入')


@app.get('/api/manual-references')
def manual_history(species: str):
    if species not in manual_references.labels():
        raise HTTPException(400, 'Invalid species / 樹種無效')
    return manual_references.rows(species, history=True)


@app.get('/api/manual-reference/{key}')
def manual_photo(key: str):
    try:
        return FileResponse(manual_references.asset_path(key))
    except KeyError:
        raise HTTPException(404)


@app.post('/api/manual-references')
async def add_manual_reference(request: Request, image: UploadFile = File(...), species: str = Form(...),
    slot: int = Form(...), author: str = Form(...), source_url: str = Form(''), rights: str = Form(...),
    confirmed: bool = Form(False)):
    local_write(request)
    try:
        data = await image.read(MAX_BYTES+1)
    finally:
        await image.close()
    try:
        digest = hashlib.sha256(data).hexdigest()
        original = next((r for r in pool_rows() if r['species']==species and r['sha256']==digest),None)
        return await run_in_threadpool(manual_references.save, data, species, slot, author, source_url, rights, confirmed, original)
    except (ValueError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as e:
        raise HTTPException(400, str(e) if isinstance(e, ValueError) else 'Invalid image / 圖片無效')


@app.post('/api/manual-references/{key}/status')
def manual_status(key: str, request: Request, active: bool = Form(...)):
    local_write(request)
    try:
        manual_references.set_active(key, active)
    except KeyError:
        raise HTTPException(404)
    return {'ok':True, 'active':active}


@app.post('/api/predict')
async def predict(image: UploadFile = File(...), scenario: str = Form('street'), source: str = Form('image'), projection: str = Form('auto'), view_policy: str = Form('context'), recognition_scope: str = Form('hong_kong')):
    if scenario not in {'street', 'conservation'}:
        await image.close()
        raise HTTPException(400, 'Invalid scenario / 識別場景無效')
    if source != 'image':
        await image.close()
        raise HTTPException(409, 'Use the image + cloud upload endpoint for fusion. / 融合請使用影像＋點雲上傳入口。')
    data = await image.read(MAX_BYTES + 1)
    await image.close()
    try:
        geometry=tree_detection.panorama.resolve_projection(tree_detection.decode(data).size,projection)
    except (ValueError,OSError,Image.DecompressionBombError,Image.DecompressionBombWarning):
        raise HTTPException(400,'Invalid image or projection / 影像或投影類型無效')
    if geometry['kind']=='equirectangular':
        raise HTTPException(409,'Detect and choose a perspective tree crop first. / 請先偵測並選擇透視校正的樹木切片。')
    if view_policy not in {'context', 'single'}:
        raise HTTPException(400, 'Invalid image view policy / 影像取景模式無效')
    if recognition_scope not in flora_scope.SCOPES:
        raise HTTPException(400, 'Invalid recognition scope')
    return await run_in_threadpool(infer, data, scenario, 'sstv1', view_policy, recognition_scope, True)


@app.post('/api/detect-trees')
async def detect_trees(image: UploadFile = File(...), source: str = Form('image'), projection: str = Form('auto')):
    try:
        if source != 'image':
            raise HTTPException(409, 'Tree selection is image-only. / 選樹流程僅供單源影像使用。')
        data = await image.read(MAX_BYTES+1)
    finally:
        await image.close()
    try:
        result=await run_in_threadpool(tree_detection.detect, data, projection)
        # Validate inside the guarded block, before FastAPI response serialization.
        try:
            json.dumps(result,allow_nan=False)
        except (ValueError,TypeError) as exc:
            raise RuntimeError('Detector returned invalid JSON numbers') from exc
        return result
    except BlockingIOError as e:
        raise HTTPException(429, str(e))
    except (ValueError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise HTTPException(400, 'Invalid image / 影像無效')
    except Exception:
        import logging
        logging.getLogger(__name__).exception('Tree detection failed')
        raise HTTPException(503, 'Tree detector unavailable; please retry. / 樹木偵測器暫不可用，請重試。')


@app.get('/api/sam/status')
def sam_status():
    return sam_assist.status()


@app.post('/api/sam/segment')
async def sam_segment(image: UploadFile = File(...), x: float = Form(...), y: float = Form(...),
                      projection: str = Form('auto')):
    try:
        data = await image.read(MAX_BYTES + 1)
    finally:
        await image.close()
    try:
        return await run_in_threadpool(sam_assist.segment, data, x, y, projection)
    except BlockingIOError as exc:
        raise HTTPException(429, str(exc)) from exc
    except (ValueError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc


@app.post('/api/identify-tree')
async def identify_tree(image: UploadFile = File(...), token: str = Form(...),
                        scenario: str = Form('street'), source: str = Form('image'), recognition_scope: str = Form('hong_kong')):
    try:
        if source != 'image' or scenario not in {'street', 'conservation'}:
            raise HTTPException(400, 'Invalid source or scenario / 來源或場景無效')
        data = await image.read(MAX_BYTES+1)
    finally:
        await image.close()
    try:
        crop, meta, preview = await run_in_threadpool(tree_detection.selected, data, token)
    except (ValueError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as e:
        raise HTTPException(400, str(e))
    result = await run_in_threadpool(infer, crop, scenario, 'sstv1', 'context', recognition_scope, True)
    result.update(tree_detection=True, selected_tree=meta, crop_preview=preview)
    if meta.get('projection_view'):
        result['preprocessing']='Spherical panorama → perspective → selected tree. / 球面全景 → 透視校正 → 所選樹木。 '+result['preprocessing']
        result['warnings'].append('Reprojection corrects panorama geometry, not measured tree height or missing detail. / 重投影校正全景幾何，不代表量測樹高，亦不補造缺失細節。')
        if meta.get('edge_clipped'):
            result['warnings'].append('Tree touches the view edge; parts may be missing. Try another view. / 樹木接近視圖邊界，可能不完整，請補充另一視角。')
        if meta['projection'].get('inferred_from_ratio'):
            result['warnings'].append('Assumed a full 2:1 spherical panorama. Use Photo mode for ordinary wide photos. / 按 2:1 比例推定為完整球面全景；普通寬幅照片請選擇照片模式。')
    result['warnings'].append('Check the selected crop contains your intended tree. / 請確認所選切片包含目標樹木。')
    return result


@app.post('/api/export-target')
async def export_target(image: UploadFile = File(...), token: str = Form(...)):
    # Repeat the signed native crop used for inference, not the 960px UI thumbnail.
    try:
        data = await image.read(MAX_BYTES + 1)
    finally:
        await image.close()
    try:
        crop, _, _ = await run_in_threadpool(tree_detection.selected, data, token)
    except (ValueError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise HTTPException(400, 'Target expired or image changed. Identify again before exporting. / 目標已過期或影像已變更，請重新識別後匯出。')
    return Response(crop, media_type='image/jpeg', headers={
        'Content-Disposition': 'attachment; filename="SpaceScan-selected-target.jpg"',
        'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff',
        'X-Target-SHA256': hashlib.sha256(crop).hexdigest(),
    })


@app.post('/api/predict-with-cloud')
async def predict_with_cloud(image: UploadFile = File(...), cloud: UploadFile = File(...),
                             paired: bool = Form(False), units: str = Form('unknown'),
                             z_up: bool = Form(False), scenario: str = Form('street'),
                             projection: str = Form('auto')):
    try:
        if not paired:
            raise HTTPException(400, 'Confirm the image and cloud represent the same tree. / 請確認影像與點雲為同一株樹。')
        if scenario not in {'street', 'conservation'}:raise HTTPException(400, 'Invalid scenario / 場景無效')
        data=await image.read(MAX_BYTES+1);points=await cloud.read(point_assist.MAX_BYTES+1)
    finally:
        await image.close();await cloud.close()
    try:
        geometry=tree_detection.panorama.resolve_projection(tree_detection.decode(data).size,projection)
        if geometry['kind']=='equirectangular':
            raise HTTPException(409, 'For cloud assistance, upload a confirmed single-tree perspective crop. / 點雲輔助請使用已確認同株的單木透視切片。')
        evidence,point=await run_in_threadpool(point_recognition.infer_upload,points,cloud.filename or '',units,z_up)
    except (ValueError,OSError,EOFError,point_assist.zipfile.BadZipFile):
        raise HTTPException(400,'Invalid cloud or image. Check format, XYZ and units. / 點雲或影像無效，請檢查格式、XYZ 及單位。')
    except RuntimeError:
        raise HTTPException(503,'Point model unavailable; no fusion result was produced. / 點雲模型暫不可用，未產生融合結果。')
    # Keep the previously evaluated dual-source image branch unchanged.
    result=await run_in_threadpool(infer,data,scenario,'teacher')
    result.update(source='image_point',point_evidence=evidence,fusion_used=False,
                  image_score_policy='Unchanged BioCLIP image scores; point geometry provided separately.',
                  paired_user_confirmed=True,input_image_sha256=hashlib.sha256(data).hexdigest())
    if point is None:
        result['warnings'].append('Cloud quality / vertical-axis check failed. Image-only results retained. / 點雲品質或垂直軸檢查未通過，保留單源影像結果。')
        return result
    try:
        image_scores,cloud_scores,fused,mass=point_recognition.fuse(result['image_distribution'],point)
    except RuntimeError:
        raise HTTPException(503,'Full image distribution unavailable; restart the teacher service. / 影像完整分數不可用，請重新啟動教師服務。')
    def ranked(scores):
        return [dict(species=point_recognition.CLASSES[i],score=float(scores[i]),
                     **bilingual_names(point_recognition.CLASSES[i]),
                     references=references(point_recognition.CLASSES[i],result['input_image_sha256']))
                for i in sorted(range(7),key=lambda i:-scores[i])[:3]]
    full_image=result['top3']
    result.update(top3=ranked(fused),model='BioCLIP 2.5 + PTv2 · 7 classes / 七類',
                  fusion_used=True,point_model=point,
                  comparison=dict(image=ranked(image_scores),point=ranked(cloud_scores),fusion=ranked(fused)),
                  image_full_catalog_top3=full_image,image_catalog_mass=mass,
                  class_scope=[dict(species=c,**bilingual_names(c)) for c in point_recognition.CLASSES],
                  fusion_policy=dict(image_weight=.5,point_weight=.5,calibrated=False),
                  image_score_policy='Image scores conditioned on the seven supported classes; equal-weight score fusion. / 影像分數限於七個支援類別後重新正規化，再等權融合。',
                  preprocessing=result['preprocessing']+' PTv2: 2,048 measured points, four Z rotations, mean features. / PTv2：2,048 個實測點，四次 Z 軸旋轉後平均特徵。',
                  inference_seconds=(result.get('inference_seconds') or 0)+point['inference_seconds'])
    evidence.update(fusion_used=True,model_scores_changed=True)
    result['warnings'].append('Seven-class experimental ranking; not a 148-class fusion result or a measured accuracy gain. Absolute height is not a classifier feature. / 七類實驗性排序，並非 148 類融合或已量測準確率增益；分類器不直接使用絕對樹高。')
    if full_image[0]['species'] == presentation_groups.BAUHINIA_GROUP:
        result['warnings'].append('The full image result groups three Bauhinia species, but the point model supports only B. x blakeana. Its fusion score must not be read as a three-species group score or a confirmed Hong Kong Orchid Tree. / 完整影像將三種洋紫荊合併顯示，但點雲模型只支援 B. × blakeana；融合分數不可當作三種合併分數或香港市花的確認。')
    elif full_image[0]['scientific_name'] not in point_recognition.CLASSES:
        result['warnings'].append('The full image model favours a species outside these seven classes. Check the original image candidates. / 完整影像模型首選不在此七類內，請核對原影像候選。')
    if fused.max()<.5:
        result['warnings'].append('Low-confidence fusion: add a clearer view or review the tree. / 融合置信度偏低，請補充清晰視角或人工核驗。')
    return result


@app.post('/api/video/jobs')
async def create_video_job(video: UploadFile = File(...), profile: str = Form('balanced')):
    if profile not in video_pipeline.PROFILES:
        await video.close()
        raise HTTPException(400, 'Invalid video analysis profile / 影片分析模式無效')
    try:
        job_id, target = VIDEO_JOBS.reserve(video.filename or 'video.mp4')
    except ValueError as exc:
        await video.close()
        raise HTTPException(400, str(exc)) from exc
    size = 0
    try:
        with target.open('wb') as stream:
            while True:
                chunk = await video.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > video_pipeline.MAX_VIDEO_BYTES:
                    raise ValueError(video_pipeline.video_size_error())
                stream.write(chunk)
        if size < 1024:
            raise ValueError('Video is empty or incomplete. / 影片為空或不完整。')
        VIDEO_JOBS.start(
            job_id, profile,
            lambda crop: infer(crop, 'street', 'sstv1', 'single'),
        )
        return {'job_id': job_id, 'status_url': f'/api/video/jobs/{job_id}',
                'source_video_retained': False}
    except (ValueError, OSError) as exc:
        VIDEO_JOBS.discard(job_id)
        raise HTTPException(400, str(exc)) from exc
    finally:
        await video.close()


@app.get('/api/video/jobs/{job_id}')
def video_job(job_id: str):
    if not re.fullmatch(r'[0-9a-f]{32}', job_id):
        raise HTTPException(404)
    try:
        return VIDEO_JOBS.status(job_id)
    except KeyError:
        raise HTTPException(404, 'Video job not found / 找不到影片任務')


@app.get('/api/video/jobs/{job_id}/views/{name}')
def video_job_view(job_id: str, name: str):
    if not re.fullmatch(r'[0-9a-f]{32}', job_id):
        raise HTTPException(404)
    try:
        path = VIDEO_JOBS.view_path(job_id, name)
    except KeyError:
        raise HTTPException(404)
    return FileResponse(path, media_type='image/jpeg', headers={
        'Cache-Control': 'private, max-age=3600', 'X-Content-Type-Options': 'nosniff',
    })


@app.get('/api/vmms/corridor/catalog')
def vmms_corridor_catalog():
    try:
        return vmms_corridor.catalog()
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc


@app.post('/api/vmms/corridor/folder/select')
async def select_vmms_corridor_folder(request: Request):
    """Use the native Windows picker so hundreds of source panoramas stay in place."""
    try:
        body = await request.json()
    except json.JSONDecodeError:
        body = {}
    try:
        folder = await run_in_threadpool(
            vmms_corridor.choose_panorama_folder, str(body.get('initial_folder', '')).strip() or None)
        if not folder:
            return {'cancelled': True}
        return {'cancelled': False, **await run_in_threadpool(
            vmms_corridor.public_folder_inspection, folder)}
    except (ValueError, RuntimeError, OSError) as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post('/api/vmms/corridor/folder/inspect')
async def inspect_vmms_corridor_folder(request: Request):
    try:
        body = await request.json()
        return await run_in_threadpool(
            vmms_corridor.public_folder_inspection, str(body.get('folder', '')))
    except (ValueError, TypeError, json.JSONDecodeError, OSError) as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post('/api/vmms/corridor/jobs')
async def create_vmms_corridor_job(request: Request):
    try:
        body = await request.json()
        source_mode = str(body.get('source_mode', 'catalog'))
        if source_mode == 'folder':
            job_id = CORRIDOR_JOBS.create_folder(
                str(body.get('folder', '')), int(body.get('start_index', 0)),
                int(body.get('frame_step', 1)), int(body.get('max_frames', 300)),
                lambda crop: infer(crop, 'street', 'sstv1', 'single'),
            )
        elif source_mode == 'catalog':
            job_id = CORRIDOR_JOBS.create(
                str(body.get('segment_id', '')), float(body.get('offset_m', 0)),
                float(body.get('length_m', 200)), float(body.get('spacing_m', 5)),
                lambda crop: infer(crop, 'street', 'sstv1', 'single'),
            )
        else:
            raise ValueError('Unknown corridor source mode / 未知走廊資料來源')
        return {'job_id': job_id, 'status_url': f'/api/vmms/corridor/jobs/{job_id}',
                'source_panoramas_copied': False, 'model_weights_changed': False}
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get('/api/vmms/corridor/jobs/{job_id}')
def vmms_corridor_job(job_id: str):
    if not re.fullmatch(r'[0-9a-f]{32}', job_id):
        raise HTTPException(404)
    try:
        return CORRIDOR_JOBS.status(job_id)
    except KeyError:
        raise HTTPException(404, 'Corridor job not found / 找不到走廊測試任務')


@app.get('/api/vmms/corridor/jobs/{job_id}/{group}/{name}')
def vmms_corridor_asset(job_id: str, group: str, name: str):
    if not re.fullmatch(r'[0-9a-f]{32}', job_id):
        raise HTTPException(404)
    try:
        path = CORRIDOR_JOBS.asset(job_id, group, name)
    except KeyError:
        raise HTTPException(404)
    return FileResponse(path, media_type='image/jpeg', headers={
        'Cache-Control': 'private, max-age=3600', 'X-Content-Type-Options': 'nosniff',
    })


app.mount('/static', StaticFiles(directory=ROOT / 'static'), name='static')
