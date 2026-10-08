"""Inference-only tree selection. Local checkpoints; no uploaded images persisted.

Scores are detector outputs, not calibrated probabilities. Background veto and
central-target policy are heuristics, not verified single-tree ground truth.
"""
import base64
import hashlib
import hmac
import io
import json
import math
import secrets
import sys
import threading
import time
import warnings
from pathlib import Path

from PIL import Image, ImageOps
from . import panorama

PROJECT = Path(__file__).resolve().parents[1]
WEIGHT = PROJECT / 'hk-tree-vmms/derived/training_runs/combined_tree_segmentation/yolo11m_seg_1024_20260901_104757/weights/best.pt'
RUNTIME = PROJECT / 'outputs/tree_detector_benchmark_20260909_v1/runtime'
CLIP_WEIGHT = PROJECT / 'outputs/tree_detector_benchmark_20260909_v1/assets/clip/ViT-B-32.pt'
DINO_DIR = PROJECT / 'outputs/tree_detector_benchmark_20260909_v1/assets/grounding_dino_base'
THRESHOLD = .4
SECRET = secrets.token_bytes(32)
LOCK = threading.Lock()
ENGINE = None
TREE = ['a tree', 'a palm tree', 'a tree canopy with leaves', 'a tree trunk and branches',
        'a dark tree in shadow', 'a leafless tree', 'a street tree in front of a building', 'a blurry photograph of a tree']
BACKGROUND = ['the underside of a bridge', 'a concrete ceiling', 'a building roof', 'an architectural canopy',
              'a building facade with windows', 'a concrete wall', 'a metal fence', 'a road with cars',
              'a bus', 'a street lamp', 'the sky', 'a construction site', 'a building column']


def decode(data):
    if not data or len(data) > 20 * 1024 * 1024:
        raise ValueError('Use an image up to 20 MB. / 請使用不超過 20 MB 的影像。')
    with warnings.catch_warnings():
        warnings.simplefilter('error', Image.DecompressionBombWarning)
        with Image.open(io.BytesIO(data)) as im:
            if im.format not in ('JPEG', 'PNG', 'WEBP') or im.width * im.height > 40_000_000:
                raise ValueError('Use JPG / PNG / WEBP up to 40 MP. / 請使用不超過四千萬像素的影像。')
            return ImageOps.exif_transpose(im).convert('RGB')


def jpeg(im, edge=None):
    im = im.copy()
    if edge:
        im.thumbnail((edge, edge), Image.Resampling.LANCZOS)
    b = io.BytesIO()
    im.save(b, 'JPEG', quality=95)
    return b.getvalue()


def data_url(data):
    return 'data:image/jpeg;base64,' + base64.b64encode(data).decode()


def visibility_p90(crop):
    """Conservative crop-quality veto; never brighten inference pixels or scores."""
    gray = ImageOps.grayscale(crop)
    gray.thumbnail((96, 96), Image.Resampling.BILINEAR)
    histogram = gray.histogram()
    target = sum(histogram) * .9
    seen = 0
    for level, count in enumerate(histogram):
        seen += count
        if seen >= target:
            return level
    return 255


def area(b):
    return max(0, b[2]-b[0]) * max(0, b[3]-b[1])


def intersection(a, b):
    return area([max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])])


def clean_boxes(rows, width, height):
    valid = []
    for r in rows:
        b, s = r['box'], r['score']
        if len(b) != 4 or not all(math.isfinite(v) for v in [*b, s]) or not THRESHOLD <= s <= 1:
            continue
        b = [max(0, min(1, v)) for v in b]
        if (b[2]-b[0])*width < 48 or (b[3]-b[1])*height < 72 or (width/height >= 1.8 and area(b) > .85):
            continue
        if any(intersection(b, x['box']) / max(1e-9, area(b)+area(x['box'])-intersection(b, x['box'])) > .5 for x in valid):
            continue
        valid.append({**r, 'box': b})
    # Remove a coarse group box when it contains two separate smaller detections.
    result = []
    for r in valid:
        inside = [s for s in valid if s is not r and area(s['box']) < .65*area(r['box']) and
                  intersection(s['box'], r['box']) > .9*area(s['box'])]
        group = sum(area(s['box']) for s in inside) >= .5*area(r['box']) and any(intersection(a['box'], b['box']) < .2*min(area(a['box']), area(b['box']))
                    for i, a in enumerate(inside) for b in inside[i+1:])
        if not group:
            result.append(r)
    return result


def centered(rows, panorama=False):
    ordered = sorted(enumerate(rows), key=lambda x: area(x[1]['box']), reverse=True)
    if not ordered:
        return None
    i, r = ordered[0]
    a, b, c, d = r['box']
    if abs((a+c)/2-.5) > .12 or c-a < .12 or c-a > (.8 if panorama else 1) or d-b < .3 or not b <= .5 <= d:
        return None
    if len(ordered) > 1 and area(r['box']) < 1.6*area(ordered[1][1]['box']):
        return None
    return i


def remove_partial_boxes(rows):
    """Drop contained fragments only after both candidates pass the tree gate.

    Do not invent a larger rectangle or transfer the fragment's higher score.
    A parent containing several separate children is not treated as a single tree.
    """
    result=[]
    for r in rows:
        fragment=False
        for parent in rows:
            if parent is r:continue
            b,p=r['box'],parent['box']
            if not (area(b)<.45*area(p) and area(p)<.85 and
                    intersection(b,p)>=.9*area(b) and p[3]-p[1]>1.5*(b[3]-b[1])):continue
            peers=[s for s in rows if s is not r and s is not parent and
                   area(s['box'])<.65*area(p) and intersection(s['box'],p)>.9*area(s['box']) and
                   intersection(b,s['box'])<.2*min(area(b),area(s['box']))]
            if not peers:
                fragment=True;break
        if not fragment:result.append(r)
    return result


def crop_rect(box, size):
    w, h = size
    x0, y0, x1, y1 = box
    dx, dy = .04*(x1-x0), .04*(y1-y0)
    return [max(0, math.floor((x0-dx)*w)), max(0, math.floor((y0-dy)*h)),
            min(w, math.ceil((x1+dx)*w)), min(h, math.ceil((y1+dy)*h))]


def sign(payload):
    raw = base64.urlsafe_b64encode(json.dumps(payload, separators=(',', ':')).encode()).decode()
    return raw + '.' + hmac.new(SECRET, raw.encode(), hashlib.sha256).hexdigest()


def selected(data, token):
    try:
        if len(token) > 8000:
            raise ValueError()
        raw, signature = token.split('.')
        if not hmac.compare_digest(signature, hmac.new(SECRET, raw.encode(), hashlib.sha256).hexdigest()):
            raise ValueError()
        meta = json.loads(base64.urlsafe_b64decode(raw))
        if meta['expires'] < time.time() or meta['sha256'] != hashlib.sha256(data).hexdigest():
            raise ValueError()
        im = decode(data)
        if meta.get('projection_view'):
            im = panorama.render(im, meta['projection_view'])
        crop = im.crop(meta['crop_box'])
        return jpeg(crop), meta, data_url(jpeg(crop, 960))
    except (KeyError, ValueError, TypeError, OSError):
        raise ValueError('Selection expired or image changed. Detect again. / 選擇已過期或影像已變更，請重新偵測。')


class Engine:
    def __init__(self):
        if not WEIGHT.is_file() or not CLIP_WEIGHT.is_file():
            raise RuntimeError('Local detector checkpoints missing')
        if str(RUNTIME) not in sys.path:
            sys.path.insert(0, str(RUNTIME))
        import torch
        import clip
        from ultralytics import YOLO
        self.torch, self.clip = torch, clip
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        torch.set_num_threads(6)
        self.yolo = YOLO(str(WEIGHT))
        self.tree_classes = [i for i, n in self.yolo.names.items() if 'tree' in n.lower()]
        if not self.tree_classes:
            raise RuntimeError('Checkpoint has no tree class')
        self.gate, self.preprocess = clip.load(str(CLIP_WEIGHT), device=self.device, jit=False)
        # The background gate must not leak half-precision NaN/Inf into decisions/JSON.
        # Keep the same weights but run this small gate in float32.
        self.gate.float().eval().requires_grad_(False)
        prompts = ['a photo of '+p.removeprefix('a ') for p in TREE+BACKGROUND]
        with torch.inference_mode():
            self.text = self.gate.encode_text(clip.tokenize(prompts).to(self.device)).float()
            self.text = self.normalized_features(self.text)
        self.weights = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [WEIGHT, CLIP_WEIGHT]}
        self.dino = None

    def boxes(self, image):
        w, h = image.size
        result = self.yolo.predict(image, imgsz=1024, conf=THRESHOLD, iou=.5, device=self.device,
                                   classes=self.tree_classes, agnostic_nms=True, max_det=80, verbose=False)[0]
        return [{'box': [v/d for v, d in zip(b, [w, h, w, h])], 'score': float(s), 'detector': 'YOLO11m-seg'}
                for b, s in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist())]

    def supplemental_boxes(self, image):
        """Local open-vocabulary supplement for panorama / missing targets only."""
        import inspect
        if self.dino is None:
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            self.processor = AutoProcessor.from_pretrained(DINO_DIR, local_files_only=True, trust_remote_code=False)
            self.dino = AutoModelForZeroShotObjectDetection.from_pretrained(
                DINO_DIR, local_files_only=True, trust_remote_code=False).to(self.device).eval().requires_grad_(False)
            self.weights['grounding_dino_base'] = hashlib.sha256((DINO_DIR/'model.safetensors').read_bytes()).hexdigest()
        inputs = self.processor(images=image, text='tree. palm tree.', return_tensors='pt').to(self.device)
        with self.torch.inference_mode():
            output = self.dino(**inputs)
        method = self.processor.post_process_grounded_object_detection
        parameter = 'threshold' if 'threshold' in inspect.signature(method).parameters else 'box_threshold'
        result = method(output, inputs.input_ids, **{parameter: THRESHOLD, 'text_threshold': .25,
                        'target_sizes': [(image.height, image.width)]})[0]
        return [{'box': [v/d for v, d in zip(b, [image.width, image.height]*2)],
                 'score': float(s), 'detector': 'Grounding DINO Base'}
                for b, s in zip(result['boxes'].cpu().tolist(), result['scores'].cpu().tolist())]

    def normalized_features(self, features):
        norms=features.norm(dim=-1,keepdim=True)
        if not self.torch.isfinite(features).all() or not self.torch.isfinite(norms).all() or (norms<=1e-12).any():
            raise RuntimeError('Nonfinite background-filter features; no tree decision produced')
        return features/norms

    def background_checks(self, crops):
        if not crops:
            return []
        tensors = []
        for im in crops:
            tensors += [self.preprocess(ImageOps.pad(im, (224, 224), color=(124, 116, 104))), self.preprocess(im)]
        with self.torch.inference_mode():
            x = self.gate.encode_image(self.torch.stack(tensors).to(self.device).float()).float()
            x = self.normalized_features(x)
            scores = (x @ self.text.T).cpu().numpy().reshape(len(crops), 2, -1)
        checks = []
        for views in scores:
            margins = [float(v[len(TREE):].max()-v[:len(TREE)].max()) for v in views]
            if not all(math.isfinite(m) for m in margins):
                raise RuntimeError('Nonfinite background-filter scores; no tree decision produced')
            # Never let NaN comparisons silently pass the background gate.
            checks.append({'keep': not all(m >= 0 for m in margins), 'background_minus_tree': margins})
        return checks


def detect_panorama(im, data, engine, projection):
    """Detect only AFTER spherical-to-perspective reprojection, not on ERP tiles."""
    start=time.perf_counter();rows=[];suppressed=0;low_visibility_removed=0
    specs=panorama.detection_views(im.size)
    for spec in specs:
        view=panorama.render(im,spec,max_edge=1280)
        found=engine.boxes(view)+engine.supplemental_boxes(view)
        found=clean_boxes(sorted(found,key=lambda r:r['score'],reverse=True),*view.size)[:24]
        crops=[view.crop(crop_rect(r['box'],view.size)) for r in found]
        checks=[]
        for i in range(0,len(crops),8):checks.extend(engine.background_checks(crops[i:i+8]))
        for r,crop,check in zip(found,crops,checks):
            if visibility_p90(crop)<24:
                low_visibility_removed+=1
                continue
            if not check['keep']:
                suppressed+=1;continue
            spherical=panorama.sphere_box(r['box'],spec)
            # Prefer interior detections over clipped duplicates of the same tree.
            a,b,c,d=r['box'];edge=min(a,b,1-c,1-d)<.015
            rows.append({**r,'view_box':r['box'],'box':spherical,'projection_view':spec,
                         'background_check':check,'edge_clipped':edge,
                         'thumbnail':data_url(jpeg(crop,360))})
    # Spherical matching wraps horizontally; no hard split at the panorama seam.
    ordered=sorted(rows,key=lambda r:(r['edge_clipped'],-r['score']))
    unique=[]
    for r in ordered:
        if not any(panorama.circular_iou(r['box'],s['box'])>.45 for s in unique):unique.append(r)
    candidates=sorted(unique[:12],key=lambda r:((r['box'][0]+r['box'][2])/2)%1)
    digest=hashlib.sha256(data).hexdigest()
    for i,r in enumerate(candidates):
        spec=r['projection_view'];rect=crop_rect(r['view_box'],(spec['width'],spec['height']))
        r.update(id=i+1,crop_box=rect,native_size=[rect[2]-rect[0],rect[3]-rect[1]])
        payload={k:v for k,v in r.items() if k!='thumbnail'}
        payload.update(sha256=digest,expires=time.time()+1800,weight_sha256=engine.weights,
                       threshold=THRESHOLD,policy='spacescan-perspective-select-v2',projection=projection,
                       dimensions_are_resampled=True,source_size=list(im.size))
        r['token']=sign(payload)
    # Never automatically identify a pole/seam view as the image's central target.
    central_rows=[{**r,'box':r['box']} for r in candidates]
    auto=centered(central_rows,panorama=True)
    if auto is not None and (candidates[auto]['edge_clipped'] or not 0 <= candidates[auto]['box'][0] < candidates[auto]['box'][2] <= 1):auto=None
    return {'action':'auto' if auto is not None else 'choose' if candidates else 'no_target',
            'auto_id':candidates[auto]['id'] if auto is not None else None,
            'candidates':candidates,'detector':'YOLO11m-seg + Grounding DINO Base','threshold':THRESHOLD,
            'background_filter':'CLIP ViT-B/32','suppressed':suppressed+low_visibility_removed,
            'low_visibility_removed':low_visibility_removed,
            'additional_candidates':max(0,len(unique)-12),'tiled':False,'perspective_view_count':len(specs),
            'projection':{**projection,'method':'spherical_to_pinhole','view_count':len(specs)},
            'width':im.width,'height':im.height,'seconds':time.perf_counter()-start,
            'scores_calibrated':False,'single_instance_guaranteed':False}


def detect(data, projection='auto'):
    global ENGINE
    im = decode(data)
    projection = panorama.resolve_projection(im.size, projection)
    # Bounded inference queue: no unbounded GPU work after repeated UI requests.
    if not LOCK.acquire(blocking=False):
        raise BlockingIOError('Detector busy. Please retry shortly. / 偵測器忙碌，請稍後重試。')
    try:
        start = time.perf_counter()
        if ENGINE is None:
            ENGINE = Engine()
        if projection['kind']=='equirectangular':
            result=detect_panorama(im,data,ENGINE,projection)
            result['seconds']=time.perf_counter()-start
            return result
        view = im.copy()
        view.thumbnail((4096, 4096), Image.Resampling.LANCZOS)
        rows = ENGINE.boxes(view)
        tiled = view.width/view.height >= 1.8
        if tiled:
            for left, right in [(0, .55), (.225, .775), (.45, 1)]:
                x0, x1 = round(left*view.width), round(right*view.width)
                for row in ENGINE.boxes(view.crop((x0, 0, x1, view.height))):
                    a, b, c, d = row['box']
                    rows.append({**row, 'box': [(x0+a*(x1-x0))/view.width, b, (x0+c*(x1-x0))/view.width, d]})
        rows = clean_boxes(sorted(rows,key=lambda r:-r['score']),*im.size)
        # A solitary/uncentered YOLO hit does not establish adequate tree coverage.
        supplemental = tiled or len(rows)<2 or centered(rows) is None
        supplement_reason = 'wide_view' if tiled else 'few_candidates' if len(rows)<2 else 'no_clear_center' if supplemental else None
        if supplemental:
            rows += ENGINE.supplemental_boxes(view)
        rows = clean_boxes(sorted(rows, key=lambda r: r['score'], reverse=True), *im.size)[:24]
        crops = [im.crop(crop_rect(r['box'], im.size)) for r in rows]
        checks = []
        for offset in range(0, len(crops), 8):
            checks.extend(ENGINE.background_checks(crops[offset:offset+8]))
        visibility = [visibility_p90(crop) for crop in crops]
        low_visibility_removed = sum(level < 24 for level in visibility)
        accepted = [{**r, 'background_check': c} for r, c, level in zip(rows, checks, visibility)
                    if c['keep'] and level >= 24]
        before_fragments=len(accepted)
        accepted=remove_partial_boxes(accepted)
        fragments_removed=before_fragments-len(accepted)
        # A compact chooser; never silently imply the list is exhaustive.
        candidates = sorted(accepted[:12], key=lambda r: r['box'][0])
        digest = hashlib.sha256(data).hexdigest()
        for i, row in enumerate(candidates):
            rect = crop_rect(row['box'], im.size)
            row.update(id=i+1, crop_box=rect, native_size=[rect[2]-rect[0], rect[3]-rect[1]])
            meta = {**row, 'sha256': digest, 'expires': time.time()+1800,
                    'weight_sha256': ENGINE.weights,
                    'threshold': THRESHOLD, 'policy': 'spacescan-tree-select-recall-v2'}
            row['token'] = sign(meta)
            row['thumbnail'] = data_url(jpeg(im.crop(rect), 360))
        auto = centered(candidates, panorama=tiled)
        return {'action': 'auto' if auto is not None else 'choose' if candidates else 'no_target',
                'auto_id': candidates[auto]['id'] if auto is not None else None,
                'candidates': candidates, 'detector': 'YOLO11m-seg' + (' + Grounding DINO Base' if supplemental else ''), 'threshold': THRESHOLD,
                'background_filter': 'CLIP ViT-B/32', 'suppressed': len(rows)-len(accepted),
                'low_visibility_removed': low_visibility_removed,
                'supplement_reason': supplement_reason, 'partial_boxes_removed': fragments_removed,
                'additional_candidates': max(0, len(accepted)-12), 'tiled': tiled,
                'width': im.width, 'height': im.height, 'seconds': time.perf_counter()-start,
                'projection': projection,
                'scores_calibrated': False, 'single_instance_guaranteed': False}
    finally:
        LOCK.release()
