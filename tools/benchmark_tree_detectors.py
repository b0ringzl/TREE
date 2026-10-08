"""Isolated inference-only detector comparison; no training or production writes."""
from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/tree_detector_benchmark_20260909_v1'
DATA = ROOT / 'hk-tree-vmms/derived/tree_detector_iteration2_full_20260906'
OLD_DATA = ROOT / 'hk-tree-vmms/derived/combined_tree_segmentation/effective_yolo_20260901_103256'
sys.path.insert(0, str(OUT / 'runtime'))
# Child-local network/runtime settings; never modify the installed environment.
os.environ.update(NO_PROXY='*', HF_HUB_DISABLE_XET='1', HF_HUB_DOWNLOAD_TIMEOUT='45',
                  HF_HUB_ETAG_TIMEOUT='25', YOLO_AUTOINSTALL='false',
                  YOLO_CONFIG_DIR=str(OUT / 'runtime_settings'), TOKENIZERS_PARALLELISM='false')
MODELS = {
    'project_baseline': ('yolo', ROOT / 'hk-tree-vmms/derived/training_runs/combined_tree_segmentation/yolo11m_seg_1024_20260901_104757/weights/best.pt'),
    'project_review_v2': ('yolo', ROOT / 'hk-tree-vmms/derived/training_runs/combined_tree_segmentation/yolo11m_seg_active_review_v2_20260906/weights/best.pt'),
    'urban_tree_early': ('yolo', ROOT / 'models/yolo_runs/urban_tree_seg_v1/weights/best.pt'),
    'yolo_world_l': ('world', 'yolov8l-worldv2.pt'),
    'yoloe_11_l': ('yoloe', 'yoloe-11l-seg.pt'),
    'yoloe_26_l': ('yoloe', 'yoloe-26l-seg.pt'),
    'grounding_dino_tiny': ('gdino', 'IDEA-Research/grounding-dino-tiny'),
    'grounding_dino_base': ('gdino', 'IDEA-Research/grounding-dino-base'),
    'owlv2_base': ('owl', 'google/owlv2-base-patch16-ensemble'),
}
QUERIES = [['tree'], ['tree', 'palm tree']]


def now():
    return datetime.now(timezone.utc).isoformat()


def dump(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def label_boxes(path):
    boxes = []
    for line in Path(path).read_text(encoding='utf-8-sig').splitlines():
        v = [float(x) for x in line.split()]
        if not v:
            continue
        if len(v) == 5:
            _, x, y, w, h = v
            boxes.append([x-w/2, y-h/2, x+w/2, y+h/2])
        elif len(v) >= 7 and len(v) % 2 == 1:
            xs, ys = v[1::2], v[2::2]
            boxes.append([min(xs), min(ys), max(xs), max(ys)])
        else:
            raise ValueError(f'Malformed label: {path}')
    if any(not (0 <= b[0] < b[2] <= 1.00001 and 0 <= b[1] < b[3] <= 1.00001) for b in boxes):
        raise ValueError(f'Invalid reference box: {path}')
    return boxes


def prepare():
    if (OUT / 'manifest.json').exists():
        return
    OUT.mkdir(parents=True, exist_ok=True)
    with (DATA / 'manifest.csv').open(encoding='utf-8-sig', newline='') as f:
        all_rows = list(csv.DictReader(f))
    training_blocks = {r['route_block_id'] for r in all_rows if r['split'] == 'train'}
    rows, provenance = [], {}
    for r in all_rows:
        if r['split'] not in ('val', 'test'):
            continue
        p, label = Path(r['image']), Path(r['label'])
        old = OLD_DATA / 'records' / r['split'] / (p.stem.split('_', 2)[2] + '.json')
        record = read(old) if old.exists() else {}
        source_kind = record.get('source_kind', 'unverified_legacy')
        provenance[source_kind] = provenance.get(source_kind, 0) + 1
        snap = OUT / 'reference_snapshot' / r['split'] / label.name
        snap.parent.mkdir(parents=True, exist_ok=True)
        # Data snapshot, not a model backup. Never alter original labels.
        snap.write_bytes(label.read_bytes())
        rows.append({**r, 'id': p.stem, 'image_sha256': sha(p), 'label_sha256': sha(label),
                     'label_snapshot': str(snap), 'reference_provenance': source_kind,
                     'overlaps_training_block': r['route_block_id'] in training_blocks})
    if any(r['overlaps_training_block'] for r in rows):
        raise RuntimeError('Route block overlap detected; audit before continuing')
    split_hashes = {s: {r['image_sha256'] for r in rows if r['split'] == s} for s in ['val', 'test']}
    if split_hashes['val'] & split_hashes['test']:
        raise RuntimeError('Exact val/test image overlap')
    protected = [Path(v) for k, v in MODELS.values() if k == 'yolo']
    protected += [ROOT / p for p in ['tree_species_teacher_demo/engine.py', 'tree_species_demo/engine.py',
                  'tools/tree_selection_preview_server.py', 'outputs/tree_target_demo_v1/selection.js']]
    dump(OUT / 'protected_artifacts.json', [{'path': str(p), 'sha256': sha(p), 'bytes': p.stat().st_size} for p in protected])
    dump(OUT / 'manifest.json', rows)
    dump(OUT / 'protocol.json', {'created_at': now(), 'seed': 20260909, 'models': {k: [v[0], str(v[1])] for k, v in MODELS.items()},
         'queries': QUERIES, 'strategies': ['full', 'full_plus_tiles'], 'prediction_floor': .05,
         'score_threshold': .4, 'nms_iou': .5, 'matching_iou': .5,
         'selection': 'validation reference F1@0.4, then recall, then lower latency',
         'reference_provenance_counts': provenance,
         'warning': 'Legacy, partly draft-derived, potentially incomplete reference boxes. NOT fresh human-ground-truth accuracy.',
         'manifest_sha256': sha(OUT / 'manifest.json'), 'training': False})
    dump(OUT / 'STATUS.json', {'stage': 'prepared', 'updated_at': now()})
    print('Prepared', len(rows), provenance, flush=True)


def area(b):
    return max(0, b[2]-b[0]) * max(0, b[3]-b[1])


def intersection(a, b):
    return max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))


def iou(a, b):
    inter = intersection(a, b)
    return inter / max(1e-12, area(a)+area(b)-inter)


def nms(preds, threshold=.5):
    result = []
    for p in sorted(preds, key=lambda p: -p['score']):
        if not any(iou(p['box'], q['box']) > threshold for q in result):
            result.append(p)
    return result


def metrics(records, threshold=.4):
    tp = fp = nref = covered = vertical = 0
    ranked, times = [], []
    per_image = []
    for r in records:
        gt, predictions = r['reference'], r['predictions']
        nref += len(gt)
        used, image_tp = set(), 0
        for p in sorted(predictions, key=lambda p: -p['score']):
            matches = [(iou(p['box'], b), j) for j, b in enumerate(gt) if j not in used]
            score, j = max(matches, default=(0, -1))
            good = score >= .5
            if good:
                used.add(j)
            ranked.append((p['score'], int(good)))
            if p['score'] >= threshold:
                tp += int(good)
                fp += int(not good)
                image_tp += int(good)
        selected = [p for p in predictions if p['score'] >= threshold]
        for b in gt:
            candidates = [p['box'] for p in selected if iou(p['box'], b) >= .5 and area(p['box']) <= 2*area(b)]
            covered += int(any(intersection(b, p)/max(area(b), 1e-12) >= .9 for p in candidates))
            vertical += int(any(p[1] <= b[1]+.05*(b[3]-b[1]) and p[3] >= b[3]-.05*(b[3]-b[1]) for p in candidates))
        times.append(r['seconds'])
        per_image.append({'id': r['id'], 'reference_count': len(gt), 'prediction_count': len(selected), 'matched': image_tp})
    precision, recall = tp/max(1, tp+fp), tp/max(1, nref)
    acc_tp = acc_fp = 0
    pr = []
    for _, good in sorted(ranked, reverse=True):
        acc_tp += good
        acc_fp += 1-good
        pr.append((acc_tp/max(1, nref), acc_tp/(acc_tp+acc_fp)))
    ap = sum(max((p for r, p in pr if r >= t/100), default=0) for t in range(101))/101 if nref else 0
    sorted_times = sorted(times[1:] or times)
    return {'frames': len(records), 'reference_instances': nref, 'tp_reference': tp, 'unmatched_predictions': fp,
            'missed_reference': nref-tp, 'reference_precision': precision, 'reference_recall': recall,
            'reference_f1': 2*precision*recall/max(1e-12, precision+recall), 'reference_ap50_floor005': ap,
            'reference_box_coverage90': covered/max(1, nref), 'reference_vertical_coverage': vertical/max(1, nref),
            'median_seconds': sorted_times[len(sorted_times)//2] if sorted_times else None,
            'p95_seconds': sorted_times[min(len(sorted_times)-1, int(.95*len(sorted_times)))] if sorted_times else None,
            'per_image': per_image}


class Detector:
    def __init__(self, key):
        import torch
        self.torch = torch
        self.kind, spec = MODELS[key]
        self.spec = str(spec)
        self.key = key
        self.query = None
        assets = OUT / 'assets'
        assets.mkdir(exist_ok=True)
        os.chdir(assets)
        self.meta = {'kind': self.kind, 'spec': self.spec, 'torch': torch.__version__, 'started_at': now()}
        torch.set_num_threads(6)
        if self.kind in ('yolo', 'world', 'yoloe'):
            import ultralytics
            self.meta['ultralytics'] = ultralytics.__version__
            if self.kind != 'yolo':
                import clip
                from functools import partial
                clip.load = partial(clip.load, download_root=str(assets / 'clip'))
            cls = {'yolo': ultralytics.YOLO, 'world': ultralytics.YOLOWorld, 'yoloe': ultralytics.YOLOE}[self.kind]
            self.model = cls(self.spec)
            # Move before encoding prompts: otherwise cached CLIP device metadata
            # remains CPU after predictor moves its parent to CUDA.
            self.model.to('cuda')
            self.meta['names_before_prompt'] = self.model.names
            if self.kind == 'yolo' and not any('tree' in str(n).lower() for n in self.model.names.values()):
                raise RuntimeError('Local checkpoint does not expose a tree category')
            weights = Path(self.spec)
            self.meta['weight_sha256'] = sha(weights)
        else:
            from huggingface_hub import HfApi, snapshot_download
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            import transformers
            self.meta['transformers'] = transformers.__version__
            model_dir = assets / key
            if os.environ.get('TREE_BENCH_OFFLINE') == '1':
                self.meta['revision'] = completed(key, 'val')['metadata']['revision']
            else:
                revision = HfApi().model_info(self.spec, timeout=25).sha
                self.meta['revision'] = revision
                snapshot_download(self.spec, revision=revision, local_dir=model_dir, max_workers=2,
                                  allow_patterns=['*.json', '*.txt', '*.safetensors', '*.model', 'README.md', 'LICENSE*'])
            self.processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_dir, local_files_only=True,
                            trust_remote_code=False).to('cuda').eval()
            self.meta['weights'] = [{'path': str(p), 'sha256': sha(p)} for p in model_dir.glob('*.safetensors')]
        torch.cuda.reset_peak_memory_stats()

    def set_query(self, query):
        if query == self.query:
            return
        if self.kind in ('world', 'yoloe'):
            self.model.set_classes(query)
        self.query = query

    def predict(self, image):
        t = self.torch
        w, h = image.size
        if self.kind in ('yolo', 'world', 'yoloe'):
            r = self.model.predict(image, imgsz=1024, conf=.05, iou=.5, device=0, verbose=False,
                                   agnostic_nms=True, max_det=200)[0]
            boxes = r.boxes.xyxy.cpu().tolist()
            scores = r.boxes.conf.cpu().tolist()
        else:
            text = '. '.join(self.query) + '.' if self.kind == 'gdino' else [self.query]
            inputs = self.processor(images=image, text=text, return_tensors='pt').to('cuda')
            with t.inference_mode():
                output = self.model(**inputs)
            if self.kind == 'gdino':
                method = self.processor.post_process_grounded_object_detection
                parameter = 'threshold' if 'threshold' in inspect.signature(method).parameters else 'box_threshold'
                r = method(output, inputs.input_ids, **{parameter: .05, 'text_threshold': .25, 'target_sizes': [(h, w)]})[0]
            else:
                r = self.processor.post_process_object_detection(output, threshold=.05, target_sizes=t.tensor([[h, w]], device='cuda'))[0]
            boxes, scores = r['boxes'].cpu().tolist(), r['scores'].cpu().tolist()
        result = []
        for b, s in zip(boxes, scores):
            b = [max(0., min(1., v/d)) for v, d in zip(b, [w, h, w, h])]
            if area(b) > 0 and s >= .05:
                result.append({'box': b, 'score': float(s)})
        return nms(result)


def image_predictions(detector, image, tiles):
    import torch
    torch.cuda.synchronize()
    start = time.perf_counter()
    full = detector.predict(image)
    torch.cuda.synchronize()
    full_seconds = time.perf_counter()-start
    joined = list(full)
    if tiles:
        w, h = image.size
        for x0, x1 in [(0, .55), (.225, .775), (.45, 1)]:
            left, right = round(x0*w), round(x1*w)
            crop = image.crop((left, 0, right, h))
            for p in detector.predict(crop):
                a, b, c, d = p['box']
                joined.append({'box': [(left+a*(right-left))/w, b, (left+c*(right-left))/w, d], 'score': p['score']})
    torch.cuda.synchronize()
    return full, nms(joined), full_seconds, time.perf_counter()-start


def worker(key, split):
    from PIL import Image
    rows = [r for r in read(OUT / 'manifest.json') if r['split'] == split]
    folder = OUT / 'attempts' / key / (split + '_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    folder.mkdir(parents=True, exist_ok=False)
    dump(folder / 'START.json', {'pid': os.getpid(), 'created_at': now(), 'split': split, 'model': key, 'script_sha256': sha(__file__)})
    try:
        detector = Detector(key)
        queries = QUERIES if detector.kind != 'yolo' else [QUERIES[0]]
        locked_strategy = None
        if split == 'test':
            lockfile = read(OUT / 'selection_lock.json')
            if lockfile['manifest_sha256'] != sha(OUT / 'manifest.json'):
                raise RuntimeError('Manifest changed after validation selection')
            lock = lockfile['models'][key]
            queries = [lock['query']]
            locked_strategy = lock['strategy']
        results = {}
        for query in queries:
            detector.set_query(query)
            qkey = '_'.join(query).replace(' ', '-')
            modes = [locked_strategy] if locked_strategy else ['full', 'full_plus_tiles']
            for mode in modes:
                results[qkey+'__'+mode] = {'query': query, 'strategy': mode, 'records': []}
            for i, row in enumerate(rows):
                if sha(row['image']) != row['image_sha256'] or sha(row['label_snapshot']) != row['label_sha256']:
                    raise RuntimeError('Input content changed after snapshot')
                with Image.open(row['image']) as original:
                    original_size = original.size
                    image = original.convert('RGB')
                image.thumbnail((4096, 4096), Image.Resampling.LANCZOS)
                full, tiled, seconds, total = image_predictions(detector, image, 'full_plus_tiles' in modes)
                # Test references are read only after the global validation selection lock exists.
                gt = label_boxes(row['label_snapshot'])
                for mode in modes:
                    rec = {'id': row['id'], 'reference': gt, 'predictions': full if mode == 'full' else tiled,
                           'seconds': seconds if mode == 'full' else total, 'original_size': original_size,
                           'inference_source_size': image.size, 'route_block_id': row['route_block_id'],
                           'reference_provenance': row['reference_provenance']}
                    results[qkey+'__'+mode]['records'].append(rec)
                dump(folder / 'partial.json', results)
                print(key, split, qkey, f'{i+1}/{len(rows)}', 'full boxes', len(full), 'tiled', len(tiled), flush=True)
        for val in results.values():
            val['metrics'] = metrics(val['records'])
            if split == 'val':
                val['threshold_sweep'] = {str(t): metrics(val['records'], t) for t in [.2, .3, .4, .5, .6]}
        detector.meta['peak_cuda_allocated_bytes'] = detector.torch.cuda.max_memory_allocated()
        dump(folder / 'results.json', {'model': key, 'split': split, 'metadata': detector.meta, 'policies': results})
        dump(folder / 'COMPLETE.json', {'finished_at': now()})
        dump(OUT / 'completed' / (key+'_'+split+'.json'), {'attempt': str(folder), 'results': str(folder / 'results.json')})
    except Exception:
        dump(folder / 'ERROR.json', {'failed_at': now(), 'traceback': traceback.format_exc()})
        traceback.print_exc()
        raise


def completed(key, split):
    p = OUT / 'completed' / (key+'_'+split+'.json')
    return read(read(p)['results']) if p.exists() else None


def lock_selection():
    if (OUT / 'selection_lock.json').exists():
        return
    selected = {}
    for key in MODELS:
        result = completed(key, 'val')
        if not result:
            continue
        best = max(result['policies'].values(), key=lambda x: (x['metrics']['reference_f1'], x['metrics']['reference_recall'], -x['metrics']['median_seconds']))
        selected[key] = {'query': best['query'], 'strategy': best['strategy'], 'validation_metrics': best['metrics']}
    if not selected:
        raise RuntimeError('No successful model to lock')
    dump(OUT / 'selection_lock.json', {'locked_at': now(), 'manifest_sha256': sha(OUT / 'manifest.json'),
                                     'criterion': 'validation reference F1 at 0.4', 'models': selected})


def overlays():
    from PIL import Image, ImageDraw
    rows = {r['id']: r for r in read(OUT / 'manifest.json')}
    for key in MODELS:
        for split in ('val', 'test'):
            result = completed(key, split)
            if not result:
                continue
            policies = result['policies']
            if split == 'val':
                best = max(policies.values(), key=lambda p: p['metrics']['reference_f1'])
            else:
                best = next(iter(policies.values()))
            dest = OUT / 'overlays' / key / split
            dest.mkdir(parents=True, exist_ok=True)
            cases = best['records'] if split == 'test' else best['records'][::max(1, len(best['records'])//8)][:8]
            for rec in cases:
                target = dest / (rec['id']+'.jpg')
                if target.exists():
                    continue
                with Image.open(rows[rec['id']]['image']) as im:
                    im.thumbnail((1600, 900))
                    canvas = im.convert('RGB')
                draw = ImageDraw.Draw(canvas)
                w, h = canvas.size
                for b in rec['reference']:
                    draw.rectangle([b[0]*w, b[1]*h, b[2]*w, b[3]*h], outline='#26d06f', width=3)
                for p in rec['predictions']:
                    if p['score'] < .4:
                        continue
                    b = p['box']
                    draw.rectangle([b[0]*w, b[1]*h, b[2]*w, b[3]*h], outline='#ff9e25', width=2)
                    draw.text((b[0]*w, b[1]*h), f"{p['score']:.2f}", fill='white', stroke_width=1, stroke_fill='black')
                draw.rectangle((0, h-25, w, h), fill='black')
                draw.text((5, h-20), f'{key} | green: historical reference (possibly incomplete); orange: prediction >=0.4', fill='white')
                canvas.save(target, quality=88)


def report():
    data = []
    for key in MODELS:
        for split in ['val', 'test']:
            r = completed(key, split)
            if r:
                for name, policy in r['policies'].items():
                    data.append({'model': key, 'split': split, 'policy': name, **{k: v for k, v in policy['metrics'].items() if k != 'per_image'}})
    checks = [{**p, 'unchanged': sha(p['path']) == p['sha256']} for p in read(OUT / 'protected_artifacts.json')]
    errors = [{'path': str(p), **read(p)} for p in (OUT / 'attempts').glob('*/*/ERROR.json')]
    dump(OUT / 'results_summary.json', {'updated_at': now(), 'results': data, 'protected_artifacts': checks, 'errors': errors})
    lines = ['# 树木检测器多模型实测报告（自动汇总，待案例审查）', '', f'更新时间：{now()}', '',
             '## 重要边界', '',
             '- 本轮仅推理，无训练、无权重替换、无 Demo 部署。',
             '- 61 帧验证与 74 帧历史回归；验证选择参数后锁定，回归结果不用于重新挑选参数。',
             '- 历史参考含草稿来源且可能漏标。以下 P/R/F1/AP 是参考标签一致性，不是人工穷尽真值精度；额外框不一定误检。',
             '- 40% 是检测分数阈值，不是校准后的正确概率。不同模型分数不可直接解释为同等可靠性。',
             '- Coverage90 / Vertical 是相对参考框的覆盖代理，不证明真实树干和树冠完整。',
             '- 图像上限 4096 像素；YOLO 输入 1024；HF 使用官方处理器。计时包含处理器和推理，不含磁盘读图/首次加载，排除每策略第一帧。',
             '- 完整比较与人工案例审查前不宣布最佳部署模型。', '',
             '## 固定 0.4 阈值结果', '',
             '| 模型 | 数据 | 策略 | 参考 P | 参考 R | 参考 F1 | AP50* | Coverage90 | 中位秒/帧 |',
             '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for r in data:
        lines.append(f"| {r['model']} | {r['split']} | {r['policy']} | {r['reference_precision']:.1%} | {r['reference_recall']:.1%} | {r['reference_f1']:.1%} | {r['reference_ap50_floor005']:.1%} | {r['reference_box_coverage90']:.1%} | {r['median_seconds']:.3f} |")
    lines += ['', '*AP50 为统一匹配实现、预测保留分数下限 0.05 的 101 点插值，不等同官方完整 COCO mAP。', '',
              '## 自检与异常', '', f'- 受保护文件全部未变：{all(x["unchanged"] for x in checks)}',
              f'- 已记录失败尝试：{len(errors)}。详细原因见 results_summary.json。',
              '- 下载失败不算模型精度差；不把失败模型排到精度榜末尾。', '',
              '## 案例图', '', '见 overlays/：绿色为历史参考框，橙色为真实模型预测。图像只在本地，不公开发布。', '',
              '## 来源', '',
              '- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO)',
              '- [YOLO-World](https://docs.ultralytics.com/models/yolo-world)',
              '- [YOLOE](https://docs.ultralytics.com/models/yoloe)',
              '- [OWLv2](https://huggingface.co/google/owlv2-base-patch16-ensemble)', '',
              '本文件由脚本逐阶段更新；最终人工/agent 审查结论另见 FINAL_REPORT.md。']
    (OUT / 'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def run():
    prepare()
    # A controller can be restarted after a code fix while its last worker
    # finishes. Wait for that exact experiment worker instead of double-running GPU jobs.
    import psutil
    for process in psutil.process_iter(['pid', 'cmdline']):
        command = process.info['cmdline'] or []
        if process.pid != os.getpid() and '--worker' in command and any('benchmark_tree_detectors.py' in s for s in command):
            print('Waiting for existing benchmark worker', process.pid, flush=True)
            try:
                process.wait(timeout=1200)
            except psutil.TimeoutExpired:
                raise RuntimeError('Existing worker still alive; no duplicate controller work started')
    for split in ['val', 'test']:
        if split == 'test':
            lock_selection()
            keys = list(read(OUT / 'selection_lock.json')['models'])
        else:
            keys = list(MODELS)
        for key in keys:
            if completed(key, split):
                continue
            dump(OUT / 'STATUS.json', {'stage': split, 'model': key, 'pid': os.getpid(), 'updated_at': now()})
            log = OUT / 'logs' / (key+'_'+split+'_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'.log')
            log.parent.mkdir(exist_ok=True)
            with log.open('w', encoding='utf-8') as f:
                try:
                    p = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', key, '--split', split],
                         cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, timeout=1200, env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
                    print(key, split, 'exit', p.returncode, flush=True)
                except subprocess.TimeoutExpired:
                    print(key, split, 'TIMEOUT', flush=True)
                    dump(OUT / 'controller_errors' / (key+'_'+split+'.json'), {'reason': 'worker exceeded 20 minutes', 'log': str(log)})
            report()
    overlays()
    for key in read(OUT / 'selection_lock.json')['models']:
        if (OUT / 'offline_checks' / (key+'.json')).exists():
            continue
        with (OUT / 'logs' / (key+'_offline.log')).open('w', encoding='utf-8') as f:
            try:
                subprocess.run([sys.executable, str(Path(__file__).resolve()), '--offline-smoke', key],
                               cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, timeout=180)
            except subprocess.TimeoutExpired:
                dump(OUT / 'offline_checks' / (key+'.json'), {'success': False, 'reason': 'offline check timed out'})
    report()
    dump(OUT / 'STATUS.json', {'stage': 'experiments_complete_needs_case_audit', 'updated_at': now(), 'pid': os.getpid()})


def offline_smoke(key):
    """One validation-image inference with socket connections deliberately blocked."""
    import socket
    from PIL import Image
    os.environ.update(TREE_BENCH_OFFLINE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    def deny(*args, **kwargs):
        raise RuntimeError('Network deliberately blocked for offline smoke test')
    socket.socket.connect = deny
    result = {'model': key, 'checked_at': now(), 'network_blocked': True}
    try:
        d = Detector(key)
        d.set_query(['tree'])
        row = next(r for r in read(OUT / 'manifest.json') if r['split'] == 'val')
        with Image.open(row['image']) as im:
            im = im.convert('RGB')
            im.thumbnail((2048, 2048))
            preds = d.predict(im)
        result.update(success=True, prediction_count=len(preds))
    except Exception:
        result.update(success=False, traceback=traceback.format_exc())
    dump(OUT / 'offline_checks' / (key+'.json'), result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--report', action='store_true')
    parser.add_argument('--overlays', action='store_true')
    parser.add_argument('--worker', choices=list(MODELS))
    parser.add_argument('--offline-smoke', choices=list(MODELS))
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    args = parser.parse_args()
    if args.prepare:
        prepare()
    if args.worker:
        worker(args.worker, args.split)
    if args.offline_smoke:
        offline_smoke(args.offline_smoke)
    if args.run:
        run()
    if args.report:
        report()
    if args.overlays:
        overlays()
