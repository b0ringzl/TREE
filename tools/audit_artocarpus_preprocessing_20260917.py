"""Fixed, paired inference audit: legacy center crop versus actual Demo pixels.

No fitting, no threshold search, no test images, no deployed-file writes.
Both branches are re-encoded with the same frozen encoder and deployed MLPs.
The 150-label branch only appends the two pre-existing, fixed text vectors.
"""
import argparse
import ast
import gc
import hashlib
import io
import os
import sys
import time
import warnings
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

os.environ['HF_HUB_OFFLINE'] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageOps

from bioclip_probe import SOURCE, TEACHER
from distillation_core import (bootstrap_gain, canonical_image, collapse_logits,
                               digest, grouped_metrics, metrics, read_json, save_json)
from train_sstv1_incremental import prepare_pixels
from training_safety import sha256
from tree_species_teacher_demo.engine import Engine, SSTV1_MANIFEST, decode

PREVIOUS = ROOT / 'outputs/SST_Conserve_Browser_20260917_v3'
DEFAULT_OUT = ROOT / 'outputs/artocarpus_preprocess_audit_20260917_v1'
SEED = 20260917
BRANCHES = ('center_crop', 'demo_letterbox')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def exact_demo_prepare():
    """Execute the actual pure prepare function, without importing server routers."""
    path = ROOT / 'spacescan_trees_demo_v1/server.py'
    parsed = ast.parse(path.read_text(encoding='utf-8'))
    nodes = [n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name == 'prepare']
    require(len(nodes) == 1, 'Expected exactly one production prepare function')
    scope = dict(io=io, warnings=warnings, Image=Image, ImageOps=ImageOps, MAX_BYTES=20 * 1024 * 1024)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)
    return scope['prepare']


def protected_paths():
    dep = read_json(SSTV1_MANIFEST)
    code = ['spacescan_trees_demo_v1/server.py', 'spacescan_trees_demo_v1/tree_detection.py',
            'spacescan_trees_demo_v1/panorama.py', 'spacescan_trees_demo_v1/static/polygon-crop.js',
            'spacescan_trees_demo_v1/static/tree-selection.js', 'tree_species_teacher_demo/engine.py',
            'tools/train_sstv1_incremental.py', 'tools/distillation_core.py', 'tools/run_distillation.py',
            'tools/bioclip_probe.py', 'tools/audit_artocarpus_preprocessing_20260917.py']
    paths = [ROOT / x for x in code] + [SSTV1_MANIFEST, TEACHER,
        SOURCE / 'planning_rows.json', SOURCE / 'classes.json', SOURCE / 'aliases.json',
        SOURCE / 'teacher_cache/bioclip25/planning.npz',
        SOURCE / 'teacher_cache/bioclip25/metadata.json',
        PREVIOUS / 'new_manifest.json', PREVIOUS / 'new_features.npz',
        PREVIOUS / 'labels.json', PREVIOUS / 'accepted_replay_manifest.json']
    paths += [SSTV1_MANIFEST.parent / r['file'] for r in dep['members']]
    return paths


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    old = read_json(SOURCE / 'planning_rows.json')
    new = [r for r in read_json(PREVIOUS / 'new_manifest.json') if r['phase'] == 'planning']
    require(len(old) == 2640 and len(new) == 15, 'Fixed audit cohort changed')
    require(all(r['phase'] == 'planning' for r in old), 'Non-planning row in cohort')
    rows = [dict(r, audit_cohort='old_planning') for r in old]
    rows += [dict(r, audit_cohort='new_source_development') for r in new]
    require(len({r['sample_id'] for r in rows}) == len(rows), 'Duplicate sample identifier')
    replay = read_json(PREVIOUS / 'accepted_replay_manifest.json')
    replay += [r for r in read_json(PREVIOUS / 'new_manifest.json') if r['phase'] == 'train']
    fit_sha = {r.get('sha256', r.get('source_sha256')) for r in replay}
    fit_groups = {r['group_id'] for r in replay}
    for row in rows:
        path = Path(row['source_image'])
        require(path.is_file(), 'Missing fixed audit image: ' + str(path))
        actual = sha256(path)
        expected = row.get('sha256', row.get('source_sha256'))
        require(actual == expected, 'Source image differs from fixed manifest: ' + str(path))
        row['audit_sha256'] = actual
        row['fit_sha_overlap'] = actual in fit_sha
        row['fit_group_overlap'] = row['group_id'] in fit_groups
        if row.get('materialized_path'):
            materialized = sha256(row['materialized_path'])
            require(actual == materialized, 'Legacy materialized image differs; cannot isolate preprocessing')
            row['materialized_sha256'] = materialized
    require(not any(r['fit_sha_overlap'] or r['fit_group_overlap'] for r in rows),
            'Audit cohort overlaps the previous study fitting set')
    before = {str(p.resolve()): sha256(p) for p in protected_paths()}
    dep = read_json(SSTV1_MANIFEST)
    labels150 = read_json(PREVIOUS / 'labels.json')
    require(len(labels150) == 150 and labels150[16] == 'Artocarpus hypargyreus', 'Unexpected taxonomy')
    save_json(out / 'rows.json', rows)
    save_json(out / 'labels150.json', labels150)
    save_json(out / 'deployment_snapshot.json', dep)
    groups, hashes = defaultdict(list), defaultdict(list)
    for i, r in enumerate(rows):
        groups[r['group_id']].append(i)
        hashes[r['audit_sha256']].append(i)
    save_json(out / 'duplicate_groups.json', {
        'source_groups': dict(groups), 'exact_sha_groups': dict(hashes),
        'individual_identity_warning': 'Source observations/galleries are conservative proxies, not verified independent trees.',
        'within_cohort_sha_duplicates': {k: v for k, v in hashes.items() if len(v) > 1},
        'fit_sha_overlap': 0, 'fit_source_group_overlap': 0})
    protocol = {
        'created_at': datetime.now(timezone.utc).isoformat(), 'seed': SEED,
        'purpose': 'Paired classifier-entry preprocessing audit; no model fitting or model selection.',
        'cohort': {'old_planning': 2640, 'new_source_development': 15},
        'cohort_selection': 'All rows of the two fixed planning cohorts, before inference; no test set read.',
        'rows_digest': digest(rows), 'protected_sha256_before': before,
        'branches': {
            'center_crop': 'Current distillation_core.canonical_image: EXIF/RGB, bicubic shortest side224, center224. Same source bytes as materialized legacy image.',
            'demo_letterbox': 'Actual server.prepare via AST: EXIF/RGB, longest edge<=1024, gray128 square letterbox bicubic, JPEG95, then exact engine.decode224.'},
        'pixels_parity_check': 'For every row, actual server.prepare+decode must be bitwise equal to train_sstv1_incremental.prepare_pixels.',
        'classifier': 'Same frozen encoder and same three deployed SSTv1 MLP members, calibrated probability mean; x=e.half().float and raw z from float32 encoder outputs.',
        'batch_size': 1, 'temperature': dep['temperature'], 'deployed_version': dep['version'],
        'fair150': 'Append the two already-existing extension text vectors from prior new_features.npz; float32 renormalize, fixed scale and temperature, zero learned residual. Not deployed.',
        'new15_authority': 'Institutional source plus agent visual/source consistency; not all human-certified, not a fresh test set. No label changes in this audit.',
        'scope': 'Input to classifier after any target selection. Does not rerun detector or infer manual polygons; does not measure detector quality or geometric reprojection quality.',
        'manual_crop': 'Browser keeps native bbox pixels, polygon exterior gray128, outputs PNG. Manual panorama crop is not perspective-corrected.',
        'automatic_crop': 'Detector selects full-resolution image/view crop with 4% context; JPEG95 before server.prepare. Thumbnail resize is preview-only.',
        'panorama': 'ERP converted to perspective views before automatic detection; selected perspective crop then same classifier preparation.',
        'bootstrap': 'Paired source-group cluster bootstrap, 2000 repeats, seed20260917. Groups are not proven independent trees.',
        'limitations': ['Historical planning data and previously inspected new development galleries, not unbiased final test evidence.',
                       'Padding changes retained context, effective object resolution and JPEG pixels together; no assertion it must improve.',
                       'Only two fixed preprocessing branches. No threshold, temperature, head or encoder optimization.',
                       'Native148 lacks Rhodoleia championii and Keteleeria fortunei; their five images excluded from native multiclass accuracy, included in binary white-guimu false-positive diagnostics.'],
        'trained': False, 'deployed': False}
    save_json(out / 'protocol.json', protocol)
    print('PREPARED', len(rows), 'pairs; protected files', len(before), flush=True)
    return rows, protocol


def pixels_for_row(row, production_prepare):
    path = Path(row['source_image'])
    content = path.read_bytes()
    require(hashlib.sha256(content).hexdigest() == row['audit_sha256'], 'Image mutated during audit')
    padded_bytes, width, height = production_prepare(content)
    padded, quality, _ = decode(padded_bytes)
    center = canonical_image(path)
    helper = prepare_pixels(path)
    require(torch.equal(padded, helper), 'Production/helper pixel mismatch: ' + row['sample_id'])
    with Image.open(io.BytesIO(content)) as original:
        exif_orientation = int(original.getexif().get(274, 1))
    ratio = min(width, height) / max(width, height)
    diagnostic = {'sample_id': row['sample_id'], 'sha256': row['audit_sha256'],
        'width': width, 'height': height, 'exif_orientation': exif_orientation,
        'center_retained_area_fraction_ideal': ratio, 'letterbox_empty_area_fraction_ideal': 1 - ratio,
        'aspect_band': 'elongated_below_0.75' if ratio < .75 else 'near_square_at_least_0.75',
        'demo_jpeg_sha256': hashlib.sha256(padded_bytes).hexdigest(),
        'pixel_hash_format': 'C-contiguous float32 RGB CHW [0,1],224x224, before CLIP normalization',
        'center_pixel_sha256': hashlib.sha256(center.numpy().tobytes()).hexdigest(),
        'demo_pixel_sha256': hashlib.sha256(padded.numpy().tobytes()).hexdigest(),
        'production_helper_pixels_equal': True, 'demo_decode_quality': quality}
    return center, padded, diagnostic


def probabilities(engine, embedding, raw, extension):
    x = embedding.half().float()
    scale = engine.model.logit_scale.exp().float()
    extra = x @ engine.extra_bank.T * scale
    novel = x @ extension.T * scale
    members = torch.stack([torch.cat([collapse_logits(h(x, raw), engine.classes['groups']), extra, novel], -1)
                           / engine.choice['calibration_temperature'] for h in engine.sstv1_heads])
    return members[..., :148].softmax(-1).mean(0), members.softmax(-1).mean(0)


def pixel_image(tensor):
    return Image.fromarray(np.rint(tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8))


def extract(out, rows, protocol):
    chunks = out / 'chunks'; chunks.mkdir(exist_ok=False)
    previews = out / 'white_guimu_inputs'; previews.mkdir(exist_ok=False)
    production_prepare = exact_demo_prepare()
    engine = Engine(); engine.load()
    dep = read_json(out / 'deployment_snapshot.json')
    labels = read_json(out / 'labels150.json')
    require(engine.sstv1_state == 'ready' and engine.sstv1_version == dep['version'], 'Deployed model mismatch')
    require(engine.labels == labels[:148] and engine.encoder_sha256 == dep['encoder_sha256'], 'Model contract mismatch')
    with np.load(PREVIOUS / 'new_features.npz') as previous:
        require(str(previous['encoder_sha256']) == engine.encoder_sha256, 'Novel text encoder mismatch')
        extension = F.normalize(torch.tensor(previous['extension_bank'], device='cuda', dtype=torch.float32), dim=-1)
        require(np.isclose(float(previous['logit_scale']), engine.model.logit_scale.exp().float().item(), atol=1e-5), 'Scale mismatch')
    diagnostics = []
    started = time.perf_counter()
    try:
        with ThreadPoolExecutor(max_workers=4) as pool, torch.inference_mode():
            for start in range(0, len(rows), 64):
                current = rows[start:start + 64]
                inputs = list(pool.map(lambda r: pixels_for_row(r, production_prepare), current))
                arrays = {f'{b}_{k}': [] for b in BRANCHES for k in ('x', 'z', 'p148', 'p150')}
                for offset, (center, padded, diagnostic) in enumerate(inputs):
                    row = current[offset]
                    diagnostics.append(diagnostic)
                    if row['target'] == 16:
                        canvas = Image.new('RGB', (448, 246), 'white')
                        canvas.paste(pixel_image(center), (0, 22)); canvas.paste(pixel_image(padded), (224, 22))
                        ImageDraw.Draw(canvas).text((4, 3), 'Center crop                  Demo letterbox', fill='black')
                        canvas.save(previews / (row['sample_id'] + '.png'))
                    for branch, pixel in zip(BRANCHES, (center, padded)):
                        embedding = engine.model.encode_image(((pixel[None].cuda() - engine.mean) / engine.std).to(engine.dtype), normalize=True).float()
                        raw = embedding @ engine.bank.T * engine.model.logit_scale.exp().float()
                        p148, p150 = probabilities(engine, embedding, raw, extension)
                        for key, value in [('x', embedding.half()), ('z', raw), ('p148', p148), ('p150', p150)]:
                            arrays[f'{branch}_{key}'].append(value.cpu().numpy())
                stop = start + len(current)
                np.savez_compressed(chunks / f'{start:06d}_{stop:06d}.npz',
                    **{k: np.concatenate(v) for k, v in arrays.items()}, rows_hash=digest(current))
                save_json(chunks / f'{start:06d}_{stop:06d}_diagnostics.json', diagnostics[start:stop])
                print('ENCODED', stop, '/', len(rows), 'pairs; seconds', round(time.perf_counter() - started, 1), flush=True)
    finally:
        del engine, extension
        gc.collect(); torch.cuda.empty_cache()
    combined = defaultdict(list)
    for path in sorted(chunks.glob('*.npz')):
        with np.load(path) as data:
            for key in data.files:
                if key != 'rows_hash': combined[key].append(data[key])
    arrays = {k: np.concatenate(v) for k, v in combined.items()}
    np.savez_compressed(out / 'paired_features_predictions.npz', **arrays, rows_hash=digest(rows))
    save_json(out / 'pixel_diagnostics.json', diagnostics)
    return arrays, diagnostics


def binary(y, p, target=16):
    true = np.asarray(y) == target; pred = p.argmax(1) == target
    tp = int((true & pred).sum()); fp = int((~true & pred).sum())
    fn = int((true & ~pred).sum()); tn = int((~true & ~pred).sum())
    return dict(tp=tp, fn=fn, fp=fp, tn=tn, precision=tp/(tp+fp) if tp+fp else 0.,
                recall=tp/(tp+fn) if tp+fn else 0., f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)


def paired(rows, a, b):
    y = np.array([r['target'] for r in rows]); ai = a.argmax(1); bi = b.argmax(1)
    result = {'center': metrics(y, a), 'demo': metrics(y, b),
        'center_white_guimu': binary(y, a), 'demo_white_guimu': binary(y, b),
        'center_aquilaria': binary(y, a, 11), 'demo_aquilaria': binary(y, b, 11),
        'changed_top1': int((ai != bi).sum()), 'corrected': int(((ai != y) & (bi == y)).sum()),
        'regressed': int(((ai == y) & (bi != y)).sum()),
        'corrected_ids': [r['sample_id'] for r, ok in zip(rows, (ai != y) & (bi == y)) if ok],
        'regressed_ids': [r['sample_id'] for r, ok in zip(rows, (ai == y) & (bi != y)) if ok],
        'paired_bootstrap': bootstrap_gain(y, a, b, [r['group_id'] for r in rows], seed=SEED, repeats=2000),
        'center_group_metrics': grouped_metrics(rows, a), 'demo_group_metrics': grouped_metrics(rows, b)}
    return result


def evaluate(out, rows, arrays, diagnostics):
    results = {}
    for cohort in ('old_planning', 'new_source_development'):
        for nclass in (148, 150):
            indices = [i for i, r in enumerate(rows) if r['audit_cohort'] == cohort and r['target'] < nclass]
            rr = [rows[i] for i in indices]
            a, b = [arrays[f'{branch}_p{nclass}'][indices] for branch in BRANCHES]
            results[f'{cohort}_{nclass}'] = paired(rr, a, b)
        indices = [i for i, r in enumerate(rows) if r['audit_cohort'] == cohort]
        y = [rows[i]['target'] for i in indices]
        results[f'{cohort}_148_binary_all'] = {
            branch: binary(y, arrays[f'{branch}_p148'][indices]) for branch in BRANCHES}
    for band in ('elongated_below_0.75', 'near_square_at_least_0.75'):
        indices = [i for i, (r, d) in enumerate(zip(rows, diagnostics)) if r['audit_cohort'] == 'old_planning' and d['aspect_band'] == band]
        results[f'old_planning_148_{band}'] = paired([rows[i] for i in indices],
            arrays['center_crop_p148'][indices], arrays['demo_letterbox_p148'][indices])
    labels = read_json(out / 'labels150.json')
    decisions = []
    for i, row in enumerate(rows):
        result = {'sample_id': row['sample_id'], 'source_image': row['source_image'],
            'sha256': row['audit_sha256'], 'cohort': row['audit_cohort'], 'group_id': row['group_id'],
            'truth_target': row['target'], 'truth_label': labels[row['target']],
            'label_authority': row.get('label_authority', 'existing source label; not independently re-reviewed'),
            'native148_truth_in_catalog': row['target'] < 148}
        for branch in BRANCHES:
            result[branch] = {}
            for nclass in (148, 150):
                p = arrays[f'{branch}_p{nclass}'][i]; order = np.argsort(-p, kind='stable')[:3]
                result[branch][str(nclass)] = {
                    'top3': [{'index': int(j), 'species': labels[j], 'score': float(p[j])} for j in order],
                    'white_guimu_probability': float(p[16]),
                    'correct': bool(order[0] == row['target']) if row['target'] < nclass else None}
        decisions.append(result)
    save_json(out / 'per_image_diagnostics.json', decisions)
    save_json(out / 'evaluation.json', results)
    lines = ['# 白桂木预处理配对审计', '',
        '仅重新推理，未训练、未部署。固定 2640 张旧 planning 与 15 张已查看的新来源开发图；不是新独立测试集。', '',
        '两分支对同一原始字节运行同一冻结编码器、同一 3 个当前部署 MLP：旧短边缩放＋中央裁切，当前 Demo 等比灰补边＋JPEG95＋224。', '',
        '| 数据与目录 | 中央裁切 Top-1 | Demo 补边 Top-1 | 改正 / 退步 | 白桂木 TP / FP：裁切 → 补边 |',
        '|---|---:|---:|---:|---|']
    for key in ['old_planning_148', 'old_planning_150', 'new_source_development_148', 'new_source_development_150']:
        r = results[key]; a, b = r['center'], r['demo']; wa, wb = r['center_white_guimu'], r['demo_white_guimu']
        lines.append(f'| {key} (n={a["n"]}) | {a["top1"]:.2%} | {b["top1"]:.2%} | {r["corrected"]} / {r["regressed"]} | {wa["tp"]}/{wa["fp"]} → {wb["tp"]}/{wb["fp"]} |')
    for key in ['old_planning_148', 'new_source_development_150']:
        r = results[key]; bs = r['paired_bootstrap']
        lines += ['', f'{key}：配对准确率差 {bs["point"]:+.2%}，来源分组 bootstrap 95% 区间 [{bs["ci95"][0]:+.2%}, {bs["ci95"][1]:+.2%}]。',
            f'按来源组平均概率计分：中央裁切 {r["center_group_metrics"]["top1"]:.2%}；补边 {r["demo_group_metrics"]["top1"]:.2%}。']
    lines += ['', '## 边界与结论使用', '',
        '- 本审计是分类器入口对照；未评估检测器、选树、多边形是否包全、全景重投影准确性。',
        '- 新 15 张中 5 张为原 148 类目录以外物种；原目录多分类指标只纳入白桂木 10 张，二元误检诊断仍纳入全部 15 张。150 类只追加上一轮固定文本向量，未重新训练。',
        '- 图库组不是已证实的独立单木。新开发图为来源与形态一致性标记，不冒充全部经过人工鉴定。',
        '- 等比补边保留内容，但树在 224 输入中的有效分辨率可能下降；不能预设补边一定提高精度。',
        '- 历史模型训练混用旧中央裁切缓存与新 Demo 补边数据；如后续统一训练缓存，必须重新提取训练与开发集并在新备份版本拟合，不能据本审计直接替换生产。',
        '- 原图、像素及模型文件 SHA，全部分布与来源分组分别保存在 rows.json、pixel_diagnostics.json、protocol.json、paired_features_predictions.npz、duplicate_groups.json。',
        '- white_guimu_inputs/ 包含全部 16 张白桂木两种真实 224 输入对照（固定真值选图，不按预测优劣筛选）。', '']
    (out / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.out.resolve()
    require(out.is_relative_to(ROOT / 'outputs'), 'Audit output must remain within outputs')
    require(not out.exists(), 'Audit output exists; no silent reuse or overwrite')
    torch.set_num_threads(4)
    torch.manual_seed(SEED); np.random.seed(SEED)
    rows, protocol = prepare(out)
    arrays, diagnostics = extract(out, rows, protocol)
    result = evaluate(out, rows, arrays, diagnostics)
    after = {path: sha256(path) for path in protocol['protected_sha256_before']}
    require(after == protocol['protected_sha256_before'], 'Protected source or deployed weight changed during audit')
    for row in rows:
        require(sha256(row['source_image']) == row['audit_sha256'], 'Source image changed during audit')
    complete = {'completed_at': datetime.now(timezone.utc).isoformat(), 'images': len(rows),
        'encoder_inferences': 2 * len(rows), 'trained': False, 'deployed': False,
        'production_and_source_hashes_unchanged': True, 'all_production_helper_pixels_equal': True,
        'probabilities': 'paired_features_predictions.npz; arrays contain all 148/150 probabilities in labels150 order',
        'old_planning148': {key: result['old_planning_148'][key] for key in ('corrected', 'regressed', 'paired_bootstrap')},
        'protected_sha256_after': after}
    save_json(out / 'COMPLETE.json', complete)
    print('COMPLETE', out, flush=True)
    print((out / 'REPORT.md').read_text(encoding='utf-8'), flush=True)


if __name__ == '__main__':
    main()
