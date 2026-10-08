"""CPU-only fixed two-view DEVELOPMENT diagnostic; never fit, select or deploy.

Both branches use previous best58 + zero novel residuals and the same 150
classes. Equal averaging is fixed in advance, not optimized against these data.
Only the historical 2640+15 development source images/cache are read.
"""
import argparse
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_OFFLINE'] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]

import numpy as np
import torch
from torch.nn.functional import normalize

from bioclip_probe import TEACHER
from distillation_core import digest, metrics, read_json, save_json
from train_conservation_iteration_20260918 import Conservation150
from training_safety import sha256

STUDY = ROOT / 'outputs/SST_Conservation_Iterative_20260918_v1'
PREVIOUS = ROOT / 'outputs/SSTv1_Artocarpus_User_20260918_v1'
CACHE = ROOT / 'outputs/artocarpus_preprocess_audit_20260917_v1'
EXTENSION_SOURCE = ROOT / 'outputs/SST_Conserve_Browser_20260917_v3/new_features.npz'
OUT = ROOT / 'outputs/conservation_fourclass_view_audit_20260918_v1'
FOCUS = [11, 16, 148, 149]
MIX_CENTER_WEIGHT = 0.5
EXPECTED_BEST58_SHA = '9d18b096feb4ef7af027f4ecc921517503df50873f5e84eb0b3cc09ad0eb34a7'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def binary(y, p, target):
    truth, prediction = np.asarray(y) == target, p.argmax(1) == target
    tp, fn = int((truth & prediction).sum()), int((truth & ~prediction).sum())
    fp, tn = int((~truth & prediction).sum()), int((~truth & ~prediction).sum())
    return dict(support=tp + fn, tp=tp, fn=fn, fp=fp, tn=tn,
                precision=tp / (tp + fp) if tp + fp else None,
                recall=tp / (tp + fn) if tp + fn else None,
                f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None)


def measure(rows, p):
    y = np.array([r['target'] for r in rows])
    summary = metrics(y, p)
    summary['correct'] = int((p.argmax(1) == y).sum())
    return dict(overall=summary, focus={str(k): binary(y, p, k) for k in FOCUS},
                source_groups=len({r['group_id'] for r in rows}))


def preflight():
    study_protocol = read_json(STUDY / 'protocol.json')
    require(len(study_protocol['recipes']) == 3, 'Expected the frozen three recipes; diagnostic must not redefine them')
    previous_lock = read_json(PREVIOUS / 'selection_lock.json')
    selected = Path(previous_lock['selected_checkpoint']).resolve()
    require(selected.is_relative_to(PREVIOUS) and previous_lock['epoch'] == 58,
            'Only the previous best58 initialization is permitted, never this iteration run weights')
    require(previous_lock['sha256'] == EXPECTED_BEST58_SHA and sha256(selected) == EXPECTED_BEST58_SHA,
            'Previous initialization checkpoint changed')
    cache_protocol = read_json(CACHE / 'protocol.json')
    require(cache_protocol['cohort'] == {'old_planning': 2640, 'new_source_development': 15}, 'Unexpected cohort')
    require(cache_protocol['batch_size'] == 1, 'Unexpected cached extraction batch')
    require(cache_protocol['branches'] == {
        'center_crop': 'Current distillation_core.canonical_image: EXIF/RGB, bicubic shortest side224, center224. Same source bytes as materialized legacy image.',
        'demo_letterbox': 'Actual server.prepare via AST: EXIF/RGB, longest edge<=1024, gray128 square letterbox bicubic, JPEG95, then exact engine.decode224.'},
        'Cached preprocessing definition changed')
    complete = read_json(CACHE / 'COMPLETE.json')
    require(complete['images'] == 2655 and complete['all_production_helper_pixels_equal'] is True
            and complete['production_and_source_hashes_unchanged'] is True, 'Prior paired extraction incomplete')
    # Hash only an explicit DEV/model-constant allowlist. Do not loop through the
    # entire study protected list, because it also names unrelated test caches.
    protected = [CACHE / f for f in ('rows.json', 'protocol.json', 'COMPLETE.json', 'paired_features_predictions.npz')]
    protected += [selected, PREVIOUS / 'selection_lock.json', TEACHER,
                  ROOT / 'tools/train_conservation_iteration_20260918.py',
                  ROOT / 'tools/audit_artocarpus_preprocessing_20260917.py',
                  ROOT / 'tools/distillation_core.py', ROOT / 'tools/bioclip_probe.py',
                  ROOT / 'tree_species_teacher_demo/engine.py', ROOT / 'tools/train_sstv1_incremental.py',
                  ROOT / 'spacescan_trees_demo_v1/server.py', EXTENSION_SOURCE]
    for path in protected:
        require(str(path.resolve()) in study_protocol['protected_sha256'], 'Source was not pinned by the frozen study: ' + str(path))
        require(sha256(path) == study_protocol['protected_sha256'][str(path.resolve())], 'Pinned source changed: ' + str(path))
    study_files = ['train_features.npz', 'labels.json', 'classes.json', 'teacher_choice.json',
                   'feature_contract.json', 'planning_rows.json', 'newdev_rows.json',
                   'planning_features.npz', 'newdev_features.npz']
    for name in study_files:
        require(sha256(STUDY / name) == study_protocol['study_inputs'][name], 'Frozen study input changed: ' + name)
    before = {str(path.resolve()): sha256(path) for path in protected + [STUDY / p for p in study_files]
              + [STUDY / 'protocol.json', Path(__file__).resolve()]}
    encoder_sha = sha256(TEACHER)
    require(cache_protocol['protected_sha256_before'][str(TEACHER)] == encoder_sha, 'Paired cache encoder differs')
    rows = read_json(CACHE / 'rows.json')
    require(len(rows) == 2655 and digest(rows) == cache_protocol['rows_digest'], 'Cached row ordering changed')
    require(Counter(r['audit_cohort'] for r in rows) == Counter(old_planning=2640, new_source_development=15),
            'Development cohort changed')
    require(all(r['phase'] == 'planning' for r in rows), 'Non-development row is forbidden')
    require(len({r['sample_id'] for r in rows}) == len(rows), 'Duplicate sample IDs')
    for row in rows:
        require(sha256(row['source_image']) == row['audit_sha256'], 'Development source content changed')
    with np.load(CACHE / 'paired_features_predictions.npz', allow_pickle=False) as f:
        require(str(f['rows_hash']) == digest(rows), 'Paired cache row digest changed')
        # Deliberately never read the old p148/p150 fields: these are different heads.
        arrays = {branch: (f[branch + '_x'].copy(), f[branch + '_z'].copy())
                  for branch in ('center_crop', 'demo_letterbox')}
    for x, z in arrays.values():
        require(x.shape == (2655, 1024) and z.shape == (2655, 148), 'Unexpected feature shapes')
        require(x.dtype == np.float16 and z.dtype == np.float32, 'Unexpected feature precision')
        require(np.isfinite(x).all() and np.isfinite(z).all(), 'Nonfinite cached feature')
    labels, classes = read_json(STUDY / 'labels.json'), read_json(STUDY / 'classes.json')
    require(len(labels) == 150 and [labels[k] for k in FOCUS] ==
        ['Aquilaria sinensis', 'Artocarpus hypargyreus', 'Rhodoleia championii', 'Keteleeria fortunei'], '150-label map changed')
    require(labels == read_json(CACHE / 'labels150.json'), 'Historical 150-class order differs')
    before[str((CACHE / 'labels150.json').resolve())] = sha256(CACHE / 'labels150.json')
    with np.load(STUDY / 'train_features.npz', allow_pickle=False) as f:
        require(str(f['encoder_sha256']) == encoder_sha, 'Text constant encoder differs')
        extra, extension, scale = f['extra_bank'].copy(), f['extension_bank'].copy(), float(f['scale'])
    require(extra.shape == (1, 1024) and extension.shape == (2, 1024) and len(classes['groups']) == 147,
            'Conservation150 dimensions changed')
    require(extension.dtype == np.float32 and np.allclose(np.linalg.norm(extension, axis=1), 1., atol=1e-6),
            '150-class extension must be float32 unit-normalized')
    with np.load(EXTENSION_SOURCE, allow_pickle=False) as f:
        normalized = normalize(torch.tensor(f['extension_bank'], dtype=torch.float32), dim=-1).numpy()
        require(np.array_equal(extension, normalized), 'Extension differs from fixed normalized source bank')
        require(np.isclose(scale, float(f['logit_scale']), atol=1e-5), 'Text logit scale changed')
    temperature = read_json(STUDY / 'teacher_choice.json')['calibration_temperature']
    require(temperature == cache_protocol['temperature'], 'Temperature changed')
    params = dict(extra_bank=torch.tensor(extra, dtype=torch.float32),
                  extension_bank=torch.tensor(extension, dtype=torch.float32),
                  scale=scale, groups=classes['groups'], temperature=temperature)
    index = {r['sample_id']: i for i, r in enumerate(rows)}
    for phase, cohort in [('planning', 'old_planning'), ('newdev', 'new_source_development')]:
        expected = read_json(STUDY / (phase + '_rows.json')); ids = []
        for row in expected:
            i = index[row['sample_id']]; old = rows[i]
            require(old['audit_cohort'] == cohort and old['target'] == row['target']
                    and old['group_id'] == row['group_id'] and old['audit_sha256'] == row['sha256'],
                    'Study development mapping differs from paired cache')
            ids.append(i)
        with np.load(STUDY / (phase + '_features.npz'), allow_pickle=False) as f:
            require(str(f['rows_hash']) == digest(expected), 'Study development row order changed')
            require(np.array_equal(f['x'], arrays['demo_letterbox'][0][ids])
                    and np.array_equal(f['z'], arrays['demo_letterbox'][1][ids]), 'Study letterbox tensors differ')
    state = torch.load(selected, map_location='cpu', weights_only=False)
    require(state['epoch'] == 58 and state['classes'] == classes and state['protocol']['encoder_sha256'] == encoder_sha,
            'Previous best58 architecture/encoder changed')
    model = Conservation150().cpu().eval().requires_grad_(False)
    model.heads.load_state_dict(state['model_state'], strict=True)
    require(all(bool((p == 0).all()) for p in model.novel.parameters()), 'Novel residuals must remain zero')
    require(all(p.device.type == 'cpu' for p in model.parameters()), 'CPU-only audit contract violated')
    return rows, arrays, labels, params, model, before, cache_protocol


def infer(model, x, z, params):
    blocks = []
    with torch.inference_mode():
        for start in range(0, len(x), 128):
            blocks.append(model.logprob(torch.tensor(x[start:start + 128].astype(np.float32)),
                                       torch.tensor(z[start:start + 128]), params).exp().numpy())
    p = np.concatenate(blocks)
    require(p.shape == (2655, 150) and np.isfinite(p).all() and np.allclose(p.sum(1), 1., atol=1e-5),
            'Invalid Conservation150 probabilities')
    return p


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args(); out = args.out.resolve()
    require(out.is_relative_to(ROOT / 'outputs') and not out.exists(), 'Audit output must be new; never overwrite')
    torch.set_num_threads(2)
    rows, arrays, labels, params, model, before, cache_protocol = preflight()
    out.mkdir(parents=True, exist_ok=False)
    save_json(out / 'protocol.json', dict(created_at=datetime.now(timezone.utc).isoformat(),
        source_hashes=before, dataset='All fixed 2640 old-planning + 15 source-development photos; no test reads',
        source_rows_hash=digest(rows), initialization='Previous locked best58 heads + zero new-class residuals',
        previous_checkpoint_sha256=EXPECTED_BEST58_SHA, vocabulary=labels,
        comparison='Same Conservation150 model: Demo-letterbox vs arithmetic mean of center and letterbox probabilities',
        center_weight=MIX_CENTER_WEIGHT, letterbox_weight=1. - MIX_CENTER_WEIGHT,
        weight_search=False, temperature=params['temperature'], scale=params['scale'],
        encoder_sha256=sha256(TEACHER), device='cpu', cpu_threads=2,
        preprocessing=cache_protocol['branches'], batch1_cached_embeddings=True,
        normalized_extension_reverified=True, current_iteration_fit_models_read=False,
        changes_to_frozen_recipes_or_selection=False, encoded=False, fitted=False, deployed=False,
        scope='Diagnostic only; previously inspected development, NOT training gain or fresh generalization evidence.',
        limitations=['Two framings are not new camera views or independent trees.',
                     'Only 4 Rhodoleia and 1 Keteleeria development positives, from previously seen galleries.',
                     'Framing changes context, effective target scale and JPEG together; cannot establish the single principal cause.',
                     'This audit cannot alter the frozen three training recipes, their selected epochs, fusion coefficient or deployment.']))
    save_json(out / 'rows.json', rows)
    center = infer(model, *arrays['center_crop'], params)
    letterbox = infer(model, *arrays['demo_letterbox'], params)
    mixed = MIX_CENTER_WEIGHT * center + (1. - MIX_CENTER_WEIGHT) * letterbox
    probabilities = {'letterbox': letterbox, 'fixed_equal_view_mean': mixed}
    results = {}
    for cohort in ('all_development', 'old_planning', 'new_source_development'):
        ids = [i for i, r in enumerate(rows) if cohort == 'all_development' or r['audit_cohort'] == cohort]
        subset = [rows[i] for i in ids]; truth = np.array([r['target'] for r in subset])
        a, b = letterbox[ids].argmax(1), mixed[ids].argmax(1)
        results[cohort] = {name: measure(subset, p[ids]) for name, p in probabilities.items()}
        results[cohort]['paired'] = dict(corrected=int(((a != truth) & (b == truth)).sum()),
            regressed=int(((a == truth) & (b != truth)).sum()), changed=int((a != b).sum()),
            corrected_ids=[r['sample_id'] for r, ok in zip(subset, (a != truth) & (b == truth)) if ok],
            regressed_ids=[r['sample_id'] for r, ok in zip(subset, (a == truth) & (b != truth)) if ok])
    predictions = []
    for i, row in enumerate(rows):
        record = dict(sample_id=row['sample_id'], source_image=row['source_image'], sha256=row['audit_sha256'],
                      group_id=row['group_id'], cohort=row['audit_cohort'], truth_target=row['target'],
                      truth_label=labels[row['target']],
                      label_authority=row.get('label_authority', 'Existing source labels; not new human certification'))
        for name, p in dict(center_for_reproducibility=center, **probabilities).items():
            top = int(p[i].argmax())
            record[name] = dict(top1_target=top, top1_species=labels[top], top1_score=float(p[i, top]),
                                focus_scores={str(k): float(p[i, k]) for k in FOCUS})
        predictions.append(record)
    for path, expected in before.items():
        require(sha256(path) == expected, 'Frozen source changed during audit: ' + path)
    for row in rows:
        require(sha256(row['source_image']) == row['audit_sha256'], 'Development image changed during audit')
    save_json(out / 'results.json', results)
    save_json(out / 'predictions.json', predictions)
    np.savez_compressed(out / 'predictions.npz', center_for_reproducibility=center, **probabilities,
                        target=np.array([r['target'] for r in rows]), rows_hash=digest(rows))
    def pct(value):
        return 'N/A' if value is None else f'{value:.2%}'
    lines = ['# Conservation150 fixed two-view development diagnostic', '',
             'Same previous best58 + zero-novel baseline in both branches. No encoder inference, fitting or deployment.',
             'Fixed 50% center + 50% letterbox probability mean; no coefficient search. All computation on CPU.',
             'Only previously inspected development data; this is NOT a training improvement claim.', '',
             '| Cohort | Letterbox Top-1 | Equal-view Top-1 | Corrected | Regressed |',
             '|---|---:|---:|---:|---:|']
    for cohort, result in results.items():
        lines.append(f"| {cohort} | {pct(result['letterbox']['overall']['top1'])} | {pct(result['fixed_equal_view_mean']['overall']['top1'])} | {result['paired']['corrected']} | {result['paired']['regressed']} |")
    lines += ['', '| Cohort | Species | Support | Letterbox TP/FP/FN/TN | Equal-view TP/FP/FN/TN | Recall | F1 |',
              '|---|---|---:|---|---|---|---|']
    for cohort, result in results.items():
        for target in FOCUS:
            a, b = [result[branch]['focus'][str(target)] for branch in probabilities]
            counts = lambda m: '/'.join(str(m[k]) for k in ('tp', 'fp', 'fn', 'tn'))
            lines.append(f"| {cohort} | {labels[target]} | {a['support']} | {counts(a)} | {counts(b)} | {pct(a['recall'])} → {pct(b['recall'])} | {pct(a['f1'])} → {pct(b['f1'])} |")
    lines += ['', '## Interpretation boundaries', '',
              '- These two framings use identical source bytes and weights, so differences isolate this paired preprocessing strategy, not training.',
              '- A gain demonstrates a relevant framing sensitivity; it does not prove preprocessing is the sole or principal accuracy limit.',
              '- Only four Rhodoleia and one Keteleeria development positives are available; source galleries and observations are not independent trees.',
              '- All source/development/cache/encoder checks passed. No test set, current fitted iteration checkpoint, threshold search or fusion search was used.',
              '- The frozen three training recipes and their selection remain untouched. This diagnostic does not authorize deployment.']
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    save_json(out / 'COMPLETE.json', dict(completed_at=datetime.now(timezone.utc).isoformat(),
        images=2655, cpu_only=True, test_read=False, encoder_inference=False, fitted=False, deployed=False,
        source_files_unchanged=True, frozen_recipes_unchanged=True,
        results_sha256=sha256(out / 'results.json'), predictions_sha256=sha256(out / 'predictions.npz')))
    compact = {name: dict(letterbox=result['letterbox']['overall']['top1'],
                         fixed_equal_view_mean=result['fixed_equal_view_mean']['overall']['top1'],
                         corrected=result['paired']['corrected'], regressed=result['paired']['regressed'],
                         focus={str(k): {'letterbox': result['letterbox']['focus'][str(k)],
                                        'mix': result['fixed_equal_view_mean']['focus'][str(k)]} for k in FOCUS})
               for name, result in results.items()}
    print('FIXED VIEW AUDIT COMPLETE', compact, str(out), flush=True)


if __name__ == '__main__':
    main()
