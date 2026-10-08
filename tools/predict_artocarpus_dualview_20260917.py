"""Read-only, locked 50/50 same-image dual-view experimental inference.

Re-encodes original files; no detector, training, deployment, score override, or
reference-label lookup during inference. Optional cache comparison happens only
after every live prediction has been computed. Outputs never overwrite a run.
"""
import argparse
import gc
import hashlib
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ['HF_HUB_OFFLINE'] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'tools'))

import numpy as np
import torch
import torch.nn.functional as F

from audit_artocarpus_preprocessing_20260917 import exact_demo_prepare, probabilities, pixel_image
from distillation_core import canonical_image, digest, read_json, save_json
from train_sstv1_incremental import prepare_pixels
from training_safety import sha256
from tree_species_teacher_demo.engine import Engine, SSTV1_MANIFEST, decode

V1 = ROOT/'outputs/artocarpus_preprocess_audit_20260917_v1'
V2 = ROOT/'outputs/artocarpus_preprocess_audit_20260917_v2'
EXTENSION_SOURCE = ROOT/'outputs/SST_Conserve_Browser_20260917_v3/new_features.npz'
SEED = 20260917
BRANCHES = ('center_crop', 'demo_letterbox', 'equal_dualview')
PARITY_TOLERANCE = 1e-5


def require(condition, message):
    if not condition: raise RuntimeError(message)


def verify_contract():
    v2complete = read_json(V2/'COMPLETE.json')
    require(sha256(V2/'protocol.json') == v2complete['protocol_sha256'], 'Dual-view protocol changed')
    require(v2complete['fit'] is False and v2complete['deployed'] is False, 'Unexpected dual-view study status')
    protocol = read_json(V2/'protocol.json')
    require(protocol['only_candidate'] == {'name': 'equal_dualview',
        'center_crop_probability_weight': .5, 'demo_letterbox_probability_weight': .5}, 'Only fixed50/50 is supported')
    require(protocol['no_search'] and protocol['no_training'] and protocol['no_deployment'], 'Invalid study scope')
    # Verify the immutable audit sources before trusting its model/preprocess pin.
    for path, expected in protocol['source_hashes'].items():
        require(sha256(path) == expected, 'Locked v2 source changed: '+path)
    v1protocol = read_json(V1/'protocol.json')
    protected = v1protocol['protected_sha256_before']
    for path, expected in protected.items():
        require(sha256(path) == expected, 'Pinned model or preprocessing changed: '+path)
    require(read_json(V1/'COMPLETE.json')['production_and_source_hashes_unchanged'], 'v1 did not complete safely')
    labels = read_json(V2/'labels150.json')
    require(labels == read_json(V1/'labels150.json') and len(labels) == 150, 'Taxonomy mismatch')
    current = read_json(SSTV1_MANIFEST)
    require(current == read_json(V1/'deployment_snapshot.json'), 'Deployment differs from audited model')
    return protocol, v1protocol, labels, current


def distribution(p, labels):
    p = np.asarray(p, dtype=np.float32)
    require(p.shape == (len(labels),) and np.isfinite(p).all() and (p >= 0).all(), 'Invalid probabilities')
    require(np.isclose(p.sum(), 1., atol=1e-5), 'Distribution not normalized')
    ids = np.argsort(-p, kind='stable')[:3]
    return {'top3': [{'index': int(i), 'species': labels[i], 'score': float(p[i])} for i in ids],
            'probabilities': p.tolist()}


def infer_one(path, index, out, engine, extension, labels, production_prepare):
    content = path.read_bytes(); input_sha = hashlib.sha256(content).hexdigest()
    padded_bytes, width, height = production_prepare(content)
    padded = decode(padded_bytes)[0]
    center = canonical_image(path)
    require(torch.equal(padded, prepare_pixels(path)), 'Production/helper preprocessing mismatch')
    pixels = {'center_crop': center, 'demo_letterbox': padded}
    arrays = {}; pixel_info = {}; start = time.perf_counter()
    with torch.inference_mode():
        for name, tensor in pixels.items():
            embedding = engine.model.encode_image(((tensor[None].cuda()-engine.mean)/engine.std).to(engine.dtype), normalize=True).float()
            raw = embedding @ engine.bank.T * engine.model.logit_scale.exp().float()
            p148, p150 = probabilities(engine, embedding, raw, extension)
            arrays[f'{name}_p148'] = p148[0].cpu().numpy()
            arrays[f'{name}_p150'] = p150[0].cpu().numpy()
            pixel_path = out/f'{index:02d}_{input_sha[:12]}_{name}.png'
            pixel_image(tensor).save(pixel_path)
            pixel_info[name] = {'tensor_sha256': hashlib.sha256(tensor.numpy().tobytes()).hexdigest(),
                'tensor_format': 'C-contiguous float32 RGB CHW [0,1],224x224 before CLIP normalization',
                'preview': str(pixel_path), 'preview_sha256': sha256(pixel_path)}
    for n in (148, 150):
        arrays[f'equal_dualview_p{n}'] = .5*arrays[f'center_crop_p{n}']+.5*arrays[f'demo_letterbox_p{n}']
    require(sha256(path) == input_sha, 'Input changed during inference')
    record = {'input': str(path), 'input_sha256': input_sha, 'input_dimensions': [width, height],
        'pixels': pixel_info, 'demo_prepared_jpeg_sha256': hashlib.sha256(padded_bytes).hexdigest(),
        'production_helper_pixels_identical': True, 'encoder_calls': 2,
        'elapsed_seconds': time.perf_counter()-start,
        'native148': {b: distribution(arrays[f'{b}_p148'], labels[:148]) for b in BRANCHES},
        'extension150': {b: distribution(arrays[f'{b}_p150'], labels) for b in BRANCHES},
        'ground_truth_used_for_prediction': False,
        'scope': 'Classifier input only. No tree detection, panorama reprojection, crop selection, point cloud, training, deployment or per-image routing.',
        'scores_notice': 'Model distributions are not measured image-specific accuracy. This is a development-stage research candidate.'}
    return record, arrays


def compare_cache(records, all_arrays):
    """Post-prediction consistency check only, never a prediction source."""
    rows = read_json(V2/'rows.json'); matching = {}
    for i, row in enumerate(rows): matching.setdefault(row['audit_sha256'], []).append(i)
    with np.load(V2/'fixed_equal_dualview_predictions.npz') as cache:
        require(str(cache['rows_hash']) == digest(rows), 'Reference cache order changed')
        results = []
        for record, arrays in zip(records, all_arrays):
            ids = matching.get(record['input_sha256'], [])
            require(bool(ids), '--cache-parity requires a byte-identical v2 audit image')
            checks = []
            for i in ids:
                for n in (148, 150):
                    for branch in BRANCHES:
                        key = f'{branch}_p{n}'; old = cache[key][i]; live = arrays[key]
                        delta = np.abs(old-live)
                        checks.append({'catalog': n, 'branch': branch, 'same_top1': int(old.argmax()) == int(live.argmax()),
                            'cache_top1_index': int(old.argmax()), 'live_top1_index': int(live.argmax()),
                            'max_abs_probability_error': float(delta.max()), 'mean_abs_probability_error': float(delta.mean())})
            passed = all(c['same_top1'] and c['max_abs_probability_error'] <= PARITY_TOLERANCE for c in checks)
            results.append({'input': record['input'], 'sha256': record['input_sha256'],
                'reference_sample_ids': [rows[i]['sample_id'] for i in ids],
                'reference_labels_for_post_prediction_audit_only': [rows[i]['target'] for i in ids],
                'absolute_probability_tolerance': PARITY_TOLERANCE, 'checks': checks, 'passed': passed,
                'is_new_independent_validation': False})
        return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, action='append', required=True, help='Repeat this argument for more original images.')
    parser.add_argument('--out', type=Path, required=True, help='New output directory; existing directories are refused.')
    parser.add_argument('--cache-parity', action='store_true', help='After prediction, compare known original files against immutable v2 cached probabilities.')
    args = parser.parse_args()
    images = [p.resolve() for p in args.image]; out = args.out.resolve()
    require(all(p.is_file() for p in images), 'Missing input file')
    require(not out.exists() and out.is_relative_to(ROOT/'outputs'), 'Choose a new output directory within outputs')
    protocol, v1protocol, labels, dep = verify_contract()
    # Do not permit this standalone tool to append to the two locked audits.
    require(not out.is_relative_to(V1) and not out.is_relative_to(V2), 'Locked v1/v2 outputs are read-only')
    protected = dict(v1protocol['protected_sha256_before'])
    for path in [Path(__file__).resolve(), V2/'protocol.json', V2/'COMPLETE.json',
                 V2/'fixed_equal_dualview_predictions.npz', V2/'rows.json', V2/'labels150.json']:
        protected[str(path)] = sha256(path)
    out.mkdir(parents=True, exist_ok=False)
    model = {'deployment_version': dep['version'], 'encoder_sha256': dep['encoder_sha256'],
        'members': dep['members'], 'temperature': dep['temperature'], 'native_count': 148,
        'extended_count': 150, 'labels150': labels,
        'extension_cache': str(EXTENSION_SOURCE), 'extension_cache_sha256': sha256(EXTENSION_SOURCE),
        'operation': 'Fixed mean of center-crop and Demo-letterbox probability distributions,50/50; no weight-search option.',
        'extension': 'Same fixed two text vectors as v1/v2, float32 renormalization; zero new learned residual.',
        'preprocessing': v1protocol['branches'], 'source_model_preprocess_sha256': protected,
        'input_files_sha256': {str(p): sha256(p) for p in images},
        'tool': str(Path(__file__).resolve()), 'tool_sha256': sha256(__file__),
        'created_at': datetime.now(timezone.utc).isoformat(), 'fit': False, 'deploy': False,
        'cache_reference_used_for_inference': False}
    save_json(out/'run_contract.json', model)
    torch.set_num_threads(4); torch.manual_seed(SEED); np.random.seed(SEED)
    engine = Engine(); records = []; all_arrays = []; extension = None
    try:
        engine.load()
        require(engine.sstv1_state == 'ready' and engine.sstv1_version == dep['version'], 'Expected deployed heads unavailable')
        require(engine.encoder_sha256 == dep['encoder_sha256'] and engine.labels == labels[:148], 'Live model contract mismatch')
        with np.load(EXTENSION_SOURCE) as previous:
            require(str(previous['encoder_sha256']) == engine.encoder_sha256, 'Text extension encoder mismatch')
            require(np.isclose(float(previous['logit_scale']), engine.model.logit_scale.exp().float().item(), atol=1e-5), 'Scale mismatch')
            extension = F.normalize(torch.tensor(previous['extension_bank'], device='cuda', dtype=torch.float32), dim=-1)
        production_prepare = exact_demo_prepare()
        for i, path in enumerate(images):
            record, arrays = infer_one(path, i, out, engine, extension, labels, production_prepare)
            records.append(record); all_arrays.append(arrays)
            save_json(out/f'{i:02d}_{record["input_sha256"][:12]}_prediction.json', record)
            print('PREDICTED', path.name, {b: record['extension150'][b]['top3'][0] for b in BRANCHES}, flush=True)
    finally:
        del engine, extension
        gc.collect(); torch.cuda.empty_cache()
    if args.cache_parity:
        parity = compare_cache(records, all_arrays)
        save_json(out/'cache_parity.json', parity)
        require(all(p['passed'] for p in parity), 'Live/cache parity failed; see cache_parity.json')
    else: parity = None
    np.savez_compressed(out/'live_probabilities.npz',
        **{k: np.stack([a[k] for a in all_arrays]) for k in all_arrays[0]},
        input_sha256=np.array([r['input_sha256'] for r in records]))
    for path, expected in protected.items():
        require(sha256(path) == expected, 'Read-only model or audit artifact changed: '+path)
    for path, expected in model['input_files_sha256'].items():
        require(sha256(path) == expected, 'Input image changed')
    save_json(out/'COMPLETE.json', {'completed_at': datetime.now(timezone.utc).isoformat(),
        'images': len(images), 'encoder_calls': 2*len(images), 'fit': False, 'deploy': False,
        'protected_files_unchanged': True, 'gpu_released': True,
        'cache_parity': all(p['passed'] for p in parity) if parity is not None else 'not_requested',
        'new_independent_accuracy_evidence': False})
    print('COMPLETE', out, flush=True)


if __name__ == '__main__':
    main()
