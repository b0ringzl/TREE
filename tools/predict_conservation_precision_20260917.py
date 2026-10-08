"""Read-only live-image comparison for a globally locked precision study.

No fitting, threshold tuning, deployment, image editing or output overwrite.
Scientific names are presented without translated parenthetical annotations;
the original full label mapping and its hash remain available for provenance.
"""
import argparse
import gc
import importlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ['HF_HUB_OFFLINE'] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

import train_conservation_browser_20260917 as base
from distillation_core import digest, read_json
from train_sstv1_incremental import prepare_pixels
from training_safety import sha256


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def scientific_name(label):
    # Strip only translated annotations, retaining the scientific pooled names.
    return re.sub(r'\s*\([^)]*[\u3400-\u9fff][^)]*\)', '', label).strip()


def selected_implementation(lock):
    """Only the two explicitly supported, locally audited architectures."""
    module_name = ('optimize_conservation_kernel_20260917' if 'beta' in lock['recipe']
                   else 'optimize_conservation_precision_20260917')
    return importlib.import_module(module_name)


def verify_locked_run(run):
    require(run.is_dir(), 'Training run directory does not exist')
    lock_file = run / 'selection_lock.json'
    require(lock_file.is_file(), 'No global model-selection lock: inference refused')
    lock = read_json(lock_file)
    optimization = selected_implementation(lock)
    require(lock.get('selection_uses_validation_only') is True,
            'Model selection was not explicitly validation-only')
    require(lock.get('all_recipes_finished_before_regression') is True,
            'The whole registered study must be locked before inference')
    checkpoint = Path(lock['checkpoint']).resolve()
    require(checkpoint.is_relative_to(run), 'Selected checkpoint is outside the study')
    require(checkpoint.is_file() and sha256(checkpoint) == lock['sha256'],
            'Selected checkpoint failed SHA-256 verification')
    data = run / 'data'
    protocol = read_json(run / 'protocol.json')
    data_config = read_json(data / 'config.json')
    labels = read_json(data / 'labels.json')
    classes = read_json(data / 'classes.json')
    require(len(labels) == 150 and protocol['output_count'] == 150,
            'Unexpected output dimension')
    require(labels[:147] == classes['labels'] and labels[147] == 'Wodyetia bifurcata',
            'Existing output ordering changed')
    require([labels[i] for i in (16, 148, 149)] == base.TARGETS,
            'Conservation target mapping changed')
    require(lock['recipe'] in protocol['recipes'], 'Selected recipe was not registered')
    require(bool(lock.get('safe')), 'The selected candidate did not pass its declared development gate')
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    require(state['labels'] == labels, 'Checkpoint label mapping mismatch')
    require(state['config'] == dict(protocol, recipe=lock['recipe']),
            'Checkpoint configuration differs from the registered study')
    require(int(state['epoch']) == int(lock['epoch']), 'Checkpoint epoch differs from the lock')
    deployment = read_json(base.DEPLOY)
    require(state['encoder_sha256'] == deployment['encoder_sha256'], 'Encoder contract mismatch')
    require(data_config['base_version'] == deployment['version'], 'Deployed baseline version changed')
    require(data_config['temperature'] == deployment['temperature'], 'Calibration temperature changed')
    # The implementation that produces logits must match this fit's backup.
    backup_manifest = Path(lock['backup']) / 'backup_manifest.json'
    saved_sources = {str(Path(r['source']).resolve()): r for r in read_json(backup_manifest)['files']}
    for module_path in (Path(optimization.__file__).resolve(), Path(base.__file__).resolve()):
        entry = saved_sources.get(str(module_path))
        require(entry is not None and sha256(module_path) == entry['sha256'],
                'Inference implementation differs from the training snapshot: ' + str(module_path))
    if 'beta' in lock['recipe']:
        support_manifest = run / 'support_manifest.json'
        support_entry = saved_sources.get(str(support_manifest.resolve()))
        require(support_entry is not None and sha256(support_manifest) == support_entry['sha256'],
                'Kernel support provenance changed since fitting')
        support_info = read_json(support_manifest)
        support = state['model_state']['support']
        require(tuple(support.shape) == (support_info['n'], 1024) and support_info['n'] > 0,
                'Kernel support shape differs from the saved training manifest')
        require(bool(torch.isfinite(support).all()) and
                torch.allclose(support.norm(dim=1), torch.ones(support_info['n']), atol=1e-5),
                'Kernel support vectors must be finite unit-normalized training embeddings')
        require(tuple(state['model_state']['linear.weight'].shape) == (3, support_info['n']) and
                'linear.bias' not in state['model_state'], 'Unexpected kernel weight architecture')
    base.verify_contract(data)
    return lock, protocol, data_config, labels, classes, state, optimization


def branch_record(probabilities, labels):
    p = np.asarray(probabilities, dtype=np.float64)
    require(p.ndim == 1 and len(p) in (148, 150), 'Unexpected probability dimension')
    require(np.isfinite(p).all() and (p >= 0).all(), 'Invalid model probabilities')
    require(np.isclose(p.sum(), 1., atol=1e-5), 'Model probabilities are not normalized')
    names = labels[:len(p)]
    distribution = [dict(index=i, scientific_name=scientific_name(label), score=float(p[i]))
                    for i, label in enumerate(names)]
    order = np.argsort(-p, kind='stable')[:3]
    return {
        'candidate_count': len(p),
        'top3': [distribution[int(i)] for i in order],
        'distribution': distribution,
    }


def predict(run, image):
    run, image = run.resolve(), image.resolve()
    require(image.is_file(), 'Input image does not exist')
    lock, protocol, cfg, labels, classes, state, optimization = verify_locked_run(run)
    data = run / 'data'
    image_sha = sha256(image)
    protected = [run / 'selection_lock.json', Path(lock['checkpoint']), run / 'protocol.json',
                 data / 'config.json', data / 'labels.json', data / 'classes.json',
                 data / 'new_features.npz', Path(optimization.__file__), Path(base.__file__)]
    hashes_before = {str(p.resolve()): sha256(p) for p in protected}
    from tree_species_teacher_demo.engine import Engine
    engine = Engine()
    engine.load()
    require(engine.sstv1_state == 'ready' and engine.sstv1_version == cfg['base_version'],
            'Engine does not provide the pinned baseline')
    require(engine.encoder_sha256 == state['encoder_sha256'], 'Live encoder hash mismatch')
    require(engine.classes == classes and engine.labels == labels[:148], 'Live engine taxonomy mismatch')
    require(engine.choice['calibration_temperature'] == cfg['temperature'], 'Live temperature mismatch')
    encoder_sha = engine.encoder_sha256
    pixels = embedding = raw = None
    try:
        with torch.inference_mode():
            pixels = prepare_pixels(image)[None].cuda()
            embedding = engine.model.encode_image(
                ((pixels - engine.mean) / engine.std).to(engine.dtype), normalize=True).float()
            # Raw logits retain full encoder precision, exactly as the training cache.
            raw = embedding @ engine.bank.T * engine.model.logit_scale.exp().float()
            x = embedding.half().float().cpu().numpy()
            z = raw.cpu().numpy()
        require(x.shape == (1, 1024) and z.shape == (1, 148), 'Live feature dimensions changed')
        require(np.isfinite(x).all() and np.isfinite(z).all(), 'Nonfinite live features')
    finally:
        # Encoder and its GPU tensors are not retained during small-head inference.
        del pixels, embedding, raw, engine
        gc.collect()
        torch.cuda.empty_cache()

    # Both calls validate the frozen deployment/label contracts again.
    selected, current_lock = optimization.load_selected(run)
    require(current_lock == lock, 'Selection changed during feature extraction')
    model = base.models(data)
    zero = optimization.BoundedResidual(lock['recipe']['cap']).cuda().eval().requires_grad_(False)
    row = dict(sample_id='live_' + image_sha, source_image=str(image), sha256=image_sha,
               group_id='live_input_not_verified_tree', target=-1,
               label_authority='unlabelled_inference_no_ground_truth')
    tensor = base.tensor_data(data, ([row], x, z), model)
    try:
        probabilities = {
            'native148': base.scores(zero, tensor, native=True)[0],
            'baseline150': base.scores(zero, tensor)[0],
            'candidate150': base.scores(selected, tensor)[0],
        }
    finally:
        del selected, model, zero, tensor
        gc.collect()
        torch.cuda.empty_cache()
    require(sha256(image) == image_sha, 'Input image changed during inference')
    for path, expected in hashes_before.items():
        require(sha256(path) == expected, 'Study artifact changed during inference: ' + path)
    base.verify_contract(data)
    return {
        'created_at': datetime.now(timezone.utc).isoformat(),
        'run': str(run), 'image': str(image), 'image_sha256': image_sha,
        'selected_recipe': lock['recipe'], 'selected_epoch': lock['epoch'],
        'residual_architecture': 'local_rbf_kernel' if 'beta' in lock['recipe'] else 'bounded_linear',
        'selected_checkpoint': lock['checkpoint'], 'selected_checkpoint_sha256': lock['sha256'],
        'encoder_sha256': encoder_sha, 'baseline_version': cfg['base_version'],
        'temperature': cfg['temperature'], 'label_mapping_sha256': sha256(data / 'labels.json'),
        'class_group_digest': digest(classes), 'original_label_mapping': labels,
        'display': 'Scientific names; translated parenthetical annotations omitted. Pooled scientific taxa remain pooled.',
        'preprocessing': 'EXIF/RGB, gray letterbox <=1024, JPEG95, decode224; batch1 frozen encoder; x=e.half().float, raw z from unrounded e.',
        'branches': {key: branch_record(p, labels) for key, p in probabilities.items()},
        'scope': 'Single-image experimental comparison. Scores are not measured accuracy; no detection, multiview fusion or point-cloud inference.',
        'trained': False, 'deployed': False, 'source_artifacts_unchanged': True,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if args.out is not None:
        require(not args.out.exists(), 'Output already exists; choose a new filename')
    result = predict(args.run, args.image)
    encoded = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation also prevents races from overwriting an existing file.
        with args.out.open('x', encoding='utf-8') as output:
            output.write(encoded)
    print(encoded, end='')
