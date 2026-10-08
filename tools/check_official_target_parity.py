"""Compare the real web-selected target with the offline Android ONNX classifier.

Runs ONNX on this Windows host, not on a physical Android phone. No fitting.
"""
import argparse
import hashlib
import io
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT / 'outputs/android_toolchain_20260918_v1/python_deps')]
import numpy as np
import onnxruntime as ort
import requests
from PIL import Image
from spacescan_trees_demo_v1.server import prepare


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    def post(route, fields=None):
        with args.image.open('rb') as stream:
            response = requests.post('http://127.0.0.1:8043' + route, files={'image': (args.image.name, stream, 'image/jpeg')}, data=fields or {}, timeout=180)
        response.raise_for_status()
        return response

    detection = post('/api/detect-trees', {'projection': 'photo'}).json()
    if detection['action'] != 'auto':
        raise RuntimeError('No automatic centered target; do not choose a crop by its species result.')
    target = next(c for c in detection['candidates'] if c['id'] == detection['auto_id'])
    web = post('/api/identify-tree', {'token': target['token']}).json()
    crop_response = post('/api/export-target', {'token': target['token']})
    crop = crop_response.content
    crop_hash = hashlib.sha256(crop).hexdigest()
    assert crop_hash == web['release_provenance']['target_bytes_sha256'] == crop_response.headers['X-Target-SHA256']
    (args.output / 'SpaceScan-selected-target.jpg').write_bytes(crop)
    pixels, _, _ = prepare(crop)
    assert hashlib.sha256(pixels).hexdigest() == web['release_provenance']['padded_jpeg_sha256']
    im = Image.open(io.BytesIO(pixels)).convert('RGB').resize((224, 224), Image.Resampling.BICUBIC)
    x = np.ascontiguousarray(np.asarray(im, dtype=np.float32).transpose(2, 0, 1)[None] / 255)
    pack = PROJECT / 'outputs/SpaceScan_Android_SSTv1_20260919_v1/model_pack'
    manifest = json.loads((pack / 'model_manifest.json').read_text(encoding='utf-8'))
    species = json.loads((pack / 'species.json').read_text(encoding='utf-8'))
    assert manifest['desktop_version'] == web['image_model_version']
    for row in manifest['graph_files']:
        path = pack / row['file']
        assert sha(path) == row['sha256'], path
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        opts.enable_cpu_mem_arena = False
        opts.enable_mem_pattern = False
        opts.add_session_config_entry('session.disable_prepacking', '1')
        session = ort.InferenceSession(str(path), sess_options=opts, providers=['CPUExecutionProvider'])
        x = session.run(None, {'input': x})[0]
        del session
        print(row['file'], 'OK', flush=True)
    scores = x[0]
    mobile_top = np.argsort(-scores)[:3]
    expected = np.asarray([web['image_distribution'][row['label']] for row in species])
    result = {
        'training_performed': False,
        'test_kind': 'Real web automatic target selection, then Windows-hosted Android ONNX on identical RGB preprocessing; not physical-phone validation.',
        'source_path': str(args.image.resolve()), 'source_sha256': sha(args.image),
        'native_crop_size': list(Image.open(io.BytesIO(crop)).size), 'crop_sha256': crop_hash,
        'crop_box': web['selected_tree']['crop_box'], 'detection_candidate_count': len(detection['candidates']),
        'web_model_version': web['image_model_version'], 'android_model_version': manifest['version'],
        'web_top3': [{'species': row['species'], 'score': row['score']} for row in web['top3']],
        'android_onnx_same_target_top3': [{'species': species[i]['label'], 'score': float(scores[i])} for i in mobile_top],
        'same_top1': species[mobile_top[0]]['label'] == web['top3'][0]['species'],
        'max_absolute_score_difference': float(np.abs(expected - scores).max()),
        'export_is_byte_identical_to_inference_target': True,
        'full_android_pipeline_parity': False,
    }
    (args.output / 'same_target_parity.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    assert result['same_top1'], 'Classifier regression: first candidate differs.'
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
