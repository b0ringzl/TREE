"""Inference-only H120 Android pack: reuse verified encoder stages, replace heads."""
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'outputs/android_toolchain_20260918_v1/python_deps'), str(ROOT / 'tools')]
import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort
import torch
from training_safety import sha256, verify_backup

RUN = ROOT / 'outputs/android_h120_ui_sync_20260927_v1'
OLD = ROOT / 'outputs/SpaceScan_Android_SSTv1_20260919_v1/model_pack'
DEPLOY = ROOT / 'tree_species_teacher_demo/models/SSTv1_Aquilaria_UserCrops_20260919_H120/deployment.json'
ASSETS = ROOT / 'spacescan_android_sstv1/app/src/main/assets'
PACK = RUN / 'SSTv1-H120-20260927-Offline-Model.zip'


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    verify_backup(RUN / 'pre_training_backup')
    deployment = json.loads(DEPLOY.read_text(encoding='utf-8'))
    old_manifest = json.loads((OLD / 'model_manifest.json').read_text(encoding='utf-8'))
    assert deployment['encoder_sha256'] == old_manifest['source_deployment']['encoder_sha256']
    assert deployment['labels_sha256'] == old_manifest['source_deployment']['labels_sha256']
    assert deployment['temperature'] == old_manifest['source_deployment']['temperature']
    assert deployment['output_count'] == old_manifest['class_count'] == 148
    assert len(deployment['members']) == 3
    graph = onnx.load(str(OLD / 'stage_17.onnx'))
    initializers = {value.name: value for value in graph.graph.initializer}
    for member_index, member in enumerate(deployment['members']):
        checkpoint = DEPLOY.parent / member['file']
        assert sha256(checkpoint) == member['sha256']
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        assert state['encoder_sha256'] == deployment['encoder_sha256']
        for name, value in state['model_state'].items():
            target = f'heads.{member_index}.{name}'
            before = initializers[target]
            replacement = numpy_helper.from_array(value.detach().float().numpy(), name=target)
            assert tuple(before.dims) == tuple(replacement.dims)
            before.CopyFrom(replacement)
    tail = RUN / 'stage_17_h120.onnx'
    onnx.save(graph, str(tail))
    onnx.checker.check_model(str(tail))
    files = json.loads(json.dumps(old_manifest['graph_files']))
    files[-1]['sha256'] = sha256(tail)
    files[-1]['bytes'] = tail.stat().st_size
    manifest = dict(old_manifest)
    manifest.update(version='sstv1-aquilaria-user-crops-20260919-h120-android-fp32-v1',
                    desktop_version=deployment['version'],
                    pack_id='sstv1-h120-20260927-fp32-v1', graph_files=files,
                    model_sha256=hashlib.sha256(''.join(x['sha256'] for x in files).encode()).hexdigest(),
                    model_bytes=sum(x['bytes'] for x in files), source_deployment=deployment,
                    scope='H120 weights; two-view mean for original/manual crops; one pass for SAM tight crop; offline',
                    training_performed=False)
    # Actual sequential ONNX inference on the established deterministic self-test image.
    yy, xx = np.mgrid[:224, :224]
    rgb = np.stack(((xx * 7 + yy * 3) % 256,
                    (xx * 2 + yy * 5) % 256,
                    (xx + yy * 11) % 256), axis=0).astype(np.float32)[None] / 255
    values = rgb
    for i in range(18):
        path = tail if i == 17 else OLD / f'stage_{i:02d}.onnx'
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        options.enable_cpu_mem_arena = False
        options.enable_mem_pattern = False
        session = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
        values = session.run(None, {'input': values})[0]
        del session
        print(f'SELFTEST_STAGE {i:02d}', flush=True)
    p = values[0].astype(float)
    assert len(p) == 148 and np.isfinite(p).all() and abs(p.sum() - 1) < .002
    selftest = dict(width=224, height=224,
                    recipe='r=(x*7+y*3)%256,g=(x*2+y*5)%256,b=(x+y*11)%256',
                    probabilities=p.tolist(), tolerance=.0002)
    dump(RUN / 'model_manifest.json', manifest)
    dump(RUN / 'selftest.json', selftest)
    for name in ('species.json',):
        assert sha256(OLD / name) == sha256(ASSETS / name)
    assert not PACK.exists()
    with zipfile.ZipFile(PACK, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as z:
        for i in range(18):
            source = tail if i == 17 else OLD / f'stage_{i:02d}.onnx'
            assert sha256(source) == files[i]['sha256']
            z.write(source, arcname=f'stage_{i:02d}.onnx')
        z.write(OLD / 'species.json', arcname='species.json')
        z.write(RUN / 'selftest.json', arcname='selftest.json')
        z.write(RUN / 'model_manifest.json', arcname='model_manifest.json')
    with zipfile.ZipFile(PACK) as z:
        assert set(z.namelist()) == {*(f'stage_{i:02d}.onnx' for i in range(18)),
                                     'species.json', 'selftest.json', 'model_manifest.json'}
        assert z.testzip() is None
    dump(RUN / 'model_export.json', dict(model_zip=str(PACK), zip_sha256=sha256(PACK),
                                        new_tail_sha256=sha256(tail), old_tail_sha256=sha256(OLD / 'stage_17.onnx'),
                                        selftest_top1=int(p.argmax()), model_version=deployment['version'],
                                        old_stages_unchanged=True, training_performed=False))
    verify_backup(RUN / 'pre_training_backup')
    print('PACK_READY', PACK, flush=True)


if __name__ == '__main__':
    main()
