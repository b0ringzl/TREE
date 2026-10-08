"""Compare the new Android ONNX tail with the live H120 service on fixed photos."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'outputs/android_toolchain_20260918_v1/python_deps')]
import numpy as np
import onnxruntime as ort
import requests
from tree_species_teacher_demo.engine import decode
from spacescan_trees_demo_v1.server import prepare

RUN = ROOT / 'outputs/android_h120_ui_sync_20260927_v1'
OLD = ROOT / 'outputs/SpaceScan_Android_SSTv1_20260919_v1/model_pack'
TAIL = RUN / 'stage_17_h120.onnx'
LABELS = [row['label'] for row in json.loads((OLD / 'species.json').read_text(encoding='utf-8'))]
PHOTOS = [
    Path('C:/Users/25041725G/Desktop/验证/微信图片_20260916084639_130_2.jpg'),
    Path('C:/Users/25041725G/Desktop/Aquilaria sinensis/B9A2B0F30924AEF1.jpg'),
]


def main():
    inputs = []
    truth = []
    for path in PHOTOS:
        data = path.read_bytes()
        padded, _, _ = prepare(data)
        whole = decode(padded)[0].numpy()[None].astype(np.float32)
        detail = decode(data)[0].numpy()[None].astype(np.float32)
        inputs.extend((whole, detail))
        response = requests.post('http://127.0.0.1:8039/api/predict-context',
            files={'image': ('image.jpg', padded, 'image/jpeg'), 'detail': ('image.jpg', data, 'image/jpeg')},
            timeout=180)
        response.raise_for_status()
        truth.append((path, response.json()))
    values = inputs
    for i in range(18):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        options.enable_cpu_mem_arena = False
        options.enable_mem_pattern = False
        session = ort.InferenceSession(str(TAIL if i == 17 else OLD / f'stage_{i:02d}.onnx'),
                                       sess_options=options, providers=['CPUExecutionProvider'])
        values = [session.run(None, {'input': x})[0] for x in values]
        del session
        print('PARITY_STAGE', i, flush=True)
    rows = []
    for i, (path, desktop) in enumerate(truth):
        combined = .5 * (values[2*i][0] + values[2*i+1][0])
        expected = np.array([desktop['sstv1_distribution'][s] for s in LABELS], dtype=np.float32)
        error = np.abs(combined - expected)
        top_mobile = LABELS[int(combined.argmax())]
        top_desktop = desktop['comparison']['sstv1'][0]['species']
        rows.append(dict(photo=str(path), android_onnx_top1=top_mobile, desktop_h120_top1=top_desktop,
                         same_top1=top_mobile == top_desktop, max_abs=float(error.max()),
                         l1=float(error.sum())))
    result = {'cases': rows, 'pass': all(x['same_top1'] and x['max_abs'] <= .04 for x in rows),
              'scope': 'Fixed inspected conversion agreement, not an independent species-accuracy estimate'}
    (RUN / 'h120_weight_parity.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if not result['pass']:
        raise SystemExit('H120 conversion gate failed')


if __name__ == '__main__':
    main()
