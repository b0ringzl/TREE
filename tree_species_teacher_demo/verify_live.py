"""Explicit local smoke test; no fitting or deployment writes."""
import json
import sys
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from training_safety import sha256

experiment = ROOT / 'outputs/distillation/bioclip_probe_20260908_0227'
reference = json.loads((experiment / 'raw_image_inference.json').read_text(encoding='utf-8'))
image = Path(reference['image'])
results = {}
for mode in ('adapted', 'teacher'):
    with image.open('rb') as handle:
        response = requests.post('http://127.0.0.1:8039/api/predict', files={'image': ('input.jpg', handle, 'image/jpeg')}, data={'mode': mode}, timeout=90)
    response.raise_for_status()
    result = response.json()
    assert len(result['top3']) == 3
    assert all(0 <= row['score'] <= 1 for row in result['top3'])
    assert result['mode'] == mode
    results[mode] = {k: result[k] for k in ('model', 'top3', 'inference_seconds', 'encoder_sha256')}
assert results['adapted']['top3'][0]['species'] == reference['top3'][0]['species']
assert abs(results['adapted']['top3'][0]['score'] - reference['top3'][0]['score']) < .02
assert requests.post('http://127.0.0.1:8039/api/predict', files={'image': ('bad.jpg', b'bad')}, timeout=10).status_code == 400
assert requests.get('http://127.0.0.1:8037/api/health', timeout=10).json()['status'] == 'ok'
protocol = json.loads((experiment / 'protocol.json').read_text(encoding='utf-8'))
protected = {path: sha256(Path(path)) == expected for path, expected in protocol['protected_sources'].items()}
assert all(protected.values()), protected
report = {'results': results, 'production_unchanged': protected, 'offline_reference_top1_matches': True,
          'training_performed': False, 'old_demo_healthy': True, 'sample_count': 1,
          'notice': 'Integration smoke test, not an accuracy evaluation.'}
print(json.dumps(report, ensure_ascii=False, indent=2))
output = Path(__file__).parent / '.runtime/live_verification.json'
output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
