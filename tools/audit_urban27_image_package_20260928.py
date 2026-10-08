"""Read-only, resumable replay of each packaged image through the live web classifier.

This is a source-label agreement audit, NOT an independent test of accuracy.
No fitting, relabeling, or deployed-file writes occur.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'outputs/urban27_image_quality_20260928_v1'
BACKUP = RUN / 'pre_curation_backup'
MODEL_VERSION = 'sstv1-aquilaria-user-crops-20260919-h120'
PIPELINE = 'sstv1-confirmed-target-whole-center-mean-v2'

SPECIES = [
    ('01_柱狀南洋杉', 'Araucaria columnaris', 'Araucaria columnaris(Araucaria heterophylla)', 'merged'),
    ('02_假檳榔', 'Archontophoenix alexandrae', None, 'exact'),
    ('03_秋楓', 'Bischofia javanica', None, 'exact'),
    ('04_串錢柳', 'Callistemon viminalis', None, 'exact'),
    ('05_木麻黃', 'Casuarina equisetifolia', None, 'exact'),
    ('06_陰香', 'Cinnamomum burmannii', None, 'exact'),
    ('07_樟', 'Cinnamomum camphora', None, 'exact'),
    ('08_水石榕', 'Elaeocarpus hainanensis', None, 'exact'),
    ('09_垂葉榕', 'Ficus benjamina', 'Ficus microcarpa / Ficus benjamina (榕樹)', 'merged'),
    ('10_細葉榕', 'Ficus microcarpa', 'Ficus microcarpa / Ficus benjamina (榕樹)', 'merged'),
    ('11_銀樺', 'Grevillea robusta', None, 'exact'),
    ('12_藍花楹', 'Jacaranda mimosifolia', None, 'exact'),
    ('13_大花紫薇', 'Lagerstroemia speciosa', None, 'exact'),
    ('14_蒲葵', 'Livistona chinensis', None, 'exact'),
    ('15_荷花玉蘭', 'Magnolia grandiflora', None, 'exact'),
    ('16_白楸', 'Mallotus paniculatus', None, 'exact'),
    ('17_白千層', 'Melaleuca cajuputi', None, 'exact'),
    ('18_樂昌含笑', 'Michelia chapensis', None, 'unsupported'),
    ('19_大王椰子', 'Roystonea regia', None, 'exact'),
    ('20_鵝掌柴', 'Schefflera heptaphylla', None, 'exact'),
    ('21_火焰樹', 'Spathodea campanulata', None, 'exact'),
    ('22_假蘋婆', 'Sterculia lanceolata', None, 'exact'),
    ('23_黃鐘木', 'Tabebuia chrysantha', None, 'exact'),
    ('24_紅花風鈴木', 'Tabebuia rosea', None, 'unsupported'),
    ('25_欖仁', 'Terminalia catappa', None, 'exact'),
    ('26_小葉欖仁', 'Terminalia mantaly', None, 'exact'),
    ('27_二喬木蘭', 'Yulania x soulangeana', None, 'unsupported'),
]


def read_existing(path):
    seen = set()
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            if line.strip():
                seen.add(json.loads(line)['relative_path'])
    return seen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    verification = json.loads((RUN / 'pre_curation_backup_verification.json').read_text(encoding='utf-8'))
    if not verification['sha256_verified'] or Path(verification['backup']) != BACKUP:
        raise RuntimeError('Backup verification is required before replay')
    images = []
    for folder, taxon, merged_label, lane in SPECIES:
        directory = BACKUP / folder
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        for path in sorted(directory.iterdir()):
            if path.is_file():
                images.append((path, taxon, merged_label or taxon, lane))
    if len(images) != verification['files']:
        raise RuntimeError('Backup image count no longer matches verification')

    output = RUN / 'predictions.jsonl'
    seen = read_existing(output)
    with requests.Session() as session:
        health = session.get('http://127.0.0.1:8043/api/health', timeout=20).json()
        if not (health['ready'] and health['classifier_alignment'] == 'aligned'
                and health['single_image_version'] == MODEL_VERSION
                and health['inference_pipeline'] == PIPELINE):
            raise RuntimeError(f'Unexpected live classifier: {health}')
        protocol = {
            'model_version': MODEL_VERSION,
            'inference_pipeline': PIPELINE,
            'endpoint': 'http://127.0.0.1:8043/api/predict',
            'request': {'scenario': 'street', 'source': 'image', 'projection': 'photo', 'view_policy': 'context'},
            'scope': 'Each backed-up image is sent as one already-selected target. Tree detection, panorama conversion, SAM, and manual target selection are NOT evaluated.',
            'interpretation': 'Agreement with source labels from a mixed training/reference/pending archive; not independent accuracy.',
            'image_count': len(images),
            'training_performed': False,
            'deployed_weights_changed': False,
        }
        (RUN / 'audit_protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding='utf-8')
        processed_this_run = 0
        with output.open('a', encoding='utf-8') as handle:
            for number, (path, source_taxon, expected, lane) in enumerate(images, 1):
                relative = str(path.relative_to(BACKUP))
                if relative in seen:
                    continue
                data = path.read_bytes()
                record = {
                    'relative_path': relative,
                    'source_taxon': source_taxon,
                    'expected_model_label': expected if lane != 'unsupported' else None,
                    'evaluation_lane': lane,
                    'bytes': len(data),
                    'sha256': hashlib.sha256(data).hexdigest(),
                }
                for attempt in range(3):
                    try:
                        with session.post('http://127.0.0.1:8043/api/predict',
                                          files={'image': (path.name, data)},
                                          data=protocol['request'], timeout=180) as response:
                            response.raise_for_status()
                            result = response.json()
                        if (result['image_model_version'] != MODEL_VERSION
                                or result['inference_pipeline'] != PIPELINE):
                            raise RuntimeError('Model or pipeline changed mid-audit')
                        distribution = result['image_distribution']
                        raw_top3 = result['raw_model_top3']
                        record.update({
                            'status': 'ok',
                            'top1': raw_top3[0]['species'],
                            'top1_score': raw_top3[0]['score'],
                            'top3': raw_top3,
                            'expected_score': distribution.get(expected) if lane != 'unsupported' else None,
                            'expected_rank': (1 + sum(v > distribution[expected] for v in distribution.values())) if lane != 'unsupported' else None,
                            'source_label_agreement': raw_top3[0]['species'] == expected if lane != 'unsupported' else None,
                            'width': result['width'],
                            'height': result['height'],
                            'warnings': result['warnings'],
                            'view_agreement': result.get('view_evidence', {}).get('agreement'),
                            'inference_seconds': result['inference_seconds'],
                        })
                        break
                    except (requests.RequestException, ValueError, KeyError) as exc:
                        if attempt == 2:
                            record.update(status='error', error=str(exc))
                        else:
                            time.sleep(1 + attempt)
                handle.write(json.dumps(record, ensure_ascii=False) + '\n')
                handle.flush()
                processed_this_run += 1
                if processed_this_run % 50 == 0:
                    print(f'Replayed {len(seen) + processed_this_run}/{len(images)}', flush=True)
                if args.limit and processed_this_run >= args.limit:
                    break
    print(f'Done this run: {processed_this_run}; total recorded: {len(read_existing(output))}/{len(images)}', flush=True)


if __name__ == '__main__':
    main()
