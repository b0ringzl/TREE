"""Read-only live whole-image replay, never a training or held-out accuracy test."""
import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean, median

import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--usage', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    usage = json.loads(args.usage.read_text(encoding='utf-8'))
    args.out.mkdir(parents=True, exist_ok=False)
    rows = []
    base = 'http://127.0.0.1:8043'
    with requests.Session() as client:
        health = client.get(base + '/api/health', timeout=10).json()
        assert health['ready'] and health['single_image_version'] == usage['model_version']
        for item in usage['rows']:
            path = Path(item['path'])
            data = path.read_bytes()
            assert hashlib.sha256(data).hexdigest() == item['sha256'], str(path)
            response = client.post(base + '/api/predict', files={'image': (path.name, data)},
                                   data={'scenario': 'street', 'source': 'image', 'projection': 'photo'}, timeout=120)
            response.raise_for_status()
            result = response.json()
            assert result['image_model_version'] == usage['model_version']
            ranked = sorted(result['image_distribution'].items(), key=lambda x: x[1], reverse=True)
            expected = item['species']
            rows.append(dict(filename=item['filename'], path=item['path'], species=expected,
                lane=item['lane'], sha256=item['sha256'], exact_train='train' in item['latest_exact_roles'],
                related_train='train' in item['latest_related_roles'],
                expected_score=result['image_distribution'][expected],
                expected_rank=next(i+1 for i, (s, _) in enumerate(ranked) if s == expected),
                top3=[{k: r[k] for k in ('species', 'score')} for r in result['top3']],
                correct=result['top3'][0]['species'] == expected,
                warnings=result['warnings'], width=result['width'], height=result['height'],
                provenance=result['release_provenance']))
            if len(rows) % 20 == 0:
                print('Replayed', len(rows), '/', len(usage['rows']), flush=True)
    summaries = []
    for species in sorted({r['species'] for r in rows}):
        for cohort in ('folder_all', 'exact_added_training', 'historical_training_duplicate'):
            part = [r for r in rows if r['species'] == species and
                    (cohort == 'folder_all' or
                     (cohort == 'exact_added_training' and r['exact_train']) or
                     (cohort == 'historical_training_duplicate' and r['related_train']))]
            if not part:
                continue
            summaries.append(dict(species=species, cohort=cohort, n=len(part),
                correct=sum(r['correct'] for r in part),
                expected_score_mean=mean(r['expected_score'] for r in part),
                expected_score_median=median(r['expected_score'] for r in part),
                expected_score_below_50=sum(r['expected_score'] < .5 for r in part),
                expected_score_at_least_80=sum(r['expected_score'] >= .8 for r in part),
                lowest=[{k:r[k] for k in ('filename','expected_score','expected_rank','top3')}
                        for r in sorted(part,key=lambda r:r['expected_score'])[:5]],
                wrong_top1=dict(Counter(r['top3'][0]['species'] for r in part if not r['correct']))))
    payload = dict(created_at=datetime.now().astimezone().isoformat(),
        interpretation='Whole-image replay with actual Demo classifier and prepare; bypasses automatic target detection/cropping. Training replay, NOT independent accuracy.',
        model_version=usage['model_version'], training_performed=False, weights_changed=False,
        summary=summaries, images=rows)
    (args.out / 'results.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    lines = ['# 用户补充照片：当前 Demo 整图分类重放', '',
        '采用实际 /api/predict 接口、Photo 几何、原图等比例补边；没有执行自动选树或多边形裁剪。',
        '这不是完整上传选树流程的复现，也不是独立验证准确率。未训练、未改权重。', '',
        '| 文件夹 | 文件名 | 用途 | 正确标签分数 | 正确标签排名 | 首选预测 |',
        '|---|---|---|---:|---:|---|']
    for r in rows:
        lines.append(f"| {r['species']} | [{r['filename']}](<{Path(r['path']).as_posix()}>) | {r['lane']} | {r['expected_score']:.1%} | {r['expected_rank']} | {r['top3'][0]['species']} |")
    (args.out / 'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(summaries,ensure_ascii=False,indent=2),flush=True)


if __name__ == '__main__':
    main()
