"""Chinese readout, cohort separation and paired error audit; no model selection."""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from fresh_bioclip146_benchmark_20261006 import SEED, load, metrics, save, sha256, verify_backup


def report(out):
    backup = verify_backup(out / 'pre_training_backup')
    rows = load(out / 'manifest.json')
    import json
    predictions = [json.loads(line) for line in (out / 'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
    if len(rows) != len(predictions) or any(r['id'] != p['id'] for r, p in zip(rows, predictions)):
        raise RuntimeError('Incomplete or misaligned predictions')
    labels = [r['label'] for r in load(out / 'catalog146.json')]
    results = load(out / 'RESULTS.json')
    for cohort in sorted({r.get('cohort', 'primary_20260525_20261006') for r in rows}):
        indices = [i for i, r in enumerate(rows) if r.get('cohort', 'primary_20260525_20261006') == cohort]
        results['subsets'][cohort] = metrics([rows[i] for i in indices], [predictions[i] for i in indices], labels)
    corrected, regressed = [], []
    for row, p in zip(rows, predictions):
        before = p['top3']['teacher_context'][0] == row['label']
        after = p['top3']['sstv1_context'][0] == row['label']
        if before == after: continue
        detail = dict(row)
        detail.update({'bioclip_top3': p['top3']['teacher_context'], 'sst_top3': p['top3']['sstv1_context']})
        (corrected if after else regressed).append(detail)
    save(out / 'paired_errors.json', {'corrected': corrected, 'regressed': regressed})
    # Supplementary uncertainty check: resample observers, keeping all of their
    # images across classes together. No class/model is selected using this CI.
    groups = sorted({r['observer_id'] for r in rows})
    aggregates = []
    for group in groups:
        indices = [i for i, r in enumerate(rows) if r['observer_id'] == group]
        delta = sum(int(predictions[i]['top3']['sstv1_context'][0] == rows[i]['label']) -
                    int(predictions[i]['top3']['teacher_context'][0] == rows[i]['label']) for i in indices)
        aggregates.append((delta, len(indices)))
    a = np.asarray(aggregates)
    rng = np.random.default_rng(SEED + 2)
    sampled = a[rng.integers(0, len(a), (10000, len(a)))].sum(axis=1)
    results['observer_cluster_bootstrap95_gain_pp'] = (np.quantile(sampled[:, 0]/sampled[:, 1], [.025, .975])*100).tolist()
    results['distinct_observers'] = len(groups)
    results['production_verification'] = {'unchanged': True, 'protected_files': len(backup['files']),
                                           'backup_manifest': str(out / 'pre_training_backup/backup_manifest.json')}
    save(out / 'FINAL_RESULTS.json', results)
    save(out / 'production_verification.json', results['production_verification'])
    all_stats = results['all']; before = all_stats['modes']['teacher_context']; after = all_stats['modes']['sstv1_context']
    paired = all_stats['paired_comparisons']['teacher_context_vs_sstv1_context']
    ci = results['observer_cluster_bootstrap95_gain_pp']
    before_center = all_stats['modes']['teacher_center']; after_center = all_stats['modes']['sstv1_center']
    cohort_names = {'primary_20260525_20261006': '近期照片（2026-05-25 至 10-06）',
                    'older_20250101_20260524': '较早照片补采（2025-01-01 至 2026-05-24）',
                    'hong_kong': '香港本地照片', 'outside_hong_kong': '香港以外照片',
                    'recent_after_base_project': '2026-09-08 以来照片'}
    lines = ['# 原始 BioCLIP 与当前 SST：新采样对照结果', '',
        f"结论：在本轮 {len(rows)} 张、覆盖 {all_stats['covered_classes']}/146 个展示类别的样本上，同取景对照 Top-1 从 **{before['top1']:.2%}** 变为 **{after['top1']:.2%}**，变化 **{paired['gain_percentage_points']:+.2f} 个百分点**。",
        '', '**这不是全部 146 类的香港街景准确率。** 样本标签来自多名 iNaturalist 社区鉴定者，不是本项目专家复核的真值；世界各地的枝叶、花果和全树照片混合，尚不能代表香港实际街景。', '',
        '## 同一批图片、同一类别目录的比较', '',
        '| 对照方式 | 原 BioCLIP | 当前 SST | 变化 |', '| --- | ---: | ---: | ---: |',
        f"| 全图＋中心细节，两边相同处理（主要模型对照） | {before['correct']}/{len(rows)} = {before['top1']:.2%} | {after['correct']}/{len(rows)} = {after['top1']:.2%} | {paired['gain_percentage_points']:+.2f} 个百分点 |",
        f"| 只用原图中心取景，两边相同处理 | {before_center['top1']:.2%} | {after_center['top1']:.2%} | {all_stats['paired_comparisons']['teacher_center_vs_sstv1_center']['gain_percentage_points']:+.2f} 个百分点 |",
        f"| 原始单次中心取景 → 当前正式双取景流程 | {before_center['top1']:.2%} | {after['top1']:.2%} | {all_stats['paired_comparisons']['teacher_center_vs_sstv1_context']['gain_percentage_points']:+.2f} 个百分点 |",
        '', f"Top-3：{before['top3']:.2%} → {after['top3']:.2%}。各已覆盖类别等权的平均召回率：{before['macro_recall_covered']:.2%} → {after['macro_recall_covered']:.2%}。",
        f"原来错、现在对：{paired['corrected']} 张；原来对、现在错：{paired['regressed']} 张。没有删除识错图片。",
        f"按观察者成组重采样的提升 95% 区间：[{ci[0]:+.2f}, {ci[1]:+.2f}] 个百分点；共 {len(groups)} 名观察者。该区间仅反映当前采样波动，不涵盖标签错误、地点和取景偏差。", '',
        '## 分批与地域结果', '', '| 样本批次 | 图片 / 覆盖类别 | 原 BioCLIP Top-1 | SST Top-1 | 变化 |', '| --- | ---: | ---: | ---: | ---: |']
    for key, stats in results['subsets'].items():
        if not stats['n']: continue
        lines.append(f"| {cohort_names.get(key, key)} | {stats['n']} / {stats['covered_classes']} | {stats['modes']['teacher_context']['top1']:.2%} | {stats['modes']['sstv1_context']['top1']:.2%} | {stats['paired_comparisons']['teacher_context_vs_sstv1_context']['gain_percentage_points']:+.2f} 个百分点 |")
    lines += ['', f"香港子集仅 {results['subsets']['hong_kong']['n']} 张，不足以给出可靠的香港地区识别率。每类最多 5 张，也不足以精确估计单个树种的准确率。", '',
        '## 模型与验证范围', '',
        '- 原始模型指项目原先使用的 **BioCLIP 2.5 零样本分类器**，不是另一个 BioCLIP 1 检查点。两边使用相同图像编码器、冻结提示词及类别概率合并规则。',
        '- 当前模型为正式 H120 三分类头集成，版本 `sstv1-aquilaria-user-crops-20260919-h120`。图像编码器没有因本次验证而改变。',
        '- 148 个原输出合并为 146 个展示类别；洋紫荊、榕树与南洋杉是合并类别，这些成绩不证明能区分组内树种。',
        '- 主比较用两边相同的全图补边＋中心细节等权平均，隔离分类头带来的变化；另列最初单次中心取景与当前流程的差异。',
        '- 在首、中、末三张图片上，独立计算与正式 Demo 的实际 `infer(..., context)` 路径逐项对照，允许误差不超过 1e-6。',
        '- 本次衡量单目标图片分类，不含自动检测、SAM、视频/走廊跟踪或点云融合，不能据此宣布这些完整流程的准确率。', '',
        '## 新数据如何获得', '',
        '- 按固定随机种子、固定日期范围和每类上限采集；先找香港，再补全球资料。较早照片只按类别缺样情况补采，发生在所有模型评分之前。',
        '- 仅收 CC0 / CC BY / CC BY-SA 原始图片；保留署名、许可、原始来源链接和水印。用途为本地验证，不发布图片包。',
        '- 使用研究级且至少两名当前独立一致鉴定者的记录；仍不是专家核准的训练标签。没有更改 `human_review.sqlite3`，所有图片均不具备自动训练资格。',
        '- 排除已登记历史观察记录、照片 ID、SHA-256 重复和 pHash 近重复；不按模型分数或预测对错过滤。锁定清单后才推理。',
        '- “新采到/项目未登记”不等于能够证明从未进入基础模型预训练，也不能绝对排除不同视角拍到同一棵树。', '',
        '## 逐类结果', '', '| 展示类别 | 样本数 | BioCLIP 识对 | SST 识对 | 净变化（张） |', '| --- | ---: | ---: | ---: | ---: |']
    for row in all_stats['per_class']:
        a_count = row['modes']['teacher_context']['correct']; b_count = row['modes']['sstv1_context']['correct']
        lines.append(f"| {row['label']} | {row['n']} | {a_count} | {b_count} | {b_count-a_count:+d} |")
    lines += ['', '无合格新样本的类别（不能据此评估）：', '', ', '.join(results['missing_classes']), '',
        '## 文件与安全', '',
        '- `FINAL_RESULTS.json`：完整指标、分批结果与缺样类别；`paired_errors.json`：所有纠正/退步图片及原始来源。',
        '- `manifest.json`：逐图许可、标签来源、哈希、地域与日期；`predictions.jsonl`：六种处理方式的全部原始分布。',
        '- `collection_lock.json`：评分前冻结清单；`contact_sheets/`：可人工复核的图片总览。',
        f"- 已重新校验 {len(backup['files'])} 个受保护来源文件及物理备份，正式部署权重、编码器与代码均未改变；本次没有训练、调阈值或换模型。",
        f"- 备份：`{out / 'pre_training_backup/backup_manifest.json'}`。", '']
    (out / 'SUMMARY_ZH.md').write_text('\n'.join(lines), encoding='utf-8')
    print(lines[2], flush=True)
    print('SOURCE/DEPLOYMENT VERIFICATION: unchanged', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--out', type=Path, required=True)
    report(parser.parse_args().out.resolve())
