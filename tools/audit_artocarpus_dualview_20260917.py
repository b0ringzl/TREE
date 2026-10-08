"""One locked 50/50, same-image probability ensemble; inference-cache audit only.

This proposal follows the v1 development audit. It is not independent evidence,
does not double the number of observations, and never changes the Demo.
"""
import argparse
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))

import numpy as np
from distillation_core import bootstrap_gain, digest, grouped_metrics, metrics, read_json, save_json
from training_safety import sha256

SOURCE = ROOT / 'outputs/artocarpus_preprocess_audit_20260917_v1'
DEFAULT_OUT = ROOT / 'outputs/artocarpus_preprocess_audit_20260917_v2'
SEED = 20260917
REPEATS = 2000
BRANCHES = ('center_crop', 'demo_letterbox', 'equal_dualview')
# Descriptive visual audit of every fixed positive, not rules used by a model.
# Reviewed on v1's exact input pairs, before computing any mixed probabilities.
ORGANS = {
    '36dc28ae799053fc59be': ('crown_partial', '枝葉／局部冠層；仰視，背景建築'),
    '387c74502e80346dbfb5': ('trunk_partial_crown', '樹幹與分叉／局部冠層；非完整單木'),
    '5daf3da6107fe535e809': ('fruit_leaves', '綠色果實與葉片近景'),
    '80092a9723ed7989f319': ('leaves', '林下葉片／小植株近景'),
    '933ccb50ac69b4aa72df': ('leaves', '葉片／小植株近景，樹幹與落葉背景'),
    'd91eea1ea89d22daf65b': ('trunk_partial_crown', '粗樹幹、分叉與仰視局部冠層'),
    'cons_browser_ed5912be3fd4a6f0f076': ('fruit_leaves', '綠色果實、葉片正背面與枝條'),
    'cons_browser_11714b6f5fd6aa08372f': ('leaves_reproductive', '葉片及腋生小型球狀生殖構造'),
    'cons_browser_c3642d712623ce4d6e7e': ('fruit_leaves', '多個果實、枝條及成熟葉片'),
    'cons_browser_7e04cd949b2292894806': ('leaves_reproductive', '成熟葉片正背面與小型球狀生殖構造'),
    'cons_browser_148cfd190d4127b544d9': ('fruit_leaves', '果串、逆光葉片與枝條'),
    'cons_browser_bdbc82efa22c9d100623': ('crown_tree_row', '樹冠與下段樹幹；鄰樹樹列背景，非隔離單木'),
    'cons_browser_1fc9277a3d80a4adae65': ('whole_tree', '花園整株樹木、樹冠與樹幹；背景另有樹木'),
    'cons_browser_ff6f00d57d8772d8dc7a': ('leaves', '葉片與枝條近景'),
    'cons_browser_14dd45f03d81c0df66e2': ('reproductive_leaves', '白綠球形生殖構造與葉片近景；不另判花果發育階段'),
    'cons_browser_ef18eb572761613ab0ea': ('fruit_leaves', '黄色果實、葉片與枝條')}


def require(condition, message):
    if not condition: raise RuntimeError(message)


def binary_counts(y, p):
    truth = np.asarray(y) == 16; pred = p.argmax(1) == 16
    tp = int((truth & pred).sum()); fp = int((~truth & pred).sum())
    fn = int((truth & ~pred).sum()); tn = int((~truth & ~pred).sum())
    return dict(tp=tp, fp=fp, fn=fn, tn=tn,
        precision=tp/(tp+fp) if tp+fp else None,
        recall=tp/(tp+fn) if tp+fn else None,
        f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None)


def f1_bootstrap(rows, a, b):
    groups = defaultdict(list)
    for i, row in enumerate(rows): groups[row['group_id']].append(i)
    y = np.array([r['target'] for r in rows]); ap = a.argmax(1); bp = b.argmax(1)
    counts = []
    for indices in groups.values():
        truth = y[indices] == 16; aa = ap[indices] == 16; bb = bp[indices] == 16
        counts.append([[int((truth & p).sum()), int((~truth & p).sum()), int((truth & ~p).sum())] for p in (aa, bb)])
    counts = np.array(counts); rng = np.random.default_rng(SEED); deltas = []
    for _ in range(REPEATS):
        chosen = rng.integers(0, len(counts), len(counts)); total = counts[chosen].sum(0)
        denominator = 2*total[:, 0] + total[:, 1] + total[:, 2]
        if (denominator == 0).any(): continue
        f1 = 2*total[:, 0] / denominator
        deltas.append(float(f1[1]-f1[0]))
    ba, bb = binary_counts(y, a), binary_counts(y, b)
    return {'metric': 'white_guimu_f1_difference_paired_source_group_bootstrap',
        'point': bb['f1']-ba['f1'] if bb['f1'] is not None and ba['f1'] is not None else None,
        'ci95': np.quantile(deltas, [.025, .975]).tolist() if deltas else None,
        'requested_repeats': REPEATS, 'valid_repeats': len(deltas),
        'undefined_denominator_repeats_excluded': REPEATS-len(deltas),
        'interpretation': 'Exploratory intervals on reused development groups, not independent-confirmation confidence.'}


def comparison(rows, baseline, mixed):
    y = np.array([r['target'] for r in rows]); aa = baseline.argmax(1); bb = mixed.argmax(1)
    return {'n': len(rows), 'corrected': int(((aa != y) & (bb == y)).sum()),
        'regressed': int(((aa == y) & (bb != y)).sum()), 'changed_top1': int((aa != bb).sum()),
        'corrected_ids': [r['sample_id'] for r, ok in zip(rows, (aa != y) & (bb == y)) if ok],
        'regressed_ids': [r['sample_id'] for r, ok in zip(rows, (aa == y) & (bb != y)) if ok],
        'accuracy_bootstrap': bootstrap_gain(y, baseline, mixed, [r['group_id'] for r in rows], seed=SEED, repeats=REPEATS),
        'white_guimu_f1_bootstrap': f1_bootstrap(rows, baseline, mixed)}


def top3(p, labels):
    ids = np.argsort(-p, kind='stable')[:3]
    return [{'index': int(i), 'species': labels[i], 'score': float(p[i])} for i in ids]


def summarize(rows, arrays, nclass):
    y = [r['target'] for r in rows]
    result = {'branches': {name: {'classification': metrics(y, p), 'white_guimu': binary_counts(y, p),
        'group_classification': grouped_metrics(rows, p)} for name, p in arrays.items()},
        'comparisons': {f'equal_dualview_vs_{name}': comparison(rows, arrays[name], arrays['equal_dualview'])
                        for name in ('center_crop', 'demo_letterbox')},
        'class_count': nclass}
    return result


def format_top3(items):
    return '; '.join(f'{x["species"]} {x["score"]:.1%}' for x in items)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(); out = args.out.resolve()
    require(out.is_relative_to(ROOT / 'outputs') and not out.exists(), 'Choose a new output directory under outputs')
    require(read_json(SOURCE/'COMPLETE.json')['production_and_source_hashes_unchanged'], 'v1 audit incomplete')
    rows = read_json(SOURCE/'rows.json'); labels = read_json(SOURCE/'labels150.json')
    require(len(rows) == 2655 and len(labels) == 150, 'Fixed cohort/taxonomy changed')
    require({r['sample_id'] for r in rows if r['target'] == 16} == set(ORGANS), 'Positive-case audit must include all fixed16')
    paths = [SOURCE / x for x in ('protocol.json', 'rows.json', 'labels150.json', 'COMPLETE.json',
                                 'paired_features_predictions.npz', 'pixel_diagnostics.json', 'duplicate_groups.json')]
    paths += [Path(__file__).resolve()]
    before = {str(p): sha256(p) for p in paths}
    out.mkdir(parents=True, exist_ok=False)
    protocol = {
        'created_at': datetime.now(timezone.utc).isoformat(), 'source_audit': str(SOURCE),
        'source_hashes': before, 'row_digest': digest(rows), 'seed': SEED,
        'hypothesis_timing': 'This candidate is proposed after inspecting v1 development results. The sole weight is locked here before loading/combining cached probabilities, not preregistered before v1.',
        'only_candidate': {'name': 'equal_dualview', 'center_crop_probability_weight': .5, 'demo_letterbox_probability_weight': .5},
        'operation': 'Arithmetic probability mean, separately for native148 and unchanged150; no logits mixing, no recalibration, no threshold fitting.',
        'no_search': True, 'no_training': True, 'no_gpu': True, 'no_deployment': True,
        'number_of_unique_original_rows': 2655, 'views_per_original': 2,
        'views_are_independent_samples': False,
        'cohort': 'All2640 old planning plus all15 already-inspected new-source development; no test set read.',
        'organ_labels': 'Descriptive agent visual review of all16 exact v1 input pairs, not botanical re-identification, not used for routing or selection.',
        'metrics': 'Top1, macroF1, white-guimu TP/FP/F1, source-group mean probabilities, paired source-group bootstrap with2000 draws.',
        'native148_outside_catalog': 'New5 non-white-guimu species unavailable in148; excluded only from148 multiclass accuracy, retained in all15 binary diagnostics.',
        'validity': 'Same-image multi-crop inference, not multi-angle or multi-temporal evidence. Development-only exploration; deployment requires independent validation.',
        'all_white_guimu': sorted(ORGANS)}
    # Lock the only blend before computing it. There are no alternate weights.
    save_json(out/'protocol.json', protocol)
    lock_sha = sha256(out/'protocol.json')
    arrays = {}
    with np.load(SOURCE/'paired_features_predictions.npz') as cached:
        require(str(cached['rows_hash']) == digest(rows), 'Cache rows do not match')
        for nclass in (148, 150):
            a = cached[f'center_crop_p{nclass}']; b = cached[f'demo_letterbox_p{nclass}']
            require(a.shape == b.shape == (2655, nclass), 'Probability shape mismatch')
            require(np.isfinite(a).all() and np.isfinite(b).all(), 'Nonfinite input')
            require(np.allclose(a.sum(1), 1., atol=1e-5) and np.allclose(b.sum(1), 1., atol=1e-5), 'Unnormalized input')
            arrays[f'center_crop_p{nclass}'] = a.copy()
            arrays[f'demo_letterbox_p{nclass}'] = b.copy()
            arrays[f'equal_dualview_p{nclass}'] = .5*a + .5*b
    np.savez_compressed(out/'fixed_equal_dualview_predictions.npz', **arrays, rows_hash=digest(rows))
    save_json(out/'rows.json', rows); save_json(out/'labels150.json', labels)
    results = {}
    for cohort in ('old_planning', 'new_source_development'):
        for nclass in (148, 150):
            ids = [i for i, r in enumerate(rows) if r['audit_cohort'] == cohort and r['target'] < nclass]
            results[f'{cohort}_{nclass}'] = summarize([rows[i] for i in ids],
                {b: arrays[f'{b}_p{nclass}'][ids] for b in BRANCHES}, nclass)
        ids = [i for i, r in enumerate(rows) if r['audit_cohort'] == cohort]
        results[f'{cohort}_native148_binary_all'] = {b: binary_counts([rows[i]['target'] for i in ids], arrays[f'{b}_p148'][ids]) for b in BRANCHES}
    save_json(out/'evaluation.json', results)
    cases = []
    for i, row in enumerate(rows):
        if row['target'] != 16: continue
        category, description = ORGANS[row['sample_id']]
        case = {'sample_id': row['sample_id'], 'cohort': row['audit_cohort'], 'group_id': row['group_id'],
            'source_image': row['source_image'], 'sha256': row['audit_sha256'], 'truth': labels[16],
            'organ_category': category, 'organ_description': description,
            'organ_authority': 'Agent visual review of the exact inference input pair; not human species confirmation.',
            'source_recorded_organ': row.get('visual_qa', {}).get('organ'),
            'image_pair': str(SOURCE/'white_guimu_inputs'/(row['sample_id']+'.png')),
            'by_catalog': {str(n): {b: {'top3': top3(arrays[f'{b}_p{n}'][i], labels),
                'white_guimu_probability': float(arrays[f'{b}_p{n}'][i, 16]),
                'white_guimu_rank': int(np.flatnonzero(np.argsort(-arrays[f'{b}_p{n}'][i], kind='stable') == 16)[0])+1,
                'correct': int(arrays[f'{b}_p{n}'][i].argmax()) == 16}
                for b in BRANCHES} for n in (148, 150)}}
        cases.append(case)
    save_json(out/'all16_white_guimu_cases.json', cases)
    lines = ['# 全部16张白桂木：同一图像双视图审计', '',
        '所有白桂木固定真值样本均列出；未按正确与否筛选。器官标签只是可见内容描述，不重新确认树种，也不用于模型分流。', '']
    for case in cases:
        lines += [f'## {case["sample_id"]}', '', f'{case["cohort"]}；{case["organ_description"]}。来源组：`{case["group_id"]}`。', '',
            f'![中央裁切与Demo补边输入](<{case["image_pair"]}>)', '']
        for nclass in (148, 150):
            lines += [f'目录 {nclass}：', '']
            for branch in BRANCHES:
                pred = case['by_catalog'][str(nclass)][branch]
                lines += [f'- {branch}: {format_top3(pred["top3"])}。白桂木第 {pred["white_guimu_rank"]}，分数 {pred["white_guimu_probability"]:.1%}。']
            lines.append('')
    (out/'WHITE_GUIMU_CASES.md').write_text('\n'.join(lines), encoding='utf-8')
    lines = ['# 固定50/50同图双视图：研究候选审计', '',
        '未训练、未调混合权重、未重跑编码器、未接入Demo。唯一候选为同图中央裁切与当前Demo补边的分类概率各50%。', '',
        '| 固定开发集 | 中央裁切 Top1 | Demo补边 Top1 | 50/50 Top1 | 白桂木 TP/FP/F1：裁切 → 补边 → 双视图 |',
        '|---|---:|---:|---:|---|']
    for key in ['old_planning_148', 'old_planning_150', 'new_source_development_148', 'new_source_development_150']:
        r = results[key]['branches']; acc = [r[b]['classification']['top1'] for b in BRANCHES]
        white = ' → '.join(f'{r[b]["white_guimu"]["tp"]}/{r[b]["white_guimu"]["fp"]}/{r[b]["white_guimu"]["f1"]:.3f}' for b in BRANCHES)
        lines.append(f'| {key} | {acc[0]:.2%} | {acc[1]:.2%} | {acc[2]:.2%} | {white} |')
    for key in ['old_planning_148', 'new_source_development_150']:
        lines += ['', f'## {key}', '']
        for name, item in results[key]['comparisons'].items():
            bs = item['accuracy_bootstrap']; f1 = item['white_guimu_f1_bootstrap']
            lines += [f'- {name}：改正 {item["corrected"]}、退步 {item["regressed"]}；Top1差 {bs["point"]:+.2%}，来源组bootstrap95%区间 [{bs["ci95"][0]:+.2%}, {bs["ci95"][1]:+.2%}]。',
                f'  白桂木F1差 {f1["point"]:+.3f}，区间 {f1["ci95"]}。']
        group_acc = [results[key]['branches'][b]['group_classification']['top1'] for b in BRANCHES]
        lines += ['', f'来源组平均概率Top1（裁切／补边／双视图）：{group_acc[0]:.2%}／{group_acc[1]:.2%}／{group_acc[2]:.2%}。']
    organ_counts = Counter(c['organ_category'] for c in cases)
    lines += ['', '## 全部白桂木病例与限制', '',
        f'全部16张均保留。器官描述计数：{dict(organ_counts)}。各图148/150三分支完整Top3和输入对照见 WHITE_GUIMU_CASES.md；完整分布见NPZ。', '',
        '- 本候选是在看过v1开发结果后提出，不是独立新数据验证。没有继续搜索其他混合权重；不得凭此直接部署。',
        '- 同一原图的两个预处理视图不是两棵树，也不是多角度或多时相采集；样本数仍是2655。',
        '- 少量来源图库不是独立单木。bootstrap按既有来源组重采样，仅供开发诊断；缺少独立采集和物种鉴定误差建模。',
        '- 新15中的5张物种不在原148目录，原148多分类只算其余10张；二元白桂木误检检查仍包括15张。150目录追加文本向量沿用v1。',
        '- 叶、果、枝干占多数。整株/树冠能力仍须额外独立数据验证，不能用本16张总体成绩代替。',
        '- v1全部产物只读复用；生产模型、预处理和Demo均未修改。', '']
    (out/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    require(sha256(out/'protocol.json') == lock_sha, 'Locked50/50 protocol changed')
    require({p: sha256(p) for p in before} == before, 'Read-only source artifact changed')
    save_json(out/'COMPLETE.json', {'completed_at': datetime.now(timezone.utc).isoformat(),
        'protocol_sha256': lock_sha, 'images': 2655, 'new_images': 0, 'encoder_calls': 0,
        'fit': False, 'deployed': False, 'source_artifacts_unchanged': True, 'candidate_weights_evaluated': 1,
        'white_guimu_images': 16, 'independent_confirmation': False})
    print('COMPLETE', out, flush=True)
    for key in ['old_planning_148', 'old_planning_150', 'new_source_development_148', 'new_source_development_150']:
        print(key, {b: {'top1': results[key]['branches'][b]['classification']['top1'],
                       'white_guimu': results[key]['branches'][b]['white_guimu']} for b in BRANCHES}, flush=True)


if __name__ == '__main__':
    main()
