"""Honest paired recall report for the locked rare-tree audit (no fitting)."""
import html
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import fresh_bioclip146_benchmark_20261006 as base
from training_safety import sha256
OUT = ROOT / 'outputs/rare_species_fresh_validation_20261006_v1'
TARGETS = ('Aquilaria sinensis', 'Artocarpus hypargyreus')
CN = dict(zip(TARGETS, ('土沉香', '白桂木')))
SEED = 2026100607


def stats(rows, predictions, before='teacher_context', after='sstv1_context'):
    n = len(rows)
    if not n: return {'n': 0}
    a = np.array([p['top3'][before][0] == r['label'] for r, p in zip(rows, predictions)])
    b = np.array([p['top3'][after][0] == r['label'] for r, p in zip(rows, predictions)])
    groups = defaultdict(list); observations = defaultdict(list)
    for i, r in enumerate(rows):
        groups[r['cluster_id']].append(i); observations[r['source_group']].append(i)
    keys = sorted(groups)
    na = np.array([a[groups[k]].sum() for k in keys]); nb = np.array([b[groups[k]].sum() for k in keys])
    counts = np.array([len(groups[k]) for k in keys])
    rng = np.random.default_rng(SEED)
    indices = rng.integers(0, len(keys), size=(10000, len(keys)))
    gain = 100 * (nb[indices].sum(1) - na[indices].sum(1)) / counts[indices].sum(1)
    ci = (np.quantile(gain, [.025, .975])).tolist() if len(keys) > 1 else None
    return {'n': n, 'source_observations_or_papers': len(observations), 'observer_or_paper_clusters': len(keys),
            'before_correct': int(a.sum()), 'after_correct': int(b.sum()),
            'before_top1': float(a.mean()), 'after_top1': float(b.mean()),
            'before_top3': sum(r['label'] in p['top3'][before] for r, p in zip(rows, predictions)) / n,
            'after_top3': sum(r['label'] in p['top3'][after] for r, p in zip(rows, predictions)) / n,
            'gain_percentage_points': float(100 * (b.mean() - a.mean())),
            'corrected': int((~a & b).sum()), 'regressed': int((a & ~b).sum()),
            'both_correct': int((a & b).sum()), 'both_wrong': int((~a & ~b).sum()),
            'cluster_bootstrap95_percentage_points': ci,
            'equal_observation_before': float(np.mean([a[v].mean() for v in observations.values()])),
            'equal_observation_after': float(np.mean([b[v].mean() for v in observations.values()])),
            'equal_observer_before': float(np.mean([a[v].mean() for v in groups.values()])),
            'equal_observer_after': float(np.mean([b[v].mean() for v in groups.values()])),
            'cluster_bootstrap_samples': 10000,
            'before_mode': before, 'after_mode': after}


def main():
    paths = [OUT / 'REPORT_ZH.md', OUT / 'REPORT_ZH.html', OUT / 'RARE_RESULTS.json', OUT / 'CASE_RESULTS.json']
    if any(p.exists() for p in paths): raise FileExistsError('Completed report is immutable; use a new version')
    rows = base.load(OUT / 'manifest.json')
    predictions = [json.loads(line) for line in (OUT / 'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
    if len(rows) != len(predictions) or any(r['id'] != p['id'] for r, p in zip(rows, predictions)):
        raise RuntimeError('Incomplete or misaligned evaluation')
    lock = base.load(OUT / 'collection_lock.json')
    if sha256(OUT / 'manifest.json') != lock['manifest_sha256']: raise RuntimeError('Unlocked data')
    def subset(condition, **kwargs):
        indices = [i for i, r in enumerate(rows) if condition(r)]
        return stats([rows[i] for i in indices], [predictions[i] for i in indices], **kwargs)
    primary = lambda r: r['source_kind'] == 'community_observation'
    result = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'baseline': 'BioCLIP2.5 zero-shot, same project prompts/catalog/temperature',
        'candidate': 'SSTv1-H120 deployed, same frozen BioCLIP2.5 encoder + residual classifier',
        'training_performed': False, 'deployment_changed': False, 'data_source_labelled_not_expert_ground_truth': True,
        'primary': subset(primary), 'per_species': {label: subset(lambda r: primary(r) and r['label'] == label) for label in TARGETS},
        'center_only_diagnostic': {label: subset(lambda r: primary(r) and r['label'] == label,
            before='teacher_center', after='sstv1_center') for label in TARGETS},
        'paper_challenge': subset(lambda r: r['source_kind'] == 'paper_photo_challenge'),
        'paper_per_source': {k: subset(lambda r: r['source_group'] == k) for k in sorted({r['source_group'] for r in rows if not primary(r)})},
        'subsets': {name: subset(lambda r: primary(r) and test(r)) for name, test in (
            ('hong_kong', lambda r: r['hong_kong']), ('outside_hong_kong', lambda r: not r['hong_kong']),
            ('recent_since_20260908', lambda r: r['recent_after_base_project']),
            ('cc0_by_bysa_only', lambda r: r['license'] in {'cc0', 'cc-by', 'cc-by-sa'}))},
        'manifest_sha256': sha256(OUT / 'manifest.json'), 'predictions_sha256': sha256(OUT / 'predictions.jsonl'),
        'confidence_is_not_accuracy': True, 'precision_false_positive_not_estimated': True,
        'release_id': base.load(ROOT / 'spacescan_trees_demo_v1/release.json')['release_id']}
    result['macro_two_species_before'] = float(np.mean([s['before_top1'] for s in result['per_species'].values()]))
    result['macro_two_species_after'] = float(np.mean([s['after_top1'] for s in result['per_species'].values()]))
    result['macro_two_species_gain_percentage_points'] = 100 * (result['macro_two_species_after'] - result['macro_two_species_before'])
    cases = []
    for r, p in zip(rows, predictions):
        ca = p['top3']['teacher_context'][0] == r['label']; cb = p['top3']['sstv1_context'][0] == r['label']
        cases.append({'id': r['id'], 'label': r['label'], 'source_kind': r['source_kind'], 'source_group': r['source_group'],
            'file': r['file'], 'source_page': r['source_page'], 'license': r['license'], 'attribution': r['attribution'],
            'bio_top3': p['top3']['teacher_context'], 'sst_top3': p['top3']['sstv1_context'],
            'bio_correct': ca, 'sst_correct': cb, 'status': 'corrected' if cb and not ca else 'regressed' if ca and not cb else 'both_correct' if ca else 'both_wrong'})
    result['confusions'] = {label: dict(Counter(c['sst_top3'][0] for c in cases if c['label'] == label and
        c['source_kind'] == 'community_observation' and not c['sst_correct'])) for label in TARGETS}
    base.save(OUT / 'RARE_RESULTS.json', result); base.save(OUT / 'CASE_RESULTS.json', cases)
    paired = result['primary']; gain = paired['gain_percentage_points']
    takeaway = '新模型在本次新增资料上识对更多图片' if gain > 0 else '本次新增资料未显示新模型识对更多图片'
    lines = ['# 珍稀树种新增样本验证汇报', '', 'Tree Species Identification | 2026-10-07 | 仅评估，未训练、未更新正式模型', '',
        f"## 结论：{takeaway}", '',
        f"主测试共 {paired['n']} 张：原始 BioCLIP 2.5 识对 {paired['before_correct']} 张（{paired['before_top1']:.2%}），"
        f"当前 SSTv1-H120 识对 {paired['after_correct']} 张（{paired['after_top1']:.2%}），净变化 {gain:+.2f} 个百分点。",
        f"纠正 {paired['corrected']} 张，退步 {paired['regressed']} 张；论文照片另外统计，不用于抬高主结果。", '',
        '| 树种 | 图片 / 观察记录 / 观察者 | BioCLIP Top-1 | SST Top-1 | 变化（百分点） | 纠正 / 退步 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for label, s in result['per_species'].items():
        lines.append(f"| {CN[label]} {label} | {s['n']} / {s['source_observations_or_papers']} / {s['observer_or_paper_clusters']} | "
            f"{s['before_correct']}/{s['n']} = {s['before_top1']:.2%} | {s['after_correct']}/{s['n']} = {s['after_top1']:.2%} | "
            f"{s['gain_percentage_points']:+.2f} | {s['corrected']} / {s['regressed']} |")
    lines += ['', f"两树种等权平均召回率：{result['macro_two_species_before']:.2%} → {result['macro_two_species_after']:.2%}，"
              f"{result['macro_two_species_gain_percentage_points']:+.2f} 个百分点。该指标避免土沉香图片较多主导合并结果。", '',
              '## 证据可靠度与样本依赖', '',
              '同一观察记录可能含多张照片，同一观察者还可能重复拍摄同一地点。因此照片数不是独立树木数。'
              '对变化做 10,000 次观察者整组配对 bootstrap；论文按论文来源整组，结果区间仅供探索性参考。', '',
              '| 树种 | 变化的观察者整组 95% 区间（百分点） | 每个观察记录等权：BioCLIP → SST | Top-3：BioCLIP → SST |',
              '| --- | ---: | ---: | ---: |']
    for label, s in result['per_species'].items():
        ci = s['cluster_bootstrap95_percentage_points']
        lines.append(f"| {CN[label]} | {ci[0]:+.2f} 至 {ci[1]:+.2f} | {s['equal_observation_before']:.2%} → {s['equal_observation_after']:.2%} | "
                     f"{s['before_top3']:.2%} → {s['after_top3']:.2%} |")
    ci = paired['cluster_bootstrap95_percentage_points']
    lines += ['', f"合并主结果的观察者整组 95% 变化区间：{ci[0]:+.2f} 至 {ci[1]:+.2f} 个百分点。区间跨 0 时，不宣称稳定的统计显著提升。", '',
        '## 测试怎么保证公平', '',
        '1. 先备份、采集、核查来源与质量、去重并锁定清单，再运行任何预测；没有按置信度或正确与否挑选图片。',
        '2. 使用完整 146 个展示类别竞争，不是只在这两种树之间二选一。相同原图、文本提示、标签合并与温度。',
        '3. 两模型都复用正式网页的“原图中心 + 全图灰色补边”各半概率路径，输入仍为编码器原生 224×224。'
        '未使用 SAM、目标检测、点云、视频跟踪或额外高分辨率方案。',
        '4. 原始比较对象是 BioCLIP 2.5 的零样本分支，不是 BioCLIP 1，也不是任意开放词汇库的最优零样本配置。'
        '当前分支是在同一冻结编码器上增加已有残差分类头。',
        '5. 第 1、中间、末尾样本另外调用正式 infer 路径验证分布一致；逐张保存两个分支的原始概率。', '',
        '### 只比较分类头的中心图诊断', '', '| 树种 | BioCLIP 中心图 | SST 中心图 | 变化（百分点） |', '| --- | ---: | ---: | ---: |']
    for label, s in result['center_only_diagnostic'].items():
        lines.append(f"| {CN[label]} | {s['before_top1']:.2%} | {s['after_top1']:.2%} | {s['gain_percentage_points']:+.2f} |")
    lines += ['', '这些配对差异支持评估“已有分类头相对零样本分支的适配效果”，不能进一步分摊为某一次增量训练的单独贡献。', '',
        '## 来源、香港与新近照片', '',
        '社区主来源：iNaturalist 官方 API。要求当前至少两名独立的准确物种鉴定者，且至少一人不是上传者，没有冲突鉴定。'
        '这些是来源标签，不是本项目专家人工核准真值。', '',
        '| 子集 | N | BioCLIP | SST | 变化（百分点） |', '| --- | ---: | ---: | ---: | ---: |']
    names = {'hong_kong': '香港来源', 'outside_hong_kong': '香港以外', 'recent_since_20260908': '拍摄日期 ≥ 2026-09-08', 'cc0_by_bysa_only': '仅 CC0/BY/BY-SA'}
    for name, s in result['subsets'].items():
        lines.append(f"| {names[name]} | {s['n']} | " +
            (f"{s['before_top1']:.2%} | {s['after_top1']:.2%} | {s['gain_percentage_points']:+.2f} |" if s['n'] else '无可用样本 | 无可用样本 | 不估计 |'))
    lines += ['', '全部候选资料的拍摄日期截至 2026-10-06。历史项目中出现的观察记录、图片 ID、SHA256 和 pHash 近重复均排除，'
        '但“对本项目新增”不等于“基础模型从未见过”；较早公开照片可能与 BioCLIP 预训练重合。'
        '新近子集过少或为空时，不能宣称已证实基础模型未见样本上的泛化提升。', '',
        '## 论文补充测试：单独报告', '']
    paper = result['paper_challenge']
    lines += [f"合格的土沉香自然照片共 {paper['n']} 个裁片，来自 {paper['source_observations_or_papers']} 篇论文："
        f"BioCLIP 识对 {paper['before_correct']}/{paper['n']}，SST 识对 {paper['after_correct']}/{paper['n']}。"
        '裁片是实物照片的原生像素，不是重绘、截图放大或图表。', '',
        '| 来源 | 图版 / 入选对象 | BioCLIP → SST | 许可 |', '| --- | --- | ---: | --- |']
    for pmcid, description in [('PMC7050300', 'Fig.1(a)，成熟土沉香'), ('PMC10935121', 'Fig.1(A,D,E)，幼苗、果壳、种子')]:
        md = next(r for r in base.load(OUT / 'paper_cloud_sources.json') if r['pmcid'] == pmcid)
        s = result['paper_per_source']['paper_' + pmcid]
        lines.append(f"| [{md['title']}](https://doi.org/{md['doi'][0]}) | {description} | {s['before_correct']}/{s['n']} → {s['after_correct']}/{s['n']} | CC BY 4.0 |")
    lines += ['', '[白桂木属系统发育论文](https://doi.org/10.1093/sysbio/syaa073) 的 Fig.1(I) 标注明确，但原始照片不足 224 像素最短边，'
        '故未计入识别率。[土沉香奇楠幼苗论文](https://doi.org/10.3390/plants14060896) 图版混合普通与奇楠材料，未纳入主评估。'
        '微观组织、沉香产品、其他树种、图表及有歧义的材料均排除。',
        '论文图片通过 [PMC 官方公开云端数据集](https://pmc.ncbi.nlm.nih.gov/tools/pmcaws/) 取得，图注与许可由正式 XML 核查。'
        '同一论文的多个裁片不能视作多个独立来源，更不能当作街景识别准确率。', '',
        '## 剩余错误与边界', '']
    for label, counts in result['confusions'].items():
        text = '；'.join(f'{name}：{count} 张' for name, count in sorted(counts.items(), key=lambda v: (-v[1], v[0]))) or '无错误'
        lines.append(f"- {CN[label]}被输出为其他类别：{text}。")
    lines += ['', '保留所有错误、低分图片和退步案例；CASE_RESULTS.json 可逐条核查来源，predictions.jsonl 包含全部分布。'
        '枝叶、花果、树皮与整树的取景差异，是下一步应单独分层检查的问题，本次未用错误表现重新筛样。',
        '这里只含两个树种的正样本，因此能报告各树种 Top-1 召回率，不能估计在大量其他树种中误报这两类的精确率。'
        '不外推为全部珍稀植物、146 类总体或香港连续街景的准确率，也不将模型置信度当作正确率。',
        'H120 曾由用户指定部署，并非每一次增量训练都通过开发集门槛：历史记录包括土沉香误报 7→10、'
        '白桂木识对 10→9；本次新数据比较不能抹去那些已披露的退步。', '',
        '## 文件、授权与备份', '',
        '原图、水印、许可、作者及来源链接完整保留。NC 许可图片仅用于私人非商业研究验证；没有进入训练集、'
        '修改 human_review.sqlite3、更新模型或加入可分发 Demo 包。图片质量检查不是授权或专家标注。', '',
        f"- [完整逐图结果]({(OUT / 'CASE_RESULTS.json').as_posix()})",
        f"- [测试集锁定清单]({(OUT / 'manifest.json').as_posix()})",
        f"- [模型与部署文件物理备份]({(OUT / 'pre_training_backup/backup_manifest.json').as_posix()})",
        f"- [固定评估协议备份]({(OUT / 'evaluation_protocol_snapshot/pre_training_backup/backup_manifest.json').as_posix()})",
        f"- 清单 SHA256：{result['manifest_sha256']}", f"- 预测 SHA256：{result['predictions_sha256']}",
        '- 最终审计结果见 FINAL_AUDIT.json。正式模型和部署文件必须与事前备份 SHA256 一致。', '']
    (OUT / 'REPORT_ZH.md').write_text('\n'.join(lines), encoding='utf-8')
    # Self-contained local dashboard: all original files remain local; no upload.
    esc = html.escape
    def table_row(label, s):
        return f'<tr><th>{esc(label)}</th><td>{s["n"]}</td><td>{s["before_top1"]:.2%}</td><td>{s["after_top1"]:.2%}</td><td>{s["gain_percentage_points"]:+.2f}</td><td>{s["corrected"]}/{s["regressed"]}</td></tr>'
    table = ''.join(table_row(CN[l], s) for l, s in result['per_species'].items()) + table_row('合并主测试', paired)
    items = []
    for c in cases:
        items.append(f'<article data-status="{c["status"]}" data-label="{esc(c["label"])}"><img loading="lazy" src="{esc(c["file"])}">'
            f'<h3>{esc(CN[c["label"]])} · {esc(c["status"])}</h3><p>{esc(c["id"])}</p>'
            f'<p>BioCLIP：{esc(c["bio_top3"][0])}<br>SST：{esc(c["sst_top3"][0])}</p>'
            f'<a href="{esc(c["source_page"])}" target="_blank">原始来源</a><small>{esc(c["license"])}<br>{esc(c["attribution"])}</small></article>')
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>珍稀树种新增样本验证</title><style>body{{font:16px/1.7 system-ui,"Microsoft YaHei",sans-serif;color:#153e34;background:#f4f6ee;margin:0}}main{{max-width:1150px;margin:auto;padding:40px 24px}}h1{{font-size:34px}}section{{background:white;border:1px solid #dbe1d6;border-radius:16px;padding:24px;margin:24px 0}}table{{width:100%;border-collapse:collapse}}td,th{{padding:12px;text-align:left;border-bottom:1px solid #e3e8df}}.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:20px}}article{{border:1px solid #ccd8cc;border-radius:12px;padding:14px;background:white}}article img{{width:100%;height:220px;object-fit:contain;background:#f0f2ed}}small{{display:block;color:#5f7168;overflow-wrap:anywhere}}select{{font:inherit;padding:8px;max-width:100%}}a{{color:#176e55}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}@media(max-width:700px){{table{{font-size:12px}}td,th{{padding:5px}}}}</style>
<main><p>Tree Species Identification · 2026-10-07 · 未训练 / 未更新模型</p><h1>珍稀树种新增样本验证</h1><p>{esc(takeaway)}。原始 BioCLIP 2.5 vs 当前 SSTv1-H120。</p>
<section><h2>主测试：{paired['n']} 张自然植物照片</h2><table><tr><th>树种</th><th>N</th><th>BioCLIP</th><th>SST</th><th>变化 pp</th><th>纠正/退步</th></tr>{table}</table>
<p>两树种等权：{result['macro_two_species_before']:.2%} → {result['macro_two_species_after']:.2%}。论文另计：{paper['before_correct']}/{paper['n']} → {paper['after_correct']}/{paper['n']}。</p>
<p>仅此两种的来源标签正样本召回，不是全部珍稀植物或香港街景准确率。置信度不等于正确率。</p><a href="REPORT_ZH.md">完整方法、区间、限制与来源</a> · <a href="RARE_RESULTS.json">结构化统计</a></section>
<section><h2>逐张核查</h2><select id="filter"><option value="all">全部192张</option><option value="corrected">只看纠正</option><option value="regressed">只看退步</option><option value="both_wrong">两模型均错误</option><option value="both_correct">两模型均正确</option></select>
<p>图片仅在本机显示，原始水印与归属保留；NC资料不得当作可自由分发素材。</p><div class="grid">{''.join(items)}</div></section></main>
<script>document.querySelector('#filter').addEventListener('change',e=>{{document.querySelectorAll('article').forEach(a=>a.hidden=e.target.value!=='all'&&a.dataset.status!==e.target.value)}})</script></html>'''
    (OUT / 'REPORT_ZH.html').write_text(page, encoding='utf-8')
    print('REPORT COMPLETE', json.dumps(result['per_species'], ensure_ascii=False), flush=True)


if __name__ == '__main__': main()
