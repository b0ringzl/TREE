"""Verify live equivalence and report existing inference outputs. No training."""
import csv
import io
import json
import sys
from collections import Counter
from pathlib import Path
import numpy as np
import requests
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
from evaluate_whu_current_20260911 import read,write,sha256,WHU
from spacescan_trees_demo_v1.server import prepare

out=Path(sys.argv[1]).resolve()
data=read(out/'results.json'); protocol=read(out/'protocol.json'); rows=read(out/'test_manifest.json')
names=protocol['class_names']; labels=protocol['original_demo_labels']; predictions=read(out/'predictions.json')
prob=np.load(out/'all_probabilities.npz')
official={(r['road_id'],r['ply_id']) for r in csv.DictReader((ROOT/'lidar data/whu/test_split.txt').open(encoding='utf-8-sig'))}
source=read(ROOT/'experiments/whu18_resnet50_20260825/dataset/manifest.json')['records']
bad_test=[r['sample_key'] for r in source if r['split']=='test' and (r['road_id'],r['trajectory_id']) not in official]
leak=[r['sample_key'] for r in source if r['split']!='test' and (r['road_id'],r['trajectory_id']) in official]
assert not bad_test and not leak
segments=['_'.join(r['sample_key'].split('_')[:2]) for r in rows]
qa=dict(official_split_sha256=sha256(ROOT/'lidar data/whu/test_split.txt'),
        test_segments=len(set(segments)),test_not_on_official_segments=bad_test,
        train_val_on_official_test_segments=leak,live_checks=[])
for i in [0,113,1000]:
    pixels,w,h=prepare(Path(rows[i]['source_path']).read_bytes())
    reply=requests.post('http://127.0.0.1:8039/api/predict',files={'image':('image.jpg',pixels,'image/jpeg')},data={'mode':'teacher'},timeout=120)
    reply.raise_for_status(); actual=reply.json()['comparison']['teacher']
    expected=prob['demo148_original'][i]
    delta=max(abs(expected[labels.index(r['species'])]-r['score']) for r in actual)
    entry=dict(sample_key=rows[i]['sample_key'],live_top1=actual[0]['species'],batch_top1=labels[expected.argmax()],max_top3_score_delta=float(delta))
    assert entry['live_top1']==entry['batch_top1'] and delta<.005, entry
    qa['live_checks'].append(entry)
historical=list(csv.DictReader((WHU/'yolo11s_cls/run/predictions.csv').open(encoding='utf-8-sig')))
old={r['sample_key']:r for r in historical if r['split']=='test'}
assert set(old)=={r['sample_key'] for r in rows}
diff=[]
for r,p in zip(rows,predictions):
    prior=old[r['sample_key']]
    assert int(prior['true_class'])==int(r['class_index'])
    if names[int(prior['predicted_class'])]!=p['predictions']['whu_yolo11s_cls']:
        diff.append(dict(sample_key=r['sample_key'],truth=r['scientific_name'],
                         historical=names[int(prior['predicted_class'])],current=p['predictions']['whu_yolo11s_cls']))
qa['yolo_historical_disagreements']=diff
write(out/'verification.json',qa)
correct={key:np.array([p['truth']==p['predictions'][key] for p in predictions],float) for key in data['fresh']}
clusters=sorted(set(segments)); indexes=[np.flatnonzero(np.array(segments)==s) for s in clusters]
rng=np.random.default_rng(20260911); gaps=[]
for _ in range(2000):
    idx=np.concatenate([indexes[k] for k in rng.integers(len(indexes),size=len(indexes))])
    gaps.append(float((correct['whu_yolo11s_cls'][idx]-correct['bioclip_whu18_zeroshot'][idx]).mean()))
ci=np.quantile(gaps,[.025,.975]);write(out/'paired_segment_bootstrap.json',dict(clusters=len(clusters),replicates=2000,
    method='paired road-segment cluster bootstrap, descriptive regression interval',
    comparison='YOLO18 minus BioCLIP18 zero-shot accuracy',interval95=ci.tolist()))
fmt=lambda x:f'{x*100:.2f}%'
titles={'demo148_original':'现有 Demo 分类分支：原始 BioCLIP，148 类候选',
        'demo148_adapted':'旧适配教师：BioCLIP + MLP，148 类候选',
        'bioclip_whu18_zeroshot':'原始 BioCLIP：WHU 18 类零样本候选',
        'whu_yolo11s_cls':'WHU 专用 YOLO11s-cls：18 类'}
lines=['# WHU-STree 现有模型分类对照测试','',
       '日期：2026-09-11。只推理、核验和汇总；未训练、未挑选新 checkpoint、未切换 Demo。',
       '', '## 结论','',
       '当前教师版 Demo 不能直接替代 WHU 专用模型。其候选空间和训练域主要面向此前树种库；本次在 WHU 上的低分不能解读为所有场景的模型排名。',
       '', '## 1. 同一测试集，本轮重新推理','',
       'WHU-STree-NJ：18 个具名树种，3,399 株单木，26 个官方测试路段/点云序列；每株使用既有固定裁片。不是武汉城市数据，也不是全景检测精度。',
       '为与论文的单木分类任务一致，本轮直接将固定单木裁片输入分类分支，跳过 Demo 的树木预检测及用户选树交互。各模型沿用自己的输入预处理；相同源图不等于各模型预处理后的像素完全相同。',
       '', '| 模型/候选设置 | OA / Top-1 | Top-3 | Macro-F1 | 类别 mIoU |',
       '|---|---:|---:|---:|---:|']
for k,v in data['fresh'].items():lines.append(f"| {titles[k]} | {fmt(v['accuracy'])} | {fmt(v['top3'])} | {fmt(v['macro_f1'])} | {fmt(v['mean_class_iou'])} |")
lines += ['', 'mIoU 由单木类别混淆矩阵计算，与论文分类指标一致；此处没有像素、掩膜或实例分割精度。',
          '', '### 候选覆盖与公平性','',
          '当前 148 类原始键仅覆盖 WHU 18 类中的 Cinnamomum camphora（樟）、Lagerstroemia indica（紫薇）、Magnolia grandiflora（广玉兰），共 1,156 株；另外 15 类无法被当前目录直接输出。即使这三类全部判对，原样 Demo 的全测试集准确率上限也只有 34.01%。',
          'WHU18 零样本测试保留同一 BioCLIP 编码器及相同输入预处理，仅用官方 18 个类别名称构造固定 `a photo of {scientific_name}` 文本候选。不拟合分类头、不依据测试结果改提示词；这不是已经部署的 Demo，也不是在 WHU 训练过的模型。',
          '这一零样本数值仅代表固定学名提示词和当前裁片预处理配置，不代表官方完整分类学提示、其他图像预处理或所有 BioCLIP 应用的精度；本轮未做提示词搜索或在测试集上调参。',
          'YOLO 使用已有 WHU 训练权重；BioCLIP18 未在 WHU 微调。因此这个差距同时包含领域适配、候选设计和模型差异，不能作为纯架构优劣结论。',
          f"按 26 个测试路段进行配对聚类 bootstrap，YOLO18 相对 BioCLIP18 的 OA 差值 95% 描述性区间为 {fmt(ci[0])}～{fmt(ci[1])}（百分点差值，2,000 次重采样）。不作为新的无偏显著性检验。",
          '', '## 2. 原项目模型历史结果（未重新运行点云/融合）','',
          '这些记录使用相同 18 类和 3,399 株测试规模，已核对混淆矩阵的各类支持数。YOLO 另核对了全部逐样本 ID 和真值；其余模型本轮仅重算归档混淆矩阵，不冒称完成新推理。',
          '', '| 历史模型 | 模态 | OA | Macro-F1 | 类别 mIoU |','|---|---|---:|---:|---:|']
for k,mod in [('resnet50','图像'),('yolo11s_cls','图像'),('ptv2','点云'),('fusion','YOLO + PTv2 图像/点云')]:
    v=data['historical'][k];lines.append(f"| {k} | {mod} | {fmt(v['accuracy'])} | {fmt(v['macro_f1'])} | {fmt(v['mean_class_iou'])} |")
lines += ['',f'YOLO 本轮与历史 Top-1 有 {len(diff)} 株不一致，准确率净差为 {(data["fresh"]["whu_yolo11s_cls"]["accuracy"]-data["historical"]["yolo11s_cls"]["accuracy"])*100:.4f} 个百分点。权重 SHA 不变，逐样本差异列于 verification.json；本轮 FP32 与旧推理数值路径不完全相同，不能把这一微小变化当作训练提升。',
          '', '## 3. 论文 Table 5：19 类原文报告值','',
          '[WHU-STree 官方仓库](https://github.com/WHU-USI3DV/WHU-STree)；[原论文 DOI](https://doi.org/10.1016/j.isprsjprs.2026.02.011)。本轮对照本地原文第 9 页（印刷页 527）的 Table 5 及第 8 页分类口径。',
          '', '| 论文模型 | 模态 | OA | 类别 mIoU |','|---|---|---:|---:|',
          '| MinkNet | 点云 | 81.7% | 49.9% |','| PointMLP | 点云 | 84.5% | 58.0% |',
          '| PTv2 | 点云 | 88.0% | 64.1% |','| ResNet | 图像 | 83.4% | 53.7% |',
          '| TSCMDL | 图像 + 点云 | 86.2% | 60.5% |','',
          '**不能据此宣称本地融合 89.56% 超越论文 PTv2 88.0%。** 论文使用 18 类 + Others 共 19 类；本地排除了 Others，还采用本项目裁片、点采样、实现与验证选模配置。没有取得并重新运行论文各模型权重，不称为严格复现。',
          '', '## 4. 每个树种的召回率（本轮）','',
          '| 学名 | 测试株数 | 原样 Demo | BioCLIP18 零样本 | WHU YOLO18 |','|---|---:|---:|---:|---:|']
for i,n in enumerate(names):
    a=data['fresh']['demo148_original']['per_class'][i];b=data['fresh']['bioclip_whu18_zeroshot']['per_class'][i];c=data['fresh']['whu_yolo11s_cls']['per_class'][i]
    lines.append(f"| {n} | {a['support']} | {fmt(a['recall'])} | {fmt(b['recall'])} | {fmt(c['recall'])} |")
lines+=['','## 5. BioCLIP18 最常见混淆','', '| 真值 → 预测 | 株数 |','|---|---:|']
conf=Counter((p['truth'],p['predictions']['bioclip_whu18_zeroshot']) for p in predictions if p['truth']!=p['predictions']['bioclip_whu18_zeroshot'])
for (a,b),n in conf.most_common(10):lines.append(f'| {a} → {b} | {n} |')
lines+=['','## 6. 核验、安全与可追溯性','',
        '- 3,399 张测试图逐张核对 SHA-256；训练/验证/测试样本 ID 和图像哈希无交集。官方 test_split.txt 对应关系核验通过，训练/验证未进入官方测试序列。仍不能完全排除不同序列重复采集同一实体树，未将这些记录冒称新盲测。',
        '- 抽取三个固定样本，与正在运行的 8039 教师服务进行相同 SpaceScan 预处理核验：Top-1 一致，分数差异见 verification.json。批次推理与线上单图数值不要求逐位一致。',
        '- checkpoint、候选映射、原测试划分未改变。所加载教师/分类头/YOLO 权重推理前后 SHA 完全一致，详见 weight_integrity.json。',
        '- 本轮无训练，因此没有训练前备份需求，也未覆盖任何既有训练备份；正式 Demo 服务与部署配置不变。',
        '- protocol.json：事先固定测试协议；test_manifest.json：每株来源与哈希；predictions.json：所有逐株真值和预测；all_probabilities.npz：完整分数；results.json：逐类指标与混淆矩阵。',
        '- 历史模型详细来源及哈希保留在 results.json；验证区间保留在 paired_segment_bootstrap.json。',
        '', '## 下一步建议（本轮未执行）','',
        '1. 对 WHU 使用已训练的专用 YOLO/融合分支；不要自动将当前教师分支替换过去。',
        '2. 如需提升 BioCLIP，先对 WHU 训练/验证集做编码器冻结的轻量分类头适配；每次 fitting 前做经 SHA 校验的权重备份，重新建立未反复查看的评估集。',
        '3. 如需论文级直接对比，恢复 19 类（含 Others）、对齐原文数据预处理和训练协议，再评估，不能把仅改报表口径称作公平复现。']
(out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('Report and verification complete',flush=True)
print(json.dumps(qa,ensure_ascii=False,indent=2),flush=True)
