"""Report crop effects separately from learned changes and held-out regression."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import numpy as np
import requests
from distillation_core import read_json,save_json
from training_safety import verify_backup,sha256
from train_aq_ar_diagnostic_20260919 import unchanged

OUT=ROOT/'outputs/SSTv1_Aquilaria_UserCrops_20260919_v2'


def link(path,label):return f'[{label}](<{Path(path).resolve().as_posix()}>)'


def main():
    out=OUT;assert not (out/'REPORT.md').exists()
    protocol=read_json(out/'protocol.json');unchanged(protocol)
    complete=read_json(out/'COMPLETE.json');lock=read_json(out/'selection_lock.json')
    crops=read_json(out/'crop_manifest.json');train=read_json(out/'train_rows.json')
    labels=read_json(out/'labels.json');base=read_json(out/'baseline_evaluation.json')
    reg=read_json(out/'regression_results.json');features=read_json(out/'FEATURES_COMPLETE.json')
    last_single=read_json(out/'last_steps_single_dev.json')['results']
    with np.load(out/'baseline_predictions.npz') as f:
        bp=f['train_single'];bd=f['train_dual']
    selected=read_json(out/'selected_crop_replay_single.json')
    selecteddual=read_json(out/'selected_crop_replay_dual.json')
    parents=[next(i for i,r in enumerate(train[:-8]) if r['sha256']==c['parent_sha256']) for c in crops]
    parent_single=int((bp[parents].argmax(1)==11).sum());parent_dual=int((bd[parents].argmax(1)==11).sum())
    backups=[];experiments=[]
    for recipe in protocol['recipes']:
        run=out/recipe['name'];backup=run/'pre_training_backup';b=verify_backup(backup)
        backups.append(dict(recipe=recipe['name'],path=str(backup),files=len(b['files']),bytes=sum(r['bytes'] for r in b['files'])))
        history=read_json(run/'history.json')
        experiments.append(dict(name=recipe['name'],last=history[-1],
            eligible_steps=[h['step'] for h in history if h['eligible']],
            single=read_json(run/'crop_replay_single.json'),dual=read_json(run/'crop_replay_dual.json')))
    details=[]
    for i,c in enumerate(crops):
        details.append(dict(filename=c['filename'],image_path=c['source_image'],parent_path=c['parent_source_image'],
            size=[c['width'],c['height']],parent_top1=labels[int(bp[parents[i]].argmax())],parent_score=float(bp[parents[i],11]),
            cropped_baseline=selected['images'][i]['baseline'],selected=selected['images'][i]['candidate'],
            G_last=experiments[0]['single']['images'][i]['candidate'],H_last=experiments[1]['single']['images'][i]['candidate']))
    save_json(out/'per_image_comparison.json',dict(scope='Training replay; no generalization claim',images=details))
    lines=['# 土沉香：用户裁剪图补充训练', '', '日期：2026-09-19', '', '## 本轮完成内容', '',
        '已将用户明确确认的 8 张裁剪图加入独立版本的训练数据，并完成两组各 120 步的分类头继续训练。原图和本次裁片均保留，不覆盖原训练数据或正式模型。',
        f"锁定结果：`{lock['recipe']}`，step {lock['step']}。"+('没有新训练权重通过既定替换门槛，保留原权重双取景对照。' if lock['step']==0 else '候选仅按固定开发集选择，尚未部署。'), '',
        '## 数据与安全', '',
        '- 8 张文件均与历史训练原图建立父子关系；像素模板匹配相关系数均超过 0.998，辅助确认它们确实来自对应原图，而不是仅按文件名推断。',
        '- 裁片与原图保持同一训练分组：8 个原文件、7 个既有来源分组；不是新增 8 株独立树，也没有进入验证或测试。',
        '- 原训练集 4,198 张 + 8 张裁片 = 4,206 条训练记录。原图仍参与回放；组内权重分摊避免把裁片当独立样本重复放大。',
        '- 3 张裁片短边不足 224 像素（192、150、157）；按用户确认保留并标注低分辨率。未用生成式超分辨率制造植物细节。',
        '- 未删除其他低质图片：用户尚未逐一指明要排除哪些文件。未依据模型预测更改树种标签。',
        '- 本次确认记录独立保存；既有人审 SQLite 决策只读核验，未把 pending/rejected 样本混入训练。保留原来源权利说明，未公开发布照片或实验权重。',
        f"- 实际 Demo 两路输入共 16 次 API 对照，缓存概率最大误差 {max(r['max_probability_error'] for r in features['live_api_checks']):.3g}，排除了这一批数据的预处理/模型版本错配。", '',
        '## 裁剪作用与训练作用要分开', '',
        '以下都是这 8 张已参与训练的图片的回放，不是验证精度。单路指正式 Demo 的整图补边分类输入；双取景指整图与中心裁片概率等权平均。本轮没有改变正式页面的检测/选树逻辑。', '',
        '| 输入与模型 | 单路识对 / 8 | 双取景识对 / 8 |', '|---|---:|---:|',
        f'| 裁剪前原图，原正式权重 | {parent_single} | {parent_dual} |',
        f"| 用户裁剪图，尚未补训的原权重 | {base['crop_single_correct']} | {base['crop_dual_correct']} |"]
    for e in experiments:
        lines.append(f"| {e['name']}，120 步实验权重 | {e['single']['candidate_correct']} | {e['dual']['candidate_correct']} |")
    lines.append(f"| 最终开发集锁定结果：{lock['recipe']} / {lock['step']} | {selected['candidate_correct']} | {selecteddual['candidate_correct']} |")
    lines += ['', 'G：双取景、类别/来源组均衡；H：相同训练加 0.02 的裁片来源组均衡损失。二者都从当前正式三个 MLP 分类头初始化，编码器与温度冻结；保留全部旧类回放与 0.5 KL 约束。', '',
        '## 逐张结果（单路整图补边输入）', '',
        '百分比是正确类别“土沉香”的模型分数，不是准确率。括号内是 Top-1；“土沉香”表示识对。H120 为预先设定的最后一步实验权重，不代表通过部署门槛。', '',
        '| 图片 | 尺寸 | 原图原权重 | 裁片原权重 | 裁片 H120 |', '|---|---|---|---|---|']
    def score(p):return f"{p['true_score']:.1%}（{'土沉香' if p['correct'] else p['top1']}）"
    for d in details:
        old=dict(true_score=d['parent_score'],top1=d['parent_top1'],correct=d['parent_top1']=='Aquilaria sinensis')
        lines.append(f"| {link(d['image_path'],d['filename'])} | {d['size'][0]}×{d['size'][1]} | {score(old)} | {score(d['cropped_baseline'])} | {score(d['H_last'])} |")
    lines += ['', '## 固定开发集', '',
        '仍使用 2,655 张开发数据（2,650 张目录内、5 张仅作误报检查）。土沉香 22 张、白桂木 16 张。没有使用这 8 张训练裁片挑选 checkpoint。', '',
        '| 方案 | 土沉香 TP / 22；FP | 白桂木 TP / 16；FP | 总体识对 / 2650 |', '|---|---|---|---:|']
    def row(name,m):
        a,w=m['aquilaria'],m['white']
        return f"| {name} | {a['tp']}/22；{a['fp']} | {w['tp']}/16；{w['fp']} | {m['all']['correct']} |"
    lines += [row('原权重整图',base['single']['combined']),row('原权重双取景',base['dual']['combined'])]
    for e in experiments:
        lines.append(row(e['name']+' / 120 单路',last_single[e['name']]['combined']))
        lines.append(row(e['name']+' / 120 双取景',e['last']['metrics']['combined']))
    lines.append(row('锁定结果',lock['metrics']['combined']))
    lines += ['',
        '使用与上一轮一致的保护门槛：对照原整图及原双取景，在新旧开发子集中两目标识对数/F1 不降、误报不增；总体识对数不降、宏 F1 降幅不超过 0.001。必须取得严格排序改进才接纳新 checkpoint。', '',
        '两个预设末步的单路开发集指标是锁定之后补做的诊断，只为与上述单路训练回放作同口径比较；未据此改选 checkpoint。训练图学会更多，不等于在其他树木上更可靠。', '',
        '## 锁定之后的历史回归', '',
        '5,146 张历史回归图与训练隔离，但已多次被检查，不是新的盲测泛化证据。只评价已锁定候选，不反过来用回归集选权重。', '',
        '| 方案 | 土沉香识对 / 25；误报 | 白桂木识对 / 22；误报 | 总体 Top-1 |', '|---|---|---|---:|']
    for key,name in [('baseline_single','原权重整图'),('baseline_dual','原权重双取景'),('candidate_single','锁定权重整图'),('candidate_dual','锁定权重双取景')]:
        m=reg[key];a,w=m['aquilaria'],m['white']
        lines.append(f"| {name} | {a['tp']}/25；{a['fp']} | {w['tp']}/22；{w['fp']} | {m['all']['top1']:.2%} |")
    lines += ['', '## 备份、产物与部署', '', '两轮拟合前均新建物理复制备份，包含初始化/部署权重、标签、数据清单、分组、参数、基线结果及 8 张裁片和父图；SHA-256 已验证。没有覆盖正式权重，也没有切换 Demo。', '']
    for b in backups:lines.append('- '+link(b['path'],b['recipe']+' 训练前备份')+f"：{b['files']} 文件，{b['bytes']/1024**3:.2f} GiB。")
    lines += ['', '- '+link(out/'crop_manifest.json','裁片—原图对应、用户确认与训练分组'),
        '- '+link(out/'per_image_comparison.json','逐图原始结果'),
        '- '+link(out/'selection_lock.json','仅按开发集锁定的结果'),
        '- '+link(out/'regression_results.json','历史回归完整结果'),
        '- '+link(out/'last_steps_single_dev.json','末步单路开发诊断结果'),
        '- '+link(ROOT/'tools/train_aquilaria_user_crops_20260919_v2.py','训练与审计入口'), '',
        '执行记录：v1 完成 G 的 120 步后，在生成裁片回放报告时遇到标签路径错误，H 尚未开始。v1 权重、日志和备份完整保留；修正报告路径后另建 v2，两组均从正式初始化重新训练并分别备份，没有覆盖或续写旧训练。该故障不涉及权重计算或训练数据。']
    with requests.Session() as client:
        r=client.get('http://127.0.0.1:8039/api/health',timeout=15);r.raise_for_status()
        assert r.json()['sstv1_version']==protocol['deployment_manifest']['version']
        page=client.get('http://127.0.0.1:8043/',timeout=15);page.raise_for_status()
    unchanged(protocol)
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    save_json(out/'HANDOFF.json',dict(**complete,backup_verification=backups,
        original8_single_correct=parent_single,original8_dual_correct=parent_dual,
        crops8_before_single_correct=base['crop_single_correct'],crops8_before_dual_correct=base['crop_dual_correct'],
        experiments=[dict(name=e['name'],last_single_correct=e['single']['candidate_correct'],last_dual_correct=e['dual']['candidate_correct'],eligible_steps=e['eligible_steps']) for e in experiments],
        selected8_single_correct=selected['candidate_correct'],selected8_dual_correct=selecteddual['candidate_correct'],
        webpage_http=page.status_code,teacher_version=r.json()['sstv1_version']))
    print('REPORT_COMPLETE',str(out/'REPORT.md'),flush=True)


if __name__=='__main__':main()
