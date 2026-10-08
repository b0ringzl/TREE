"""Summarize completed experiments without selecting another model or fitting."""
import sys
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import numpy as np
import requests
from distillation_core import read_json,save_json
from training_safety import verify_backup,sha256
from train_aq_ar_diagnostic_20260919 import unchanged

RUN=ROOT/'outputs/SSTv1_AQ_AR_Dualview_20260919_v1'


def link(path,label):
    return f'[{label}](<{Path(path).resolve().as_posix()}>)'


def cell(m,s):
    v=m[s]
    return f"{v['tp']}/{v['tp']+v['fn']}；误报 {v['fp']}"


def main():
    assert not (RUN/'REPORT.md').exists()
    protocol=read_json(RUN/'protocol.json');unchanged(protocol)
    base=read_json(RUN/'baseline_evaluation.json');reg=read_json(RUN/'regression_results.json')
    lock=read_json(RUN/'selection_lock.json');assert lock['recipe']=='initial_dual'
    backups=[];histories={}
    for recipe in protocol['recipes']:
        name=recipe['name'];backup=RUN/name/'pre_training_backup/backup_manifest.json'
        verify_backup(backup.parent)
        b=read_json(backup)
        backups.append(dict(recipe=name,path=str(backup.parent),files=len(b['files']),
            bytes=sum(f['bytes'] for f in b['files']),verified=True,manifest_sha256=sha256(backup)))
        histories[name]=read_json(RUN/name/'history.json')
    hard=read_json(RUN/'hard_negative_manifest.json')
    source_counter=Counter(r['source_species'] if 'source_species' in r else str(r['target']) for r in hard)
    external=read_json(RUN/'external_checks/results.json')
    user=read_json(ROOT/'outputs/user_training_view_audit_20260919_v2/results.json')
    # These are already-training examples, not a new validation set or new labels.
    problems=[r for r in user['images'] if not r['dual']['correct']]
    save_json(RUN/'user_training_remaining_cases.json',dict(
        role='Already-training user photos; target/foreground inspection only. Not validation. Do not relabel from predictions.',
        model='Original deployed weights, equal-probability whole-image/center combination',images=problems))
    text=['# 用户训练图片中仍未识对的样本', '',
          '这不是待重标的“错误标签”清单。保留用户确认的树种；仅检查目标是否明确、背景是否含其他植物、是否应补做目标区域裁剪。', '',
          '这些图片已属于训练数据，任何后续裁片必须随原图留在训练分组；不得移作验证。当前共 '+str(len(problems))+' 张，均为土沉香。', '',
          '| 图片 | 整图预测 | 双取景预测 | 土沉香双取景分数 |', '|---|---|---|---:|']
    for r in problems:
        text.append(f"| {link(r['path'],r['filename'])} | {r['pad']['top1']} | {r['dual']['top1']} | {r['dual']['true_score']:.1%} |")
    (RUN/'USER_TRAINING_CASES.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    lines=['# 土沉香与白桂木：双取景一致训练及难负样本对照', '', '日期：2026-09-19', '',
        '## 结论', '',
        '完成 D/E/F 三组真实训练，各 120 个优化步，共 360 步。没有训练后的权重通过预先冻结的替换门槛；锁定结果仍是原正式权重的双取景对照（initial_dual，step 0），不能把它称为本轮训练的新模型。正式 Demo、Windows/安卓发布文件和图像/点云/融合权重均未替换。', '',
        '确认了两件事：①整图加中心细节的取景方式有效影响结果；②当前训练集上增加目标类权重或加强负样本惩罚，会在漏识与误报之间产生取舍，尚未得到可放心替换的全面收益。', '',
        '## 实验约束', '',
        '- 4,198 张原训练图片，土沉香 215 张、白桂木 131 张；没有混入验证、历史测试或新下载隔离图片。每图两个取景仍算一个原始样本，原来源分组不变。',
        '- 在当前正式 SSTv1 白桂木增强版的三个 MLP 分类头上暖启动；BioCLIP 2.5 编码器、标签顺序、文本分支及温度保持冻结。',
        '- 全程以等权平均的两路概率计算训练损失：正式整图补边 + 短边 224 中心裁剪。中心特征按实际推理 batch=1 提取。',
        '- 学习率 0.00003，KL 保持项 0.5。D：类别训练权重不加倍；E：两目标类别训练权重 2 倍；F：2 倍类别权重并增加 0.1 难负样本约束。每组各自恢复正式初始化。',
        f'- 仅在训练数据内挖掘 320 条困难负样本记录，来自 {len(set(r["sha256"] for r in hard))} 张不同文件；每目标类最多 160 条、每来源分组最多 3 张；原始树种标签未改动。',
        '- 选择门槛同时对照原整图与原双取景：新旧开发子集内两个目标的 TP/F1 不降、FP 不增；总体识对数不降、宏 F1 降幅不超过 0.001。历史回归集只在锁定之后评估。',
        '- 开发集反复用于实验选择，旧测试集反复检查，因此均不能冒充新的盲测泛化证据。', '',
        '## 开发集：为什么没有接纳训练权重', '',
        '开发集共 2,655 张，其中 2,650 张为目录内分类样本，另 5 张仅参与误报检查。土沉香 22 张、白桂木 16 张。', '',
        '| 方案 / 步数 | 土沉香识对 / 总数；误报 | 白桂木识对 / 总数；误报 | 目录内识对数 / 2650 |',
        '|---|---|---|---:|']
    for name,m in [('原权重：整图',base['single']['combined']),('原权重：双取景',base['dual']['combined'])]:
        lines.append(f"| {name} | {cell(m,'aquilaria')} | {cell(m,'white')} | {m['all']['correct']} |")
    for name,steps in [('D_matched_views',[10,120]),('E_focus2',[10,120]),('F_focus2_hardneg',[20,120])]:
        for step in steps:
            item=next(r for r in histories[name] if r['step']==step);m=item['metrics']['combined']
            lines.append(f"| {name} / {step} | {cell(m,'aquilaria')} | {cell(m,'white')} | {m['all']['correct']} |")
    lines += ['',
        'D 的早期结果把两个目标的误报各减少 1 次，但总体少识对 4 张，且没有增加两个目标的识对数，因此未通过事先定好的总体保护门槛。E 增加类别权重后误报更多；F 虽然减少误报，却明显增加漏识。表中的中间步仅为开发日志诊断，未按回归结果挑选或补训。', '',
        '## 锁定后历史回归：取景收益，不是训练收益', '',
        '下表使用相同的 5,146 张历史回归图片。两个取景都是同一张图片，不是新增观测角度或独立树木。', '',
        '| 方案 | 土沉香识对 | 土沉香误报 | 白桂木识对 | 白桂木误报 | 全目录 Top-1 |',
        '|---|---:|---:|---:|---:|---:|']
    for key,name in [('baseline_single','原权重＋整图'),('baseline_dual','原权重＋双取景（锁定对照）')]:
        m=reg[key];a=m['aquilaria'];w=m['white']
        lines.append(f"| {name} | {a['tp']}/25（{a['recall']:.1%}） | {a['fp']} | {w['tp']}/22（{w['recall']:.1%}） | {w['fp']} | {m['all']['top1']:.2%} |")
    lines += ['',
        f"双取景相对整图，在全部历史回归图片中纠正 {reg['corrected']} 张，同时把 {reg['regressed']} 张原来正确的图片改错。白桂木精确率从 {reg['baseline_single']['white']['precision']:.1%} 降到 {reg['baseline_dual']['white']['precision']:.1%}；不能只宣传召回上升而隐去误报增加。",
        '本轮锁定的初始权重与原双取景对照完全相同；两者差值为 0，并非训练产生了额外收益。未对不合格候选反复跑回归来挑最好看的结果。', '',
        '## 用户原图回放与剩余问题', '',
        '已入训练的用户新增照片并非验证集：原权重双取景在土沉香 76 张中识对 52 张（整图为 46 张）；白桂木 9 张中识对 9 张。仍有 24 张土沉香训练图未识对。分数是未校准模型输出，不等于正确率。',
        link(RUN/'USER_TRAINING_CASES.md','逐张打开剩余的 24 个训练案例')+'。下一阶段应先检查这些训练图的目标区域与可辨识细节，再做保持来源分组的目标裁剪训练；不是根据预测把用户标签改成别的物种。', '',
        '## 新来源核验', '',
        '- 本轮 iNaturalist 按精确树种、研究级记录、多位来源鉴定者、照片许可、新观察者/观察记录、哈希与近重复排除进行有界检索。新增 10 张菠萝蜜 Artocarpus heterophyllus 负样本；没有找到通过全部隔离条件的新土沉香或白桂木正样本。不能把同一观察的转载或重复照片算成新验证数据。',
        '- 逐张查看后，在推理前排除了 1 张主体主要为树干且叠有物种俗名文字的图片；余下 9 张全部保留。它们是来源社区鉴定，不是独立专家复核。',
        '- 原权重整图和双取景在这 9 张上均识对菠萝蜜，均未误报土沉香或白桂木。这说明这小批果实/叶片负例不难，不能证明街景上性能提升。',
        '- 另只读检查了先前已采集的香港植物标本室两张叶果细节示例，各目标 1 张；两种取景都识对。它们未参与训练或选模型，也不是本轮新采集正样本，更不是足够大的街景验证集。',
        '  来源：[土沉香官方物种页](https://www.herbarium.gov.hk/en/publications/books/book2/text/aquilaria-sinensis/index.html)、[白桂木官方物种页](https://www.herbarium.gov.hk/en/publications/books/book2/text/artocarpus-hypargyreus/index.html)。',
        '- 上述图片全部保持训练不合格/隔离状态，保留署名、原始水印和许可证据；未放入正式 Demo、应用安装包或任何公开发布。', '',
        '## 回滚与正式入口', '',
        '三轮都在拟合前进行了物理复制备份，保存原绝对路径、复制路径、字节数和 SHA-256；训练前、每轮结束及本报告生成时均验证成功。正式模型及受保护发布文件哈希未变。', '']
    for b in backups:
        lines.append('- '+link(b['path'],b['recipe']+' 训练前备份')+f"：{b['files']} 文件，{b['bytes']/1024**3:.2f} GiB。")
    lines += ['', '## 复现与记录', '',
        '- '+link(ROOT/'tools/aq_ar_dualview_next_20260919.py','准备、编码、训练、锁定及回归脚本'),
        '- '+link(RUN/'protocol.json','训练与选择协议'),
        '- '+link(RUN/'selection_lock.json','仅由开发集产生的锁定记录'),
        '- '+link(RUN/'regression_results.json','历史回归完整指标'),
        '- '+link(RUN/'external_checks/results.json','外部来源逐图核验结果'),
        '- '+link(RUN/'hard_negative_manifest.json','仅来自训练集的困难负例清单'), '',
        '建议：不部署这三轮权重，也不通过调低温度把分数变高。当前可测试的产品改进是有边界的双取景及不确定性提示；模型层面的下一步是用户已确认训练图的目标裁剪与新增来源正例，仍须用隔离数据验证。']
    with requests.Session() as client:
        health=client.get('http://127.0.0.1:8039/api/health',timeout=15);health.raise_for_status()
        webpage=client.get('http://127.0.0.1:8043/',timeout=15);webpage.raise_for_status()
        assert health.json()['sstv1_version']==protocol['deployment_manifest']['version']
    unchanged(protocol)
    save_json(RUN/'HANDOFF.json',dict(selected=lock['recipe'],new_trained_candidate_accepted=False,
        optimizer_steps=360,backups=backups,formal_files_unchanged=True,formal_service_version=health.json()['sstv1_version'],
        webpage_status=webpage.status_code,remaining_user_training_cases=len(problems),
        fresh_focal_positive_images=0,new_negative_images=10,evaluated_new_negatives=9,
        independent_street_tree_accuracy_established=False,external_check_results=str(RUN/'external_checks/results.json')))
    (RUN/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('REPORT_COMPLETE',str(RUN/'REPORT.md'),flush=True)
    print('BACKUPS_GIB',sum(b['bytes'] for b in backups)/1024**3,flush=True)


if __name__=='__main__':main()
