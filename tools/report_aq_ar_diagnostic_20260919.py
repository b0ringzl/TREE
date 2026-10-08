"""Assemble read-only experiment evidence and a non-deployed candidate bundle."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import torch
import numpy as np
from distillation_core import read_json, save_json
from training_safety import sha256,verify_backup
from train_aq_ar_diagnostic_20260919 import unchanged
from train_artocarpus_user_20260918 import logprob
from bioclip_probe import ResidualHead

OUT=ROOT/'outputs/SSTv1_AQ_AR_Diagnostic_20260919_v1'
NAMES={'aquilaria':'土沉香 Aquilaria sinensis','white':'白桂木 Artocarpus hypargyreus'}


def metric_row(label,m):
    a,b=m['aquilaria'],m['white']
    return f"| {label} | {a['tp']}/{a['tp']+a['fn']}（{a['recall']:.1%}） | {a['fp']} | {a['precision']:.1%} | {b['tp']}/{b['tp']+b['fn']}（{b['recall']:.1%}） | {b['fp']} | {b['precision']:.1%} | {m['all']['top1']:.2%} |"


def main():
    protocol=read_json(OUT/'protocol.json');lock=read_json(OUT/'selection_lock.json')
    complete=read_json(OUT/'COMPLETE.json'); exact=read_json(OUT/'exact_regression/COMPLETE.json')
    assert exact['selection_unchanged'] and sha256(lock['checkpoint'])==lock['sha256']
    result=read_json(OUT/'exact_regression/results.json');base=read_json(OUT/'baseline_evaluation.json')
    views=read_json(ROOT/'outputs/user_training_view_audit_20260919_v2/results.json')
    user=read_json(OUT/'user_training_replay.json'); labels=read_json(OUT/'labels.json')
    bundle=OUT/'candidate_bundle';bundle.mkdir(exist_ok=False)
    ckpt=torch.load(lock['checkpoint'],map_location='cpu',weights_only=False)
    members=[]
    for i in range(3):
        state={key[len(str(i))+1:]:value for key,value in ckpt['model_state'].items() if key.startswith(str(i)+'.')}
        path=bundle/f'member_{i+1}.pt'
        torch.save(dict(model_state=state,config=dict(architecture='mlp'),classes=read_json(OUT/'classes.json'),
                        encoder_sha256=protocol['deployment_manifest']['encoder_sha256']),path)
        members.append(dict(file=path.name,sha256=sha256(path)))
    save_json(bundle/'inference_contract.json',dict(status='EXPERIMENTAL_NOT_DEPLOYED',
        version='sstv1-aq-ar-candidate-20260919-v1',members=members,labels=labels,
        initialization_version=protocol['deployment_manifest']['version'],
        encoder_sha256=protocol['deployment_manifest']['encoder_sha256'],temperature=protocol['temperature'],
        input='User-confirmed single-tree photo, organ detail or already reprojected selected panorama crop; not unselected full panorama',
        view1='spacescan_trees_demo_v1.server.prepare -> teacher.engine.decode (letterbox 1024 JPEG95 ->224)',
        view2='teacher.engine.decode on original bytes (EXIF RGB bicubic short side224, center224)',
        aggregate='Within each view mean of 3 temperature-calibrated member probabilities; mean of two views',
        selection=lock,encoder_passes=2,automatic_deployment=False,
        license='Local experiment only; no permission for redistributing downloaded training images or experimental weights inferred'))
    # Verify the exported head bundle actually reproduces the locked predictor.
    torch.set_num_threads(4)
    exported=torch.nn.ModuleList()
    for i,item in enumerate(members):
        restored=torch.load(bundle/item['file'],map_location='cpu',weights_only=False)
        for key,value in restored['model_state'].items():
            assert torch.equal(value,ckpt['model_state'][f'{i}.{key}'])
        head=ResidualHead('mlp').eval();head.load_state_dict(restored['model_state']);exported.append(head)
    exported.eval()
    with np.load(ROOT/'outputs/SSTv1_Artocarpus_User_20260918_v1/train_features.npz') as f:
        params=dict(groups=read_json(OUT/'classes.json')['groups'],temp=protocol['temperature'],
                    extra=torch.tensor(f['extra_bank']),scale=float(f['scale']))
    with np.load(OUT/'exact_regression/center_features.npz') as f:
        x=torch.tensor(f['x'][:32].astype(np.float32));z=torch.tensor(f['z'][:32])
    with torch.inference_mode():
        actual=logprob(exported,x,z,**params).exp().numpy()
    with np.load(OUT/'exact_regression/predictions.npz') as f:
        expected=f['candidate_center'][:32]
    error=float(abs(actual-expected).max());assert error<5e-5
    save_json(bundle/'VERIFICATION.json',dict(member_state_exactly_matches_lock=True,
        cpu_export_replay_max_probability_error=error,n=32,passed=True,no_deployment=True))
    unchanged(protocol)
    backup_bytes=0
    for folder in complete['backups']:
        data=verify_backup(folder)
        backup_bytes+=sum(f['bytes'] for f in data['files'])
    flags=[]
    for sp in NAMES:
        a,b=result['baseline'][sp],result['candidate'][sp]
        if b['fp']>a['fp']:flags.append(f"{NAMES[sp]}回归误报 {a['fp']}→{b['fp']}")
        if b['precision']<a['precision']:flags.append(f"{NAMES[sp]}回归precision {a['precision']:.1%}→{b['precision']:.1%}")
    save_json(OUT/'handoff.json',dict(production_unchanged=True,deployed=False,
        recommended_action='Keep production unchanged. Review crop-strategy benefit and white-guimu false positives before any authorized switch.',
        regression_flags=flags,candidate_bundle=str(bundle),verified_backups=complete['backups'],backup_bytes=backup_bytes,
        final_result=str(OUT/'exact_regression/results.json'),legacy_result_superseded=True))
    lines=['# 土沉香／白桂木：逐项诊断与三组受控训练', '', '日期：2026-09-19。当前正式模型保持 `sstv1-artocarpus-user-20260918-v1`，网页、Windows、安卓配置和点云／融合模型均未切换。', '',
      '## 结论', '',
      '已完成三次真实训练，每次 120 个全批量优化步（共 360 步）。验证选出 A 组第 10 步＋固定双取景作为候选。不是把正确树种的显示分数手动调高，也没有修改温度或标签。', '',
      '取景方式带来的改善比本轮继续训练本身更明确：整图保留轮廓，中心取景提供较大的局部细节，两者固定各占 50%。这是同一照片的两种取景，不是两个独立样本，也不是不同角度的多帧。', '',
      '本轮不建议立即换掉正式模型：训练的额外收益尚不稳定，而且白桂木在回归集中有误报代价。下文同时列出改善与退步。', '',
      '## 逐项排查', '',
      '| 项目 | 证据与判断 |', '|---|---|',
      '| 图片是否漏训 | 两个文件夹 102 张原图的 SHA 与导入记录一致；76 张土沉香和 9 张白桂木新增照片均在当前模型训练清单。另有 14 张旧训练近重复、1 张旧回归重叠和2张排除图，未重新塞入训练。不是“完全没有训练这些图片”。 |',
      '| Demo 是否加载旧权重 | 实际接口的85张新增训练照片，其正确标签分数与当前已部署权重的保存预测最大差值不到 2e-7；初始化三个头也逐一校验 SHA。排除这批照片上的未加载问题。 |',
      '| 取景／背景／细节 | 固定相同模型，只改整图补边与中心取景，开发集和用户训练重放均有变化。说明输入表达是实质因素；同时改变上下文、有效目标分辨率和JPEG像素，不能声称单独证明某一个因素。 |',
      '| 学习率过小 | A为3e-5，B为1e-4，除此之外配置相同。B的训练拟合更好，但两目标开发指标没有共同优于被选中的A早期结果，不能简单认定放大学习率会更准。 |',
      '| 重点树种权重不够 | C与B只相差重点类CE权重×4。训练集目标召回提高，但开发集误报明显增加，0个检查点通过安全门槛；不采用。 |',
      '| 原模型约束／遗忘 | 三组都使用初始分布KL约束以保护其它类；未单独比较去除KL。旧白桂木更新曾让一张土沉香由正确变错误，本轮也保留所有回归明细，不能把置信度升高当成全局改善。 |',
      '| 网络标签和独立样本量 | 当前训练4198张，其中土沉香215张／141个来源组，白桂木131张／57个来源组。来源组不等于已确认独立树；许多旧网络标签没有逐张独立人工植物学鉴定。未凭模型预测重标它们。 |',
      '| 特征骨干能力 | 本轮冻结BioCLIP，仅训练现有三个MLP。没有做骨干解冻实验，因此不能断言剩余错误都源于骨干能力，也不能声称已穷尽所有原因。 |',
      '| 自动选树／手动裁剪 | 本轮指标是分类阶段：明确输入整图或固定中心取景；不是包含用户选树决策的端到端测评。不能直接替代全景检测、畸变校正及目标匹配的验收。 |',
      '| 分数与正确率 | 维持同一温度1.8877486253633864。分数是模型输出，不是保证正确的概率；评价看正确数、漏检、误报及F1，而非只追求80%以上的显示分数。 |', '',
      '## 固定开发集（只在这里选择检查点）', '',
      '2655张：2650张属于现有148类，另外5张目录外树种只计入两重点类的误报检查，不计入148类总体准确率。土沉香22张、白桂木16张。既有数据已多次查看，不是新盲测。', '',
      '| 方案 | 土沉香正确／总数 | 土沉香误报 | 土沉香precision | 白桂木正确／总数 | 白桂木误报 | 白桂木precision | 全体Top-1 |',
      '|---|---:|---:|---:|---:|---:|---:|---:|',
      metric_row('正式模型＋整图',base['letterbox']['combined']),
      metric_row('不改权重＋固定双取景',base['dual']['combined']),
      metric_row('验证选出：A第10步＋双取景',lock['metrics']['combined']), '',
      '## 最终逐张回归（模型锁定后才计算）', '',
      '5146张历史回归图，土沉香25张、白桂木22张。所有中心取景使用当前Demo相同BF16编码器、batch=1重新编码；整图缓存已与当前已部署模型核对。', '',
      '| 方案 | 土沉香正确／总数 | 土沉香误报 | 土沉香precision | 白桂木正确／总数 | 白桂木误报 | 白桂木precision | 全体Top-1 |',
      '|---|---:|---:|---:|---:|---:|---:|---:|',
      *[metric_row(label,result[k]) for k,label in [('baseline','正式模型＋整图'),('baseline_dual','不改权重＋双取景（消融对照）'),('candidate_single','A第10步＋整图（消融对照）'),('candidate','验证锁定候选＋双取景')]], '',
      f"锁定候选相对正式整图：纠正 {result['corrected']} 张，退步 {result['regressed']} 张。全体准确率差的来源组bootstrap：{result['bootstrap']['point']*100:+.2f} 个百分点，95%区间 {result['bootstrap']['ci95'][0]*100:+.2f}～{result['bootstrap']['ci95'][1]*100:+.2f} 个百分点。该区间不是两个树种各自的显著性结论，也不能消除历史重复看测试集、未知预训练重叠和未知同株的局限。", '',
      '风险：'+('；'.join(flags) if flags else '未触发两重点类的回归误报/precision下降标记；仍需新的独立样本确认。'), '',
      '选择规则没有在看到回归结果后改动。不要因为某个消融对照在回归集更好，就回头重新选择权重；所有原记录保留。', '',
      '## 用户新增训练照片：为什么看起来仍不够准', '',
      '下面只用于解释拟合和取景，不是独立验证准确率。保持原正式模型权重，真实接口测试两种取景：', '',
      '| 树种 | 张数 | 整图正确 | 中心正确 | 两者综合正确 | 综合后正确树种平均分数 |', '|---|---:|---:|---:|---:|---:|']
    for r in views['summary']:
        v=r['views'];lines.append(f"| {r['species']} | {r['n']} | {v['pad']['correct']} | {v['center']['correct']} | {v['dual']['correct']} | {v['dual']['true_score_mean']:.1%} |")
    lines+=['',f"单独换成A第10步权重、仍只用整图：85张新增照片正确数从 {user['baseline']['all']['correct']} 到 {user['candidate']['all']['correct']}。这进一步说明不能把本次改善全部归因于继续训练。",'',
      '| 照片 | 现整图预测 | 双取景预测 | 正确标签分数：整图→双取景 |', '|---|---|---|---:|']
    for r in views['images']:
        if r['pad']['correct']!=r['dual']['correct']:
            lines.append(f"| [{r['filename']}](<{Path(r['path']).as_posix()}>) | {r['pad']['top1']} | {r['dual']['top1']} | {r['pad']['true_score']:.1%}→{r['dual']['true_score']:.1%} |")
    lines+=['','## 后续改善路线与交付状态','',
      '1. 已验证的方向：目标确认后保留整图结构并补充中心细节，不在未知目标的整张全景上直接使用中心裁切。代价是两次编码器推理，需另做移动端速度和内存验证。',
      '2. 未解决的重点：白桂木误报，以及土沉香剩余难例。需逐图区分小目标、背景干扰、视角、标签证据等因素，不能笼统归咎于画质。下一轮应采用训练集内的混淆负样本和与双取景一致的训练增强；选定方案后再用新的、来源分组隔离的审核数据验收。不能把本轮回归图转成训练图。',
      '3. 本轮候选已另存为candidate_bundle，包含3个MLP与输入契约，不包含复制的下载图片；不是可直接替换旧单取景入口的部署包。正式Demo、Windows、安卓及点云模型保持不变，须用户另行同意后才切换。', '',
      '## 安全与复现','',
      '每组拟合前均物理复制初始化头、冻结编码器、部署中的影像/点云/融合权重，以及标签、清单、划分、配置、随机种子、基线预测和审核快照；SHA-256验证通过后才拟合。三份备份互不复用，无硬链接、无续跑覆写。', '',
      f'本轮备份合计 {backup_bytes/(1024**3):.2f} GiB。路径：','']
    lines += [f"- [{Path(p).parent.name} 训练前备份](<{Path(p).as_posix()}/backup_manifest.json>)" for p in complete['backups']]
    lines += ['', '主要记录：','',
      f'- [最终逐张回归结果](<{(OUT/"exact_regression/results.json").as_posix()}>)',
      f'- [重点树种逐图回归明细](<{(OUT/"exact_regression/focal_cases.json").as_posix()}>)',
      f'- [验证选择锁](<{(OUT/"selection_lock.json").as_posix()}>)',
      f'- [冻结训练方案](<{(OUT/"protocol.json").as_posix()}>)',
      f'- [本次训练脚本](<{(ROOT/"tools/train_aq_ar_diagnostic_20260919.py").as_posix()}>)',
      '',f"旧批量中心缓存与逐张中心重算有 {result['legacy_cache_parity']['changed_center_top1']} 个首选预测不同，最大概率差 {result['legacy_cache_parity']['max_probability_error']:.6f}。因此目录根部 regression_results.json 仅保留作初步缓存诊断，最终结论统一采用 exact_regression/results.json。",'']
    report=OUT/'REPORT.md'; assert not report.exists();report.write_text('\n'.join(lines),encoding='utf-8')
    # Every still-wrong confirmed training image is listed for traceability, not relabeled.
    remain=['# 当前正式模型固定双取景仍未识别正确的用户训练照片','',
            '标签仍采用用户鉴定；这里只记录模型错误，不自动更改标签。不是新测试集。','',
            '| 原图 | 已确认树种 | 当前预测 | 正确标签分数 |','|---|---|---|---:|']
    for r in views['images']:
        if not r['dual']['correct']:
            remain.append(f"| [{r['filename']}](<{Path(r['path']).as_posix()}>) | {r['species']} | {r['dual']['top1']} | {r['dual']['true_score']:.2%} |")
    (OUT/'REMAINING_TRAINING_ERRORS.md').write_text('\n'.join(remain)+'\n',encoding='utf-8')
    print(report,flush=True)


if __name__=='__main__':
    main()
