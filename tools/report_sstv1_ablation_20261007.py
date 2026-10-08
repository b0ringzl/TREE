"""Consolidate completed, immutable ablation evidence; never select/deploy."""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'tools')]
import numpy as np
import torch
import ablate_sstv1_accuracy_20261007 as a
from training_safety import verify_backup, sha256


def main(inference, fits, out):
    for directory in (inference, fits):
        assert (directory/'production_verification.json').exists()
        verify_backup(directory/'pre_training_backup')
    out.mkdir(parents=True, exist_ok=False)
    first = a.load(inference/'results.json'); second = a.load(fits/'results.json')
    modes = first['subsets']['common398']['modes']; train_modes = second['subsets']['common398']['modes']
    paired = second['subsets']['common398']['paired']
    stage_names = a.load(inference/'stages.json')
    runs = a.load(fits/'fits.json')
    reproduction = []
    for run, original in zip([r for r in runs if r['recipe']=='control_mlp'], stage_names['adapted_mlp']['members']):
        current = torch.load(run['checkpoint'], map_location='cpu', weights_only=False)['model_state']
        previous = torch.load(original['path'], map_location='cpu', weights_only=False)['model_state']
        max_error = max(float((current[k]-previous[k]).abs().max()) for k in current)
        reproduction.append({'seed': run['seed'], 'max_parameter_error': max_error, 'bitwise_same': all(torch.equal(current[k],previous[k]) for k in current)})
    with np.load(fits/'regression_predictions.npz') as c, np.load(inference/'predictions.npz') as b:
        parity = {v: float(np.max(np.abs(c['control_mlp_'+v]-b['adapted_mlp_'+v]))) for v in ('center','whole','context')}
        labels = a.load(inference/'inference_complete.json')['labels']
        same_top1 = {v: bool(np.array_equal(a.grouped_array(c['control_mlp_'+v], labels)[0].argmax(1),
                                         a.grouped_array(b['adapted_mlp_'+v], labels)[0].argmax(1))) for v in ('center','whole','context')}
    a.save(out/'reproducibility.json', {'control_refits_vs_historical_initial_mlp': reproduction,
                                      'batched_vs_single_image_inference_max_abs_error': parity,
                                      'same_grouped_top1_on_all590_images': same_top1,
                                      'floating_point_note': 'Same mathematical recipe; current weighting normalization uses CPU mean before CUDA transfer, original uses CUDA mean. Tiny numerical drift is reported, not hidden or called bitwise reproduction.',
                                      'inference_benchmark_parity_max_error': a.load(inference/'parity.json')['max_error']})
    lines = ['# 树种识别精度提升：消融实验结论', '',
             '完成两类实验：固定历史检查点/推理组件对照，以及四种训练配方×三个固定种子的12次重新拟合。正式Demo未改动。', '',
             '## 核心结论', '',
             f"在398张固定回归图片上，原BioCLIP2.5为{modes['zero_context']['correct']}/398（{modes['zero_context']['top1']:.2%}）；初期香港残差MLP分类头已达到{modes['adapted_mlp_context']['correct']}/398（{modes['adapted_mlp_context']['top1']:.2%}）；当前H120仍为{modes['h120_context']['correct']}/398（{modes['h120_context']['top1']:.2%}）。所以总体+6.78个百分点主要已在初期分类头适配阶段形成，不是后续SAM或编码器解冻带来的。", '',
             '### 一、历史训练阶段：相同图片、相同取景', '',
             '| 阶段 | 识对/398 | Top-1 | 与上一阶段的差值 |', '| --- | ---: | ---: | ---: |']
    before = None
    for stage in ('zero','adapted_mlp','sst_sep14','aq_sep15','ar_sep18','h120'):
        s = modes[stage+'_context']; delta = '—' if before is None else f"{100*(s['top1']-before):+.2f}个百分点"
        lines.append(f"| {a.STAGE_NAMES[stage]} | {s['correct']}/398 | {s['top1']:.2%} | {delta} |")
        before = s['top1']
    lines += ['', '阶段差分包含多项改动，仅定位效果发生的阶段，不把数据增补、优化步数、KL或增强的作用混为单变量因果结论。', '',
              '### 二、重新训练的留一消融：同起点、同学习率、同验证分割', '',
              '| 配方 | 识对/398 | Top-1 | 完整配方减去此配方 | 成组95%区间（百分点） |', '| --- | ---: | ---: | ---: | --- |']
    descriptions = {'control_mlp':'完整初期配方（重跑对照）','no_class_balance':'撤掉类别均衡，其余保持','no_herbarium':'撤掉127张已审核标本图片','linear_same_lr':'换成线性头，同学习率与优化器'}
    for recipe, description in descriptions.items():
        s = train_modes[recipe+'_context']
        if recipe == 'control_mlp': delta, ci = '—', '—'
        else:
            v = paired[recipe+'_vs_control_mlp']; delta = f"{v['gain_pp']:+.2f}个百分点"; ci = f"[{v['cluster_ci95_pp'][0]:+.2f}, {v['cluster_ci95_pp'][1]:+.2f}]"
        lines.append(f"| {description} | {s['correct']}/398 | {s['top1']:.2%} | {delta} | {ci} |")
    tuned_linear = modes['linear_lr01_context']
    tuned_pair = first['subsets']['common398']['paired']['linear_lr01_context_vs_adapted_mlp_context']
    lines += ['', '正差值表示该组件对完整初期配方有益；负差值表示撤掉后反而更好。这些留一差值不能直接相加，也不是从最终H120完整重训后的独立贡献比例。MLP/线性是整体结构差异，含非线性、LayerNorm和Dropout，不能只归因于某一层。', '',
              f"重要优化预算限定：历史线性头在验证集选择的lr=0.01下为{tuned_linear['correct']}/398（{tuned_linear['top1']:.2%}），已经比零样本高{100*(tuned_linear['top1']-modes['zero_context']['top1']):+.2f}个百分点；与MLP仅差{tuned_pair['gain_pp']:+.2f}个百分点，成组95%区间[{tuned_pair['cluster_ci95_pp'][0]:+.2f}, {tuned_pair['cluster_ci95_pp'][1]:+.2f}]跨0。所以同lr=0.003、80步的+4.02只代表这一固定优化预算下的结构配方差异，不能宣称MLP显著优于已经调优的线性头，更不能把全部+6.78归给非线性结构。主要结论是监督分类头适配的整体作用。", '',
              f"训练集{second['plan']['train_count']}张，127张标本重新核对human_review.sqlite3为accepted且哈希一致；拒绝/待审图片未参与。每次拟合80步，AdamW，lr=0.003，三个原固定种子。仅在planning验证集选最佳轮次，selection验证集记录结果；所有配方都报告，没有根据398张回归结果挑选或部署方案。", '',
              '### 三、推理组件：不重新训练', '',
              '| 对照 | 准确率/作用 |', '| --- | --- |',
              f"| H120仅中心取景 | {modes['h120_center']['top1']:.2%}（{modes['h120_center']['correct']}/398） |",
              f"| H120全图＋中心 | {modes['h120_context']['top1']:.2%}；相对中心仅+{100*(modes['h120_context']['top1']-modes['h120_center']['top1']):.2f}个百分点，成组区间跨0 |",
              f"| H120仅全图补边 | {modes['h120_whole']['top1']:.2%}；不支持把中心细节一概取消 |"]
    single = np.mean([modes[f'h120_member{i}_context']['top1'] for i in range(1,4)])
    ensemble_gain = 100*(modes['h120_context']['top1']-single)
    if abs(ensemble_gain) < 1e-8: ensemble_gain = 0.
    lines += [f"| 三头平均 vs 三个固定单头平均成绩 | 差值{ensemble_gain:+.2f}个百分点；这批数据不支持集成为主要来源 |",
              '| SAM / 检测 / 跟踪 / 点云 | 原+6.78pp评测未经过这些模块，本轮无法衡量它们的贡献 |',
              '| 编码器解冻 | 所有历史版本为同一个编码器SHA256，本轮也未解冻 |', '',
              '### 四、类别目录规则的影响', '']
    g = first['bauhinia_presentation_only']
    lines += [f"同样不合并洋紫荊时，模型间提升为{g['ungrouped_148_gain_pp']:+.2f}个百分点；同样合并为146展示类别时为{g['grouped_146_gain_pp']:+.2f}个百分点。两者相差{g['grouped_146_gain_pp']-g['ungrouped_148_gain_pp']:+.2f}个百分点，说明统计口径与分类分数存在交互。合并本身不是模型学得更好，也没有证明能分开组内物种。", '',
              '所以“两个模型都用相同合并规则”是公平对照条件，但不意味着目录规则对差值完全没有影响。榕树和南洋杉已在原始映射里合并，这轮不能拆开检验。', '',
              '### 五、土沉香：补充训练有专项作用', '',
              '| 版本（相同双取景） | 识对/154 | 正类召回率 |', '| --- | ---: | ---: |']
    aq = first['subsets']['aquilaria_primary']['modes']
    for stage in ('zero','adapted_mlp','aq_sep15','h120'):
        s = aq[stage+'_context']; lines.append(f"| {a.STAGE_NAMES[stage]} | {s['correct']}/154 | {s['top1']:.2%} |")
    lines += ['', '后续训练在常见树这组没有净增，并不意味着“没有用”：土沉香从初期142张识对到当前147张。此154张只有正类，百分比是召回率，不能代替包含其他树种误报的精确率。', '',
              '## 复现与边界', '',
              f"- 重跑完整配方与历史初期MLP的最大参数差为{max(r['max_parameter_error'] for r in reproduction):.3g}，非位级相同；最大推理概率差为{max(parity.values()):.3g}。590张图片的三个取景Top-1是否分别完全一致：{same_top1}。详见reproducibility.json。",
              '- 数学配方相同，但本轮权重均值在CPU算后传GPU，历史代码在GPU算，存在微小浮点实现差异；没有声称逐bit复现。所有新训练对照共用同一实现。线性头零初始化、无Dropout，三个种子得到一致结果，不当作三次独立数据验证。',
              '- 398张数据已被检查过，此轮是归因/回归实验，不是新盲测；仅覆盖101/146展示类别，香港照片17张，不能推出全目录香港街景整体精度。',
              '- 置信度不作为准确率；没有删除识错样本，没有改正式温度、改标签或人工改预测。另有T=1的推理敏感性对照，未用于选择或校准。',
              '- 原教师先验移除仅作诊断，残差网络没有按独立分类器训练；且狐尾椰零样本扩展仍保留，不能把该项大跌幅解释成纯先验因果贡献。',
              '- 完整逐类指标、改对/改错列表、每种设置与种子、训练/验证/测试分割和权重均保留。', '',
              '## 文件与安全', '',
              f"- 固定检查点/推理报告：{inference/'REPORT_ZH.md'}",
              f"- 重新拟合组件报告：{fits/'REPORT_ZH.md'}",
              f"- 第一轮备份：{inference/'pre_training_backup'}",
              f"- 训练方案备份：{fits/'pre_training_backup'}",
              f"- 每次拟合独立备份：{fits/'runs'}内各种子目录的pre_training_backup。",
              '- 所有物理权重副本已SHA256核验，没有硬链接；正式网页、安卓、点云与融合权重未改动。没有公开发布图片或实验权重。']
    (out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    a.save(out/'evidence.json', {'inference_results': str(inference/'results.json'), 'inference_results_sha256': sha256(inference/'results.json'),
                               'training_results': str(fits/'results.json'), 'training_results_sha256': sha256(fits/'results.json'),
                               'training_performed': True, 'fit_count': len(runs), 'production_changed': False})
    print('SUMMARY', out/'REPORT_ZH.md')


if __name__=='__main__':
    p = argparse.ArgumentParser();p.add_argument('--inference',type=Path,required=True);p.add_argument('--fits',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    r=p.parse_args();main(r.inference.resolve(),r.fits.resolve(),r.out.resolve())
