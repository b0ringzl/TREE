"""Read-only evaluation of all trained trials; assemble transparent review artifacts."""
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
from hk_point_training_20260912 import OUT,read,write,metrics
from hk_point_model import predict_bundle
from training_safety import verify_backup,sha256
def main():
    import joblib
    plan=read(OUT/'plan.json');rows=read(OUT/'manifest.json');lookup={r['key']:r for r in rows};f=np.load(OUT/'features/all.npz');selected=[lookup[k] for k in f['keys']];y=np.array([r['class_index'] for r in selected]);test=np.array([i for i,r in enumerate(selected) if r['split']=='test_regression']);runs=[];predictions={}
    names={'00_centroid_baseline':'PTv2 centroid / 類別原型基線','01_point_linear_c1':'PTv2 linear C1 / 線性分類頭','02_point_linear_c10':'PTv2 linear C10 / 線性分類頭','03_point_geometry_c1':'PTv2 + geometry / 幾何特徵分類頭','04_rotation_mlp':'PTv2 + rotation MLP / 旋轉增強神經分類頭','05_confirmed_las_c10':'PTv2 + 2 verified LAS / 加入兩份核驗 LAS'}
    backup_files=0;backup_bytes=0
    for run in sorted((OUT/'runs').iterdir()):
        if not run.is_dir():continue
        manifest=verify_backup(run/'pre_training_backup');backup_files+=len(manifest['files']);backup_bytes+=sum(x['bytes'] for x in manifest['files'])
        if (run/'validation.json').exists():
            r=read(run/'validation.json');bundle=joblib.load(run/'model.joblib');p=predict_bundle(bundle,f['point'],f['geometry']);m=metrics(y[test],p[test],[selected[i] for i in test],plan['classes']);preds=[]
            for i,row in enumerate(selected):
                order=np.argsort(-p[i]);preds.append(dict(key=row['key'],split=row['split'],truth=row['species'],prediction=plan['classes'][order[0]],correct=bool(order[0]==y[i]),top3=[dict(species=plan['classes'][k],score=float(p[i,k])) for k in order[:3]]))
        else:r=read(run/'report.json');m=r['test_regression'];preds=read(run/'predictions.json')
        model=run/('best.pt' if run.name=='04_rotation_mlp' else 'model.joblib')
        runs.append(dict(id=run.name,name=names[run.name],validation=r['validation'],test=m,training=r['training'],model=str(model),model_sha256=sha256(model),backup=str(run/'pre_training_backup')));predictions[run.name]=preds
    chosen=max(range(len(runs)),key=lambda i:(runs[i]['validation']['macro_f1_supported_only'],runs[i]['validation']['group_accuracy'],-i))
    # Final decision repeats the predeclared validation-only rule; do not rank by regression test.
    summary=dict(plan=plan,runs=runs,selected_run=runs[chosen]['id'],selection_rule='validation macro F1, then group accuracy, then earlier simpler model; regression scores never select',accepted_source_views=95,new_las_cleaned=72,new_las_agent_confirmed=2,new_las_train_only_augmented=2,new_independent_trees=0,unlinked_las=60,fragment_las_excluded=10,backups_verified=6,backup_files=backup_files,backup_bytes=backup_bytes,production_unchanged=True,google_street_view_used=False)
    write(OUT/'final_summary.json',summary);write(OUT/'all_predictions.json',predictions)
    extra=read(OUT/'runs/05_confirmed_las_c10/manifest.json')[-2:];write(OUT/'viewer_manifest.json',rows+extra)
    s=runs[chosen];text=['# 香港点云单源训练实测报告（2026-09-12）','',f"已完成 6 组可回退实验。按验证集预定规则保留 `{s['id']}` 作为实验基线，**没有替换生产 Demo**。","",'## 结论','',f"本轮选择结果：验证集 {s['validation']['correct_views']}/7（{s['validation']['view_accuracy']:.1%}），测试回归集 {s['test']['correct_views']}/7（{s['test']['view_accuracy']:.1%}）；按树木分组合并后为 {s['test']['correct_groups']}/5。目前不能声称达到可靠的点云独立树种识别精度，也不能据此声称点云已经提升了 BioCLIP 的准确率。",'', '这是真实 XYZ 单源实验，图片仅用于标签来源和核验；没有向点云模型输入图像或图像预测结果。训练的是冻结 WHU PTv2 编码器上的分类头；包括 384→128→8 的神经网络分类头，**不是端到端微调整个 PTv2**。','', '## 数据与标签','', '- 原有 95 份接受记录：源图像、点云 SHA-256 与验收清单核对。人工确认是标签来源；联系图复看不冒称独立植物学鉴定。','- 新增 72 份单木 LAS：全部执行有限值检查、精确去重、保守孤立点清理和最多 2048 个真实点的高度/方位分层采样；保留米制坐标及归一化参数。','- 找到 12 份新旧点云坐标重合线索，并生成原始全景投影。经视觉核查将 01_TreeID_18、04_TreeID_5 标为石栗，依据是对应已验收单木的坐标和影像，而非用模型给自己制造标签。','- 这 2 份是已有训练树的补充提取版本，不是新增独立树；第 6 组实验加入后训练数为 56。其余 10 份有明显局部树冠/分枝或稀疏问题，60 份未建立可信影像对应，不盲目赋树种。','- 不删除原文件、不覆盖原标签。没有新做地面语义分割，也没有凭空补齐树干和树冠；“观测垂直跨度”不是实测完整树高。','', '## 划分和防泄漏','', '- 同一测区按原树组、5 米树轴邻近、20 厘米容差下至少 20% 点重合合并保守分组；每组最多保留两个视角。','- 训练与验证/测试树轴距离至少 10 米；不能排除所有远端共享树冠，分组不是实测单木身份真值。','- 初轮训练 54 份/39 组，验证 7 份/5 组，测试 7 份/5 组，空间缓冲排除 2 份、重复视角排除 24 份、最新单木几何核验未确认排除 1 份。','- 5 类有验证/测试：大叶合欢、石栗、榕树、蒲葵、狐尾椰子。凤凰木、银合欢、南洋杉仅训练；不得报告为八类整体准确率。南洋杉沿用原合并分类标签，未证实精确到单一物种。','- 数据属于已阅人工便利样本，测试称为回归集，不是全新无偏验证。后两轮已知第一阶段回归结果，均明确记录，不重新宣称盲测。','', '## 同一回归集的全部结果','', '| 实验 | 训练正确/总数 | 验证正确/7 | 测试正确/7 | 测试树组正确/5 |','|---|---:|---:|---:|---:|']
    for r in runs:text.append(f"| {r['name']} | {r['training']['correct_views']}/{r['training']['n_views']} | {r['validation']['correct_views']} | {r['test']['correct_views']} | {r['test']['correct_groups']} |")
    text+=['','验证集并列按预先规则选较早简单模型；未按测试结果反向挑选更好看的模型。旋转增强和新增 LAS 本轮没有可靠提升证据。历史旧点云模型的 4/10 使用另一份划分，不能直接与本轮 3/7 作提升/下降判断。','', '## 当前错误与下一步','', '基线在石栗 2/2、狐尾椰子 1/1 上正确；大叶合欢 0/2、榕树 0/1、蒲葵 0/1。这些分母极小，不代表物种真实性能排序。验收页保留所有错误，不只展示成功例。','', '主要待解决项：扩大独立单木数量而不是同树视角数量；补齐榕树冠干一致的采集；检查大叶合欢与石栗的形态相似和不同路线域差异；为 60 份未关联 LAS 找到采集任务/位姿/影像对应。新 PCD/BAG 场景级数据本轮没有全部重分割，不把 72 份 LAS 的完成等同于所有原始采集全部完成。','', '## 可回退性与交付','',f"6 个训练前备份逐个验证通过：{backup_files} 项物理副本，共 {backup_bytes/1e9:.2f} GB。备份清单包含原绝对路径、副本路径、大小、SHA-256。初始化权重、相关部署权重、标签/划分、配置和原评估快照均保留。",'', '- runs/*/pre_training_backup/backup_manifest.json：逐次备份。','- runs/*/model.joblib 或 04_rotation_mlp/best.pt：实际拟合模型。','- manifest.json / agent_label_review.json：数据划分与新标签证据。','- data/、new_las/：清洗结果、2048 点文件及三视图。','- new_las_evidence/：12 份新旧关联全景投影。','- final_summary.json / all_predictions.json：完整指标与每个样本预测。','', '本轮没有使用 Google 街景；已确认的两份新增标签用本地全景即可建立对应。原始数据、人工审查数据库以及部署权重保持不变。']
    (OUT/'REPORT.md').write_text('\n'.join(text)+'\n',encoding='utf-8');print([(r['id'],r['validation']['correct_views'],r['test']['correct_views']) for r in runs]);print('selected',summary['selected_run'])
if __name__=='__main__':main()
