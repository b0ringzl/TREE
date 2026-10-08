"""Verify every run backup and publish the unselected experiment results."""
import sys,json,subprocess,shutil
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/bauhinia_ptv3_fewshot_20260912_v1';sys.path.insert(0,str(ROOT/'tools'))
from training_safety import verify_backup,sha256
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def main():
 models={m:read(OUT/(m+'_results.json')) for m in ['ptv2','ptv3']};manifest=read(OUT/'manifest.json');config=read(OUT/'config.json');checks=[]
 for runs in models.values():
  for r in runs:
   b=verify_backup(r['backup']);checks.append(dict(run=r['run'],backup=r['backup'],verified_files=len(b['files']),verified_bytes=sum(x['bytes'] for x in b['files']),source_hashes_unchanged=True))
 for r in manifest:
  assert sha256(r['source'])==r['source_sha256'];assert sha256(OUT/r['derived'])==r['derived_sha256']
 assert read(OUT/'serialization_compat_check.json')['passed']
 save(OUT/'safety_verification.json',dict(runs=checks,source_clouds_verified=len(manifest),training_runs=len(checks),production_unchanged=True))
 env=OUT/'environment';env.mkdir(exist_ok=False);repo=ROOT/'models/vendor/sonata-official-fewshot-v1'
 (env/'sonata_upstream_commit.txt').write_text(subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True),encoding='utf8')
 (env/'sonata_compatibility.patch').write_bytes(subprocess.check_output(['git','-C',str(repo),'diff']))
 (env/'container_packages.txt').write_bytes(subprocess.check_output(['docker','exec','spacescan-ptv3-fewshot-v1','python3','-m','pip','freeze']))
 (env/'container_image.json').write_bytes(subprocess.check_output(['docker','inspect','--format','{{json .Image}}','spacescan-ptv3-fewshot-v1']))
 for name in ['prepare_bauhinia_ptv3_fewshot.py','extract_bauhinia_point_features.py','train_bauhinia_point_fewshot.py','check_sonata_serialization_compat.py','training_safety.py']:shutil.copy2(ROOT/'tools'/name,env/name)
 summary={};positive={}
 for m,runs in models.items():
  folds=runs[:3];pp=[p for f in folds for p in f['validation']['predictions'] if p['truth']==config['positive_species']];positive[m]=pp
  summary[m]=dict(heldout_bauhinia_correct=sum(p['correct'] for p in pp),heldout_bauhinia_n=3,fold_validation_accuracy=[f['validation']['accuracy'] for f in folds],mean_validation_accuracy=float(np.mean([f['validation']['accuracy'] for f in folds])),mean_validation_macro_f1=float(np.mean([f['validation']['macro_f1_supported'] for f in folds])),negative_test_regression_accuracy=[f['test_regression']['accuracy'] for f in folds])
 final=models['ptv3'][-1];save(OUT/'comparison.json',dict(status='completed_not_deployed',summary=summary,heldout_positive_predictions=positive,final_three_shot_model=str(Path(final['run'])/'model.joblib'),final_validation=final['validation'],final_negative_test_regression=final['test_regression'],decision='Do not replace deployed model: PTv3 validation regressed relative to matched PTv2 baseline',backbone_training='frozen official Sonata PTv3; classification head fitted only',unique_validation_note='Five legacy validation trees are reused in each fold. Eighteen fold predictions are not eighteen independent trees.',fresh_positive_test_available=False))
 names={'Albizia lebbeck':'大葉合歡','Aleurites moluccana':'石栗','Bauhinia x blakeana':'洋紫荊','Ficus microcarpa':'榕樹（細葉榕）','Livistona chinensis':'蒲葵','Wodyetia bifurcata':'狐尾椰子'}
 lines=['# 三株洋紫荆：PTv3 小样本点云实验','', '**结论：实验已实际完成，但本轮 PTv3 不优于相同数据上的 PTv2，不建议替换 Demo。**','', '## 方法与样本','', '- 用户本轮明确确认 01_TreeID_13、14、15 为洋紫荆；原始标签库未改写，确认记录单独保存。','- 使用官方 Sonata 预训练 Point Transformer V3（108.46M 参数），冻结骨干，仅拟合六类线性分类头。不是三株点云从零训练 PTv3，也不是全网络微调。','- 3 株洋紫荆＋43 个已有接受树组，共 46 个组。对照类别：石栗、榕树、蒲葵、狐尾椰子、大叶合欢。每个旧组只保留一个视图。','- 每折：33 个其他树种训练组＋2 株洋紫荆训练；留出的 1 株洋紫荆＋固定 5 个旧验证组验证；固定 5 个旧回归测试组只用于报告。','- 三株两两没有 5 mm 内相同原始点；脚本同时检查与旧样本的空间缓冲和点重合。仍不能消除同排、同次采集的场景偏差。','- 每株最多 2048 个实测点，归一化、PCA 法线、四个固定水平旋转；旋转特征取均值，不计为四株。RGB 置零，不使用图像特征、经纬度或绝对位置。','- PTv2 对照使用现有 WHU 预训练骨干＋同一六类分类头、同一数据和 C=1。原部署分类头没有洋紫荆，不能直接作同标签空间对照。','', '## 三折留一株结果','', '| 留出洋紫荆 | PTv2 预测 | PTv2 洋紫荆分数 | PTv3 预测 | PTv3 洋紫荆分数 |','|---|---|---:|---|---:|']
 for a,b in zip(positive['ptv2'],positive['ptv3']):lines.append(f"| {a['id']} | {names[a['prediction']]} | {a['scores'][config['positive_species']]:.1%} | {names[b['prediction']]} | {b['scores'][config['positive_species']]:.1%} |")
 lines+=['','以上分数未经校准，不是植物鉴定正确概率。','', '| 指标 | PTv2 对照 | PTv3 候选 |','|---|---:|---:|',f"| 留出洋紫荆识别正确数 | {summary['ptv2']['heldout_bauhinia_correct']}/3 | {summary['ptv3']['heldout_bauhinia_correct']}/3 |",f"| 三折平均验证准确率（每折 6 个组） | {summary['ptv2']['mean_validation_accuracy']:.1%} | {summary['ptv3']['mean_validation_accuracy']:.1%} |",f"| 三折平均验证宏 F1 | {summary['ptv2']['mean_validation_macro_f1']:.3f} | {summary['ptv3']['mean_validation_macro_f1']:.3f} |",'| 固定旧回归集（每折相同 5 个组，无洋紫荆） | 每折 4/5 | 每折 4/5 |','','**不可把三折 18 次验证预测说成 18 株独立树，也不可把 3/3 或 2/3 外推为实际街景识别精度。** 旧回归集此前已查看，不是新鲜盲测；3 株洋紫荆来自同一排，没有第四株独立正例测试。','', '## 各类别验证表现（三折召回率均值）','', '| 类别 | PTv2 | PTv3 |','|---|---:|---:|']
 for c in config['classes']:
  a=np.mean([r['validation']['per_class'][c]['recall'] for r in models['ptv2'][:3]]);b=np.mean([r['validation']['per_class'][c]['recall'] for r in models['ptv3'][:3]]);lines.append(f'| {names[c]} / {c} | {a:.1%} | {b:.1%} |')
 lines+=['', '完整 precision、recall、F1、混淆矩阵和每株预测见 ptv2_results.json、ptv3_results.json，不只报告洋紫荆正例。','', '## 保存的三样本模型','', f"最终分类头：`{Path(final['run'])/'model.joblib'}`",f"骨干：`{ROOT/'models/sonata_official_v1/sonata.pth'}`",'- 最终头使用全部 3 株洋紫荆和 33 个旧训练组拟合；旧验证和回归集未纳入拟合。',f"- 最终头的旧验证结果为 {final['validation']['correct']}/{final['validation']['n']}，旧回归测试结果为 {final['test_regression']['correct']}/{final['test_regression']['n']}。没有独立洋紫荆正例测试，不能把最终头对三株训练样本的预测当作泛化证据。",'- C、预处理、旋转次数、骨干汇聚方式均在评估前固定；没有挑最佳折，也没有按测试结果切换 Demo。','', '## 为什么没有改善','', '本轮只能确认性能差异，不能确认唯一原因。候选原因包括：三株同排数据不足；Sonata 预训练任务与树种分类不同；本实验无颜色、只有估计法线；点云残缺和邻冠混入；冻结特征＋线性头的适配能力有限。复杂骨干并不自动胜过已有领域适配骨干。','', '建议先增加不同路段、不同冠形的洋紫荆独立整株，并补充石栗等难负例；预留未看过的跨场景正例。届时再试最后一层或少量 block 微调，而不是对本次三株反复调参。','', '## 实际运行环境与兼容性','', '- 原 Windows Demo 环境未修改。新增独立 Docker 容器 spacescan-ptv3-fewshot-v1，基于本机已有 GPU 镜像；PTv3 特征在 Linux CUDA 上提取，分类头在 Windows sklearn 1.7.2 拟合。','- 使用的容器 PyTorch 1.9/CUDA 11.1 低于官方推荐环境；spconv 对 4090 架构发出了预编译架构警告，但本轮前向计算成功并输出有限特征。完整警告保存在 ptv3_feature_runtime.log，不能称为官方标准环境复现。','- 兼容修补仅涉及 Hilbert 整数位打包、dtype 重解释及 torch.concat→torch.cat；编码在 5 个深度、10,240 点上与未修改上游逐位一致。没有改变网络层、官方权重或稀疏卷积实现。','- environment/ 保存依赖列表、上游 commit 和兼容补丁。初始 WSL 环境下载方案已停止，没有切换生产 Python。','', '## 回退与数据安全','', f'- 共 {len(checks)} 次拟合，每次都有独立 pre_training_backup，包含物理复制权重、SHA-256、大小、来源、初始化分类头、数据清单、标签映射、划分、配置和原评估。所有备份及源文件已再次通过校验。',f'- 备份根目录：`{OUT/"runs"}`（各运行子目录内的 pre_training_backup）。','- 原始点云、已有人工验收库、部署权重均保持不变，未发布或部署实验模型。','', '## 官方资料','', '- PTv3：https://github.com/Pointcept/PointTransformerV3','- Sonata 预训练 PTv3：https://github.com/facebookresearch/sonata','- 权重及版本散列：models/sonata_official_v1/provenance.json','']
 (OUT/'REPORT.md').write_text('\n'.join(lines),encoding='utf8');print(json.dumps(summary,indent=2));print('ALL BACKUPS VERIFIED',len(checks),flush=True)
if __name__=='__main__':main()
