"""Verify rollback artifacts and real-input inference, then report all outcomes."""
import json,sys
from pathlib import Path
import numpy as np,joblib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
from training_safety import verify_backup,sha256
from predict_aquilaria_point_candidate import predict
OUT=ROOT/'outputs/aquilaria_ptv2_fewshot_20260913_44056021'
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))

def main():
 results=read(OUT/'results.json');base=read(OUT/'baseline_evaluation.json');rows=read(OUT/'manifest.json');summary=read(OUT/'summary.json');bundle=joblib.load(summary['model']);f=np.load(OUT/'features.npz');x=f['features'];x=x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-8)*10
 checks=[]
 for run in results:
  info=verify_backup(run['backup']);checks.append(dict(run=run['run'],backup_verified=True,source_files_unchanged=True,files=len(info['files'])))
 smoke=[]
 for r in rows:
  if r['split']!='fewshot_positive':continue
  p=predict(summary['model'],r['derived']);i=f['keys'].tolist().index(r['key']);expected=bundle['head'].predict_proba(x[[i]])[0];actual=np.array([next(v['score'] for v in p['candidates'] if v['species']==c) for c in bundle['classes']]);error=float(np.max(abs(expected-actual)))
  assert error<1e-4,(r['key'],error)
  assert sha256(r['source'])==r['source_sha256']
  smoke.append(dict(key=r['key'],probability_max_abs_error=error,scope='Training-positive replay / NOT held-out evaluation',**p))
 (OUT/'verification.json').write_text(json.dumps(dict(backups=checks,real_input_smoke=smoke),ensure_ascii=False,indent=2),encoding='utf8')
 names={'Albizia lebbeck':'大葉合歡','Aleurites moluccana':'石栗','Aquilaria sinensis':'土沉香','Bauhinia x blakeana':'洋紫荊','Ficus microcarpa':'榕樹（細葉榕）','Livistona chinensis':'蒲葵','Wodyetia bifurcata':'狐尾椰子'}
 text=['# 三株土沉香：点云小样本训练报告','',
 '**已完成真实训练，得到七类点云候选模型。三折留一株验证正确 2/3；不能据此宣称已获得稳定或跨场景的土沉香识别精度。Demo 未切换。**','',
 '## 数据与方法','',
 '- PTv2 现有领域骨干冻结，仅训练七类 LogisticRegression 分类头；不是从三株数据训练整个网络。本轮没有运行 PTv3。',
 '- 共 49 株：46 株历史数据＋3 株本轮明确确认土沉香。旧类训练 35 株、验证 6 株、回归 5 株，保持已有 fold_1 划分。每折加 2 株土沉香训练，剩下 1 株验证。',
 '- A/C 修正单独保存：旧 A 对应原圈选图右株，旧 B 中株，旧 C 左株；不改写历史导出编号。',
 '- 清洗仅包括三株共享原始索引、两侧距离小于 5 mm 的跨株点、非有限点及重复坐标；不是完美单木语义清洗，底部支撑物／植被可能残留。',
 '- 每株最多 2048 个实测点，单位球归一化，0/90/180/270 度水平旋转取平均特征。旋转不当作独立树，不输入图片、经纬度或绝对高度。',
 '- C=1、类别平衡、seed=20260913，在评分前固定，没有根据留出结果反复调参或挑最好一折。','',
 '| 历史编号 | 清洗前点数 | 清洗后点数 | 留出验证结果 | 土沉香分数 |','|---|---:|---:|---|---:|']
 for r,h in zip([r for r in rows if r['split']=='fewshot_positive'],summary['held_out_predictions']):text.append(f"| {r['key']} | {r['points_before']:,} | {r['points_after']:,} | {names[h['prediction']]} | {h['scores']['Aquilaria sinensis']:.1%} |")
 text+=['','分数未经概率校准。B 的首选为大叶合欢 28.5%，土沉香 24.3%，应视为拿不准，而不是可靠鉴定。','',
 '## 与旧模型比较','',
 '| 固定数据 | 旧六类头 | 新七类最终头 |','|---|---:|---:|','| 旧验证 6 株 | 5/6 | 5/6 |','| 旧回归 5 株 | 4/5 | 4/5 |','',
 '- 三折包含留出土沉香的验证正确数依次为 6/7、5/7、6/7；每折旧回归都是 4/5。重复使用的旧树不能累计成新的独立样本。',
 '- 旧分类头不包含土沉香，因此旧模型在该类的 0/3 是标签空间缺失；不能把新模型的 2/3 包装为公平的同任务 66.7 个百分点提升。',
 '- 最终三样本头使用全部 3 株土沉香训练，旧验证和回归不参与拟合。真实 NPZ 输入端已回放验证并与训练预处理结果核对一致；训练样本回放不计为泛化精度。','',
 '## 每类旧验证表现：precision / recall / F1（支持数）','',
 '| 树种 | 旧模型 | 最终候选 |','|---|---|---|']
 for c in summary['classes']:
  a=base['old_validation']['per_class'][c];b=results[-1]['old_validation']['per_class'][c]
  def fmt(v):return f"{v['precision']:.3f} / {v['recall']:.3f} / {v['f1-score']:.3f} ({int(v['support'])})"
  text.append(f'| {c} / {names[c]} | {fmt(a)} | {fmt(b)} |')
 text+=['','土沉香在旧验证中没有支持样本，表中零值不是该类留出表现。所有折的分类报告、混淆矩阵、逐株预测见 results.json。','',
 '## 局限与下一步','',
 '三株都来自同一次采集、同一小区域，树种与场景、密度、支撑架等因素存在混杂。B 株形态更难迁移；现有结果不足以判定错误的唯一原因。建议补充不同地点、不同树形的土沉香，并保留至少一组不参与调参的独立实测树。也需要彻底清理支撑架后复测，避免学习人工结构。当前不能证明加入图像＋点云融合必然改善，也没有新鲜跨地点测试。','',
 '## 产物与回退','',f"- 最终候选：[model.joblib]({summary['model'].replace(chr(92),'/')})",f"- 备份目录：`{OUT/'runs'}`；三个 fold 和 final_3shot 子目录各有独立 `pre_training_backup`。",'- 4 次拟合均在复制并验证权重、初始化头、数据清单、标签、划分、配置、seed、原评估之后进行。所有备份和源文件已再次验证。','- 原点云、人工圈选、旧模型权重和 Demo 未改；未自动部署。','- 独立推理入口：tools/predict_aquilaria_point_candidate.py；接受含 xyz/metric_xyz/metric 的 NPZ 或 Nx3 NPY，不是按文件名返回固定答案。']
 (OUT/'REPORT.md').write_text('\n'.join(text)+'\n',encoding='utf8');print('VERIFIED',OUT/'REPORT.md',flush=True)

if __name__=='__main__':main()
