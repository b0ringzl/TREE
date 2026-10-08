"""Summarize fixed regression honestly; no selection, training or deployment."""
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
from distillation_core import read_json,save_json

def intervals(rows):
    # Observation-group bootstrap, not a claim of verified tree/site independence.
    rng=np.random.default_rng(20260917);by=defaultdict(list)
    for r in rows:by[(r['truth'],r['group_id'])].append(r)
    result={}
    for truth,label in ((1,'recall'),(0,'fpr')):
        sums=[]
        for (y,_),rs in by.items():
            if y==truth:sums.append([len(rs),sum(r['baseline_positive'] for r in rs),sum(r['specialist_positive'] for r in rs)])
        data=np.asarray(sums,float);sample=data[rng.integers(0,len(data),(2000,len(data)))].sum(1)
        baseline=sample[:,1]/sample[:,0];candidate=sample[:,2]/sample[:,0]
        result[label]={'baseline_95':np.quantile(baseline,[.025,.975]).tolist(),'specialist_95':np.quantile(candidate,[.025,.975]).tolist(),'paired_gain_95':np.quantile(candidate-baseline,[.025,.975]).tolist(),'source_groups':len(data)}
    return result

def main(out):
    done=read_json(out/'COMPLETE.json');cfg=read_json(out/'config.json');summary=read_json(out/'data_summary.json');results=done['reports'];ci=intervals(read_json(out/'test_predictions.json'));save_json(out/'observation_bootstrap.json',ci)
    table=[]
    for phase in ('planning','test'):
        for k,label in [('baseline','现 Demo 专项分数阈值'),('specialist','二分类专用复核器'),('baseline_argmax','现 Demo 148 类首选（不同判定规则，仅参考）')]:
            r=results[phase][k];table.append(f"| {phase} | {label} | {r['tp']}/{r['positive']} ({r['recall']:.1%}) | {r['fp']}/{r['negative']} ({r['fpr']:.2%}) | {r['precision']:.1%} |")
    text=['# SST-Conserve v1：土沉香专用复核试验','',
        '已完成真实专项训练和固定旧数据回归。未接入生产 Demo，未使用任何本轮新下载且待人工确认的图片。网络新批次用于下一步来源/标签复核，不能将当前回归当作新地点泛化证据。','',
        '## 已执行的技术方案','',
        f"基线为当前部署 {cfg['baseline_version']}，不是旧弱基线。复用冻结的本地 BioCLIP 2.5 图像特征，在该模型的土沉香对数几率上学习独立线性残差头。该头仅回答土沉香/非土沉香复核，不把原有148类分数强行改写。",
        f"训练 {summary['train']['photos']} 张，其中土沉香 {summary['train']['positive_photos']} 张，反例为训练侧随机样本和训练侧高混淆样本；没有使用测试误例选反例。仅保留之前已确认或明确记录来源审核的数据，新增网络待审核图完全隔离。",
        '三个预先固定 L2 配方，各160轮；每个配方独立初始化/物理备份，按 planning 的1%误报预算选择轮次和阈值，先锁定唯一候选，再提取与评分 test。基线同样在 planning 选择1%预算阈值，避免以改变判定规则冒充模型优势。',
        '图像均按当前Demo的灰色等比例补边、JPEG95、224像素模型输入预处理。旧中心裁切缓存仅用于训练侧反例排序；正式比较使用同一图像、同一预处理。',
        '## 固定回归结果','',
        '| 数据 | 分支 | 土沉香检出 | 非土沉香误报 | 阳性精确率 |','|---|---|---:|---:|---:|',*table,'',
        f"是否达到预定回归门槛（检出率提高至少10个百分点且误报率不增加）：**{'是' if done['regression_target_met'] else '否'}**。即使通过，也不代表已经通过新来源/新地点验证。",
        f"按来源观察组重采样，检出率变化95%区间为 {ci['recall']['paired_gain_95'][0]:.1%} 至 {ci['recall']['paired_gain_95'][1]:.1%}；误报率变化区间为 {ci['fpr']['paired_gain_95'][0]:.2%} 至 {ci['fpr']['paired_gain_95'][1]:.2%}。该区间仍不解决旧数据拍摄者/地点重叠问题。",
        '所有旧测试和既有3张来源留出图均已在前轮查看，属于回归集。观察组不是已确认的独立实测树，网络照片也可能进入过通用编码器预训练。当前标签包含来源审核而非全部人工鉴定；与基线配对使用同一标签，不把标签清理算训练增益。','',
        '## 多视角、点云与Demo状态','',
        '已提供独立单图/多图实验推理工具 tools/predict_conservation_pilot.py。多图要求 --same-tree 明确确认同株，采用预先固定等权平均，没有以最大分数选最好视角。尚无可靠的新同株多视角盲测，不能宣称已验证融合提升，也未将单图阈值冒称多图校准。',
        '现有991三株点云均已用于已有点云模型训练，三株旋转不构成新树，留一株历史结果2/3正确，不能作新独立融合测试。公开数据线索须确认真实单木标签、影像配对和权限后才能补入；没有凭网络图像生成假点云。',
        '生产Demo保留当前SSTv1土沉香增强版及原点云融合；本轮仅生成新专用候选与独立复核工具。待人工来源/标签确认、新划分评估，以及用户看到比较后授权再切换。','',
        '## 下一步人工协作','',
        '在本机复核页8051优先查看来源标签为Aquilaria sinensis的候选，选择土沉香/非土沉香/无法判断/图片不可用；无法判断不进入监督训练。负例也保持原来源物种，用户“非土沉香”判断仅确认二分类，不自动改成另一具体物种。',
        '人工复核结束后冻结照片SHA与决策快照，依据拍摄者、观察、地点及近重复关系分组。存在旧训练作者重叠的图只用于训练或回归，不能充当新的作者独立测试。新测试先锁定不看分数，之后只评分一次。','',
        '## 产物、备份与回退','',f"选定候选：{done['selected']['path']}",
        *[f"- {r['backup']}" for r in read_json(out/'selection_lock.json')['candidates']],
        '每次拟合前复制并核对编码器、部署分类头、点云模型及编码器、初始化、清单、划分、种子、配置、基线验证结果；未使用硬链接。生产文件哈希在训练/评估后复查未改变。',
        '逐图结果：test_predictions.json；阈值与混淆计数：test_evaluation.json；新候选无自动部署、无原权重覆盖。']
    (out/'REPORT_ZH.md').write_text('\n'.join(text)+'\n',encoding='utf-8');print(out/'REPORT_ZH.md')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.out.resolve())
