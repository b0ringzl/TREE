"""Verify real HK artifacts, evaluate demo behavior, package quality-first holdout."""
import sys,json,zipfile,hashlib
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from build_hk_lidar_pairs import read,write,TARGETS
WORK=Path(__file__).resolve().parents[2];sys.path.insert(0,str(WORK/'tree_species_demo'))
from engine import Engine,image_input
from hk_three_species import NAMES
DATA=WORK/'hk-tree-vmms/derived/hk_dualsource_three_species_20260907'

def main():
    records=read(DATA/'manifest.json');summary=read(DATA/'summary.json');experiment=DATA/'experiment';split=read(experiment/'split.json');report=read(experiment/'report.json');lookup={r['id']:r for r in records};valid=[r for r in records if r['status']=='geometry_candidate_pending_review']
    checks=[]
    for r in valid:
        with np.load(r['point'],allow_pickle=False) as f:
            xyz=f['points_xyz'];assert np.isfinite(xyz).all() and xyz.shape[1]==3
            assert len(xyz)==len(np.unique(xyz,axis=0))==r['quality']['sample_unique_points']
            assert np.linalg.norm(xyz,axis=1).max()<1.00001
        checks.append(r['id'])
    train=[lookup[r['id']] for r in split if r['split']=='train'];test=[lookup[r['id']] for r in split if r['split']=='heldout'];distances=[]
    for r in train:
        other=[t for t in test if t['survey']==r['survey']]
        if other:distances.append(float(np.linalg.norm(np.array([t['quality']['axis_local_xy'] for t in other])-r['quality']['axis_local_xy'],axis=1).min()))
    assert min(distances)>=25
    # Real inference, including input-quality and unknown gates, on every held-out pair.
    engine=Engine();predictions=[];samples=[];seen=set()
    ranked=[]
    for r in test:
        _,q=image_input(Path(r['image']).read_bytes());quality=(not q['severe'],r['quality']['trunk_bin_occupancy'],min(q['sharpness'],1000),r['quality']['sample_unique_points'])
        ranked.append((quality,r,q))
    for quality,r,q in sorted(ranked,key=lambda v:(v[0],v[1]['id']),reverse=True):
        group=next(x['tree_group'] for x in split if x['id']==r['id'])
        if group in seen or sum(s['truth']==NAMES[TARGETS.index(r['species'])] for s in samples)>=2:continue
        seen.add(group);samples.append({'id':'hk3_'+r['id'],'pair_id':r['id'],'domain':'hongkong_vmms','truth':NAMES[TARGETS.index(r['species'])],'image':f'hk-pair-file/{r["id"]}/image.jpg','point':f'hk-pair-file/{r["id"]}/point.npz','split':'experimental_group_holdout_weak_labels','purpose':'待验收配对示例，不是独立人工真值','quality_caution':q['severe'],'selection':'input quality and unique group, never prediction correctness'})
    write(experiment/'demo_samples.json',samples)
    for i,r in enumerate(test):
        truth=NAMES[TARGETS.index(r['species'])];ib=Path(r['image']).read_bytes();pb=Path(r['point']).read_bytes()
        output=engine.predict('hongkong_vmms',ib,pb,'point.npz',True,False)
        predictions.append({'id':r['id'],'truth':truth,'decision':output['decision'],'results':output['results'],'quality':output['quality'],'warnings':output['warnings']})
        if i%10==0:print('end-to-end holdout',i+1,'/',len(test),flush=True)
    known=[p for p in predictions if p['decision']['label']!='Unknown'];correct=sum(p['truth']==p['decision']['label'] for p in known)
    write(experiment/'demo_holdout_predictions.json',predictions)
    behavior={'pairs':len(predictions),'known_outputs':len(known),'unknown_outputs':len(predictions)-len(known),'correct_known_against_transfer_labels':correct,'known_coverage':len(known)/len(predictions),'known_agreement':correct/len(known) if known else None,'scope':'Includes input-quality and unknown gates; weak-label agreement only, not human-accepted accuracy'}
    write(experiment/'demo_behavior.json',behavior)
    write(DATA/'verification.json',{'finite_unique_unit_sphere_samples':len(checks),'min_train_holdout_stem_distance_m':min(distances),'no_train_holdout_group_overlap':not(set(r['tree_group'] for r in split if r['split']=='train')&set(r['tree_group'] for r in split if r['split']=='heldout')),'scope':'format, geometry and split checks; does not certify species or complete 3D alignment'})
    lines=['# 香港三类双源：数据、训练和验收交付','','## 数据处理结果','',f"人工标注跨帧实例 {summary['input_annotations']}；保留 {len(valid)} 对待验收候选，初步合并 {summary['candidate_tree_groups']} 个单木组。拒绝结果保留原因，不修改源全景标注或原始VMMS。",'','|类别|跨帧候选|','|---|---:|']
    for sp,n in summary['candidate_species'].items():lines.append(f'|{sp}|{n}|')
    lines+=['','采用人工全景多边形和位姿发现近景非平面树冠，再定位原始密集PCD或对应帧彩色点云。按树干轴和树冠约束提取单木，去平面背景、统计孤立点，检查树干高度分箱覆盖；10厘米体素去重，分树干/树冠采样最多2048个真实独立点。训练只使用达到2048点且无未解决同组标签冲突的候选。','', '何文田采用对应帧附近彩色点云；尖沙咀和司徒拔道采用原始密集地图。记录含原始文件路径、标签来源、投影状态及拒绝原因。微小叶尖、混合树冠与精细外参仍需验收。','', '## 迁移训练（隔离实验）','',f"选择 {report['selected_pairs']} 对；训练 {report['train_pairs']} 对/{report['train_groups']} 个保守分组；留出 {report['heldout_pairs']} 对/{report['heldout_groups']} 组；空间缓冲排除 {report['buffer_excluded']} 对。最小训练—留出树干距离 {min(distances):.2f} 米，同组不跨集合。",'','YOLO图像编码器、WHU PTv2点云编码器冻结；训练三个正则化分类头：图像1280维、点云384维、各自L2归一化后拼接的融合特征。不是端到端微调；武汉和通用香港权重未覆盖。','','|实验输出|转移标签留出一致率（拒识前）|','|---|---:|']
    for mode,title in [('image','图像'),('point','点云'),('fusion','双源融合')]:lines.append(f"|{title}|{report['metrics'][mode]['accuracy']:.2%}|")
    palm_support=report['metrics']['point']['per_class']['Livistona chinensis']['support'];palm_recall=report['metrics']['point']['per_class']['Livistona chinensis']['recall']
    lines+=['', f"**上述数字不是人工验收后的识别精度。** 点云标签来自图像转移，可能包含错配；留出集中蒲葵只有{palm_support:.0f}例，点云单源召回仅{palm_recall:.1%}，不能证明其泛化已达标。不得把少数测试结果当作香港全域精度。",'',f"结构抽查后，额外将{summary.get('quality_audit_flagged',0)}个混入背景或棕榈树干覆盖不足的实例标为待返工，排除训练及推荐测试；旧实验保存在 experiment/before_structure_audit。清洗改变了分组划分，前后数字不能当作同一测试集上的精度提升。",'',f"加入实际输入质检和未知拒识后：{behavior['known_outputs']}/{behavior['pairs']} 对返回已知类别，{behavior['unknown_outputs']} 对返回未知；已知输出中 {behavior['correct_known_against_transfer_labels']} 对与转移标签一致。这个口径不同于上面的闭集一致率。",'','## 打开和验收','','识别：http://127.0.0.1:8037/ ，选择“香港街景 · 榕树 / 蒲葵 / 狐尾椰子 · 双源实验”。支持图像、点云单源与确认同木后的融合；质量不足会停用融合或返回未知。','','验收：http://127.0.0.1:8037/hk-pairs 。默认每个单木组显示一个较好视角；可关闭去重查看全部跨帧候选。接受、退回、修改标签都保存在 pair_reviews.json，不改原全景记录。训练命令默认只用接受项；本轮明确以实验弱监督模式训练，不自动将候选标为接受。','',f"香港三类测试包提供 {len(samples)} 个按输入质量挑选的留出单木，未按预测对错挑选。其参考标签仍为待验收转移标签；不要与旧武汉测试包混淆。",'', '清洗后全量点云 cleaned_local.npz 保留米制局部坐标；point.npz 为单位球归一化模型输入，含原中心和scale。禁止将单位球坐标直接解释为地理米制坐标。','','## 后续优先事项','','优先验收蒲葵独立单木及疑似混冠样本，补充独立地段的人工确认配对集；再进行端到端微调和正式精度评估。当前unknown是启发式拒识，不是训练好的第四类。']
    (DATA/'香港三类双源交付说明.md').write_text('\n'.join(lines),encoding='utf-8')
    with zipfile.ZipFile(DATA/'香港三类_双源测试包.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for name in ('香港三类双源交付说明.md','summary.json','verification.json','experiment/report.json','experiment/demo_behavior.json','experiment/demo_samples.json'):
            archive.write(DATA/name,name)
        for sample in samples:
            folder=DATA/'instances'/sample['pair_id']
            for name in ('image.jpg','point.npz','cleaned_local.npz','projection.jpg','point_views.jpg','record.json'):archive.write(folder/name,sample['pair_id']+'/'+name)
    print(json.dumps({'verified_candidates':len(checks),'behavior':behavior,'test_samples':len(samples)},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
