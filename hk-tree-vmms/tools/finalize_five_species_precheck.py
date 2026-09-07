"""Record assistant visual observations; never write human review decisions."""
from collections import Counter
from build_hk_five_species_pairs import OUT,NAMES
from build_hk_lidar_pairs import read,write

def main():
    records=read(OUT/'manifest.json');shortlist=read(OUT/'automated_shortlist.json')
    inspected={r['id']:r for r in records if shortlist.get(r['id'],{}).get('disposition')=='shortlist_not_visually_checked'}
    priority={
        'hewentian_pano__003181__078c5b92b6':'投影基本落在目标冠干内，三视图可见连续的冠下主干；可优先核对。树干仍偏稀，根颈未验证；图像模型与石栗原标签冲突，树种必须由用户确认。',
        'hewentian_pano__003325__2ca9e5339a':'目标冠干轮廓较清楚，未见明显离散邻树冠；可优先核对。下干偏稀且附近有加油站设施，须检查根部和地物；模型与石栗原标签冲突，不能当作物种真值。'}
    notes={
        'hewentian_pano__003656__412bf2513d':'冠部投影向右上溢出，冠下点串断续，存在把支撑设施当成树干的风险。',
        'hewentian_pano__004287__9515e6f1a6':'树冠与下部点串之间明显断开，缺连续树干。',
        'hewentian_pano__005664__4ad9df9fb0':'点云集中在标签左缘及另一条竖向结构，未覆盖目标冠干，疑似错配。',
        'jianshazui_pano_2__007376__1b38fea760':'三视图显示两个分离锥形树冠，疑似邻树混入，需重新做单木分离。',
        'jianshazui_pano_2__007343__12e4d73b0b':'两团分离冠部且下干不完整，不能作为完整单木。',
        'stubbs_pano_0__000433__86377718ac':'投影只覆盖远处高处局部结构，下部冠干覆盖不足且图像模型标签冲突。',
        'hewentian_pano__000672__6c5bc86e54':'冠部覆盖较好，但原图清晰的下干仅有少量点，暂不满足完整树干要求。',
        'hewentian_pano__002727__0680c7e047':'三视图出现水平直线形结构伸出树冠，疑似地物混入。',
        'hewentian_pano__002408__0a7318acce':'下干点稀疏、断续，需补查原始观测；模型与凤凰木标签冲突。',
        'hewentian_pano__002500__2b516fc06b':'只提取到标注边缘的小片离散点，目标整体未覆盖。'}
    qa={}
    for ident,r in inspected.items():
        qa[ident]={'disposition':'review_candidate' if ident in priority else 'needs_rework',
            'note':priority.get(ident,notes.get(ident,'缩略投影与三视图预检可见下干缺失/断续、局部提取或非单木结构风险；暂不放入默认验收队列，需逐例原图返工。')),
            'evidence':'assistant inspected projection.jpg and point_views.jpg, individually or in precheck contact sheets',
            'species_verified':False,'total_height_verified':False,'human_accepted':False}
    write(OUT/'agent_visual_qa.json',qa)
    status=read(OUT/'delivery_status.json');status.update(visual_qa_inspected=len(qa),
        priority_review_candidates=len(priority),visual_rework=len(qa)-len(priority),
        remaining_geometry_candidates_not_visually_checked=status['geometry_candidates']-len(qa),
        visual_qa_complete=False,all_five_species_ready=False)
    write(OUT/'delivery_status.json',status)
    counts=Counter(r['species'] for r in records);valid=Counter(r['species'] for r in records if r['status']=='geometry_candidate_pending_review')
    lines=['# 五类扩展双源数据：第一轮处理交付','','已重跑人工标注定位、点云源定位、单木几何清洗、轻量采样、既有图像模型推理与预检流程。不是五类已全部验收或已完成训练。','',
           '|类别|有效标签实例|自动几何候选|','|---|---:|---:|']
    lines += [f'|{NAMES[n]}|{counts[n]}|{valid[n]}|' for n in NAMES]
    lines += ['',f'共 {len(records)} 个有效标签实例，81 个几何候选（48 个算法单木组，不是人工核实的独立树数）。预检了 34 个约 15 米间距筛选的候选，其中 2 个进入优先验收、32 个需返工；其他 47 个几何候选尚未完成视觉预检。',
        '', '## 验收', '', '打开 http://127.0.0.1:8037/hk-expansion 。默认只显示 2 个优先候选（均为石栗原标签，种类存在模型冲突，尚未确认）。可以旋转点云、查看原图投影和三视图、修改类别或选未知、接受/退回并进入下一棵。勾选全部可查看 81 个自动候选，不代表它们均可用。',
        '', '## 来源与限制', '',
        '- 何文田：对应帧号最近的三份原始着色点云，6 cm 实测点保留；仍须检查时间/投影一致性。',
        '- 尖沙咀与司徒拔：以粗地图选择对应原始地图，再完整流式读取 4 个原始 PCD（共约 4 亿点），只保存目标邻域 10 cm 实测点。完整 PCD 点数已由读取器检查。',
        '- 所有 81 份轻量点云均不超过 2,048 个独立点，已核对归一化点可逆恢复到清洗实测点，误差小于 0.1 mm。不是合成补全。',
        '- 参考 WHU 项目的可逆单位球归一化，保留中心和尺度；本轮使用分区覆盖采样，未照搬其不足点时重复采样的行为。',
        '- 旧数字帧键从 stream_id + frame_id 恢复，重复项由新审核记录优先覆盖；未修改源标签。',
        '- 南洋杉仅沿用原训练合并类，保留原始物种名；银合欢没有当前图像模型输出类，不提供虚构的该类置信度。',
        '- 何文田 V/C 原始稠密地图的首块诊断未证明与当前帧点云可靠一致，未将它们直接拼入单木，不能把首块低重合率当作整张地图错误。',
        '- 既有 3 个审核文件内容哈希保持不变；新增 human reviews 未自动写入，正式模型未更新。',
        '', '科研灵感：本轮预检提示，树干分箱有点不等于完整单木，后续可检验图像投影一致性与冠干连通性联合筛选是否能降低伪完整样本的比例。']
    write(OUT/'precheck_summary.json',status)
    (OUT/'五类双源数据处理说明.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(status)

if __name__=='__main__':main()
