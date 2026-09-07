"""Opt-in five-class experimental inference; existing model domains stay intact."""
import hashlib,time
from pathlib import Path
import numpy as np

DATA=Path(__file__).resolve().parent.parent/'hk-tree-vmms/derived/reviewed_five_species_pilot_v1_20260907'
MODEL=DATA/'model.joblib'
DOMAIN='hongkong_vmms_five'
NAMES=['Delonix regia','Aleurites moluccana','Leucaena leucocephala','Albizia lebbeck','Araucaria columnaris(Araucaria heterophylla)']
CHINESE=['凤凰木','石栗','银合欢','大叶合欢','南洋杉（原训练合并类）']
UNVALIDATED={'Delonix regia','Leucaena leucocephala'}
SOURCES={'image':'香港扩展五类 · YOLO 图像实验头','point':'香港扩展五类 · WHU PTv2 点云实验头','fusion':'香港扩展五类 · 特征拼接融合实验头'}

def decision(bundle,mode,feature,severe=False):
    from hk_three_species import normalize
    head=bundle['heads'][mode];p=head.predict_proba(normalize(feature[None]))[0];order=np.argsort(-p)
    best=int(head.classes_[order[0]]);score=float(p[order[0]]);margin=score-float(p[order[1]])
    reasons=[]
    if severe:reasons.append('输入质量严重不足')
    if score<.55:reasons.append('最高候选分数低于演示阈值 0.55')
    if margin<.15:reasons.append('前两名差值低于演示阈值 0.15')
    return {'label':'Unknown' if reasons else NAMES[best],'display_name':'未知 / 需复核' if reasons else CHINESE[best],
        'candidate_label':NAMES[best],'score':round(score,5),'margin':round(margin,5),'source':SOURCES[mode],
        'score_kind':'实验分类头分数，未经概率校准，不代表正确率','reasons':reasons,
        'reliability':'no_independent_class_evaluation' if NAMES[best] in UNVALIDATED else 'small_holdout_only',
        'top3':[{'species':NAMES[int(head.classes_[i])],'name':CHINESE[int(head.classes_[i])],
                 'score':round(float(p[i]),5),'in_catalog':True} for i in order[:3]]}

def choose(outputs,quality,paired):
    """Image-primary demo policy, not a validated quality-gating classifier."""
    available=[m for m in ('image','point') if m in outputs and not quality[m]['severe']]
    mode=available[0] if available else next(m for m in ('image','point') if m in outputs)
    chosen={**outputs[mode],'reasons':list(outputs[mode]['reasons'])};warnings=[]
    if len(available)==2:
        if not paired:warnings.append('未确认同一棵树：分别识别，不计算融合，不把两源分歧当作同木证据。')
        elif outputs['image']['candidate_label']!=outputs['point']['candidate_label']:
            chosen.update(label='Unknown',display_name='双源分歧 / 需复核')
            chosen['reasons'].append('图像与点云首选树种不一致，请核对配对关系、树冠范围和相似树种')
        else:warnings.append('两源首选一致，但一致性不是正确性的证明。')
    if 'fusion' in outputs:warnings.append('融合结果单独展示；本轮未证明优于图像，不自动用融合覆盖主结果。')
    return chosen,mode,warnings

def predict(engine,image,point,point_name,paired):
    import joblib,torch
    from engine import LOCK,image_input,point_input
    from hk_three_species import Features,IMAGE,POINT,normalize
    if image is None and point is None:raise ValueError('请至少提供图像或点云。')
    if not MODEL.is_file():raise ValueError('香港扩展五类实验权重缺失。')
    started=time.perf_counter();quality={};outputs={};cloud=[];evidence=[];im=None;xyz=None
    if image is not None:im,quality['image']=image_input(image)
    if point is not None:
        xyz,quality['point'],cloud=point_input(point,point_name,target_points=2048,preserve_sparse=True)
        quality['point']['inference_points']=len(xyz)
    warnings=['香港扩展五类为实验模式：凤凰木、银合欢没有独立类别评估，点云识别可靠性未验证。',
              '仅适用于已裁剪的单木图像 / 已提取的单木点云；不是全景多目标自动检测。']
    with LOCK,torch.inference_mode():
        stamp=MODEL.stat().st_mtime_ns
        if getattr(engine,'hk_five_stamp',None)!=stamp:
            bundle=joblib.load(MODEL)
            if bundle['classes']!=NAMES:raise ValueError('五类模型类别映射校验失败')
            for key,path in [('image_encoder',IMAGE),('point_encoder',POINT)]:
                if Path(bundle[key]).resolve()!=path.resolve():raise ValueError('五类编码器路径不一致')
                if hashlib.sha256(path.read_bytes()).hexdigest()!=bundle['encoder_hashes'][str(path)]:raise ValueError('五类编码器版本不一致')
            for head in bundle['heads'].values():
                if list(head.classes_)!=list(range(5)):raise ValueError('五类分类头顺序不一致')
            engine.hk_five_bundle=bundle;engine.hk_five_stamp=stamp
            engine.models[DOMAIN]=(None,NAMES,str(MODEL))
        if not hasattr(engine,'hk_five_features'):engine.hk_five_features=Features(engine.device)
        if im is not None:
            iv=normalize(engine.hk_five_features.encode_image(im));outputs['image']=decision(engine.hk_five_bundle,'image',iv,quality['image']['severe'])
        if xyz is not None:
            pv=normalize(engine.hk_five_features.encode_point(xyz));outputs['point']=decision(engine.hk_five_bundle,'point',pv,quality['point']['severe'])
            evidence.append(f"输入 {quality['point']['unique_count']} 个独立点，编码器实际使用 {len(xyz)} 点；不复制凑满 2048 点。")
        if im is not None and xyz is not None:
            if paired and not any(q['severe'] for q in quality.values()):outputs['fusion']=decision(engine.hk_five_bundle,'fusion',normalize(np.r_[iv,pv]))
            elif paired:warnings.append('至少一源质量严重不足，停用融合并优先采用可用单源。')
    chosen,mode,extra=choose(outputs,quality,paired);warnings+=extra
    if im is not None and xyz is not None and not paired and not any('未确认同一棵树' in w for w in warnings):warnings.append('未确认同一棵树，不执行融合。')
    if chosen['candidate_label'] in UNVALIDATED:warnings.append('本次首选属于未独立评估类别，仅供候选参考，请人工复核。')
    for q in quality.values():warnings.extend(q['warnings'])
    evidence+=['实际调用已有 YOLO / WHU PTv2 冻结特征和本轮已训练分类头；不使用文件名、参考标签或验收答案推理。',
        '图像侧依据学习到的外观特征，点云侧依据学习到的几何特征；这里未生成未经验证的叶形、花果等植物学解释。',
        '主结果采用图像优先、严重低质时单源退回的演示规则；同木双源首选冲突则要求复核。这一规则和拒识阈值尚未独立验证。',
        '点云归一化形态不能给出可靠整树高；缺失的树干或根部不会自动补造。']
    return {'domain':DOMAIN,'decision':chosen,'decision_source_mode':mode,'results':outputs,'quality':quality,'warnings':warnings,
        'evidence':evidence,'occlusion_grid':[],'point_preview':cloud,'elapsed_seconds':round(time.perf_counter()-started,2),
        'device':engine.device,'experimental':True,'model_paths':{'heads':str(MODEL),'image_encoder':str(IMAGE),'point_encoder':str(POINT)},
        'limitations':['五类独立实验入口，未合并成香港八类统一分类器。','只有三类、10 个小留出视角可评估；不代表五类整体或现场精度。',
                       '点云/图像质量为启发式检查；候选分数未校准，范围外物种仍可能误判。']}
