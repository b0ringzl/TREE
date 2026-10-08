"""Ablate measured points within the SAME trained fusion head. No fitting."""
import json,hashlib
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import joblib
from PIL import Image
from evaluate_hk_point_robustness import ROOT,BASE,CONDITIONS,Features,normalize,transform,metric,read,write,sha256
SOURCE=ROOT/'outputs/hk_point_robustness_20260911_v1'
OUT=ROOT/'outputs/hk_point_counterfactual_20260911_v1'

def main():
    OUT.mkdir(exist_ok=False)
    protocol=read(SOURCE/'protocol.json')
    for p,h in protocol['source_hashes'].items():assert sha256(Path(p))==h
    bundle=joblib.load(BASE/'model.joblib');names=bundle['classes'];split=read(BASE/'split.json');rows=read(SOURCE/'test_manifest.json')
    def point_feature(r):
        key=hashlib.sha256((r['image_sha256']+r['point_sha256']+''.join(bundle['encoder_hashes'].values())).encode()).hexdigest()[:16]
        with np.load(BASE/'features'/(r['id']+'_'+key+'.npz'),allow_pickle=False) as f:return f['point']
    xp=normalize(np.array([point_feature(r) for r in rows]));mean=normalize(np.mean(normalize(np.array([point_feature(r) for r in split if r['split']=='train'])),axis=0))
    write(OUT/'protocol.json',dict(locked_at=datetime.now(timezone.utc).isoformat(),training=False,source_protocol=str(SOURCE/'protocol.json'),
        source_protocol_sha256=sha256(SOURCE/'protocol.json'),method='Same frozen five-class fusion head, same image feature. Compare correct point, cyclically shifted point, train mean point, zero point.',
        conditions=CONDITIONS,selection='No model/condition/parameter selection; all conditions retained. Follow-up diagnostics on previously inspected regression, not a new test.',
        source_hashes={**protocol['source_hashes'],str(Path(__file__)):sha256(Path(__file__))}))
    encoder=Features('cuda');old=np.load(SOURCE/'probabilities.npz');y=[r['class_index'] for r in rows];pred=read(SOURCE/'predictions.json');result={};arrays={}
    for c in CONDITIONS:
        xi=[]
        for r in rows:
            with Image.open(SOURCE/'images'/c['id']/(r['id']+'.png')) as im:xi.append(encoder.encode_image(im.convert('RGB')))
        xi=normalize(np.array(xi));pp={}
        for k,points in [('correct',xp),('wrong_pair',np.roll(xp,1,axis=0)),('train_mean',np.tile(mean,(len(xi),1))),('zero',np.zeros_like(xp))]:
            pp[k]=bundle['heads']['fusion'].predict_proba(normalize(np.c_[xi,points]));arrays[c['id']+'__'+k]=pp[k]
        np.testing.assert_allclose(pp['correct'],old[c['id']+'__old_fusion'],atol=1e-6)
        result[c['id']]={k:metric(y,p,names) for k,p in pp.items()}
        for i,r in enumerate(rows):
            row=next(p for p in pred if p['id']==r['id'] and p['condition']==c['id'])
            row['counterfactuals']={k:dict(prediction=names[p[i].argmax()],correct=bool(p[i].argmax()==y[i]),scores=p[i].tolist()) for k,p in pp.items()}
        print(c['id'],{k:round(v['accuracy'],2) for k,v in result[c['id']].items()},flush=True)
    np.savez_compressed(OUT/'probabilities.npz',**arrays)
    write(OUT/'predictions.json',pred);write(OUT/'results.json',dict(classes=names,conditions=CONDITIONS,metrics=result,original_results=read(SOURCE/'results.json')['metrics']))
    groups=sorted({r['leakage_group'] for r in rows});g=np.array([groups.index(r['leakage_group']) for r in rows]);rng=np.random.default_rng(20260911);stats={}
    for c in CONDITIONS:
        key=c['id'];a=arrays[key+'__correct'].argmax(1)==y;stats[key]={}
        for k,p in [('image',old[key+'__yolo_image']),('wrong_pair',arrays[key+'__wrong_pair']),('train_mean',arrays[key+'__train_mean']),('zero',arrays[key+'__zero'])]:
            b=p.argmax(1)==y;delta=a.astype(int)-b.astype(int);values=[]
            for _ in range(2000):
                ids=np.concatenate([np.where(g==j)[0] for j in rng.integers(0,len(groups),len(groups))]);values.append(delta[ids].mean()*100)
            stats[key][k]=dict(delta_pp=float(delta.mean()*100),ci95_pp=np.quantile(values,[.025,.975]).tolist(),corrected=int((a&~b).sum()),regressed=int((~a&b).sum()))
    write(OUT/'paired_uncertainty.json',stats)
    for p,h in read(OUT/'protocol.json')['source_hashes'].items():assert sha256(Path(p))==h
    lines=['# 同一融合头的点云信息反事实对照','',
        '10 个已接受配对视角、7 个空间组；真值仅石栗/大叶合欢/南洋杉，五类输出；无新训练、无部署。全部原图/退化条件保留。',
        '', '| 图像条件 | YOLO图像 | 正确点云融合 | 错配点云融合 | 训练均值点云融合 | 移除点云融合 |', '|---|---:|---:|---:|---:|---:|']
    for c in CONDITIONS:
        k=c['id'];b=read(SOURCE/'results.json')['metrics'][k]['yolo_image']['accuracy'];m=result[k]
        lines.append('| '+c['name']+' | '+f'{100*b:.0f}%'+' | '+' | '.join(f"{100*m[a]['accuracy']:.0f}%" for a in ['correct','wrong_pair','train_mean','zero'])+' |')
    lines+=['','相较图像单源，配对纠正与退步数、按 7 个组的差值区间见 paired_uncertainty.json。样本极少，宽区间不得省略。',
        '均值点云取自既有训练集，未用测试标签；错配采用固定循环移位，不按结果挑选。零点云属于缺失输入反事实而非正常训练分布，因此需与错配、训练均值共同解读。',
        '若正确点云优于上述对照，说明在该受控条件和这批样本中，正确点云提供了补充信息；仍不等于自然低质图像、所有类别或完整路线上的确定性增益。',
        '原图反而下降的情况必须展示。固定概率门控未达到可靠上线条件，不以测试结果调一个更好看的门槛。当前阶段使用独立实验页展示，主模型未切换。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

if __name__=='__main__':main()
