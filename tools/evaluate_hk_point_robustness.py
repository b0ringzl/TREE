"""Predeclared image-degradation ablation; existing frozen models only, no fitting."""
import hashlib,io,json,sys,time
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import torch
import joblib
import cv2
from PIL import Image,ImageEnhance,ImageFilter,ImageOps
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tree_species_demo'))
from training_safety import sha256
from hk_three_species import Features,normalize,IMAGE,POINT
from evaluate_whu_current_20260911 import metrics
BASE=ROOT/'hk-tree-vmms/derived/reviewed_five_species_pilot_v1_20260907'
OUT=ROOT/'outputs/hk_point_robustness_20260911_v1'
CONDITIONS=[dict(id='original',name='Original / 原圖'),
 dict(id='dark_mild',name='Dark ×0.35 / 亮度 ×0.35',brightness=.35),
 dict(id='dark_severe',name='Dark ×0.12 / 亮度 ×0.12',brightness=.12),
 dict(id='blur_mild',name='Blur σ1.5 / 模糊 σ1.5',blur=1.5),
 dict(id='blur_severe',name='Blur σ4 / 模糊 σ4',blur=4.),
 dict(id='lowres_mild',name='64 px long edge / 長邊 64 像素',edge=64),
 dict(id='lowres_severe',name='24 px long edge / 長邊 24 像素',edge=24),
 dict(id='combined',name='32 px + dark ×0.2 + blur σ2 / 綜合退化',edge=32,brightness=.2,blur=2.)]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def transform(im,c):
    im=im.copy()
    if 'edge' in c:im.thumbnail((c['edge'],c['edge']),Image.Resampling.LANCZOS)
    if 'blur' in c:im=im.filter(ImageFilter.GaussianBlur(c['blur']))
    if 'brightness' in c:im=ImageEnhance.Brightness(im).enhance(c['brightness'])
    return im
def quality(im):
    gray=np.asarray(im.convert('L').resize((320,224)),dtype=np.uint8)
    brightness=float(gray.mean());sharpness=float(cv2.Laplacian(gray,cv2.CV_64F).var());small=min(im.size)<96
    bad=brightness<40 or sharpness<18 or small
    return dict(brightness=brightness,laplacian_variance=sharpness,min_dimension=min(im.size),
        point_weight=.75 if bad else 0.,low_quality=bad)
def metric(y,p,names):
    d=metrics(y,[names[i] for i in p.argmax(1)],names)
    present=sorted(set(y));d['macro_f1_supported_classes']=float(np.mean([d['per_class'][i]['f1'] for i in present]));return d
def main():
    OUT.mkdir(exist_ok=False);(OUT/'images').mkdir()
    allrows=read(BASE/'split.json');rows=[r for r in allrows if r['split']=='heldout'];assert len(rows)==10
    catalog=read(ROOT/'outputs/hk_point_assist_20260911_v1/catalog.json');byid={r['id']:r for r in catalog}
    for r in rows:
        c=byid[r['id']];assert c['current_accepted']
        for kind in ['point','image']:assert sha256(Path(r[kind]))==c['sha256'][kind]==r[kind+'_sha256']
    for p,h in read(BASE/'protected_sources.json').items():assert sha256(Path(p))==h,p
    teacher_protocol=read(ROOT/'outputs/distillation/bioclip_probe_20260908_0227/protocol.json')
    source_files=[BASE/'model.joblib',BASE/'split.json',BASE/'accepted_manifest.json',BASE/'encoder_hashes.json',
        Path(teacher_protocol['teacher_weight']),IMAGE,POINT,Path(__file__)]
    hashes={str(p):sha256(p) for p in source_files}
    bundle=joblib.load(BASE/'model.joblib');names=bundle['classes'];assert len(names)==5
    for p,h in bundle['encoder_hashes'].items():assert sha256(Path(p))==h
    write(OUT/'protocol.json',dict(locked_at=datetime.now(timezone.utc).isoformat(),conditions=CONDITIONS,
        n=10,groups=len({r['leakage_group'] for r in rows}),class_names=names,
        scope='Existing accepted 10-view, 7-block historical regression; GT only 3 of the 5 output classes; not fresh test or natural degradation evidence.',
        input_policy='Same point cloud and tree per condition. Degraded images saved losslessly PNG; no generated content or enhancement.',
        gate='Fixed before predictions: mean grayscale <40 OR Laplacian variance on 320x224 <18 OR min native dimension <96 -> point probability weight .75, otherwise 0. Point QA failure -> 0.',
        score_policy='Arithmetic probability blend, uncalibrated. All branches choose among the same fixed 5 classes. BioCLIP uses a photo of {class} and fixed existing temperature, not the full production 148-way decision.',
        no_fitting=True,no_deployment=True,source_hashes=hashes,
        negative_control='Point probabilities cyclically shifted by one heldout row, fixed independent of labels/scores.',
        reporting='All predeclared conditions and all cases; no selection by test performance; paired differences clustered by 7 leakage blocks.'))
    write(OUT/'test_manifest.json',rows)
    encoder=Features('cuda');points=[];point_qa=[]
    for r in rows:
        fingerprint=hashlib.sha256((r['image_sha256']+r['point_sha256']+''.join(bundle['encoder_hashes'].values())).encode()).hexdigest()[:16]
        with np.load(BASE/'features'/(r['id']+'_'+fingerprint+'.npz'),allow_pickle=False) as f:points.append(f['point'])
        with np.load(r['point'],allow_pickle=False) as f:xyz=f['points_xyz']
        eig=np.linalg.eigvalsh(np.cov(xyz.T));point_qa.append(len(np.unique(xyz,axis=0))>=256 and eig[0]/max(eig.sum(),1e-12)>=.01)
    xp=normalize(np.asarray(points));pp=bundle['heads']['point'].predict_proba(xp);assert list(bundle['heads']['point'].classes_)==list(range(5))
    y=[r['class_index'] for r in rows];assert abs(metric(y,pp,names)['accuracy']-.4)<1e-12
    # Check a fixed cached point embedding against actual encoder inference.
    with np.load(rows[0]['point'],allow_pickle=False) as f:actual=encoder.encode_point(f['points_xyz'])
    np.testing.assert_allclose(actual,points[0],rtol=1e-4,atol=1e-5)
    from tree_species_teacher_demo.engine import Engine,decode
    import open_clip
    engine=Engine();engine.load()
    with torch.inference_mode():bank=engine.model.encode_text(open_clip.get_tokenizer('ViT-H-14')(['a photo of '+n for n in names]).cuda(),normalize=True).float()
    all_predictions=[];reports={};arrays={};started=time.time()
    for condition in CONDITIONS:
        yp=[];bp=[];fp=[];weights=[];qualities=[]
        folder=OUT/'images'/condition['id'];folder.mkdir()
        for i,r in enumerate(rows):
            with Image.open(r['image']) as source:im=transform(source.convert('RGB'),condition)
            im.save(folder/(r['id']+'.png'));q=quality(im);weights.append(q['point_weight'] if point_qa[i] else 0);qualities.append(q)
            xi=normalize(encoder.encode_image(im));yp.append(bundle['heads']['image'].predict_proba(xi[None])[0])
            fp.append(bundle['heads']['fusion'].predict_proba(normalize(np.r_[xi,xp[i]])[None])[0])
            edge=min(1024,max(im.size));padded=ImageOps.pad(im,(edge,edge),method=Image.Resampling.BICUBIC,color=(128,128,128));buf=io.BytesIO();padded.save(buf,format='JPEG',quality=95)
            with torch.inference_mode():
                pixels=decode(buf.getvalue())[0];x=engine.model.encode_image(((pixels[None].cuda()-engine.mean)/engine.std).to(engine.dtype),normalize=True).float()
                bp.append((x@bank.T*engine.model.logit_scale.exp().float()/engine.choice['calibration_temperature']).softmax(-1)[0].cpu().numpy())
        yp=np.array(yp);bp=np.array(bp);w=np.array(weights)[:,None]
        probs=dict(yolo_image=yp,bioclip_image=bp,point=pp,old_fusion=np.array(fp),
            yolo_assisted=(1-w)*yp+w*pp,bioclip_assisted=(1-w)*bp+w*pp,
            bioclip_wrong_pair=(1-w)*bp+w*np.roll(pp,1,axis=0))
        reports[condition['id']]={k:metric(y,p,names) for k,p in probs.items()}
        reports[condition['id']]['gate_active_views']=int((w>0).sum())
        for k,p in probs.items():arrays[condition['id']+'__'+k]=p
        for i,r in enumerate(rows):
            all_predictions.append(dict(id=r['id'],condition=condition['id'],truth=names[y[i]],group=r['leakage_group'],
                image='images/'+condition['id']+'/'+r['id']+'.png',catalog_key=byid[r['id']]['key'],
                point_sha256=r['point_sha256'],quality=qualities[i],point_quality_pass=bool(point_qa[i]),point_weight=float(w[i,0]),
                outputs={k:dict(prediction=names[p[i].argmax()],correct=bool(p[i].argmax()==y[i]),scores=p[i].tolist()) for k,p in probs.items()}))
        print(condition['id'],{k:round(v['accuracy'],3) for k,v in reports[condition['id']].items() if isinstance(v,dict)},'seconds',round(time.time()-started,1),flush=True)
    for p,h in hashes.items():assert sha256(Path(p))==h
    np.savez_compressed(OUT/'probabilities.npz',**arrays);write(OUT/'predictions.json',all_predictions)
    write(OUT/'results.json',dict(class_names=names,conditions=CONDITIONS,metrics=reports,original_weights_unchanged=True,
        point_feature_reload_verified=True,training=False,production_changed=False))
    report(names,rows,reports,arrays,y)

def report(names,rows,reports,arrays,y):
    rng=np.random.default_rng(20260911);groups=sorted({r['leakage_group'] for r in rows});membership=np.array([groups.index(r['leakage_group']) for r in rows]);truth=np.array(y)
    intervals={};lines=['# 图像质量退化时的点云辅助：受控回归实验','',
        '本轮未训练、未换权重。五类输出空间，10 个已验收配对视角、7 个保守空间组；留出真值仅石栗、大叶合欢、南洋杉三个标签。旧测试已被查看，不是新盲测。',
        'BioCLIP 为固定五类提示词对照，不冒称主 Demo 的 148 类开放候选结果。YOLO 使用原五类训练头，点云使用原五类 PTv2 头。',
        '', '| 影像条件 | YOLO图像 | YOLO辅助 | BioCLIP图像 | BioCLIP辅助 | 点云单源 | 错配点云辅助 |', '|---|---:|---:|---:|---:|---:|---:|']
    for c in CONDITIONS:
        key=c['id'];m=reports[key];lines.append('| '+c['name']+' | '+' | '.join(f"{100*m[k]['accuracy']:.0f}%" for k in ['yolo_image','yolo_assisted','bioclip_image','bioclip_assisted','point','bioclip_wrong_pair'])+' |')
        a=arrays[key+'__bioclip_image'].argmax(1)==truth;b=arrays[key+'__bioclip_assisted'].argmax(1)==truth;delta=b.astype(int)-a.astype(int);draws=[]
        for _ in range(2000):
            selected=rng.integers(0,len(groups),len(groups));idx=np.concatenate([np.where(membership==g)[0] for g in selected]);draws.append(delta[idx].mean()*100)
        intervals[key]=dict(delta_pp=float(delta.mean()*100),ci95_pp=np.quantile(draws,[.025,.975]).tolist(),
            corrected=int((~a&b).sum()),regressed=int((a&~b).sum()))
    write(OUT/'paired_uncertainty.json',dict(unit='7 conservative spatial blocks',draws=2000,seed=20260911,conditions=intervals))
    lines+=['','## 如何解释','',
        '- 点云在所有退化条件下完全相同，保留哈希；图像采用固定变暗、模糊、降采样及组合退化，全部条件和错例均报告。',
        '- 门控只看输入图像的亮度、锐度和原生尺寸，不读取正确标签、退化名称或原图作判断；质量低时以固定 0.75 权重加入点云概率。点云基础几何检查不合格时禁用辅助。',
        '- 概率线性加权尚未校准；不能解释为真实正确率。类别 mIoU/F1、逐类混淆矩阵在 results.json；不涉及分割精度。',
        '- 只有在具体条件下辅助高于图像单源，才能说本小样本受控实验支持该条件下的补偿作用；若轻度退化/原图变差必须同步说明。',
        '- 错配点云是负对照：如果正确配对没有优于错配，需要进一步检查类别先验、样本量与点云分支质量，不宜归因于同株结构。',
        '- 80 个退化视角不是 80 棵独立树，不能累加后声称样本变大；7 组 bootstrap 区间见 paired_uncertainty.json。天然低质影像和新的独立树木仍需验证。',
        '- 界面为独立实验展示，主 Demo 原模型与默认图像分数未改。',
        '', '## 可追溯产物','',
        'protocol.json 在推理前锁定全部条件、门控和权重哈希；predictions.json 保留每例全部分支概率；images/ 保留实际退化输入；没有删除不利条件。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

if __name__=='__main__':main()
