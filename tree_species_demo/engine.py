"""Real local inference. Upload labels and reference truth are never model inputs."""
import io
import json
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from scipy.spatial import cKDTree
from ultralytics import YOLO
from ultralytics.data.augment import classify_transforms

ROOT=Path(__file__).resolve().parent
WORK=ROOT.parent
sys.path.insert(0,str(WORK/'Lidar/tscmdl_whu_sandbox/src'))
from tscmdl_whu import PointTransformerV2Classifier
from tscmdl_whu.fusion import TSCMDLFusionModel

HK=WORK/'full_image_species_baseline/experiments/full_web_yolo11s_cls_20260906_seed1'
WHU=WORK/'experiments/whu18_yolo11_ptv2_fusion_20260826'
LOCK=threading.Lock()

def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))

def image_input(data):
    with Image.open(io.BytesIO(data)) as source:
        if source.width*source.height>40_000_000: raise ValueError('图像过大，请缩小到4000万像素以下。')
        im=ImageOps.exif_transpose(source).convert('RGB')
    gray=np.asarray(im.convert('L').resize((320,224)))
    brightness=float(gray.mean()); sharpness=float(cv2.Laplacian(gray,cv2.CV_64F).var())
    dark=float((gray<20).mean()); bright=float((gray>245).mean())
    warnings=[]; severe=False
    if min(im.size)<160: warnings.append('分辨率不足：请补充至少224像素宽高、主体清晰的照片。'); severe=True
    if brightness<40 or dark>0.65: warnings.append('图像过暗：增加曝光，避开强逆光并补拍树冠。'); severe=severe or brightness<18 or dark>0.85
    if brightness>225 or bright>0.65: warnings.append('高光丢失：降低曝光，保留叶片与枝干纹理。'); severe=severe or brightness>245
    if sharpness<18: warnings.append('图像疑似模糊或纹理不足：对焦叶片/树冠，稳住相机后补拍。'); severe=severe or sharpness<3
    if im.width/im.height>2.7: warnings.append('宽幅图像可能包含多棵树：请裁剪目标单木后识别。')
    return im,{'width':im.width,'height':im.height,'brightness':round(brightness,2),'sharpness':round(sharpness,2),'dark_fraction':round(dark,3),'overexposed_fraction':round(bright,3),'severe':severe,'warnings':warnings,'scope':'曝光与纹理启发式检查；无法证明物种可辨或目标为树'}

def point_input(data,name,target_points=8192,preserve_sparse=False):
    suffix=Path(name).suffix.lower(); archive_normalized=False
    if suffix=='.npy': xyz=np.load(io.BytesIO(data),allow_pickle=False)
    elif suffix=='.npz':
        with np.load(io.BytesIO(data),allow_pickle=False) as f:
            key=next((k for k in ('points_xyz','points','xyz') if k in f),None)
            if key is None: raise ValueError('NPZ需要points_xyz、points或xyz数组。')
            xyz=f[key].copy()
            archive_normalized=key=='points_xyz' and 'centroid_xyz' in f and 'scale' in f
    elif suffix in ('.xyz','.txt','.csv'):
        xyz=np.loadtxt(io.BytesIO(data),delimiter=',' if suffix=='.csv' else None)
    elif suffix in ('.las','.laz'):
        import laspy
        with laspy.open(io.BytesIO(data)) as f:
            if f.header.point_count>2_000_000: raise ValueError('点云超过200万点，请先提取一棵树。')
            las=f.read(); xyz=np.column_stack((las.x,las.y,las.z))
    else: raise ValueError('支持点云格式：NPZ、NPY、XYZ、TXT、CSV、LAS、LAZ。')
    if xyz.ndim!=2 or xyz.shape[1]<3: raise ValueError('点云需要N×3（XYZ）数组。')
    if len(xyz)>2_000_000: raise ValueError('点云超过200万点，请先提取一棵树。')
    xyz=np.asarray(xyz[:,:3],dtype=np.float64)
    if len(xyz)<32: raise ValueError('有效点数少于32，无法识别。请上传完整单木点云。')
    if not np.isfinite(xyz).all(): raise ValueError('点云包含NaN或无穷值，请清理后上传。')
    count=len(xyz); unique=len(np.unique(xyz,axis=0)); warnings=[]
    span=np.ptp(xyz,axis=0); radius=float(np.linalg.norm(xyz-xyz.mean(0),axis=1).max())
    if radius<1e-9: raise ValueError('点云坐标全部重合，无法识别。')
    severe=unique<256
    eigen=np.linalg.eigvalsh(np.cov(xyz.T));eigen_fraction=float(eigen[0]/max(eigen.sum(),1e-12))
    if target_points==2048 and eigen_fraction<.01:
        warnings.append('点云接近平面或细线，可能混入墙面或缺少树冠；请重新提取完整单木。');severe=True
    if unique<min(target_points,2048): warnings.append(f'独立点数偏少：补扫树冠和树干，建议至少{target_points}个有效点。')
    if unique/count<0.6: warnings.append('重复点比例偏高：去除重复采样并补充新的扫描视角。')
    if span[2]/max(span.max(),1e-12)<0.15: warnings.append('点云高度跨度很小：检查Z轴是否朝上及树冠是否缺失。'); severe=True
    if preserve_sparse:
        # Five-class pilot was trained on <=2048 distinct real points, without padding.
        _,indices=np.unique(xyz,axis=0,return_index=True)
        pool=xyz[np.sort(indices)]
        sampled=pool[np.random.default_rng(20260907).choice(len(pool),target_points,replace=False)] if len(pool)>target_points else pool.copy()
    else:
        sampled=xyz[np.random.default_rng(20260907).choice(count,target_points,replace=count<target_points)] if count!=target_points else xyz.copy()
    # Official assets were already centered/scaled before deterministic sampling.
    is_normalized=archive_normalized and float(np.linalg.norm(xyz,axis=1).max())<1.01
    if not is_normalized:
        center=xyz.mean(0); sampled=(sampled-center)/radius
    preview=xyz[np.linspace(0,count-1,min(count,2500),dtype=int)]
    preview=(preview-xyz.mean(0))/radius
    return sampled.astype(np.float32),{'point_count':count,'unique_count':unique,'extent_xyz_input_units':[round(float(x),5) for x in span],'height_to_width_ratio':round(float(span[2]/max(span[0],span[1],1e-9)),3),'normalized_asset':is_normalized,'severe':severe,'warnings':warnings,'scope':'几何启发式检查；无法自动确认单木完整性、树种或图像配对关系'},preview.round(5).tolist()

class Engine:
    def __init__(self):
        self.catalog=read(ROOT/'catalog.json'); self.models={}
        self.device='cuda' if torch.cuda.is_available() else 'cpu'
        self.transform=classify_transforms(size=224)
        self.names={domain:{x['scientific_name']:x for x in self.catalog[domain]} for domain in ['hongkong','wuhan']}

    def load(self,domain,point=False,fusion=False):
        if domain not in self.models:
            summary=read(HK/'datasets/full/dataset_summary.json') if domain=='hongkong' else read(WHU/'yolo_dataset/dataset_summary.json')
            path=HK/'pretrain/weights/selected_by_val_group_macro_f1.pt' if domain=='hongkong' else WHU/'yolo11s_cls/run/weights/selected_by_val_macro_f1.pt'
            model=YOLO(str(path)).model.to(self.device).eval()
            expected=[summary['class_folders'][str(i)] for i in range(len(summary['class_names']))]
            if [model.names[i] for i in range(len(model.names))]!=expected: raise RuntimeError('模型类别顺序校验失败')
            self.models[domain]=(model,summary['class_names'],str(path))
        if point and 'point' not in self.models:
            path=WHU/'ptv2/run/best.pt'; ck=torch.load(path,map_location='cpu',weights_only=False)
            model=PointTransformerV2Classifier(len(ck['class_names']))
            model.load_state_dict(ck['model_state'],strict=True)
            if list(ck['class_names'])!=self.models['wuhan'][1]: raise RuntimeError('图像点云类别映射不一致')
            self.models['point']=(model.eval().to(self.device),list(ck['class_names']),str(path))
        if fusion and 'fusion' not in self.models:
            path=WHU/'fusion/run/best.pt'; ck=torch.load(path,map_location='cpu',weights_only=False); args=ck['args']
            model=TSCMDLFusionModel(torch.nn.Identity(),torch.nn.Identity(),18,point_dim=384,image_dim=1280,modal_dim=1024,normalization=args['normalization'],classifier_hidden_dims=tuple(args['classifier_hidden_dims']))
            model.load_state_dict(ck['model_state'],strict=True)
            if list(ck['class_names'])!=self.models['wuhan'][1]: raise RuntimeError('融合类别映射不一致')
            self.models['fusion']=(model.eval().to(self.device),list(ck['class_names']),str(path))

    def image_forward(self,domain,images):
        model,names,path=self.models[domain]
        captured=[]
        handle=model.model[-1].drop.register_forward_hook(lambda _m,_i,o:captured.append(o))
        try:
            values=model(torch.stack([self.transform(x) for x in images]).to(self.device))
            probs=values[0] if isinstance(values,tuple) else values
        finally: handle.remove()
        return probs,captured[0]

    def result(self,domain,probs,source,quality):
        values=probs.detach().float().cpu().numpy().reshape(-1); names=self.models['hongkong' if domain=='hongkong' else 'wuhan'][1]
        combined={}
        for name,p in zip(names,values):
            if name in ('Ficus microcarpa','Ficus benjamina'): name='榕树'
            combined[name]=combined.get(name,0)+float(p)
        ordered=sorted(combined.items(),key=lambda x:-x[1]); best,score=ordered[0]; margin=score-ordered[1][1]
        reasons=[]
        if best not in self.names[domain]: reasons.append('最高分候选不在本demo所选树种范围')
        if score<0.55: reasons.append('最高模型分数低于0.55')
        if margin<0.15: reasons.append('前两名分数差小于0.15')
        if quality.get('severe'): reasons.append('输入质量严重不足')
        return {'label':'Unknown' if reasons else best,'display_name':'未知 / 需复核' if reasons else self.names[domain][best]['chinese_name'],'score':round(score,5),'score_kind':'模型softmax分数，未经概率校准','margin':round(margin,5),'source':source,'reasons':reasons,'top3':[{'species':n,'name':self.names[domain].get(n,{}).get('chinese_name',n),'score':round(p,5),'in_catalog':n in self.names[domain]} for n,p in ordered[:3]]}

    def predict(self,domain,image=None,point=None,point_name='',paired=False,explain=True):
        if domain=='hongkong_vmms_five':
            from hk_five_species import predict
            return predict(self,image,point,point_name,paired)
        if domain=='hongkong_vmms':return self.predict_hk_vmms(image,point,point_name,paired)
        if domain not in ('hongkong','wuhan'): raise ValueError('请选择香港或武汉模型范围。')
        if image is None and point is None: raise ValueError('请至少提供图像或点云。')
        started=time.perf_counter(); qualities={}; warnings=[]; outputs={}; evidence=[]; cloud=[]; heat=[]
        im=None; xyz=None
        if image is not None: im,qualities['image']=image_input(image)
        if point is not None: xyz,qualities['point'],cloud=point_input(point,point_name)
        if im is not None and xyz is not None and not paired:
            warnings.append('未确认图像和点云属于同一棵树，本次分别识别，不执行融合。')
        with LOCK,torch.inference_mode():
            if domain=='wuhan' or im is not None: self.load(domain,point=xyz is not None and domain=='wuhan',fusion=xyz is not None and im is not None and paired and domain=='wuhan')
            if im is not None:
                ip,ifeat=self.image_forward(domain,[im])
                outputs['image']=self.result(domain,ip[0],'YOLO11s-cls 图像模型',qualities['image'])
                evidence.append('图像前两名差值为 '+str(outputs['image']['margin'])+'；候选分数保留原模型完整类别归一化结果。')
                if explain and not qualities['image']['severe']:
                    idx=int(ip[0].argmax()); view=self.transform(im).cpu()
                    model=self.models[domain][0]; occluded=[]
                    for row in range(3):
                        for col in range(3):
                            x=view.clone(); x[:,row*224//3:(row+1)*224//3,col*224//3:(col+1)*224//3]=view.mean((1,2),keepdim=True); occluded.append(x)
                    pred=model(torch.stack(occluded).to(self.device)); pred=pred[0] if isinstance(pred,tuple) else pred
                    heat=(ip[0,idx]-pred[:,idx]).cpu().float().numpy().reshape(3,3).round(5).tolist()
                    evidence.append('3×3遮挡敏感图显示：遮住该区域后，最高候选分数下降多少。它反映模型依赖区域，不是植物学鉴定证明。')
            if xyz is not None and domain=='wuhan':
                model=self.models['point'][0]; features=[]
                handle=model.classifier.register_forward_pre_hook(lambda _m,inputs:features.append(inputs[0]))
                reference=cKDTree(xyz).query(xyz,k=8)[1]
                try: logits=model(torch.from_numpy(xyz[None]).to(self.device),torch.from_numpy(reference[None]).to(self.device))
                finally: handle.remove()
                pfeat=features[0]
                outputs['point']=self.result(domain,logits.softmax(-1)[0],'PTv2 点云模型',qualities['point'])
                evidence.append(f"点云含 {qualities['point']['unique_count']} 个独立点；高宽比 {qualities['point']['height_to_width_ratio']}，仅作几何检查，不可独立证明树种。")
                if im is not None and paired:
                    fquality={'severe':qualities['point']['severe'] or qualities['image']['severe']}
                    if fquality['severe']:
                        warnings.append('至少一种输入质量严重不足，停用双源融合，采用质量可用的单源结果。')
                    else:
                        fp=self.models['fusion'][0].forward_from_features(pfeat,ifeat).softmax(-1)[0]
                        outputs['fusion']=self.result(domain,fp,'已训练 TSCMDL 特征融合（YOLO + PTv2）',fquality)
                        evidence.append('融合使用同木图像1280维特征和点云384维特征，经训练好的投影与分类器输出；没有对两个模型分数简单平均。')
            elif xyz is not None:
                outputs['point']={'label':'Unknown','display_name':'未知 / 香港点云类别未训练','source':'无覆盖这19类的香港点云权重','score':None,'reasons':['点云模型仅在所选武汉树种范围验证'],'top3':[]}
                warnings.append('香港点云可预览和质检；当前树种结果由图像模型提供。')
        if 'fusion' in outputs: decision=outputs['fusion']
        elif 'image' in outputs and not qualities['image']['severe']: decision=outputs['image']
        elif 'point' in outputs: decision=outputs['point']
        else: decision=outputs['image']
        if 'image' in outputs and 'point' in outputs and domain=='wuhan':
            agree=outputs['image']['top3'][0]['species']==outputs['point']['top3'][0]['species']
            evidence.append('图像与点云最高候选'+('一致。' if agree else '存在分歧，建议复核配对关系、树冠完整性和相似树种。'))
        for q in qualities.values(): warnings.extend(q['warnings'])
        return {'domain':domain,'decision':decision,'results':outputs,'quality':qualities,'warnings':warnings,'evidence':evidence,'occlusion_grid':heat,'point_preview':cloud,'elapsed_seconds':round(time.perf_counter()-started,2),'device':self.device,'model_paths':{k:v[2] for k,v in self.models.items() if k==domain or (domain=='wuhan' and k in ('point','fusion'))},'limitations':['这是单木分类demo；请先裁剪目标树/提取单木点云。','模型分数不是已校准正确率；未知阈值是启发式，类别范围外仍可能误判。','香港验证来自网络图像，不代表香港街景精度。']}

    def predict_hk_vmms(self,image,point,point_name,paired):
        import joblib
        from hk_three_species import Features,MODEL,IMAGE,POINT,normalize,decision
        if image is None and point is None:raise ValueError('请至少提供图像或点云。')
        started=time.perf_counter();quality={};outputs={};cloud=[];evidence=[]
        im=None;xyz=None
        if image is not None:im,quality['image']=image_input(image)
        if point is not None:xyz,quality['point'],cloud=point_input(point,point_name,target_points=2048)
        warnings=['香港街景三类为实验模块：使用人工图像标签转移的待验收点云训练，不等同于人工验收精度。']
        if not MODEL.exists():raise ValueError('香港三类双源模型尚在准备，暂无可调用权重；请先查看点云验收页面。')
        with LOCK,torch.inference_mode():
            stamp=MODEL.stat().st_mtime_ns
            if getattr(self,'hk_bundle_stamp',None)!=stamp:
                self.hk_bundle=joblib.load(MODEL);self.hk_bundle_stamp=stamp
            if not hasattr(self,'hk_features'):self.hk_features=Features(self.device)
            if im is not None:
                iv=normalize(self.hk_features.encode_image(im));outputs['image']=decision(self.hk_bundle,'image',iv,quality['image']['severe'])
            if xyz is not None:
                pv=normalize(self.hk_features.encode_point(xyz));outputs['point']=decision(self.hk_bundle,'point',pv,quality['point']['severe'])
                evidence.append(f"输入含{quality['point']['unique_count']}个独立点，PTv2使用2048点；重复采样不会增加几何信息。")
            if im is not None and xyz is not None:
                if not paired:warnings.append('未确认同一棵树，本次不执行融合。')
                elif quality['image']['severe'] or quality['point']['severe']:warnings.append('至少一种输入质量不足，停用融合。')
                else:outputs['fusion']=decision(self.hk_bundle,'fusion',normalize(np.r_[iv,pv]))
        chosen=outputs.get('fusion') or (outputs.get('image') if not quality.get('image',{}).get('severe') else None) or outputs.get('point') or outputs['image']
        for q in quality.values():warnings.extend(q['warnings'])
        evidence+=['图像使用已有YOLO特征，点云使用武汉训练的PTv2特征；香港三类分类头和拼接特征融合头单独训练。','这是冻结编码器的迁移基线，并非已完成端到端微调。未知通过分数、候选差值及特征距离拒识，阈值尚未独立校准。']
        return {'domain':'hongkong_vmms','decision':chosen,'results':outputs,'quality':quality,'warnings':warnings,'evidence':evidence,'occlusion_grid':[],'point_preview':cloud,'elapsed_seconds':round(time.perf_counter()-started,2),'device':self.device,'experimental':True,'model_paths':{'heads':str(MODEL),'image_encoder':str(IMAGE),'point_encoder':str(POINT)},'limitations':['仅覆盖榕树、蒲葵、狐尾椰子及启发式未知。','标签转移、单木分割和开放集拒识仍需独立人工验收。']}
