"""Shared, frozen feature extraction for HK experimental paired training/inference."""
from pathlib import Path
import sys
import numpy as np
import torch
from scipy.spatial import cKDTree
from ultralytics import YOLO
from ultralytics.data.augment import classify_transforms

WORK=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(WORK/'Lidar/tscmdl_whu_sandbox/src'))
from tscmdl_whu import PointTransformerV2Classifier
IMAGE=WORK/'full_image_species_baseline/experiments/full_web_yolo11s_cls_20260906_seed1/pretrain/weights/selected_by_val_group_macro_f1.pt'
POINT=WORK/'experiments/whu18_yolo11_ptv2_fusion_20260826/ptv2/run/best.pt'
MODEL=WORK/'hk-tree-vmms/derived/hk_dualsource_three_species_20260907/experiment/model.joblib'
NAMES=['榕树','Livistona chinensis','Wodyetia bifurcata']
CHINESE=['榕树','蒲葵','狐尾椰子']

def normalize(v):
    return v/np.maximum(np.linalg.norm(v,axis=-1,keepdims=True),1e-9)

class Features:
    def __init__(self,device):
        self.device=device;self.image=None;self.point=None;self.transform=classify_transforms(size=224)

    @torch.inference_mode()
    def encode_image(self,im):
        if self.image is None:self.image=YOLO(str(IMAGE)).model.to(self.device).eval()
        captures=[];handle=self.image.model[-1].drop.register_forward_hook(lambda m,i,o:captures.append(o))
        try:self.image(self.transform(im)[None].to(self.device))
        finally:handle.remove()
        return captures[0].flatten(1).cpu().float().numpy()[0]

    @torch.inference_mode()
    def encode_point(self,xyz):
        if self.point is None:
            ck=torch.load(POINT,map_location='cpu',weights_only=False)
            self.point=PointTransformerV2Classifier(len(ck['class_names']))
            self.point.load_state_dict(ck['model_state'],strict=True);self.point=self.point.to(self.device).eval()
        captures=[];handle=self.point.classifier.register_forward_pre_hook(lambda m,i:captures.append(i[0]))
        reference=cKDTree(xyz).query(xyz,k=8)[1]
        try:self.point(torch.as_tensor(xyz[None],dtype=torch.float32,device=self.device),torch.as_tensor(reference[None],dtype=torch.long,device=self.device))
        finally:handle.remove()
        return captures[0].cpu().float().numpy()[0]

def decision(bundle,mode,feature,severe=False):
    clf=bundle['heads'][mode];x=normalize(feature[None]);p=clf.predict_proba(x)[0]
    order=np.argsort(-p);best=int(clf.classes_[order[0]]);score=float(p[order[0]]);margin=score-float(p[order[1]])
    references=bundle['references'][mode];distance=float(1-np.max(x@references.T))
    reasons=[]
    if severe:reasons.append('输入质量不足')
    if score<.65:reasons.append('最高分低于实验阈值0.65')
    if margin<.15:reasons.append('前两名差值小于0.15')
    if distance>bundle['distance_limits'][mode]:reasons.append('特征与训练样本差异较大')
    source={'image':'香港街景三类 · YOLO迁移分类头','point':'香港街景三类 · WHU PTv2迁移分类头','fusion':'香港街景三类 · 图像＋PTv2特征融合'}[mode]
    return {'label':'Unknown' if reasons else NAMES[best],'display_name':'未知 / 需复核' if reasons else CHINESE[best],'score':round(score,5),'margin':round(margin,5),'score_kind':'实验模型分数，未经概率校准','source':source+'（实验）','reasons':reasons,'feature_distance':round(distance,5),'top3':[{'species':NAMES[int(clf.classes_[i])],'name':CHINESE[int(clf.classes_[i])],'score':round(float(p[i]),5),'in_catalog':True} for i in order]}
