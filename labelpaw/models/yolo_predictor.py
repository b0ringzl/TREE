import os
import cv2
import numpy as np
import torch
from PySide6.QtGui import QImage, QPolygonF
from PySide6.QtCore import QPointF, QRectF, QThread, Signal
from ultralytics import YOLO

class YoloPredictorWorker(QThread):
    finished = Signal(list)
    error = Signal(str)
    
    def __init__(self, predictor, image_path, classes=None):
        super().__init__()
        self.predictor = predictor
        self.image_path = image_path
        self.classes = classes
        
    def run(self):
        try:
            shapes = self.predictor.predict_sync(self.image_path, classes=self.classes)
            self.finished.emit(shapes)
        except Exception as e:
            self.error.emit(str(e))

class YoloPredictor:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = YOLO(model_path)
        self.task = getattr(self.model, 'task', 'detect')
        
        # 尝试从模型中提取骨架和关键点名称 (YOLOv8/v11/v26 结构)
        self.skeleton = None
        self.kpt_names = None
        try:
            # 访问底层模型的元数据
            m = self.model.model
            if hasattr(m, 'yaml'):
                self.skeleton = m.yaml.get('skeleton')
                self.kpt_names = m.yaml.get('kpt_names')
            
            # 如果 yaml 里没找到，尝试从属性里找
            if not self.skeleton and hasattr(m, 'skeleton'):
                self.skeleton = m.skeleton
        except:
            pass
            
    def predict(self, image_path, classes=None):
        """兼容旧的同步调用"""
        return self.predict_sync(image_path, classes=classes)
    
    def predict_sync(self, image_path, classes=None):
        """
        预测单张图片并解析结果。
        返回 shapes 列表，每个元素是个字典：
        {
            "type": "rect" | "poly" | "rbox" | "pose",
            "label": "class_name",
            "score": float,
            "data": ...,
            "skeleton": list[list[int]] (可选),
            "kpt_names": list[str] (可选)
        }
        """
        if not os.path.exists(image_path):
            return []
            
        if classes is not None and len(classes) > 0:
            results = self.model(image_path, classes=classes)
        else:
            results = self.model(image_path)
        if not results:
            return []
            
        result = results[0]
        names = result.names
        
        parsed_shapes = []
        
        # 目标检测 (Boxes)
        if result.boxes is not None and getattr(result, 'obb', None) is None and getattr(result, 'masks', None) is None and getattr(result, 'keypoints', None) is None:
            boxes = result.boxes
            for i in range(len(boxes)):
                box = boxes[i]
                cls_id = int(box.cls[0].item())
                label = names[cls_id]
                conf = float(box.conf[0].item())
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = xyxy
                parsed_shapes.append({
                    "type": "rect",
                    "label": label,
                    "score": conf,
                    "data": QRectF(x1, y1, x2 - x1, y2 - y1)
                })
                
        # OBB (Oriented Bounding Boxes)
        if getattr(result, 'obb', None) is not None:
            obbs = result.obb
            for i in range(len(obbs)):
                obb = obbs[i]
                cls_id = int(obb.cls[0].item())
                label = names[cls_id]
                conf = float(obb.conf[0].item())
                xyxyxyxy = obb.xyxyxyxy[0].cpu().numpy() # [4, 2]
                poly = QPolygonF()
                for pt in xyxyxyxy:
                    poly.append(QPointF(float(pt[0]), float(pt[1])))
                parsed_shapes.append({
                    "type": "rbox",
                    "label": label,
                    "score": conf,
                    "data": poly
                })
                
        # 分割 (Segmentation)
        if getattr(result, 'masks', None) is not None:
            masks = result.masks
            boxes = result.boxes
            for i in range(len(masks)):
                mask = masks[i]
                cls_id = int(boxes[i].cls[0].item())
                label = names[cls_id]
                conf = float(boxes[i].conf[0].item())
                segments = mask.xy # list of arrays
                if segments and len(segments) > 0:
                    for segment in segments:
                        if len(segment) > 2:
                            # 使用 cv2.approxPolyDP 简化多边形
                            contour = np.array(segment, dtype=np.float32)
                            epsilon = 0.005 * cv2.arcLength(contour, True)
                            approx = cv2.approxPolyDP(contour, epsilon, True)
                            approx = approx.reshape(-1, 2)
                            
                            poly = QPolygonF()
                            for pt in approx:
                                poly.append(QPointF(float(pt[0]), float(pt[1])))
                            parsed_shapes.append({
                                "type": "poly",
                                "label": label,
                                "score": conf,
                                "data": poly
                            })
                            
        # 关键点 (Keypoints/Pose)
        if getattr(result, 'keypoints', None) is not None:
            keypoints = result.keypoints
            boxes = result.boxes
            for i in range(len(keypoints)):
                kpt = keypoints[i]
                cls_id = int(boxes[i].cls[0].item())
                label = names[cls_id]
                conf = float(boxes[i].conf[0].item())
                xy = kpt.xy[0].cpu().numpy() # [N, 2]
                
                confidences = None
                if kpt.conf is not None:
                    confidences = kpt.conf[0].cpu().numpy()
                
                box_xywh = boxes[i].xywh[0].cpu().numpy()
                rect = QRectF(box_xywh[0] - box_xywh[2]/2, box_xywh[1] - box_xywh[3]/2, box_xywh[2], box_xywh[3])
                
                pts = []
                for j, point in enumerate(xy):
                    c = float(confidences[j]) if confidences is not None else 1.0
                    pts.append({"pos": QPointF(float(point[0]), float(point[1])), "vis": 2 if c > 0.25 else 0})
                    
                shape_data = {
                    "type": "pose",
                    "label": label,
                    "score": conf,
                    "data": {
                        "rect": rect,
                        "keypoints": pts
                    }
                }
                
                # 注入模型自带的骨架连接和名称
                if self.skeleton:
                    shape_data["skeleton"] = self.skeleton
                if self.kpt_names:
                    shape_data["kpt_names"] = self.kpt_names
                    
                parsed_shapes.append(shape_data)
                
        # 兼容性冗余过滤，确保无论哪种底层版本都能 100% 滤除其他类别
        if classes is not None and len(classes) > 0:
            allowed_names = {names[cid] for cid in classes if cid in names}
            parsed_shapes = [s for s in parsed_shapes if s["label"] in allowed_names]
            
        return parsed_shapes

if __name__ == "__main__":
    # 测试代码
    model_path = r"E:\11-AI\标注工具\weights\yolo26_weights\yolo26n-pose.pt"
    image_path = r"E:\11-AI\标注工具\1_github\关键点数据集测试\0c8a1fcd-fba3-4a92-a57d-fecc07825090.png"
    
    if os.path.exists(model_path) and os.path.exists(image_path):
        print(f"Loading YOLO model from {model_path}...")
        predictor = YoloPredictor(model_path)
        print(f"Predicting image {image_path}...")
        shapes = predictor.predict_sync(image_path)
        
        print(f"\n======== Prediction Results ========")
        print(f"Total shapes detected: {len(shapes)}")
        for i, shape in enumerate(shapes):
            print(f"[{i}] Type: {shape['type']}, Label: {shape['label']}, Score: {shape['score']:.4f}")
            data = shape['data']
            print(f"    Data: {data}")
    else:
        print("Model path or Image path does not exist. Please verify paths.")
