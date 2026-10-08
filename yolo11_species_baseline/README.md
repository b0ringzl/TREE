# YOLO11 十类树种检测基线

该目录独立于 WHU 工程。它复用 ResNet50 基线的无泄漏数据划分，但训练目标是
“在整张图中定位目标树并识别树种”，而不是对预先裁好的树木图块分类。

## 环境

- Python：`envs/yolo11-tree/Scripts/python.exe`
- Ultralytics：8.4.127
- PyTorch：2.11.0 + CUDA 12.8
- GPU：NVIDIA GeForce RTX 4090 D

## 启动

在仓库根目录执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\yolo11_species_baseline\run_experiment.ps1 `
  -RunName yolo11s_10class_v2
```

训练期间可另开终端监控：

```powershell
.\envs\yolo11-tree\Scripts\python.exe `
  .\yolo11_species_baseline\scripts\monitor_training.py `
  --run-dir .\yolo11_species_baseline\experiments\yolo11s_10class_v2 --watch
```

运行目录会保存数据索引、环境配置、训练曲线、最佳权重、测试指标和带框推理图。

## 本次正式基线

正式结果位于 `experiments/yolo11s_10class_v2`。YOLO11s 在 55 张独立香港测试图像
（57 个目标框）上的结果为：P=0.6583、R=0.2064、mAP@0.50=0.2579、
mAP@0.50:0.95=0.1125。流程和数据隔离检查均通过，但模型召回率不足，当前权重仅用于
验证训练、测试与可视化链路，不能作为应用交付模型。
