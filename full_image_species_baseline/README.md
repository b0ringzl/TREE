# 全量图像树种识别两阶段基线

本实验复用既有 10 类 ResNet50 图像识别权重，依次完成：

1. 合并 `inat_max150`、日期批次和母种补充集，按 photo ID/内容去重后做全图弱标签预训练；
2. 从人工 YOLO 多边形/框生成目标裁剪，继承原图划分后做精调；
3. 只在验证集选定最佳权重后读取测试集，并输出 sample 与 observation 两级指标。

数据源始终只读，产物位于 `experiments/<RunName>`。

```powershell
powershell -ExecutionPolicy Bypass -File .\full_image_species_baseline\run_experiment.ps1
```

中断后可安全续训：

```powershell
powershell -ExecutionPolicy Bypass -File .\full_image_species_baseline\run_experiment.ps1 -Resume
```

## 首次正式基线

`full_web_resnet50_20260906_seed1` 已完成并通过数据泄漏、清单哈希、分类头和权重继承检查。
全图 148 类测试 Accuracy 为 13.87%、Top-5 为 35.17%；人工裁剪 26 类精调后，
单目标 Accuracy 为 38.97%、Top-5 为 70.83%，按 observation 聚合 Accuracy 为 50.00%。

详细结果见 `reports/20260906_全量图像树种识别两阶段基线报告.md`。

## YOLO11s-cls 对照

同一份清单与划分上的 `full_web_yolo11s_cls_20260906_seed1` 已完成。全图 148 类
测试 Accuracy 为 57.09%，人工裁剪 26 类精调后为 68.87%；按 observation 聚合后
分别为 68.00% 和 82.50%。这是 YOLO 的分类模型，不包含目标检测或实例分割。

复现命令：

```powershell
powershell -ExecutionPolicy Bypass -File .\full_image_species_baseline\run_yolo11_experiment.ps1
```

脚本会把测试集放在训练目录之外，并根据验证集 observation Macro-F1 选择检查点。
中断后再次运行同一命令，会复用已完成的数据物化、训练和验证选模阶段。

详细对照见 `reports/20260906_YOLO11s-cls_全量图像对照报告.md`。

## YOLO11m-seg 实例分割

`streetview_yolo11m_seg_20260906_seed1` 已使用人工多边形在原图上完成实例分割训练，
同时预测目标位置、掩膜和 26 类树种。独立测试集上，边界框 mAP50-95 为 39.53%，
实例掩膜 mAP50-95 为 31.53%。

复现命令：

```powershell
powershell -ExecutionPolicy Bypass -File .\full_image_species_baseline\run_yolo11_seg_experiment.ps1
```

训练数据 YAML 不包含测试集；框架根据验证集指标保存 `best.pt` 后，脚本才会加载独立测试集。

详细结果见 `reports/20260906_YOLO11m-seg_实例分割基线报告.md`。
