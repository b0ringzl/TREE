# 南京四分类：YOLO11 与 WHU ResNet50 对比

该实验使用 WHU D1 冻结数据包中的同一批南京单树裁剪、同一 train/val/test
划分和相同四类顺序，对比 YOLO11s-cls 与既有 ResNet50。

这里使用 YOLO11 **分类**模型，而不是十类整图检测权重。原因是检测 mAP 与已裁剪单树
分类 Accuracy 不可直接比较，且十类权重的类别集合与南京四分类不一致。

一键运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\nj_yolo11_vs_resnet50\run_experiment.ps1
```

正式三种子复现（在全新的 experiments 目录中运行）：

```powershell
foreach ($seed in @(20260728, 20260729, 20260730)) {
  powershell -ExecutionPolicy Bypass -File .\nj_yolo11_vs_resnet50\run_experiment.ps1 `
    -RunName "nj_yolo11s_cls_seed_$seed" -Seed $seed
}
```

三种子统计和逐树概率集成：

```powershell
.\envs\yolo11-tree\Scripts\python.exe `
  .\nj_yolo11_vs_resnet50\scripts\summarize_three_seeds.py `
  --experiments-root .\nj_yolo11_vs_resnet50\experiments `
  --resnet-suite '.\lidar data\whu\derived\tscmdl\d1_resnet50_clean_repeats\20260817_protocol_v1' `
  --output-dir .\nj_yolo11_vs_resnet50\reports\20260824_three_seed_results
```

训练期间监控：

```powershell
.\envs\yolo11-tree\Scripts\python.exe `
  .\nj_yolo11_vs_resnet50\scripts\monitor_run.py `
  --run-dir .\nj_yolo11_vs_resnet50\experiments\nj_yolo11s_cls_seed_20260728
```

测试集不会参与训练或检查点选择。训练结束后，脚本逐个在验证集上评估已保存的
epoch 检查点，以验证集 Macro-F1、Accuracy 顺序选出最终权重，然后只对测试集评估一次。

当前正式结果见 `reports/20260824_three_seed_results/三种子正式汇总.md`。三次 YOLO11s-cls
测试 Accuracy 为 90.07%、92.05%、95.36%，平均为 92.49% ± 2.68%（样本标准差）。
