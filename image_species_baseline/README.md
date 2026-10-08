# 香港街景 10 类图像识别基线

本目录是独立于 `Lidar/tscmdl_whu_sandbox` 的图像分类试验。源数据只读，
训练产物写入 `experiments/`，不会修改 WHU 项目或其四分类模型。

## 试验范围

- 主干：复制 D1 最佳四分类 ResNet50 的 `best.pt`，丢弃原四分类 `fc`，
  保留其余主干参数并新建 10 类分类头。
- 训练数据：10 个共有树种的已标注网络图片目标裁剪，以及香港街景训练树实例。
- 验证/测试：仅香港街景，按树文件夹分组，避免同一棵树不同视角泄漏。
- 模型选择：验证集树实例级 macro-F1；测试集仅在训练结束并确定最佳权重后评估。

## 一键启动

在仓库根目录运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\image_species_baseline\run_experiment.ps1 `
  -RunName hk_web_resnet50_10class_v3
```

正式 v2 已存在，脚本会拒绝覆盖已完成的运行；再次训练时必须指定新的 `RunName`。

查看训练状态：

```powershell
.\envs\whu-tscmdl\python.exe .\image_species_baseline\scripts\monitor_training.py `
  --run-dir .\image_species_baseline\experiments\hk_web_resnet50_10class_v2
```

加入 `--watch` 可持续刷新。正式运行的配置、数据清单、复制的来源权重、训练曲线、
混淆矩阵、逐样本预测和最终指标都会保存在相应运行目录中。

数据清单还会对图像内容计算 SHA-256。完全相同的香港图像即使位于不同树文件夹，
也会绑定到同一分组并仅保留一份，避免内容级训练/验证/测试泄漏。
