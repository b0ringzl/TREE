# B5 PTv2 三类点云训练阶段报告

## 1. 阶段结论

B5 已完成。使用 B1 固定的三类别、同一批 `420/90/90` train/val/test
样本，完成了 PTv2 mode-1 分类编码器的原生 PyTorch 复现、GPU 正式训练、
独立验收和四模型逐样本比较。

本次最佳 checkpoint 来自第 59 轮，训练在第 99 轮因连续 40 轮验证指标
未提升而早停：

| split | Accuracy | Macro-F1 | Loss |
|---|---:|---:|---:|
| validation | 75.56% | 73.93% | 0.9170 |
| test | 96.67% | 96.70% | 0.3872 |

测试集结果很高，但验证集与测试集 Macro-F1 相差 22.77 个百分点，因此不能把
96.70% 单独解释为稳定泛化性能。当前最重要的后续问题是固定道路划分的难度差异，
而不是继续依据本次测试集调整模型。

## 2. 范围与复现边界

本阶段实现并验证了 PTv2 的核心结构：

- Grouped Vector Attention（GVA）；
- grouped linear 权重编码；
- position encoding multiplier 与 bias；
- partition-based grid pooling；
- 编码器全局平均池化和分类 MLP。

模型保持每棵树 8,192 点输入，层级配置为：

| 项目 | 配置 |
|---|---|
| Patch embedding | 48 channels，1 block，6 groups，8-NN |
| Encoder depths | `[2, 2, 6, 2]` |
| Encoder channels | `[96, 192, 384, 384]` |
| Encoder groups | `[12, 24, 48, 48]` |
| Encoder neighbours | `[16, 16, 16, 16]` |
| Grid sizes | `[0.06, 0.12, 0.24, 0.48]` |
| 参数量 | 9,389,985 |

官方代码以 Ubuntu、旧版 PyTorch 和自定义 `pointops` CUDA 扩展为主要运行条件，
且官方说明已发布的分类代码没有持续维护。因此本阶段没有修改隔离 Conda 环境，
而是用 PyTorch 原生操作复现相同数学结构。它不是官方仓库在 Linux 上的逐行运行，
但保留了论文定义 PTv2 的三个核心创新。

参考：

- 官方实现：https://github.com/Pointcept/PointTransformerV2
- 论文：https://arxiv.org/abs/2210.05666

## 3. 邻域缓存

为避免每轮重复计算 8,192 点全分辨率 kNN，先构建只依赖几何和样本顺序的缓存：

- 样本数：600；
- 邻居数：8；
- train tensor：`[420, 8192, 8]`；
- val tensor：`[90, 8192, 8]`；
- test tensor：`[90, 8192, 8]`；
- 构建时间：16.63 秒；
- 文件大小：157,300,905 字节；
- SHA-256：
  `64e7f508ace3985a801887c5403c04f9e0da9814d6945afb5e903f047f87c3b1`。

缓存绑定了 B1 `manifest.json` 和 `classes.json` 的 SHA-256。训练脚本在启动前会
检查 split、sample key 顺序、形状和哈希，防止误用其他数据版本。

训练增强只使用绕 Z 轴旋转和 0.9-1.1 各向同性缩放，两者保持欧氏邻域不变；
四级网格池化后的邻域仍在每个 batch 内动态重算。

## 4. 正式训练

锁定配置：

| 项目 | 值 |
|---|---:|
| seed | 20260728 |
| optimizer | AdamW |
| 初始/最低学习率 | 0.001 / 0.00001 |
| weight decay | 0.05 |
| label smoothing | 0.1 |
| batch size | 16 |
| 最大轮数 | 200 |
| early-stopping patience | 40 |
| AMP | 开启 |
| DataLoader workers | 0 |

执行情况：

- 实际完成 99 轮；
- 最佳轮次 59；
- 逐轮训练累计 1,768.47 秒（29.47 分钟）；
- 完整会话 1,809.49 秒（30.16 分钟）；
- 平均每轮 17.86 秒；
- 正式训练最大显存分配约 4,469.67 MiB；
- 未续训，未出现 AMP optimizer step 跳过；
- 测试集只在验证选模结束后评估。

最佳轮次 59 的训练 Accuracy 为 91.90%、Macro-F1 为 91.86%，验证
Macro-F1 为 73.93%。训练曲线继续上升而验证曲线大幅波动，说明三类别小样本下
已经出现明显过拟合和道路域差异。

## 5. 分类结果

验证集混淆矩阵：

```text
[[29,  0,  1],
 [ 9, 14,  7],
 [ 2,  3, 25]]
```

验证集主要瓶颈是 `Lagerstroemia indica`，召回率仅 46.67%；其中 9 棵被判为
`Cinnamomum camphora`，7 棵被判为 `Magnolia grandiflora`。

测试集混淆矩阵：

```text
[[30,  0,  0],
 [ 1, 29,  0],
 [ 2,  0, 28]]
```

测试集只有 3 个错误，并且全部流向 `Cinnamomum camphora`。验证与测试错误模式
明显不同，支持“道路、点云密度或树体形态分布不同”的解释，但目前尚未做正式的
道路级诊断，不能把原因写死。

## 6. 与既有基线比较

所有结果来自相同 B1 split：

| 模型 | Val Accuracy | Val Macro-F1 | Test Accuracy | Test Macro-F1 |
|---|---:|---:|---:|---:|
| PointMLP | 54.44% | 53.42% | 66.67% | 66.11% |
| ResNet50 | 91.11% | 91.10% | 80.00% | 80.19% |
| TSCMDL B4 | 80.00% | 79.93% | 83.33% | 83.32% |
| PTv2 B5 | 75.56% | 73.93% | 96.67% | 96.70% |

PTv2 相对 PointMLP：

- 验证 Macro-F1 提升 20.51 个百分点；
- 测试 Macro-F1 提升 30.59 个百分点。

PTv2 在测试集最高，但验证集低于 ResNet50 和 B4 TSCMDL。逐样本比较中，PTv2
在测试集独自纠正 5 棵树；四模型的理想 oracle 在 val/test 都为 100%，说明模态
和架构之间仍存在互补性，但 oracle 不是可部署指标。

## 7. 验收

`reports/B5_validation.json` 已独立通过：

- `best.pt` 为第 59 轮，`last.pt` 为第 99 轮；
- checkpoint 能严格加载到重建的 9,389,985 参数模型；
- 180 行 val/test 预测无重复 sample key；
- Accuracy、balanced accuracy、precision、recall、Macro-F1 和混淆矩阵均从
  `predictions.csv` 重新计算一致；
- 缓存哈希与 checkpoint、运行配置和最终指标一致；
- 训练历史没有任何 `test_*` 字段；
- 三张 PNG 图像均可验证和正常显示；
- 没有遗留 `.tmp` 文件。

项目单元测试共 34 项，全部通过。

## 8. 实时监控

本次运行可由只读脚本实时查看：

```powershell
D:\TREE\envs\whu-tscmdl\python.exe `
  D:\TREE\Lidar\tscmdl_whu_sandbox\scripts\monitor_training.py `
  --run-dir "D:\TREE\lidar data\whu\derived\tscmdl\b5_ptv2\run_batch16_seed20260728"
```

监控显示轮次、进度、最近十轮平均耗时、ETA、最新/最佳验证指标、连续未提升轮数、
GPU 利用率和显存。训练完成后显示 `99/99 100%` 与 `40/40` 早停原因。

## 9. 产物与磁盘

主要产物：

- 正式运行：
  `D:\TREE\lidar data\whu\derived\tscmdl\b5_ptv2\run_batch16_seed20260728`
- kNN 缓存：
  `D:\TREE\lidar data\whu\derived\tscmdl\b5_ptv2\knn_cache_k8.pt`
- 独立验收：`reports/B5_validation.json`
- 四模型比较：`reports/B2_B3_B4_B5_prediction_comparison.json`
- 指标比较图：`reports/B2_B3_B4_B5_metric_comparison.png`

B5 派生目录新增 22 个文件，共约 295.25 MiB：

- kNN 缓存及元数据约 150.01 MiB；
- 正式运行约 145.20 MiB；
- 三组 smoke test 约 0.04 MiB。

没有修改 WHU-STree 原始数据，没有修改 `yolo`、`pdf-reading` 或
`whu-tscmdl` 环境，也没有执行 Git 操作。

## 10. 风险与下一阶段建议

当前主要风险：

1. 只有一个 PTv2 随机种子；
2. 验证和测试 Macro-F1 相差 22.77 个百分点；
3. 验证曲线波动大，最佳模型仍有明显过拟合；
4. 当前是原生 PyTorch 结构复现，不是官方 Linux `pointops` 后端；
5. 只有三类，不代表完整 19 类难度。

建议下一小阶段为 **B5b：三随机种子重复与道路域差异诊断**。该阶段应保持配置
不变，补跑两个随机种子，并按道路、点数、网格池化后点数和类别分析 val/test
差异。完成后再决定直接进入 C1/C2，还是先调整数据划分。

B5b 尚未获得授权，本阶段到此停止。
