# B3 ResNet50 三类图像训练阶段报告

日期：2026-07-23

## 1. 阶段结论

B3 已完成。已在 B1 的 600 株平衡数据上训练三类 ResNet50 图像单模态基线，完成 300 轮正式训练，并生成最佳/末轮检查点、逐样本预测、训练曲线、验证与测试混淆矩阵、结构化指标和独立验收。

最佳模型为第 166 轮，按验证集宏平均 F1、再按 accuracy 选择。该模型在验证集上取得 91.11% accuracy、91.10% macro F1，在独立测试集上取得 80.00% accuracy、80.19% macro F1。测试集没有参与训练、调参或检查点选择。

与 B2 PointMLP 点云基线相比，ResNet50 的测试 accuracy 提高 13.33 个百分点，macro F1 提高 14.08 个百分点。结果说明当前街景裁剪中的纹理、叶色和树冠外观比单位球归一化点云提供了更强的三类判别信息，但验证与测试之间仍有明显域差异。

## 2. 论文协议核对

本阶段使用 `pdf-reading` Conda 环境中的 PyMuPDF 解析 TSCMDL 论文，生成中间文本：

`D:\TREE\paper reading\output\TSCMDL_extracted.txt`

正文双栏文本层总体正常。第 6 页 Table I 的表格结构没有进入线性文本，因此另用 PyMuPDF 渲染第 6 页并人工复核表格；没有根据缺失文本猜测超参数。

论文图像协议为：

- ImageNet 预训练 ResNet50；
- 随机 224×224 裁剪；
- 随机水平翻转；
- ImageNet mean `(0.485, 0.456, 0.406)`；
- ImageNet std `(0.229, 0.224, 0.225)`；
- batch size 8；
- 300 epochs；
- SGD、learning rate 0.1、momentum 0.9；
- CosineAnnealingLR；
- weight decay 0.0002；
- CrossEntropyLoss。

本地实现遵守以上设置。验证和测试使用确定性的中心 224×224 裁剪。执行层启用 CUDA AMP、channels-last 和 fused SGD，以缩短运行时间；这些是执行优化，不改变数据划分、模型结构和训练超参数。

需要注意，论文使用 UAV 正射影像和两类森林树种，本阶段使用 WHU-STree 街景裁剪和三类城市树种，因此本地指标不能与论文数值直接横向比较。现代 torchvision 的 `IMAGENET1K_V2` 权重也不一定与论文当年的具体 ImageNet 检查点逐位一致。

## 3. 数据与类别

数据集：

`D:\TREE\lidar data\whu\derived\tscmdl\b1_three_species\dataset`

| class | WHU label | 树种 | train | val | test |
|---:|---:|---|---:|---:|---:|
| 0 | 1 | Cinnamomum camphora | 140 | 30 | 30 |
| 1 | 2 | Lagerstroemia indica | 140 | 30 | 30 |
| 2 | 13 | Magnolia grandiflora | 140 | 30 | 30 |
| 合计 | | | 420 | 90 | 90 |

每个样本使用一张 768×512 RGB 全景裁剪。split、sample key 和类别映射与 B2 完全一致，因此两个单模态基线可逐样本比较。

Codex 视觉预检覆盖 18 张图，用户复核尚未完成；最终 val/test 预测表保留其中 12 张预检质量和备注字段。正式训练没有依据这些预检标记删除样本，避免改变 B1 划分。

## 4. 模型与运行配置

| 项目 | 配置 |
|---|---|
| 模型 | torchvision ResNet50 |
| 预训练 | `ResNet50_Weights.IMAGENET1K_V2` |
| 参数量 | 23,514,179 |
| 输入 | 224×224 RGB |
| 输出 | 3 类 |
| 优化器 | fused SGD，momentum 0.9 |
| 初始/最终学习率 | 0.1 / 接近 0 |
| 调度器 | CosineAnnealingLR，300 epochs |
| 权重衰减 | 0.0002 |
| 损失 | CrossEntropyLoss |
| train/eval batch | 8 / 16 |
| 随机种子 | 20260723 |
| DataLoader workers | 0 |
| 数值与布局 | CUDA AMP、channels-last |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU |
| PyTorch/torchvision | 2.11.0+cu128 / 0.26.0+cu128 |

正式训练 300 轮累计 train+val 时间 2372.39 秒，约 39.54 分钟；包含初始化、检查点保存和最终评估的会话时间为 2453.04 秒，约 40.88 分钟。第一轮包含 CUDA 预热，后续多数轮次约 7.7–8.0 秒。峰值 PyTorch 已分配显存约 616.39 MiB。

## 5. 最终指标

| split | accuracy | balanced accuracy | macro precision | macro recall | macro F1 |
|---|---:|---:|---:|---:|---:|
| val | 91.11% | 91.11% | 91.79% | 91.11% | 91.10% |
| test | 80.00% | 80.00% | 82.40% | 80.00% | 80.19% |

验证集逐类指标：

| 树种 | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| Cinnamomum camphora | 87.50% | 93.33% | 90.32% | 30 |
| Lagerstroemia indica | 87.88% | 96.67% | 92.06% | 30 |
| Magnolia grandiflora | 100.00% | 83.33% | 90.91% | 30 |

测试集逐类指标：

| 树种 | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| Cinnamomum camphora | 95.45% | 70.00% | 80.77% | 30 |
| Lagerstroemia indica | 68.42% | 86.67% | 76.47% | 30 |
| Magnolia grandiflora | 83.33% | 83.33% | 83.33% | 30 |

测试混淆矩阵，行为真实类别、列为预测类别：

| true/pred | 0 | 1 | 2 |
|---|---:|---:|---:|
| 0 | 21 | 8 | 1 |
| 1 | 0 | 26 | 4 |
| 2 | 1 | 4 | 25 |

测试集主要错误是 8 株香樟被判为紫薇。验证 accuracy 91.11% 而测试为 80.00%，说明 road 06/23 验证集比官方 reference 测试轨迹更容易，或者两者街景条件存在域差异。90 株小测试集不足以准确估计更广泛道路泛化。

## 6. 质量标记分析

val/test 的 12 个已人工审查样本中：

| 质量 | 样本数 | 正确数 |
|---|---:|---:|
| excellent | 5 | 5 |
| good | 3 | 2 |
| acceptable | 3 | 2 |
| poor | 1 | 0 |

唯一 poor 样本 `23_1_2502` 被公交车严重遮挡，真实 class 2 被判为 class 0。另一个天空过曝、树冠稀疏的 acceptable 样本和一个邻树枝干扰的 good 样本也分类错误。样本数很少，不能做统计因果结论，但结果支持在 B4 保留质量标记并考虑多视角或遮挡感知策略。

## 7. 产物与验收

正式运行目录：

`D:\TREE\lidar data\whu\derived\tscmdl\b3_resnet50\run_batch8_seed20260723`

主要产物：

- `best.pt`：第 166 轮最佳检查点，约 90.0 MiB；
- `last.pt`：第 300 轮完整恢复检查点，约 179.8 MiB；
- `predictions.csv`：val/test 共 180 行，含概率和质量标记；
- `final_metrics.json`：最终结构化指标；
- `training_history.json/csv`：300 轮历史；
- `training_curves.png`：loss、accuracy、macro F1 曲线；
- `confusion_matrix_val.png`、`confusion_matrix_test.png`；
- `train.log`、`run_config.json`。

独立验收结果为 passed：

- 11 个必需文件完整，无遗留 `.tmp`；
- history 连续覆盖 1–300 轮；
- best/last 分别为第 166/300 轮；
- 180 行预测无重复 sample key，val/test 各 90 行；
- 概率有限、非负、和为 1，预测类等于 argmax；
- 从预测表重算的指标和混淆矩阵与最终结果完全一致；
- 12 个质量标记及备注与 B1 人工审查文件一致；
- 三张 PNG 可正常解码并已由 Codex 做可视化预检；
- 沙盒 21 项单元测试全部通过。

正式运行目录约 270.43 MiB；包含四次 smoke 输出的全部 B3 目录约 270.46 MiB。ImageNet 预训练权重另缓存在用户 Torch cache 中，约 97.8 MiB。

## 8. 通用实时监控

新增只读监控器：

`scripts/monitor_training.py`

它统一读取 `run_config.json`、`training_history.json`、`last.pt`、`train.log` 和 `final_metrics.json`，显示：

- 当前/总轮次与进度条；
- 最新 train/val loss、accuracy、macro F1；
- 最佳验证轮次；
- 已用时间、最近平均轮时和 ETA；
- 产物更新时间及暂停/陈旧判断；
- 实时 GPU 利用率、显存和温度；
- 完成后的最终 val/test 指标。

本次正式运行已用 `--once` 验证，正确显示 300/300、最佳第 166 轮和最终指标。阶段门已增加强制规则：后续每次正式训练开始前，都必须验证该监控器可以读取训练状态。

## 9. 限制与风险

- 只运行一个随机种子，尚不能估计结果方差；
- 300 轮中验证曲线仍有明显波动，最佳单轮结果可能偏乐观；
- 验证与测试存在约 11.11 个百分点的 accuracy 差距；
- 随机 224×224 裁剪遵循论文，但对街景目标可能偶尔裁掉部分树冠或主干；
- 当前每株只有一个选定视角，没有使用多视角集成；
- 图像可能包含邻树、车辆和背景道路线索，模型不一定只学习目标树体；
- 当前结果是三类实验，不能外推到完整 19 类。

## 10. 下一阶段（未授权）

B4 为三类 TSCMDL 点云与图像融合训练。建议：

1. 分别从 B2 最佳 PointMLP 和 B3 最佳 ResNet50 加载特征提取权重；
2. 明确并测试两分支特征维度、归一化和拼接契约；
3. 先做冻结主干的融合头冒烟，再决定是否联合微调；
4. 沿用相同 split、seed、验证宏平均 F1 选模和逐样本预测；
5. 正式训练前先启动并验证通用监控器；
6. 与 B2、B3 做统一的逐样本和质量分层比较。

B3 到此停止，等待用户明确批准后再开始 B4。
