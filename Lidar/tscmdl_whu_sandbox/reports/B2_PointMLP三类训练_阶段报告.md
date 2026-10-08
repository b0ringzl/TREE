# B2 PointMLP 三类点云训练阶段报告

日期：2026-07-14

## 1. 阶段结论

B2 已完成。已在 B1 的 600 株平衡数据上训练三类 PointMLP 点云单模态基线，并完成最佳/末轮检查点、逐样本预测、训练曲线、验证与测试混淆矩阵、结构化指标和独立验收。

最佳模型为第 19 轮，按验证集宏平均 F1 选择。该模型在验证集上取得 54.44% accuracy、53.42% macro F1，在独立测试集上取得 66.67% accuracy、66.11% macro F1。测试集从未用于训练、早停或检查点选择。

本结果证明点云分支能够学习三类树种差异，但还不是完整 TSCMDL 复现。训练曲线存在显著验证波动，紫薇类别召回率偏低，当前结果应作为后续图像单模态和融合实验的基线，而不是最终性能结论。

## 2. 数据与类别

训练使用 B1 数据集：

`D:\TREE\lidar data\whu\derived\tscmdl\b1_three_species\dataset`

| 本地 class | WHU label | 树种 | train | val | test |
|---:|---:|---|---:|---:|---:|
| 0 | 1 | Cinnamomum camphora | 140 | 30 | 30 |
| 1 | 2 | Lagerstroemia indica | 140 | 30 | 30 |
| 2 | 13 | Magnolia grandiflora | 140 | 30 | 30 |
| 合计 | | | 420 | 90 | 90 |

每株点云为 `(8192, 3)` float32 单位球坐标。B1 中共有 242 株源点少于 8192，通过有放回采样补齐。其中紫薇在 train/test 分别为 140/30 株全部需要补点，val 有 14 株需要补点；这可能是该类召回率较弱的因素之一，但本阶段没有进行因果消融。

划分继续遵守 B1 协议：val 使用完整道路 06、23；test 使用 WHU 官方 reference 轨迹。样本、树和轨迹不跨 split，但官方测试按轨迹而非整条道路划分，所以部分 train/test 道路 ID 重叠，不属于严格道路域外评估。

## 3. 模型与训练配置

| 项目 | 配置 |
|---|---|
| 模型 | 上游完整 PointMLP |
| 参数量 | 13,228,995 |
| 输入 | 每株 8192 个 XYZ 点 |
| 输出 | 3 类 |
| 优化器 | SGD，momentum 0.9 |
| 初始/最低学习率 | 0.009375 / 0.00046875 |
| 调度器 | CosineAnnealingLR，40 epochs |
| 权重衰减 | 0.0002 |
| 标签平滑 | 0.2 |
| train/eval batch | 3 / 8 |
| 数值精度 | CUDA AMP，GradScaler 初始 scale 8 |
| 随机种子 | 20260714 |
| DataLoader workers | 0 |
| 增强 | Z 轴旋转、0.9-1.1 等比缩放、裁剪高斯抖动、点顺序打乱 |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU |
| PyTorch/CUDA runtime | 2.11.0+cu128 / 12.8 |

单位球归一化去除了绝对尺寸信息，模型只能使用归一化形态。这样有利于控制尺度差异，但也可能损失树种判别所需的真实树高、冠幅和密度信息。

## 4. 试跑与正式运行

先进行 batch size 4 试跑。完成 3 轮后，优化器状态使设备显存接近上限，第 3 轮耗时从约 126 秒增至 294 秒，因此主动停止该试跑。外层终止后发现子 Python 仍存在，已按 PID 清理并确认 GPU 显存释放；该试跑不作为正式结果。

随后将 batch size 降为 3。正式运行在 RTX 5060 上保持约 93%-99% GPU 利用率，PyTorch 峰值已分配显存约 6153.8 MiB，系统观测设备显存通常约 7.2-7.4 GiB。未使用多进程 DataLoader，结束后没有遗留训练 Python。

正式运行分为两段：第 1-12 轮原使用 patience 10 并提前停止；检查初始结果接近随机后，保留模型、优化器、调度器和 AMP scaler 状态，以 patience 25 恢复第 13-40 轮。恢复逻辑没有保存 DataLoader generator 和完整 RNG 状态，因此第 13 轮后的样本打乱序列从固定 seed 重新开始；这次结果可由现有 checkpoint 复核，但不应宣称与一次不中断的 40 轮运行逐位等价。

40 轮 train+val 累计耗时 4660.88 秒，约 1.295 小时，平均每轮 116.5 秒。最佳第 19 轮的训练准确率为 48.10%，验证准确率为 54.44%、宏平均 F1 为 53.42%；第 40 轮训练准确率为 52.38%，验证宏平均 F1 降为 39.08%。训练集最高准确率为 56.90%，曲线显示拟合缓慢改善但验证方差较大。

## 5. 最终指标

| split | accuracy | balanced accuracy | macro precision | macro recall | macro F1 |
|---|---:|---:|---:|---:|---:|
| val | 54.44% | 54.44% | 54.64% | 54.44% | 53.42% |
| test | 66.67% | 66.67% | 70.90% | 66.67% | 66.11% |

验证集逐类指标：

| 树种 | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| Cinnamomum camphora | 61.29% | 63.33% | 62.30% | 30 |
| Lagerstroemia indica | 52.63% | 33.33% | 40.82% | 30 |
| Magnolia grandiflora | 50.00% | 66.67% | 57.14% | 30 |

测试集逐类指标：

| 树种 | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| Cinnamomum camphora | 66.67% | 73.33% | 69.84% | 30 |
| Lagerstroemia indica | 87.50% | 46.67% | 60.87% | 30 |
| Magnolia grandiflora | 58.54% | 80.00% | 67.61% | 30 |

验证混淆矩阵，行为真实类别，列为预测类别，顺序均为 class 0/1/2：

| true/pred | 0 | 1 | 2 |
|---|---:|---:|---:|
| 0 | 19 | 4 | 7 |
| 1 | 7 | 10 | 13 |
| 2 | 5 | 5 | 20 |

测试混淆矩阵：

| true/pred | 0 | 1 | 2 |
|---|---:|---:|---:|
| 0 | 22 | 2 | 6 |
| 1 | 5 | 14 | 11 |
| 2 | 6 | 0 | 24 |

紫薇是最弱类别：测试集中 30 株只有 14 株识别正确，11 株被判为广玉兰。测试指标高于验证指标可能来自两个 90 株小样本 split 的难度差异，不能据此推断模型在更广泛道路域上达到同等性能。

## 6. 产物与验收

正式运行目录：

`D:\TREE\lidar data\whu\derived\tscmdl\b2_pointmlp\run_batch3_seed20260714`

主要产物：

- `best.pt`：第 19 轮最佳检查点，约 50.69 MiB。
- `last.pt`：第 40 轮完整恢复检查点，约 101.20 MiB。
- `predictions.csv`：val/test 共 180 行逐样本概率和预测。
- `final_metrics.json`：最终结构化指标。
- `training_history.json/csv`：40 轮完整历史。
- `training_curves.png`：loss、accuracy、macro F1 曲线。
- `confusion_matrix_val.png`、`confusion_matrix_test.png`：行归一化混淆矩阵。
- `train.log`、`run_config.json`：控制台摘要与配置。

新增 `scripts/validate_b2_run.py`，并生成 `reports/B2_validation.json`。验收结果为 passed，检查内容包括：

- 必需文件完整，没有遗留 `.tmp`；
- history 为连续 1-40 轮，最佳轮次与 final metrics、best checkpoint 一致；
- best/last checkpoint 分别为第 19/40 轮，类别映射一致；
- predictions 共 180 行，val/test 各 90 行，无重复 sample key；
- 所有类别概率有限、非负、和为 1，预测类等于概率 argmax；
- 从 predictions 重新计算的全部指标和混淆矩阵与 final metrics 完全一致；
- 三张 PNG 通过解码、尺寸和人工可读性检查；
- 沙盒 20 项单元测试全部通过。

正式运行目录约 152.33 MiB。包含 batch 4 试跑和三个 smoke 目录在内，B2 全部派生结果约 304.25 MiB。没有自动删除试跑产物，以便审计；后续如需清理应单独确认。

## 7. 隔离与限制

- 原始 WHU-STree PLY、NPY、POS 和全景图保持只读。
- 仅使用 `D:\TREE\envs\whu-tscmdl`，未修改 `yolo` 或 `pdf-reading`。
- 未执行 Git 暂存、提交、推送或仓库全量扫描；最终检查时没有 `git.exe` 进程。
- 最终检查时没有 `train_b2_pointmlp.py` 训练进程。
- 只运行一个随机种子，尚不能估计随机方差。
- 三类各 200 株，验证和测试各类仅 30 株，指标置信区间较宽。
- 低点数实例通过重复采样补齐，紫薇受影响最明显。
- 单位球归一化丢失绝对尺度；本阶段也未使用 RGB、强度、真实尺寸或全景影像。
- B1 官方测试协议允许部分道路 ID 跨 train/test，结果不是严格道路域外泛化指标。

## 8. 下一阶段（未授权）

B3 为同一三类数据的 ResNet50 图像单模态训练。建议沿用 B2 的 split、seed、最佳验证宏平均 F1 选模、逐样本预测和统一验收格式，并将 B1 的影像遮挡质量标记纳入结果分析。B2 到此停止，等待用户明确批准后再开始 B3。
