# TSCMDL WHU-STree 当前状态

更新时间：2026-08-11T16:48:46+08:00

> 本文件由 `scripts/consolidate_project_status.py` 生成，是跨对话同步时的
> 当前状态入口。阶段授权仍以 `STAGE_GATES.md` 为准，请勿手工修改本文件。

## 当前结论

- 已完成阶段：20/21。
- 当前阶段门：C3：待批准。
- 三类实验已经完成 B1 数据、B2 PointMLP、B3 ResNet50、B5 PTv2 基线、B5b 三种子复验和 B5c 道路互斥协议；C1a/C1b/C1c/C1 已完成完整清单、双轨规划、冒烟和全量共享资产验收。
- C2a 已完成 19 类 PTv2 全量 kNN 缓存、断点续跑和 batch 16 GPU 前后向验收。
- C2 完整 19 类 PTv2 已训练完成，C2b 已完成回传文件、预测、概率、检查点和类别诊断验收。
- B4c 已完成测试前锁定配置的三种子正则化选择和模态消融；fusion 测试 macro F1 均值为 83.35%。
- 完整 19 类图像/融合基线、PTv2 多随机种子和自动单树分割尚未完成。
- 当前分类数据由 WHU-STree 真实 `tree ID` 提取，不是自动分割模型的输出。

## 基线结果

| 阶段 | 模型 | 最佳轮次 | val accuracy | val macro F1 | test accuracy | test macro F1 |
|---|---|---:|---:|---:|---:|---:|
| B2 | PointMLP | 19 | 54.44% | 53.42% | 66.67% | 66.11% |
| B3 | ImageNet-pretrained ResNet50 | 166 | 91.11% | 91.10% | 80.00% | 80.19% |
| B4 | TSCMDL frozen-backbone fusion | 19 | 80.00% | 79.93% | 83.33% | 83.32% |
| B4c | TSCMDL selected three-seed fusion | 10 | 82.22% | 82.30% | 84.44% | 84.42% |
| B5 | PTv2 mode-1 classification encoder reimplementation | 59 | 75.56% | 73.93% | 96.67% | 96.70% |
| C2 | PTv2 mode-1 classification encoder reimplementation | 48 | 69.82% | 68.80% | 79.06% | 62.29% |

## B5b 三种子复验

| split | accuracy mean ± std | macro F1 mean ± std | probability ensemble accuracy | ensemble macro F1 |
|---|---:|---:|---:|---:|
| val | 76.67% ± 0.91% | 75.83% ± 1.39% | 76.67% | 75.91% |
| test | 94.44% ± 1.81% | 94.46% ± 1.82% | 96.67% | 96.70% |

固定划分中，验证集 90/90 样本来自训练未见道路；测试集 89/90 样本来自训练已见道路。因此测试结果不能视为严格道路域外泛化。

## B5c 道路分组互斥协议

- 使用 29 条道路、4829 株非官方测试样本建立三折道路分组交叉验证。
- 1156 株官方参考测试样本未进入交叉验证。
- 每轮 train/val/test 道路交集为 0；验证和道路域外测试各固定 90 株平衡样本。
- 本阶段只完成协议与清单，尚未按新协议重新训练模型。

## C1a 完整 19 类清单与道路覆盖

- 完成 110/110 条轨迹、17134 株、19 类实例清单。
- 开发池 13686 株，官方参考测试池 3448 株。
- 严格三折、每折每类 30 株可联合评估 13 类；三折、每折每类 20 株可联合评估 16 类。
- 完整 19 类只有在两折、每折每类 10 株时全部满足道路分组约束；该方案不建议作为主要道路域外结论。
- 本阶段未导出点云/图像，未启动训练。

## C1b 双轨清单与磁盘预算

- 19 类基准轨道共 17134 株：train 13116、val 570、test 3448。
- 16 类道路域外轨道共 3 轮，每轮覆盖 13189 株开发样本，train/val/test 道路交集均为 0。
- 两条轨道引用同一份共享点云与图像路径，预计重复资产为 0 字节。
- 共享资产 P95 加 25% 安全余量约 5.72 GiB，规划时可用空间约 205.7 GiB。
- 本阶段未导出点云/图像，未启动训练。

## C1c 共享资产导出器冒烟验收

- 从 3 条开发轨迹导出 19 株，19 类各 1 株。
- 通过两次运行实测断点续传，第二次核验并复用 8 对既有资产。
- 最低投影裁剪可见率 99.89%；19 张已完成 Codex 视觉预检，无严重投影失败；用户复核待完成。
- 冒烟目录约 7.56 MiB，未启动全量导出或训练。

## C1 完整 19 类共享数据集

- 完成 110/110 条轨迹、17134 株、19 类点云与图像导出。
- 基准划分为 train 13116、val 570、test 3448；16 类道路清单共 39567 行。
- 共享资产约 3.896 GiB，完整目录约 4.005 GiB。
- 34,268 个物理资产逐一通过格式和哈希验收；38 张分层预览已完成 Codex 视觉预检，无严重投影失败；用户复核待完成。
- 本阶段未生成训练缓存或模型权重。

## C2a 19 类 PTv2 缓存与 GPU 冒烟

- 完成 17134 株、19 类、8192 点、8 邻居的分片内存映射缓存。
- 缓存占用 4.183 GiB，两次运行共 403.4 秒；已实测部分运行后续跑。
- 在 NVIDIA GeForce RTX 5060 Laptop GPU 上完成 batch 16 的前向、反向和参数更新；峰值实际分配显存 4402.5 MiB。
- 46 项测试通过；未写模型权重，未启动正式训练，冒烟未载入官方测试分片。

## C2/C2b 完整 19 类 PTv2 结果收口

- 训练完成 68 轮，最佳轮次 48；模型参数量 9,394,097。
- 验证集 accuracy 69.82%、macro F1 68.80%；测试集 accuracy 79.06%、macro F1 62.29%。
- 测试 accuracy 比 macro F1 高 16.77 个百分点，说明类别不均衡和弱类漏识别明显；正式比较应优先看 macro F1。
- 训练、断点、预测、概率、混淆矩阵和可读诊断图均已通过独立验收；后续推理使用验证集选出的 `best.pt`。

## B2/B3 逐样本互补性

`B2 only` 表示只有点云模型正确，`B3 only` 表示只有图像模型正确。
`either oracle` 是理想选择器在两个模型至少一个正确时选对的上限，不是可部署指标。

| split | 样本 | 两者都对 | B2 only | B3 only | 两者都错 | either oracle |
|---|---:|---:|---:|---:|---:|---:|
| val | 90 | 45 | 4 | 37 | 4 | 95.56% |
| test | 90 | 49 | 11 | 23 | 7 | 92.22% |

完整逐类别统计见 `reports/B2_B3_prediction_comparison.json`。

## 阶段门

| 阶段 | 范围 | 状态 |
|---|---|---|
| A0 | 建立隔离沙盒、固定来源和复现基线 | 已完成 |
| A1 | 创建 `whu-tscmdl` 环境，验证 PointMLP 依赖 | 已完成，原阻塞已由 A1b 解除 |
| A1b | 为旧版 `pointnet2_ops` 做本地 Windows/CUDA 12.8 兼容验证 | 已完成 |
| A2 | 实现 PLY/NPY 分块同步读取器 | 已完成 |
| A3 | 单轨迹单树提取与8192点采样 | 已完成 |
| A4 | 全景投影原型与小样本配对质检 | 已完成 |
| B1 | 三类共有树种数据集构建 | 已完成 |
| B2 | 三类 PointMLP 单模态训练 | 已完成 |
| B3 | 三类 ResNet50 单模态训练 | 已完成 |
| B4 | 三类 TSCMDL 融合训练 | 已完成（B4a+B4b+B4c） |
| B5 | 三类 PTv2 点云基线训练 | 已完成 |
| B5b | PTv2 三随机种子复验与固定道路域诊断 | 已完成 |
| B5c | 三类道路分组互斥交叉验证协议与可行性验收 | 已完成 |
| C1a | 完整19类实例清单与道路覆盖审计 | 已完成 |
| C1b | 19类基准与16类道路域外双轨清单及磁盘预算 | 已完成 |
| C1c | 共享资产导出器适配与小规模冒烟导出验收 | 已完成 |
| C1 | 完整19类共享资产全量导出与验收 | 已完成 |
| C2a | 19类PTv2缓存、数据加载与单批次GPU冒烟 | 已完成 |
| C2 | 完整19类基线训练 | 已完成 |
| C2b | 完整19类PTv2结果验收、诊断与状态收口 | 已完成 |
| C3 | 三随机种子复验和论文指标对照 | 待批准 |

## 权威产物

- 数据集：`D:\TREE\lidar data\whu\derived\tscmdl\b1_three_species\dataset`
- B2 正式运行：`D:\TREE\lidar data\whu\derived\tscmdl\b2_pointmlp\run_batch3_seed20260714`
- B3 正式运行：`D:\TREE\lidar data\whu\derived\tscmdl\b3_resnet50\run_batch8_seed20260723`
- B4a 冒烟运行：`D:\TREE\lidar data\whu\derived\tscmdl\b4a_fusion_smoke\run_seed20260727`
- B4a 验收：`reports/B4a_validation.json`
- B4 正式运行：`D:\TREE\lidar data\whu\derived\tscmdl\b4_tscmdl\run_batch8_l2_lr001_seed20260727`
- B4 验收：`reports/B4_validation.json`
- B2/B3/B4 对比：`reports/B2_B3_B4_prediction_comparison.json`
- B4c 三种子消融：`D:\TREE\lidar data\whu\derived\tscmdl\b4c_regularization_ablation\run_3seed_20260727`
- B4c 验收：`reports/B4c_validation.json`
- B5 PTv2 正式运行：`D:\TREE\lidar data\whu\derived\tscmdl\b5_ptv2\run_batch16_seed20260728`
- B5 验收：`reports/B5_validation.json`
- B2/B3/B4/B5 对比：`reports/B2_B3_B4_B5_prediction_comparison.json`
- B5b 三种子复验：`reports/B5b_validation.json`
- B5b 逐样本诊断：`reports/B5b_sample_diagnostics.csv`
- B5c 道路互斥协议：`D:\TREE\lidar data\whu\derived\tscmdl\b5c_road_group_cv\road_cv_protocol.json`
- B5c 机器验收：`D:\TREE\lidar data\whu\derived\tscmdl\b5c_road_group_cv\validation.json`
- C1a 完整 19 类清单：`D:\TREE\lidar data\whu\derived\tscmdl\c1a_full_inventory\inventory.json`
- C1a 道路覆盖审计：`D:\TREE\lidar data\whu\derived\tscmdl\c1a_full_inventory\audit\c1a_audit.json`
- C1a 机器验收：`D:\TREE\lidar data\whu\derived\tscmdl\c1a_full_inventory\audit\validation.json`
- C1b 双轨规划：`D:\TREE\lidar data\whu\derived\tscmdl\c1b_dual_track_plan\dual_track_plan.json`
- C1b 磁盘预算：`D:\TREE\lidar data\whu\derived\tscmdl\c1b_dual_track_plan\disk_budget.json`
- C1b 机器验收：`D:\TREE\lidar data\whu\derived\tscmdl\c1b_dual_track_plan\validation.json`
- C1c 冒烟 manifest：`D:\TREE\lidar data\whu\derived\tscmdl\c1c_shared_asset_smoke\manifest.json`
- C1c 视觉总览：`D:\TREE\lidar data\whu\derived\tscmdl\c1c_shared_asset_smoke\smoke_contact_sheet.jpg`
- C1c 机器验收：`D:\TREE\lidar data\whu\derived\tscmdl\c1c_shared_asset_smoke\validation.json`
- C1 全量共享 manifest：`D:\TREE\lidar data\whu\derived\tscmdl\c1_full_shared_dataset\shared_manifest.json`
- C1 19 类基准 manifest：`D:\TREE\lidar data\whu\derived\tscmdl\c1_full_shared_dataset\benchmark_19_manifest.csv`
- C1 16 类道路 manifest：`D:\TREE\lidar data\whu\derived\tscmdl\c1_full_shared_dataset\road_domain_16_manifest.csv`
- C1 全量验收：`D:\TREE\lidar data\whu\derived\tscmdl\c1_full_shared_dataset\validation.json`
- C2a 全量 kNN 缓存验收：`D:\TREE\lidar data\whu\derived\tscmdl\c2a_ptv2\knn_memmap_k8\validation.json`
- C2a GPU 单批次结果：`D:\TREE\lidar data\whu\derived\tscmdl\c2a_ptv2\smoke_result.json`
- C2a 阶段验收：`D:\TREE\lidar data\whu\derived\tscmdl\c2a_ptv2\validation.json`
- C2 正式运行：`D:\TREE\lidar data\whu\derived\tscmdl\c2_ptv2\received_24gb_20260803\run`
- C2 最佳检查点：`D:\TREE\lidar data\whu\derived\tscmdl\c2_ptv2\received_24gb_20260803\run\best.pt`
- C2/C2b 机器验收：`D:\TREE\Lidar\tscmdl_whu_sandbox\reports\C2_validation.json`
- C2b 类别诊断：`D:\TREE\Lidar\tscmdl_whu_sandbox\reports\C2_diagnostics.json`
- C2b 阶段报告：`D:\TREE\Lidar\tscmdl_whu_sandbox\reports\C2b_完整19类PTv2结果收口_阶段报告.md`
- 实验登记：`experiment_registry.json`
- 环境锁：`environment/whu-tscmdl-environment.yml`
- B2 报告：`reports/B2_PointMLP三类训练_阶段报告.md`
- B3 报告：`reports/B3_ResNet50三类图像训练_阶段报告.md`
- B4 报告：`reports/B4_TSCMDL三类融合训练_阶段报告.md`
- B4c 报告：`reports/B4c_融合正则化与模态消融_阶段报告.md`
- B5 报告：`reports/B5_PTv2三类点云训练_阶段报告.md`
- B5b 报告：`reports/B5b_PTv2三种子与道路域诊断_阶段报告.md`
- B5c 报告：`reports/B5c_道路分组互斥划分协议_阶段报告.md`
- C1a 报告：`reports/C1a_完整19类清单与道路覆盖审计_阶段报告.md`
- C1b 报告：`reports/C1b_双轨清单与磁盘预算_阶段报告.md`
- C1c 报告：`reports/C1c_共享资产导出器冒烟验收_阶段报告.md`
- C1 报告：`reports/C1_完整19类共享资产全量导出_阶段报告.md`
- C2a 报告：`reports/C2a_19类PTv2缓存与GPU冒烟_阶段报告.md`
- C2b 报告：`reports/C2b_完整19类PTv2结果收口_阶段报告.md`

完整 SHA-256、配置和验收文件路径保存在 `experiment_registry.json`。

## 已知风险

- B2 的 batch size 3 与多层 BatchNorm 导致训练和验证波动较大。
- B3 验证与测试存在明显道路、视角或图像质量域差异。
- B2/B3 都只有一个随机种子。
- B5b 三种子验证 macro F1 为 75.83% ± 1.39%，测试为 94.46% ± 1.82%。
- 固定划分的验证集全部来自训练未见道路，测试集 89/90 来自训练已见道路；
  当前测试结果主要反映同道路分布性能，不是严格道路域外泛化。
- B5 是保留 PTv2 核心模块的原生 PyTorch 复现，未使用官方 Linux `pointops` 扩展。
- 图像可能包含邻树、车辆、天空和道路背景线索。
- 低点数树通过重复采样补齐，紫薇受影响最明显。
- B5c 已建立严格道路互斥协议，但尚未按该协议重训模型。
- C1a 证明完整 19 类不能统一满足严格三折道路评估；标签 17 只有 2 条开发道路。
- C1b 的 19 类轨道适合官方基准对照，但不保证道路互斥；16 类道路轨道
  不覆盖标签 12、16、17。
- C1c 的 Codex 视觉预检中有 4/19 张存在邻树、杆件或背景干扰；用户复核尚未完成；分类裁剪不能当作
  精确图像分割结果。
- C1 的 Codex 视觉预检中有 11/38 张存在遮挡、模糊或背景干扰；用户复核尚未完成。
- C2a 证明 PTv2 batch 16 能运行，但单批次损失不代表收敛性或精度。
- C2 测试 accuracy 79.06%，但 macro F1 仅 62.29%；Sophora japonica 测试 F1 为 0，类别不均衡和弱类混淆仍是主要风险。
- 完整 19 类 ResNet50 和 TSCMDL 尚未正式训练。

## 下一建议

C2/C2b 已结束。下一阶段建议执行 C3 三随机种子复验和论文指标对照；C3 当前仅为待批准状态，不会自动启动。

## 重新生成

```powershell
D:\TREE\envs\whu-tscmdl\python.exe .\scripts\consolidate_project_status.py `
  --dataset-root "D:\TREE\lidar data\whu\derived\tscmdl\b1_three_species\dataset" `
  --b2-run "D:\TREE\lidar data\whu\derived\tscmdl\b2_pointmlp\run_batch3_seed20260714" `
  --b3-run "D:\TREE\lidar data\whu\derived\tscmdl\b3_resnet50\run_batch8_seed20260723" `
  --b4a-run "D:\TREE\lidar data\whu\derived\tscmdl\b4a_fusion_smoke\run_seed20260727" `
  --b4-run "D:\TREE\lidar data\whu\derived\tscmdl\b4_tscmdl\run_batch8_l2_lr001_seed20260727" `
  --b4c-run "D:\TREE\lidar data\whu\derived\tscmdl\b4c_regularization_ablation\run_3seed_20260727" `
  --b5-run "D:\TREE\lidar data\whu\derived\tscmdl\b5_ptv2\run_batch16_seed20260728" `
  --b5b-validation ".\reports\B5b_validation.json" `
  --b5c-validation "D:\TREE\lidar data\whu\derived\tscmdl\b5c_road_group_cv\validation.json" `
  --c1a-validation "D:\TREE\lidar data\whu\derived\tscmdl\c1a_full_inventory\audit\validation.json" `
  --c1b-validation "D:\TREE\lidar data\whu\derived\tscmdl\c1b_dual_track_plan\validation.json" `
  --c1c-validation "D:\TREE\lidar data\whu\derived\tscmdl\c1c_shared_asset_smoke\validation.json" `
  --c1-validation "D:\TREE\lidar data\whu\derived\tscmdl\c1_full_shared_dataset\validation.json" `
  --c2a-validation "D:\TREE\lidar data\whu\derived\tscmdl\c2a_ptv2\validation.json" `
  --c2-run "D:\TREE\lidar data\whu\derived\tscmdl\c2_ptv2\received_24gb_20260803\run" `
  --c2-validation ".\reports\C2_validation.json"
```
