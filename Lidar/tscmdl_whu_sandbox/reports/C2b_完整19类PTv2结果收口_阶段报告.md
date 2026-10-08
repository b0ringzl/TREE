# C2b 完整 19 类 PTv2 结果收口阶段报告

日期：2026-08-05

## 一、阶段范围

本阶段只对 C2 完整 19 类 PTv2 训练结果做回传验收、指标复算、类别诊断、
可视化和项目状态收口。未重新训练模型，未启动 C3 三随机种子复验，未修改
WHU-STree 原始数据、C1 共享资产、C2a kNN 缓存或既有模型权重。

权威运行目录：

```text
D:\TREE\lidar data\whu\derived\tscmdl\c2_ptv2\received_24gb_20260803\run
```

## 二、数据与模型合同

- 数据：19 类，train 13,116 株、val 570 株、test 3,448 株。
- 输入：每株 8,192 点，使用 C2a 的 8 邻居精确 kNN 缓存。
- 模型：PTv2 mode-1 classification encoder 原生 PyTorch 复现。
- 参数量：9,394,097。
- 训练策略：最多 120 轮，验证集选择最佳模型，耐心值 20；正式测试只在
  训练停止并锁定 `best.pt` 后执行。
- 实际完成 68 轮，最佳轮次为第 48 轮，随后因连续 20 轮无提升提前停止。

## 三、主要结果

| 划分 | 样本数 | loss | accuracy | macro F1 |
|---|---:|---:|---:|---:|
| val | 570 | 1.4500 | 69.82% | 68.80% |
| test | 3,448 | 1.1673 | 79.06% | 62.29% |

测试 accuracy 比 macro F1 高 16.77 个百分点。accuracy 受样本量大的类别主导，
而 macro F1 对 19 个类别等权，因此后续论文对照和模型选择应优先报告 macro F1，
不能只引用 79.06% 的总体准确率。

![C2 训练曲线](./C2_training_curves_readable.png)

## 四、类别诊断

测试集表现最强的五类为：

| 类别 | support | F1 |
|---|---:|---:|
| Metasequoia glyptostroboides | 65 | 95.24% |
| Platanus x acerifolia | 1,051 | 91.14% |
| Cinnamomum camphora | 749 | 90.97% |
| Lagerstroemia indica | 354 | 84.88% |
| Osmanthus fragrans | 173 | 83.55% |

测试集最弱的五类为：

| 类别 | support | F1 |
|---|---:|---:|
| Sophora japonica | 49 | 0.00% |
| Others | 49 | 25.00% |
| Malus x micromalus | 66 | 44.44% |
| Prunus cerasifera 'atropurpurea' | 102 | 46.74% |
| Ligustrum lucidum | 70 | 48.06% |

主要定向混淆包括：`Malus x micromalus -> Lagerstroemia indica` 30/66，
`Ligustrum lucidum -> Cinnamomum camphora` 27/70，
`Sophora japonica -> Zelkova serrata` 17/49。它们比单一总体分数更能指导下一步
采样、增强和融合模型设计。

![C2 测试集逐类指标](./C2_test_per_class_metrics.png)

![C2 测试集可读混淆矩阵](./C2_confusion_matrix_test_readable.png)

## 五、恢复过程与可复现性

- 日志记录到第 6、9、22、60 轮的四次断点恢复，历史轮次仍连续为 1 至 68。
- 最佳第 48 轮检查点产生于 batch 48、eval batch 64 的训练配置。
- 第 61 轮故障后使用 batch 40、eval batch 48 完成恢复和提前停止。
- 因训练过程中存在批量大小切换，本结果可作为已验收的单次 C2 基线，但不能
  代替严格同配置的三随机种子复验。后续推理必须使用 `best.pt`，不能用
  第 68 轮的 `last.pt` 代替。

## 六、机器验收

`scripts/validate_c2_run.py` 已独立完成以下检查：

- 20 个原始结果文件完整，无残留 `.tmp` 文件；运行状态和启动器退出码正常。
- `best.pt` 与 `last.pt` 均可加载，类别顺序、模型参数量和 kNN 缓存哈希一致。
- 4,018 条 val/test 预测键唯一，概率有限、非负、和为 1，预测类别等于概率最大值。
- 由 `predictions.csv` 重算的 accuracy、balanced accuracy、precision、recall、
  macro F1 和混淆矩阵与 `final_metrics.json` 完全一致。
- 验证集选模、正式测试延迟访问、提前停止和恢复配置均有记录。
- 原始图以及本阶段生成的四张可读 PNG、两个 CSV 和诊断 JSON 均通过格式、
  尺寸、行数与指标一致性检查。

机器验收结果：`reports/C2_validation.json`，状态 `passed`。

关键 SHA-256：

- `best.pt`：`0f99373406d73e27fa4a175353417fa46cf0bfeeaeae5194b4102a7bb4b9ac71`
- `last.pt`：`c4c774eae10c3c0a89720e24f8035cb0bd13e23a4c11760e85b5b175d5ba54a8`
- `final_metrics.json`：`ce3d3ef0f0d3bf1def931fb5fac7e3613fb14d4ebcb77a8a87f1731ad568a654`
- `predictions.csv`：`ebb832eebb0c7e9ee5715d23591d25ad6bb7ddc7308db72308035ace240ae788`

## 七、阶段产物

- 机器验收：`reports/C2_validation.json`
- 类别诊断：`reports/C2_diagnostics.json`
- 逐类指标：`reports/C2_per_class_metrics.csv`
- 定向混淆：`reports/C2_top_confusions.csv`
- 可读验证混淆矩阵：`reports/C2_confusion_matrix_val_readable.png`
- 可读测试混淆矩阵：`reports/C2_confusion_matrix_test_readable.png`
- 测试逐类指标图：`reports/C2_test_per_class_metrics.png`
- 可读训练曲线：`reports/C2_training_curves_readable.png`
- 验收脚本：`scripts/validate_c2_run.py`
- 诊断脚本：`scripts/analyze_c2_results.py`

## 八、风险与限制

1. 官方测试集类别不平衡，且不是严格道路互斥划分，79.06% accuracy 不能解释为
   对所有树种或新道路都有同等识别能力。
2. `Sophora japonica` 测试 F1 为 0，说明至少一个类别尚未形成可用识别能力。
3. 当前只有一个随机种子，无法给出均值、标准差或置信范围。
4. 本模型是保留 PTv2 核心结构的原生 PyTorch 复现，未使用论文的 Linux
   `pointops` 扩展，不能直接宣称等同于论文官方实现。
5. 当前输入为 WHU-STree 提供的单树实例，不包含自动单树分割误差。

## 九、磁盘变化

- 本阶段没有新增点云缓存或模型权重，也没有复制原始训练数据。
- 新增/更新的 8 个验收与诊断数据、图片文件合计约 1.148 MiB；阶段报告、
  状态文件和脚本仅增加少量文本体积。
- 回传运行目录原有 20 个文件约 152.09 MiB，本阶段只读验收，未改写。

## 十、下一阶段

建议 C3 使用固定且不再变化的 batch/eval batch 配置，完成三个随机种子的独立
训练与均值、标准差、逐类稳定性和论文指标对照；同时保留本次 C2 作为种子
`20260731` 的历史单次基线，不将混合恢复配置伪装成严格重复实验。

C3 当前仅标记为“待批准”。本报告完成后停止，不自动启动训练。
