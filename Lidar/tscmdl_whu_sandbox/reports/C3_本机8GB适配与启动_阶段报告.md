# C3 本机 8GB 适配与启动阶段报告

## 1. 阶段结论

- C3 已适配本机 `NVIDIA GeForce RTX 5060 Laptop GPU (8151 MiB)`，并于 2026-08-10 15:24 启动正式训练。
- 固定训练配置为 `batch_size=20`、`eval_batch_size=64`、AMP 开启、`workers=0`。
- 三个随机种子 `20260728 / 20260729 / 20260730` 串行运行；每个种子从头训练，只有同一随机种子的 `last.pt` 才允许断点续训。
- 24GB 迁移包 `D:\TREE\Lidar\C3_PTv2_3Seed_Portable_20260805_READY` 未修改，可继续作为异机训练后备方案。

## 2. 本机批量探测

所有比较均使用相同数据、缓存、随机种子和 AMP，执行 10 个真实优化步骤。

| batch | 训练峰值显存（PyTorch allocated） | 训练吞吐 | 结论 |
| ---: | ---: | ---: | --- |
| 16 | 4280.8 MiB | 29.96 样本/秒 | 稳定，但低于 batch 20 |
| 20 | 5088.7 MiB | 33.80 样本/秒 | 本机最快，正式采用 |
| 24 | 6050.7 MiB | 15.21 样本/秒 | 显存压力导致吞吐明显下降，弃用 |

验证批量 `64` 的峰值为 2823.1 MiB。此前 `batch=24` 的正式短跑已完整归档到：

`D:\TREE\lidar data\whu\derived\tscmdl\c3_ptv2_repeats\output\runs\aborted_batch24_20260810_152049`

该目录没有训练检查点，不参与后续汇总。

## 3. 正式训练配置

- 模型：PTv2 mode-1 classification encoder reimplementation
- 数据集：`D:\TREE\lidar data\whu\derived\tscmdl\c1_full_shared_dataset`
- kNN 缓存：`D:\TREE\lidar data\whu\derived\tscmdl\c2a_ptv2\knn_memmap_k8`
- 输出根目录：`D:\TREE\lidar data\whu\derived\tscmdl\c3_ptv2_repeats`
- 每个种子最大轮数：120
- Early stopping：验证指标连续 20 轮不提升时停止
- 学习率：0.001，最低学习率 0.00001
- Weight decay：0.05
- Label smoothing：0.1
- 测试集策略：只在验证集选择出最佳检查点后评估锁定测试集

配置文件：

`D:\TREE\lidar data\whu\derived\tscmdl\c3_ptv2_repeats\c3_config.json`

## 4. 启动验收

正式启动后观察到：

- 活跃种子：`20260728`（第 1/3 个）
- 已稳定运行至 epoch 1 的 batch 120/656
- GPU 利用率：99%
- GPU 总显存占用：7784/8151 MiB
- 温度：59°C
- 当前单轮估计：约 9 分钟（含验证会有波动）
- Git 后台进程：0
- 训练进程结构：1 个 PowerShell 包装器 + 1 个 Python 训练进程

按当前速度，单个种子跑满 120 轮约 18 小时，三个种子的理论上限约 54 小时。Early stopping 可能缩短实际耗时。

## 5. 使用入口

启动或断点恢复：

`D:\TREE\Lidar\tscmdl_whu_sandbox\start-c3-local-training.bat`

只打开实时监控：

`D:\TREE\Lidar\tscmdl_whu_sandbox\monitor-c3-local.bat`

命令行单次查看：

```powershell
& "D:\TREE\envs\whu-tscmdl\python.exe" "D:\TREE\Lidar\tscmdl_whu_sandbox\scripts\monitor_c3.py" --root "D:\TREE\lidar data\whu\derived\tscmdl\c3_ptv2_repeats" --once
```

## 6. 中断与恢复规则

- 正常关机或暂停前，应先停止当前 Python 训练进程，避免检查点正在写入时强制断电。
- 每轮结束会更新 `last.pt`；重新运行启动脚本时，同一随机种子将从该检查点继续。
- 已生成 `final_metrics.json` 且状态为 `complete` 的种子会自动跳过。
- 三个种子结束后会生成 C3 汇总文件，报告均值、标准差以及类别稳定性。

## 7. 本阶段文件

- `scripts\run_c3_local.ps1`：本机三随机种子串行训练与恢复
- `scripts\start_c3_local.ps1`：设备检查、防重复启动和监控入口
- `scripts\monitor_c3.py`：实时训练监控
- `scripts\summarize_c3_results.py`：三随机种子结果汇总
- `start-c3-local-training.bat`：一键启动/恢复
- `monitor-c3-local.bat`：一键监控

本阶段未执行任何 Git 提交、推送或仓库级扫描。
