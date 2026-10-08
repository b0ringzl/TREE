# A1b Windows/CUDA 兼容验证阶段报告

完成日期：2026-07-14（Asia/Shanghai）

## 阶段结论

A1b 已按批准范围完成，结论为“通过”。

旧版 `pointnet2_ops` 已在隔离环境 `D:\TREE\envs\whu-tscmdl` 中成功编译，生成面向 RTX 5060 Laptop GPU 计算能力 12.0 的 Windows 原生扩展。FPS 算子、PointMLP 1024 点前向传播和 PointMLP 8192 点前向传播均通过 GPU 冒烟测试，A1 中记录的扩展阻塞已经解除。

## 本阶段范围

- 仅修改隔离沙盒及其被忽略的 PointMLP vendor 副本。
- 不修改系统级 PyTorch、CUDA Toolkit 或 Visual Studio 安装。
- 不修改 `yolo` 和 `pdf-reading` conda 环境。
- 不安装或启用 WSL2。
- 不读取、修改或扫描 WHU-STree 原始点云和影像负载。
- 不进入 A2 数据读取器实现。

## 兼容改动

1. 将上游两处旧 CUDA 架构硬编码改为可由环境变量覆盖的 `12.0` 默认值。
2. 在 `setup.py` 中加入显式的 `POINTNET2_ALLOW_UNSUPPORTED_COMPILER=1` 开关，用于向 NVCC 传入 `-allow-unsupported-compiler`。
3. 新增 `scripts/build_pointnet2_windows.py`，只在当前进程内将 PyTorch 编译器输出解码改为 UTF-8 容错模式，不修改环境内的 `cpp_extension.py`。
4. 新增 `scripts/build_pointnet2_windows.ps1`，只在子进程中加载 VS、CUDA 和 Conda 环境路径。
5. 将 vendor 源码差异固化为 `patches/pointnet2_ops_windows_cuda128.patch`。

## 编译过程与结果

修正 A1 中的本地化 `cl.exe` 输出解码问题后，构建成功进入真实 C++/CUDA 编译。未启用兼容开关时，CUDA 12.8 明确拒绝 Visual Studio 2026，并提示其官方支持范围为 VS 2017 至 2022。

随后仅对本次沙盒构建显式启用 NVIDIA 提供的 `-allow-unsupported-compiler` 开关。9 个 C++/CUDA 对象完成编译并成功链接：

```text
pointnet2_ops/_ext.cp310-win_amd64.pyd
size: 321536 bytes
sha256: 8A63C5E05F5E177BDB2361D360250A4D1F7201909FF261ACDC4C9F9EA14ECBAD
```

`cuobjdump -lelf` 显示扩展包含 4 个 `sm_120` cubin，证明目标 GPU 架构已写入二进制。完整成功构建日志位于 `reports/A1b_pointnet2_build.log`，日志约 2.4 MiB，并受 `.gitignore` 排除。

## GPU 验证结果

| 验证项 | 输入 | 结果 | 峰值分配显存 |
|---|---|---|---:|
| FPS CUDA 算子 | `2×256×3`，采样 64 点 | `2×64`，int32，索引范围 0 至 255 | 未单独计量 |
| PointMLP | `1×3×1024` | `1×40`，全部为有限值 | 94.8 MiB |
| PointMLP | `1×3×8192` | `1×40`，全部为有限值 | 347.9 MiB |

本次记录的前向时间分别为 0.3629 秒和 0.2199 秒。两次运行的 CUDA 预热条件不同，因此这些数字只证明可运行，不能用于比较 1024 点和 8192 点性能。结构化结果保存在 `reports/A1b_pointmlp_validation.json`。

## 风险与限制

- CUDA 12.8 未正式声明支持 Visual Studio 2026。虽然编译、链接和 GPU 前向传播均已通过，后续正式训练仍需监测长时间稳定性和数值异常。
- 生成的 `.pyd` 与 Windows x64、Python 3.10、PyTorch 2.11.0+cu128 和当前编译工具链绑定；升级任一关键组件后应重新编译和复测。
- 原扩展仍是较旧实现，本阶段只解决构建与运行兼容，不重写算子，也不声称完成训练复现。

## 资源与隔离检查

- `pointnet2_ops_lib/build` 约占 238.4 MiB。
- A1 结束时 D 盘可用约 78.9 GiB，本阶段结束时约 78.7 GiB。
- C 盘可用约 17.4 GiB。
- 阶段结束时未发现 `git/python/pip/conda/ninja/nvcc/cl` 残余进程。
- 未启动训练和持续监控任务。
- 本阶段未提交或推送 Git。

## 下一阶段建议

建议下一小阶段进入 A2：实现 PLY/NPY 分块同步读取器。A2 应先使用极小范围样本验证字段、块边界和同步关系，不做全量数据转换，不启动训练。

本报告不构成 A2 的自动执行授权，需等待用户明确批准。
