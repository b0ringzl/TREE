# A1 隔离环境与 PointMLP 依赖验证阶段报告

完成日期：2026-07-14（Asia/Shanghai）

## 阶段结论

A1 已按范围执行完毕，结果为“基础环境通过，PointMLP CUDA 扩展阻塞”。

隔离环境、稳定版 PyTorch、RTX 5060 GPU 计算、通用科学计算依赖和上游源码固定均已成功。PointMLP 依赖的旧版 `pointnet2_ops` 没有成功编译，因此当前环境还不能导入 PointMLP，也没有进入数据读取或训练阶段。

## 已完成事项

- 在 D 盘创建隔离 conda 环境：`D:\TREE\envs\whu-tscmdl`。
- 安装 Python 3.10.20。
- 从 PyTorch 官方 CUDA 12.8 索引安装稳定版 PyTorch 2.11.0 和 torchvision 0.26.0。
- 安装 `numpy/scipy/scikit-learn/h5py/einops/PyYAML/matplotlib/ninja/cmake` 等基础依赖。
- 验证 RTX 5060 Laptop GPU 可由 PyTorch 正确识别。
- 验证 GPU compute capability 为 `(12, 0)`。
- 完成 2048×2048 CUDA 矩阵乘法并同步返回结果。
- 将 PointMLP 固定提交克隆到被忽略的 `vendor` 目录。
- 确认 vendor HEAD 为 `1ab7f04e88d2aeb8fd16921a2b2fc4fba950b6df`。
- 确认上游 tracked 文件没有本地修改。
- 多次有限诊断 Visual Studio、CUDA_HOME、PATH 和 Python 编码问题，未循环重试。

## 环境版本

| 项目 | 版本或状态 |
|---|---|
| 环境路径 | `D:\TREE\envs\whu-tscmdl` |
| Python | 3.10.20 |
| PyTorch | 2.11.0+cu128 |
| torchvision | 0.26.0+cu128 |
| CUDA runtime | 12.8 |
| CUDA Toolkit | 12.8.93 |
| GPU | RTX 5060 Laptop GPU，8151 MiB |
| GPU capability | 12.0 |
| MSVC | 19.50.35730，x64 |
| ninja | 1.13.0 |
| cmake | 4.4.0 |

完整 Python 包清单保存于 `environment/requirements-a1.txt`。

## 成功验证

PyTorch GPU 验证结果：

```text
torch 2.11.0+cu128
torchvision 0.26.0+cu128
cuda 12.8 True
gpu NVIDIA GeForce RTX 5060 Laptop GPU
capability (12, 0)
matmul (2048, 2048)
```

该结果说明显卡、驱动、PyTorch CUDA runtime 和基本 GPU 运算链正常。

## PointMLP 扩展阻塞

PointMLP 的分类模型直接导入：

```python
from pointnet2_ops import pointnet2_utils
```

上游 `pointnet2_ops_lib/setup.py` 存在明显的旧环境假设：

- CUDA 架构硬编码为 `3.7/5.0/6.x/7.x`，不包含 RTX 5060 的 capability 12.0。
- 上游环境面向旧版 PyTorch 和 CUDA。
- 构建依赖 PyTorch `cpp_extension` 解析 MSVC 版本输出。

在确认 `cl.exe` 和 `nvcc.exe` 均可找到后，当前最深层错误为：

```text
UnicodeDecodeError: 'cp1' codec can't decode bytes ...
```

错误发生在 PyTorch `cpp_extension.py` 检查本地化 MSVC 版本输出时，尚未进入 CUDA 源码编译。由于扩展没有安装，PointMLP 导入结果为：

```text
ModuleNotFoundError: No module named 'pointnet2_ops'
```

这不是 GPU 不可用，也不是 CUDA Toolkit 缺失，而是旧扩展构建脚本、PyTorch 2.11 和本地化 Windows 编译输出之间的兼容问题。

## 隔离性检查

- `yolo` 环境仍为 `2.11.0.dev20260210+cu128`，CUDA 可用，未被修改。
- `pdf-reading` 环境未修改。
- WSL2 未安装、未启用。
- WHU-STree 原始 PLY、NPY、影像和 POS 文件未读取负载、未修改。
- 没有启动训练或后台进程。
- 没有修改 PointMLP 上游 tracked 文件。

## 磁盘变化

| 项目 | A0 | A1结束 |
|---|---:|---:|
| C盘可用空间 | 17.7 GiB | 17.3 GiB |
| D盘可用空间 | 83.7 GiB | 78.9 GiB |
| `whu-tscmdl` 环境 | 不存在 | 约 4.7 GiB |

安装使用 `--no-cache-dir`，没有额外保留约2.8GB的 PyTorch wheel 缓存。

## 建议的下一小阶段

建议新增 A1b：原生 Windows 兼容验证，而不是立即进入 A2。

A1b 仅执行以下动作：

1. 在沙盒中为 MSVC 版本检查提供 ASCII 输出兼容层，不修改系统全局设置。
2. 将旧扩展的 CUDA 架构列表改为 capability 12.0，并以独立补丁文件记录差异。
3. 尝试编译 `pointnet2_ops`。
4. 若安装成功，验证 FPS 算子和 PointMLP 的小输入、8192点 GPU 前向传播。
5. 若进入源码编译后仍因 PyTorch API 或 CUDA 代码不兼容而失败，立即停止，再请示选择 WSL2 或纯 PyTorch FPS 替代方案。

A1b 不修改 `yolo`，不安装 WSL2，不读取 WHU 点云负载。本报告不构成 A1b 的自动执行授权。

