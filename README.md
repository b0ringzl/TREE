# Tree Species Identification — current Demo source

截至 2026-10-08 的代码快照。主要入口为桌面 SpaceScan Trees 与离线安卓 SST。

## 当前功能

- 桌面单源影像：自动选树、SAM 智能裁剪、多边形裁剪、参考图片浏览、候选结果与记录导出。
- 香港范围默认使用 SSTv1-H120-DV：148 个原始模型输出，146 个展示类别。洋紫荊三类合并展示。
- 世界模型：独立植物目录候选。范围外或不确定结果弹窗提供查看世界模型的选项；用户确认后切换，可返回同一次 SST 结果，不自动切换、不混合两套模型分数。
- OneDrive / Dropbox：配置自己的公开应用 ID / Chooser App key 后，只导入用户选择的图片，不写入云端原件。
- 视频：导入、方向处理、采样、树木检测、多目标跟踪、多视角选择与识别结果导出。
- VMMS 走廊：连续全景文件夹选择、连续照片范围与距离采样、透视重投影、跨帧关联及逐帧处理结果浏览。
- 双源照片入口：同株图片与点云辅助分析，保留独立的七类点云流程，不代表 146 类点云融合。
- 安卓 SST：离线识别、轻量 SAM 选树、简约界面、物种范围与参考图浏览、低分不确定提示。

## 入口

| 用途 | 文件 / URL |
|---|---|
| 桌面单源 | `START_SPACESCAN_IMAGE.cmd` / `http://127.0.0.1:8043/image-only` |
| 桌面双源 | `START_SPACESCAN_MULTIMODAL.cmd` / `http://127.0.0.1:8043/multimodal` |
| 视频 | `http://127.0.0.1:8043/video` |
| VMMS 走廊 | `http://127.0.0.1:8043/corridor` |
| 应用版本页 | `http://127.0.0.1:8043/apps` |
| 安卓当前源码 | `spacescan_android_sstv1/` |
| 历史安卓 / 识别 Demo | `spacescan_android/`、`tree_species_demo/` |

## 安装与本地运行

这不是包含模型的大型离线安装包。Git 仅保存源码、测试、启动脚本、非敏感配置及必要说明；不上传私有训练权重、原图、点云、用户审核数据库或 APK。

1. 使用 Python 3.10/3.11，按设备安装 CUDA 兼容的 PyTorch / torchvision。
2. 安装 `requirements-demo.txt`。历史研究工具的单独依赖见对应目录。
3. 按 `ASSETS_REQUIRED.md` 恢复已授权的本地模型和元数据；不能仅凭 Git 源码获得当前 H120 权重。
4. Windows 可设置 `SPACESCAN_PYTHON` 为 Python 可执行文件，再使用上述 CMD 入口。当前 BioCLIP 服务需要 NVIDIA CUDA。
5. 未使用启动脚本时，在仓库根目录分别运行：

```powershell
python -m uvicorn tree_species_teacher_demo.server:app --host 127.0.0.1 --port 8039
python -m uvicorn spacescan_trees_demo_v1.server:app --host 127.0.0.1 --port 8043
```

仅绑定本机环回地址。不要将未经加固的 Demo 暴露到公网。云端导入凭据使用环境变量，不能提交 App secret、令牌、学校账户会话或私钥。

## 验证与限制

```powershell
python -m unittest spacescan_trees_demo_v1.test_flora_scope tree_species_teacher_demo.test_flora spacescan_trees_demo_v1.test_sam_single_view spacescan_trees_demo_v1.test_presentation_groups
```

世界模型是候选目录扩展，不是已验证的全球物种准确率。相关小测试与已知回归见 `docs/validation/`。原始模型版本、哈希与识别来源仍保留在代码及导出记录中。

历史便携 Windows 包、安卓旧包和当前网页的功能不完全相同；不把归档包冒称为最新版。历史研究脚本可能引用原机器路径，运行前需调整配置，恢复模型/数据并遵守 `AGENTS.md` 的备份要求。

新数据、参考图片、模型和第三方组件保留各自许可证；源码入库不等于有权再次分发原始图片或实验权重。
