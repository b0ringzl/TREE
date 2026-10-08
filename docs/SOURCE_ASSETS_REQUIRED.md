# 本地资产与许可证

Git 版本仅保存代码和小型配置。下面的模型、图片和运行时不包含在源码提交中；新电脑需要另行恢复已授权的本地文件。

| 本地路径 | 用途 |
|---|---|
| `external_datasets/distillation_models/bioclip25/` | BioCLIP 2.5 编码器、模型配置与 tokenizer；编码器 SHA-256 `ac2e37c2f89ef8e6b889176a9a3f418970ad9db15a218bd29e3321e95c46ae97` |
| `tree_species_teacher_demo/models/SSTv1_Aquilaria_UserCrops_20260919_H120/member_1.pt` 等三份权重 | 当前香港 SST 分类头；标签、部署配置与哈希保留在 Git 中，权重不公开分发 |
| `outputs/distillation/bioclip_probe_20260908_0227/` | `classes.json`、`protocol.json`、`cache_metadata.json`、`teacher_choice.json`、`selection_lock.json` 及现有适配模型所引用的元数据 |
| `outputs/distillation/20260908_0045/` | 原始类别别名与映射，包括 `aliases.json`；保持原哈希与类序 |
| `external_datasets/distillation_runtime/` | 本地 OpenCLIP / 检测组件运行时，或按对应版本正常安装；不得把整个虚拟环境提交到 Git |
| `external_datasets/bioclip25_flora_20261008_v1/` | 世界模型植物文本特征、目录、manifest 与源文件锁定；约 34.9 万条目，不等于全球准确率 |
| `LabelPaw-web-images/weights/sam_weights/sam2.1_hiera_tiny.pt` | 网页 SAM 2.1 Tiny 分割权重；相关组件依赖另装 |
| `spacescan_trees_demo_v1/models/ptv2_seven_20260913_v1/` | 七类点云分类器 `model.joblib`；编码器在 `experiments/whu18_yolo11_ptv2_fusion_20260826/ptv2/run/best.pt` |
| `outputs/tree_detector_benchmark_20260909_v1/` 等检测模型路径 | YOLO / Grounding DINO / 背景过滤模型，实际默认路径见 `spacescan_trees_demo_v1/tree_detection.py` |
| `hk-tree-vmms/outputs/coordinates/frame_coordinates.csv` 与仓库相邻的 `vmms/` 文件夹 | 内置 VMMS 路线目录和全景原图；未入库。读取用户自己的文件夹不等于附送该内置路线数据 |
| `tree_species_demo/samples.json` 与 `test_data/` | 本地测试示例；代码快照提供空清单，不发布原图，也不伪造示例推理 |
| `external_datasets/spacescan_whole_tree_references_20260910_v2/` 等 | 参考图库与审核索引；图片保留原作者、来源、水印及许可证，仅在授权范围使用 |
| `spacescan_android_sstv1/app/src/main/assets/` | 安卓 ONNX 模型与示例图；许可证文本和小型物种配置保留，模型和图片不入库 |

元数据如引用原机器的绝对路径，新电脑需调整为自己的资产路径，但不要修改模型哈希、输出类别顺序或用另一模型替换后仍沿用旧版本名称。

模型准备脚本涉及下载时应遵守上游许可；训练与模型切换前遵守 `AGENTS.md` 的物理备份和固定评估要求。当前 H120 训练数据及权重的公开再分发权未被本次 Git 提交批准。

云端导入只需自己的公开客户端 ID / Dropbox Chooser App key；不要把 App secret、访问令牌、个人账户会话、`.env`、Android 签名私钥或学校账户凭据放进仓库。桌面服务仅监听 `127.0.0.1`。

旧 Demo、研究工具和第三方包各有自己的依赖/许可证；本目录不是统一的商业再分发授权。完整离线包应在本地单独交付，不混进 Git 源码仓库。
