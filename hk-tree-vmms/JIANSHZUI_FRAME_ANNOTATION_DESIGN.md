# 尖沙咀全景影像逐帧人工标注项目

## 1. 标注对象

- 数据来源：`C:\Users\25041725G\Desktop\receive\vmms\20250210-jianshazui\ladybug`
- 全景流 `jianshazui_pano_1`：3,299 帧
- 全景流 `jianshazui_pano_2`：7,421 帧
- 原始影像合计：10,720 帧
- 原始 VMMS 目录只读，预览缓存和标注结果写入项目 `derived` 目录。

两条影像流存在相同的六位 `frame_id`，系统以 `stream_id__frame_id` 作为内部唯一键，避免记录、草稿和标签文件互相覆盖。导出时仍同时保留原始 `stream_id`、`frame_id`、时间、坐标和航向。

## 2. 标注流程

1. 按 0、2、5 或 10 米间隔，从每条影像流分别抽帧；默认 5 米。
2. 一帧可圈选任意数量的树，每棵树可选择不同树种。
3. 列表中没有的树种可以现场添加；无法确认时选择 `Unknown / 待定` 并在单树备注中记录候选树种。
4. 影像过亮、过暗或对比度不足时，可交互调整曝光和对比度，也可采用自动建议。
5. 调整参数、质量指标和人工接受状态与标注一同保存，供后续机器识别前处理实验使用；不会覆盖原图。
6. 当前经纬度、车辆航向可直接跳转 Google Street View 做外部核对。
7. 编辑停止 0.9 秒后自动保存；项目进度以“存在至少一个人工轮廓”为有效帧口径，不依赖提交状态。没有人工轮廓的帧不进入训练集。
8. 新建轮廓及顶点修正结束时会自动解开边界自交，避免生成无效的 YOLO segmentation 多边形。

## 3. 输出结构

根目录：`derived\jianshazui_frame_labeler`

```text
annotations/
  classes.json
  dataset_manifest.csv
  records/
    jianshazui_pano_1/<frame_id>.json
    jianshazui_pano_2/<frame_id>.json
  labels/
    jianshazui_pano_1/<frame_id>.txt
    jianshazui_pano_2/<frame_id>.txt
preview_cache/
  jianshazui_pano_1/<frame_id>.jpg
  jianshazui_pano_2/<frame_id>.jpg
runtime/
  frame_drafts.json
  frame_review_state.json
  session.json
```

`labels` 使用 YOLO segmentation 多边形格式；`records` 保存树种、轮廓、可见性、备注、影像处理参数、坐标和时间等完整信息。

双击 `导出尖沙咀有效YOLO数据.cmd` 可把 `frame_drafts.json` 优先、`frame_review_state.json` 兜底的全部有效轮廓导出到：

```text
derived/jianshazui_frame_labeler/training_exports/effective_yolo_<时间>/
```

导出包含 `data.yaml`、`class_map.json`、`manifest.csv`、完整记录以及 `train/val/test` 图像和标签。图像采用硬链接，原始影像不会被复制或改写。YOLO 数据集使用仅覆盖本段实际树种的紧凑类别编号，同时由 `class_map.json` 保留项目全局类别编号。

## 4. 数据集划分约束

相邻全景帧高度相关，后续训练、验证和测试不能随机按图像拆分。应至少按 `stream_id + 100 m route_block_id` 分组；同一路段相邻帧只能进入同一个数据划分，避免验证和测试精度虚高。

导出脚本按上述 100 m 分组进行确定性划分；仅出现在一至两个路段块的稀有树种优先保留在训练集，因此这些树种在当前尖沙咀子集上不能单独报告验证或测试精度。

## 5. 与点云后处理的关系

尖沙咀人工影像标注与 SegmentAnyTree 点云候选后处理完全分离。本阶段标注不会改变点云模型或去重阈值。后续建立三维真值时，可按时间、坐标和地图块把影像目标与点云候选建立只读关联。
