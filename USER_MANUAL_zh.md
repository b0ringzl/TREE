# LabelPaw 网络图片人工标注系统使用手册

本系统位于：

```text
D:\TREE\LabelPaw-web-images
```

它用于网络爬取图片的人工标注，默认使用多边形标注并保存为 YOLO segmentation 标签，格式与此前训练样本标签一致。

## 1. 启动

双击：

```text
D:\TREE\LabelPaw-web-images\start-labelpaw-yolo.bat
```

也可以在 PowerShell 中运行：

```powershell
cd /d D:\TREE\LabelPaw-web-images
powershell -NoProfile -ExecutionPolicy Bypass -File .\start-labelpaw-yolo.ps1
```

只检查环境、不打开界面：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\start-labelpaw-yolo.ps1 -CheckOnly
```

启动脚本固定使用 conda 的 `yolo` 环境，并把缓存写到项目内 `.runtime/`。

## 2. 当前可用模型

已确认可用：

```text
SAM2 tiny:
D:\TREE\LabelPaw-web-images\weights\sam_weights\sam2.1_hiera_tiny.pt

SAM3:
D:\TREE\LabelPaw-web-images\weights\sam_weights\sam3.pt

YOLO11 segmentation:
D:\TREE\LabelPaw-web-images\weights\yolo11_weights\yolo11n-seg.pt
```

已真实验证：

```text
SAM2 tiny 可加载到 CUDA
SAM3 可加载到 CUDA
YOLO11 segmentation 可加载，任务类型为 segment
```

默认启动时优先加载 SAM2 tiny。SAM3 已可用，但不会作为默认启动模型，避免每次启动都加载 3GB 以上的大模型。

## 3. 图片目录准备

建议每个待标注图片目录保持如下结构：

```text
某个图片目录\
  image_001.jpg
  image_002.jpg
  image_003.png
  classes.txt
```

`classes.txt` 是类别顺序文件。YOLO 标签中的 `class_id` 就来自这里的行号，从 0 开始。

示例：

```text
tree
```

如果按树种标注，也可以写：

```text
Acacia auriculiformis
Ficus microcarpa
```

注意：训练时必须保持 `classes.txt` 顺序一致。

## 4. 打开图片

1. 启动 LabelPaw。
2. 点击左侧/顶部的打开目录按钮。
3. 选择网络爬取图片所在文件夹。
4. 在左侧图片列表中点击图片开始标注。

系统会读取当前图片同名的标注文件：

```text
image_001.txt
```

如果不存在，就从空标注开始。

## 5. 人工多边形标注

默认启动后：

```text
标注模式：多边形
保存格式：YOLO
```

操作方式：

```text
左键点击：添加多边形顶点
双击：结束当前多边形
点击首个顶点附近：闭合多边形
Enter：结束当前多边形
Esc：取消当前未完成绘制
Del / Backspace：删除选中的标签
Ctrl+S：保存
Ctrl+Z / Ctrl+Y：撤销 / 重做
```

完成后的多边形可以选中并编辑：

```text
拖动顶点：调整边界
选中多边形后靠近边：可插入新顶点
拖动整体：移动标签
```

标签之间允许重叠。每个多边形会保存为一行 YOLO segmentation。

## 6. 使用 SAM2 辅助标注

当前可用的 SAM 模型是：

```text
sam2.1_hiera_tiny
sam3
```

使用建议：

1. 在模型选择器中选择 `sam2.1_hiera_tiny`。
2. 确认当前仍在多边形或矩形模式。
3. 打开智能辅助开关。
4. 在目标树木上点击，SAM2 会生成候选轮廓。
5. 如果轮廓可用，将其作为标签保存；如果不理想，继续手工多边形标注。

SAM2 tiny 速度较快、显存压力小，适合默认使用。SAM3 支持文本提示分割，但模型更大，加载和推理更吃显存。

若以后加入更大的 SAM2 权重，文件名要包含这些关键词之一：

```text
tiny
small
base_plus
large
```

## 7. 使用 YOLO11 辅助标注

当前 YOLO 权重：

```text
yolo11n-seg.pt
```

使用方式：

1. 在模型选择器中选择 `YOLO11 / yolo11n-seg`。
2. 点击预测按钮。
3. 模型会生成分割多边形。
4. 检查并手动修改不准确的多边形。
5. 保存。

YOLO11 预测结果会作为普通标签加入画布，可以继续拖点、删除、修改。

## 8. 保存格式

本系统默认保存 YOLO segmentation `.txt`。

每个图片生成一个同名标签文件：

```text
image_001.jpg
image_001.txt
```

每一行表示一个多边形：

```text
class_id x1 y1 x2 y2 x3 y3 ...
```

示例：

```text
0 0.100000 0.200000 0.300000 0.200000 0.200000 0.400000
```

含义：

```text
类别 ID = 0
顶点 1 = (0.100000, 0.200000)
顶点 2 = (0.300000, 0.200000)
顶点 3 = (0.200000, 0.400000)
```

坐标均已归一化到 `[0, 1]`，与此前训练样本输出一致。

## 9. 类别管理

右侧类别面板可以：

```text
新增类别
重命名类别
改变类别颜色
隐藏/显示某类标签
查看当前图片中每类标签数量
```

如果重命名类别，请确认训练数据中的 `classes.txt` 也保持一致。

## 10. 质量检查建议

保存前建议检查：

```text
目标树冠/树体是否被完整覆盖
多边形是否明显包含大量无关背景
类别是否正确
同一图片是否漏标目标
自动预测标签是否有明显错误
标签文件是否确实保存为 .txt
```

对于训练样本，优先保证多边形覆盖真实目标区域。矩形标签虽然也能保存，但本项目建议使用多边形。

## 11. 常见问题

### 启动后没有 SAM2

检查文件是否存在：

```text
D:\TREE\LabelPaw-web-images\weights\sam_weights\sam2.1_hiera_tiny.pt
```

然后重启程序。

### 启动后没有 YOLO11

检查文件是否存在：

```text
D:\TREE\LabelPaw-web-images\weights\yolo11_weights\yolo11n-seg.pt
```

LabelPaw 只扫描这种目录名：

```text
weights\yolo*_weights\
```

### 启动后没有 SAM3

检查文件是否存在：

```text
D:\TREE\LabelPaw-web-images\weights\sam_weights\sam3.pt
```

SAM3 权重较大，建议保留文件名 `sam3.pt`，这样 LabelPaw 会识别为 SAM3 模型。

### SAM2 点击后没结果或报错

先运行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\start-labelpaw-yolo.ps1 -CheckOnly
```

如果环境检查通过，但界面内仍报错，通常是显存、CUDA 或单张图片过大导致。可以先用较小分辨率图片测试。

### 不小心切到 JSON/XML

在格式选择器中切回：

```text
YOLO 格式
```

保存后确认图片旁边生成的是 `.txt`。

### 删除标签

选中标签后按：

```text
Del
```

或在右侧标签列表中点击删除按钮。

如果当前图片所有标签都删除，再保存时，对应空标签文件会被清理。

## 12. 备份与版本

原始 LabelPaw 克隆备份：

```text
branch: backup/upstream-pristine-e25e5d7
tag:    backup/upstream-pristine-20260608
```

当前本地配置分支：

```text
codex/network-image-labelpaw-setup
```

权重文件在 `.gitignore` 中被忽略，不会提交到 git。

## 13. 使用 `inat_max150` 作为标注来源

如果从 iNaturalist 批量图片开始标注，推荐使用：

```text
D:\TREE\external_datasets\inat_max150
```

这个目录的结构是：

```text
D:\TREE\external_datasets\inat_max150\
  Acacia auriculiformis\
    images\
      001_inat_photo_664817789.jpg
      002_inat_photo_664817829.jpg
    metadata.jsonl
  Acacia confusa\
    images\
      ...
    metadata.jsonl
```

LabelPaw 打开目录时不会递归读取子目录，所以标注时应选择具体树种下的 `images` 目录，例如：

```text
D:\TREE\external_datasets\inat_max150\Acacia auriculiformis\images
```

不要直接选择：

```text
D:\TREE\external_datasets\inat_max150
D:\TREE\external_datasets\inat_max150\Acacia auriculiformis
```

保存后，标签会生成在同一个 `images` 目录里：

```text
001_inat_photo_664817789.jpg
001_inat_photo_664817789.txt
```

导出脚本已经支持这种 `树种\images\图片` 结构，会自动根据上一级树种文件夹名生成全局类别 ID。

一键导出默认使用：

```text
输入：D:\TREE\external_datasets\inat_max150
输出：D:\TREE\models\web_tree_species_seg_dataset_v1
```

也就是说，你可以放心从 `inat_max150` 开始标注；标注时打开每个树种的 `images` 子目录，训练前再运行导出脚本。

## 14. 导出项目树种 YOLO 训练集

标注时可以逐个打开树种文件夹，例如：

```text
D:\TREE\external_datasets\inat_max150\Acacia auriculiformis\images
D:\TREE\external_datasets\inat_max150\Acacia confusa\images
```

每个树种文件夹内的标签可以只使用该树种作为本地类别。导出训练集时，脚本会按文件夹名自动生成全局 156 类，并把每个标签文件第一列 `class_id` 重写为全局类别 ID。

一键导出：

```text
D:\TREE\LabelPaw-web-images\export-species-yolo-dataset.bat
```

默认输入目录：

```text
D:\TREE\external_datasets\inat_max150
```

默认输出目录：

```text
D:\TREE\models\web_tree_species_seg_dataset_v1
```

默认类别目录：

```text
D:\TREE\选取树种
```

默认合并表：

```text
D:\TREE\tools\tree_species_label_merge_map.csv
```

导出时会以 `D:\TREE\选取树种` 的文件夹列表作为最终训练类别表，并用合并表把旧来源目录映射到当前项目类别。比如 `Bridelia insulana` 和 `Bridelia tomentosa` 都会导出为 `Bridelia tomentosa(Bridelia insulana)`。因此即使 `D:\TREE\external_datasets\inat_max150` 里还保留两个旧来源文件夹，最终 `classes.txt` 和 `data.yaml` 里也只会有合并后的一个类别。

也可以手动运行：

```powershell
cd /d D:\TREE\LabelPaw-web-images
C:\Users\57680\.conda\envs\yolo\python.exe export_species_yolo_dataset.py `
  --species-root D:\TREE\external_datasets\inat_max150 `
  --class-root D:\TREE\选取树种 `
  --merge-map D:\TREE\tools\tree_species_label_merge_map.csv `
  --output D:\TREE\models\web_tree_species_seg_dataset_v1 `
  --overwrite
```

导出后的结构：

```text
D:\TREE\models\web_tree_species_seg_dataset_v1\
  data.yaml
  classes.txt
  class_to_idx.json
  manifest.csv
  export_report.json
  warnings.txt
  images\
    train\
    val\
    test\
  labels\
    train\
    val\
    test\
```

导出后的图片不再按树种分文件夹，而是混合放入 `images/train`、`images/val`、`images/test`。树种由标签文件里的全局 `class_id` 区分。

### 导出命名规则

导出脚本会自动避免不同树种的同名图片冲突。命名规则是：

```text
全局序号__树种安全名__原文件名安全短名__原路径短哈希.jpg
全局序号__树种安全名__原文件名安全短名__原路径短哈希.txt
```

示例：

```text
000000__Acacia_auriculiformis__image_001__b82a776461.jpg
000000__Acacia_auriculiformis__image_001__b82a776461.txt

000001__Acacia_confusa__image_001__992a777683.jpg
000001__Acacia_confusa__image_001__992a777683.txt
```

即使两个树种文件夹里都有 `image_001.jpg`，导出后也不会覆盖或错配。

### 导出注意事项

导出脚本只导出有同名 `.txt` 标签的图片。例如：

```text
image_001.jpg
image_001.txt
```

如果图片没有对应 `.txt`，会被跳过，并记录到：

```text
warnings.txt
```

如果标签行不是 YOLO segmentation 多边形格式，也会跳过并记录警告。
