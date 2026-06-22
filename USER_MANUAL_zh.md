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

YOLO11 segmentation:
D:\TREE\LabelPaw-web-images\weights\yolo11_weights\yolo11n-seg.pt
```

已真实验证：

```text
SAM2 tiny 可加载到 CUDA
YOLO11 segmentation 可加载，任务类型为 segment
```

暂不使用：

```text
SAM3
```

原因：当前没有拿到 `sam3.pt` 权重。程序已调整为没有 SAM3 权重时不默认选择 SAM3。

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
```

使用建议：

1. 在模型选择器中选择 `sam2.1_hiera_tiny`。
2. 确认当前仍在多边形或矩形模式。
3. 打开智能辅助开关。
4. 在目标树木上点击，SAM2 会生成候选轮廓。
5. 如果轮廓可用，将其作为标签保存；如果不理想，继续手工多边形标注。

SAM2 tiny 速度较快、显存压力小，适合先测试流程。若以后加入更大的 SAM2 权重，文件名要包含这些关键词之一：

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
