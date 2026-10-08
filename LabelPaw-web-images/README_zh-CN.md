
<div align="center">
  <p>
    <a href="https://github.com/luohuabuxiema/LabelPaw" target="_blank">
      <img alt="X-AnyLabeling" width="200" src="assets/logo.png"></a>
  </p>
  <a href="README.md">English</a> | <a href="README_zh-CN.md">简体中文</a>
</div>


# LabelPaw - 智能图像标注系统 (v2.0.0)
## 前言
由于项目需要标注数据集，之前用过 labelme、labelimg 等工具，于是决定结合 SAM2、SAM3、YOLO 姿态估计等优秀的视觉模型，开发一个更智能、更高效的标注工具。经过多次迭代，系统迎来了全新的 **2.0.0 版本**！

源码地址：[https://github.com/luohuabuxiema/LabelPaw](https://github.com/luohuabuxiema/LabelPaw)

## 更新日志

- 2026-05-28：新增sam3批量智能标注和多个提示词标注，并增加YOLO26模型批量标注，文件列表新增删除和多选按钮等等功能优化。
- 2026-05-27：删除类别选择的窗口，统一改成在右侧的历史类别面板修改与新增类别、可在历史类别删除和隐藏指定目标框等功能优化。
- 2026-05-15：新增人脸、手部、行人等关键点模板骨架，可自定义关键点模板与连线。
- 2026-05-14：新增 SAM2.1 模型，实现智能点选标注，并集成了 Ultralytics YOLO 模型，YOLO 模型可用于矩形、分割、关键点、obb智能标注。
- 2026-05-13：完善了 JSON/XML/YOLO 格式互转、支持 JSON 转 U-Net 掩膜（Mask）、数据集一键随机划分。
- 2026-05-10：系统新增对亮色（Light）和暗黑（Dark）主题模式的支持，提供更舒适的视觉体验。
- 2026-04-12：基于 PySide6 构建的基础智能标注界面（首次发布）。
- 2026-04-10：集成最新一代 SAM3，支持鼠标悬停预览、单点极速提取轮廓、输入文本提示词全图目标自动分割。
- 2026-04-9：支持矩形 (Rect)、多边形 (Poly)、点 (Point) 标注，以及独创的 OBB 旋转框控制手柄（支持 360° 无极顺滑旋转与贴墙滑动检测）。
- 2026-04-8：支持原生保存 JSON、YOLO (.txt)、XML (Pascal VOC)。


## 系统简介

系统基于 PySide6 构建，集成了 **SAM2**、**SAM3** 以及 **Ultralytics YOLO** 视觉模型，极大地提升了标注效率：
- **智能点选与提示词分割**：开启 SAM 智能标注后，支持在多边形、矩形、OBB 等模式下进行目标快速提取。
- **关键点骨架模板与智能标注**：全新的关键点模块，内置行人、手部、面部等关键点模板，可自定义关键点模板，实现快速标注，可选 YOLO 模型实现关键点的智能检测与自动连线。

| 功能             | 界面演示                                                     |
| ---------------- | ------------------------------------------------------------ |
| sam3批量标注     | ![在这里插入图片描述](assets/0fe12a65b4064e6c964520232569302b.png) |
| YOLO模型批量标注 | ![在这里插入图片描述](assets/797b72992f5c48ac93f0097fef943efe.png) |
| 关键点标注       | ![在这里插入图片描述](assets/3907465018334ef597d142779b2b8b61-177985711229411.png) |
| OBB智能标注      | ![在这里插入图片描述](assets/52c66efbccbe4c91ba2a334cb9006939-177985711229413.png) |
| 矩形智能标注     | ![在这里插入图片描述](assets/d87d307971c9475182bb7c5a1756aed8-177985711229415.png) |
| 关键点智能标注   | ![在这里插入图片描述](assets/a8a9ebdb1e56464aac57307318446bdc-177985711229417.png) |
| 手部关键点模板   | ![在这里插入图片描述](assets/615b3d73adb84378bd87128d5869a316-177985711229419.png) |
| 内置关键点模板   | ![在这里插入图片描述](assets/b2540b59b9e34f7491d1499f3d125e76-177985711229421.png) |
| 人脸关键点模板   | ![在这里插入图片描述](assets/b2b2aebc9db1438aadb3cafddb48a8f0-177985711229423.png) |
| 手部关键点模板   | ![在这里插入图片描述](assets/f22a8de7a22a45aeaf7ecc5dea3809b4-177985711229425.png) |
| 自定义关键点模板 | ![在这里插入图片描述](assets/61f48c4cc9a44bd2980d2aa450c5f616-177985711229427.png) |
| 数据集处理工具   | ![在这里插入图片描述](assets/f3371af5d6a94f3880e2dcac8bc0b068-177985711229429.png) |



## 🙊核心功能特性

- **✨ AI 智能辅助 (SAM2/SAM3 驱动)**：鼠标悬停预览、单点快速提取轮廓、输入文本提示词全图目标自动分割。
- **🦴 关键点骨架模板与智能 (YOLO 驱动)标注**：支持矩形、分割、obb、关键点智能标注，关键点内置行人（17个关键点）、人脸（68个关键点）、手部（21个关键点），关键点标注可自定义骨架模板。
- **📐 全能标注模式**：矩形 (Rect)、多边形 (Poly)、点 (Point)、OBB 旋转框以及关键点 (Pose)。
- **🔄 极致 OBB 交互**：旋转框控制手柄，360° 无极顺滑旋转与贴墙滑动检测。
- **💾 多格式互转与导出**：原生保存 JSON、YOLO (.txt)、XML (Pascal VOC)，可一键生成 U-Net Mask。
- **🗄️ 数据集处理工作流**：支持按比例切分训练集/验证集/测试集。

---

## 🛠️ 部署与运行环境

### 1. 基础环境依赖

推荐使用 Python 3.10+。

创建虚拟环境，命令如下：

```python
conda create -n py311 python==3.11.5
```
进入刚刚创建的虚拟环境，命令如下：

```python
conda activate py311 
```


首先安装必要的 Python 依赖包：

单独安装 torch>=2.5.0，pytorch 官网地址： [https://pytorch.org/](https://pytorch.org/get-started/previous-versions/?_gl=1*r08hqw*_up*MQ..*_ga*MTg1ODQzMTE5LjE3NzU4ODk5NDI.*_ga_469Y0W5V62*czE3NzU4ODk5NDEkbzEkZzAkdDE3NzU4ODk5NDEkajYwJGwwJGgw/)
![在这里插入图片描述](assets/d5d10831b3184dfb83fba9ea8166bb7b.png)

**💡 PyTorch 安装注意事项（新手必看）**

在进行安装 PyTorch 之前，请大家务必核对以下几点，避免安装后运行报错：

**1. 确认显卡支持与 CUDA 版本（极其重要）**
* **适用系统**：本教程基于 Windows 环境。
* **如何查看**：按下 `Win + R` 键，输入 `cmd` 打开命令提示符，输入 `nvidia-smi` 并回车。在弹出的表格右上角，找到 **CUDA Version**。
![在这里插入图片描述](assets/984f810aeee94916a5ef87ba8739d4d9.png)

* **版本匹配要求**：你下载的 PyTorch CUDA 版本（例如命令中的 `cu118` 或 `cu116`），**必须小于或等于**你电脑刚刚查到的 CUDA Version。如果你的电脑没有独立 N 卡，或者查不到该信息，请到官网选择 **CPU 版本**的安装命令。


**2. Conda 与 Pip 命令二选一即可**


根据自己电脑安装指定版本，安装命令如下，如果你使用 `conda` 命令卡住，可以尝试先在终端配置好国内的清华/中科大 conda 镜像源，然后再删掉命令后面的 `-c pytorch -c nvidia`（因为带上 `-c` 会强制去国外官方频道下载）：


```bash
conda install pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0  pytorch-cuda=11.8 -c pytorch -c nvidia
```

上面的命令，大多数情况下是安装失败的，所以这里推荐使用阿里云镜像源安装，阿里云上镜像，pytorch gpu版的 whl 包可以在此链接查看：链接: [https://mirrors.aliyun.com/pytorch-wheels](https://mirrors.aliyun.com/pytorch-wheels/)

后面 cu 版本需要对应cuda 的版本号，例如安装 cuda11.8 就写  cu118

```bash
-f  https://mirrors.aliyun.com/pytorch-wheels/cu118
```
cuda11.8 安装命令：
```bash
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 -f  https://mirrors.aliyun.com/pytorch-wheels/cu118
```
cuda12.1 安装命令：

```bash
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 -f  https://mirrors.aliyun.com/pytorch-wheels/cu121
```

**3. 验证是否安装成功**
安装进度条跑完后，不要急着关掉窗口！在终端里输入 `python` ，输入以下代码：
```python
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(f"CUDA：{torch.version.cuda}")
```
如果输出了 `True`，恭喜你，cuda 可用！如果输出 `False`，说明装成了 CPU 版本或者 CUDA 不匹配，可能需要卸载重装。

---


之后在自己的虚拟环境下，使用下面命令安装所需的库

```
pip install -r requirements.txt
```
```python
pyside6~=6.4.2
numpy~=1.24.4
opencv-python~=4.11.0.86
pillow~=10.4.0
einops~=0.8.2
pycocotools~=2.0.11
scipy~=1.15.3
tqdm~=4.67.1
iopath~=0.1.10
matplotlib~=3.10.8
timm~=1.0.26
ftfy~=6.3.1
psutil~=7.2.1
torchmetrics~=1.5.0
omegaconf~=2.3.0
numba~=0.64.0
huggingface-hub~=0.36.2
pandas~=2.3.3
scikit-learn~=1.8.0
setuptools==79.0.1
git+https://github.com/facebookresearch/sam3.git
git+https://github.com/facebookresearch/sam2.git
ultralytics==8.4.49
```

>注：如果需要使用智能辅助功能，请确保你的环境中已经正确配置了 `sam3`、 `sam2`、`ultralytics`相关的库及其依赖，如果上面的sam3、sam2、ultralytics 库安装失败，下文也有源码方式安装教程。

#### 安装过程我遇到的错误

（1）创建虚拟环境时候创建不了，报错如下：
![在这里插入图片描述](assets/eab21eb3721f45cf87d161c782afc73f.png)

解决方法：去 C:\Users\你的用户下，删除 .condarc 文件 

（2）报错 ModuleNotFoundError: No module named ‘pkg_resources‘ 

![在这里插入图片描述](assets/45b21e1d50fc4853978617234eec5d32.png)

解决方法：降低 setuptools 库版本，我安装的版本是 79.0.1

```python
pip install setuptools==79.0.1
```
![在这里插入图片描述](assets/c673e9d646bc4e0989cfe40a432c2adf.png)

参考链接： [https://blog.csdn.net/u014451778/article/details/158469881](https://blog.csdn.net/u014451778/article/details/158469881/)

（3）报错 ModuleNotFoundError: No module named ‘triton‘ 
![在这里插入图片描述](assets/d4b99849318740f0b1a40cfa857af640.png)

解决方法：下载离线安装包，单独安装

triton离线安装包下载地址：[https://hf-mirror.com/madbuda/triton-windows-builds](https://hf-mirror.com/madbuda/triton-windows-builds/)

![在这里插入图片描述](assets/404ac26c33944f0f9bcfe727dbfd501c.png)

参考链接：[https://blog.csdn.net/qq_42910179/article/details/155606159](https://blog.csdn.net/qq_42910179/article/details/155606159/)

### 2.  sam3、sam2、Ultralytics 源码安装方式

为了确保 SAM2、SAM3、Ultralytics（YOLO） 能够正常工作，你需要去官方仓库下载源码并放置在 `LabelPaw` 根目录下。因为官方库在不断更新，采用源码方式能最大程度保证兼容性。

**官方源码地址**：

- **SAM2**: [https://github.com/facebookresearch/sam2](https://github.com/facebookresearch/sam2)
- **SAM3**: [https://github.com/facebookresearch/sam3](https://github.com/facebookresearch/sam3)
- **Ultralytics (YOLO)**: [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)

**操作步骤**：

1. 访问上述 GitHub 地址，点击绿色的 **Code** 按钮，选择 **Download ZIP**。
2. 解压下载的压缩包。
3. **重要**：压缩包内往往包含文档、测试用例等很多文件。只需要把解压后里面的**核心代码文件夹**（下载完可以看到有对应名称的 `sam2`、`sam3`、`ultralytics` 文件夹）。
4. 将这三个文件夹（`sam2`, `sam3`, `ultralytics`）直接粘贴到 `LabelPaw` 的根目录下。

**通过命令行直接安装（可选，推荐进阶用户）**：
如果你不想手动下载和复制文件夹，可以直接使用 pip 从 GitHub 源码或 PyPI 进行安装：

```bash
# 安装 SAM2
pip install git+https://github.com/facebookresearch/sam2.git

# 安装 SAM3
pip install git+https://github.com/facebookresearch/sam3.git

# 安装 Ultralytics
pip install ultralytics
```

> ⚠️ **`git+` 安装方式注意事项**：
> 1. **需安装 Git**：你的电脑必须提前安装并配置好 [Git](https://git-scm.com/) 环境，否则命令会直接报错。
> 2. **国内网络问题**：由于 GitHub 在国内部分地区访问不稳定，使用 `git+https://...` 时容易遇到 `Time out` 或连接失败。国内用户建议：
>    - 开启科学上网环境，并在命令行中临时设置 Git 代理。
>    - 或者优先推荐使用上方的**官方源码下载解压**方式，这种方式最稳妥。

### 3. 模型下载与配置修改

**模型下载与存放说明**：

为了启用智能标注，你需要下载相应的权重文件 (`.pt`) 并按照规范的目录结构组织。

**1. 推荐的模型存放目录结构**
建议按照以下结构在本地整理您的模型文件：
```text
 weights/
      ├── sam_weights/          <-- 存放所有 SAM 系列模型 (必须叫这个名字)
      │    ├── sam3.pt
      │    ├── sam2.1_hiera_tiny.pt
      │    └── ...
      ├── yolo26_weights/       <-- 存放 YOLO26 系列模型
      │    ├── yolo26n-pose.pt
      │    └── ...
      ├── yolov8_weights/       <-- 您也可以自己新建其他 YOLO 版本的文件夹
      │    ├── yolov8n.pt
      │    └── ...
      └── ...
```

**2. SAM 系列模型下载与存放**

- **SAM 3 模型 (3.5 GB)**：前往官方 GitHub 仓库或 HuggingFace 搜索 `sam3` 获取。存放在 `\weights\sam_weights\sam3.pt`。
- **SAM 2.1 模型**：
  - SAM 2.1 Tiny : [https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt)
  - SAM 2.1 Small: [https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt)
  - SAM 2.1 Base: [https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt)
  - SAM 2.1 Large: [https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt)
  - 存放位置：`\weights\sam_weights\` 目录下 (文件名请保持默认)。

**3. YOLO 系列模型下载与存放**
- **YOLO 模型**：可前往 YOLO 官方 GitHub 或对应框架页面下载最新权重（如 yolov8、yolo11、yolo26 等）。姿态估计推荐下载带 `-pose` 后缀的模型（如 `yolo26n-pose.pt`）。
- **存放位置**：存放在对应的文件夹内，如 `\weights\yolo26_weights\`。*(注：您也可以将自己训练好的 YOLO 模型放入对应文件夹中，软件可自动读取！)*

**模型路径修改说明**：
为了让系统找到你下载的模型，

>方法1：在项目根目录下新建一个 weights 文件夹，存放模型，模型存放结构看上文介绍

>方法2：在其他地方新建一个 weights 文件夹，存放模型，模型存放结构看上文介绍，**只需**修改代码中的一个基础路径变量：
打开 `main.py`、`labelpaw/models/sam_client.py` 以及 `ui/model_selector_dialog.py`，找到里面的 `HARDCODED_DEV_DIR` 变量，将其统一修改为您本地 `weights` 文件夹的绝对路径： **HARDCODED_DEV_DIR= r"你的绝对路径\weights"**


*(注：系统会自动扫描该目录下所有形如 `yolo*_weights` 的子文件夹并加载 YOLO 模型，因此你只需放好模型，无需再手动指定 YOLO 的子目录！)*

**【特别说明：无显卡(GPU)用户的建议】**
如果您的电脑没有独立显卡（GPU）或者配置较低，强烈建议您优先使用 YOLO 系列模型（如带有 "n" 或 "s" 的轻量级模型）。SAM 系列模型即使是 tiny 版本也相对较重，在纯 CPU 环境下运行可能会非常卡顿或导致软件未响应，而 YOLO 轻量级模型在 CPU 上也能保持不错的处理速度。

### 4. 启动系统

完成所有配置后，在根目录下运行：
```bash
python main.py
```

---

## 📖 用户操作指南

### 📋基本工作流

1. **打开目录**：点击“打开目录”选择图片文件夹。
2. **选择格式**：在左侧下拉菜单选择保存格式（JSON / YOLO / XML）。
3. **标注模式**：选择左侧工具栏，可选矩形、关键点、obb、多边形标注模式，或者使用快捷键。
4. **智能标注**：开启 **智能辅助**（快捷键 Q）。sam3/sam2 模型支持悬停预览点选，sam3 模型支持输入提示词标注，可在顶部工具栏选择其他模型，也可以使用 YOLO 智能预推理进行标注。
5. **关键点/骨架标注**：选择关键点标注模式后，可在顶部栏工具栏可选择内置的关键点骨架模板进行标注，关键点骨架模板目前内置人脸、手部、行人等骨架，也可以自定义关键点骨架模板，也可通过 YOLO 智能预推理。
6. **数据集处理**：点击工具栏的“数据集转换”，可执行格式互转、U-Net Mask 生成、以及训练/验证集比例划分。

### ⌨️ 快捷键大全

- **A / 左方向键**：上一张图片
- **D / 右方向键**：下一张图片
- **Ctrl + S**：手动保存当前标注
- **Q**：开启/关闭 SAM 智能辅助
- **R**：矩形标注 (Rect)
- **P**：多边形标注 (Poly)
- **O**：旋转框标注 (OBB)
- **T**：关键点标注 (Pose/Point)
- **M**：使用 YOLO 模型进行推理 (需开启智能辅助并加载对应模型)
- **E**：修改当前选中的标签类别
- **Del / Backspace**：删除选中的标注框/点
- **Ctrl + Z**：撤销 (支持 20 步)
- **Ctrl + Y (或 Ctrl + Shift + Z)**：重做
- **Z / X / C / V**：OBB 旋转框快捷微调角度

---

## 🤝 欢迎二次开发 

系统采用模块化设计，高内聚低耦合，前端 UI 与底层模型推理分离。
- `main.py`：主控界面与事件路由。
- `labelpaw/`：核心模块，包含绘图画布 (`graphics/canvas.py`)、数据格式导出 (`data/exporter.py`)、SAM 智能模型推理 (`models/sam_client.py`) 以及 YOLO 智能模型推理 (`models/yolo_predictor.py`)。
- `ui/`：图形化组件与主题定制。

欢迎广大开发者 Fork 并提交 PR！

## 声明

本项目已采用 GPL-3.0 协议，如果您在商业或非商业项目中使用了本代码，请遵守该协议开源您的衍生修改版本。感谢大家的支持，有帮助的话可以给仓库点个 Star！

## 引用

如果您在研究中使用该软件，请引用如下：

```bibtex
@misc{LabelPaw,
  year = {2026},
  author = {luohuabuxiema},
  publisher = {Github},
  journal = {Github repository},
  title = {LabelPaw: Intelligent image annotation system},
  howpublished = {\url{https://github.com/luohuabuxiema/LabelPaw}}
}
```

**致谢与参考模型引用**：

```bibtex
@misc{carion2025sam3segmentconcepts,
      title={SAM 3: Segment Anything with Concepts},
      author={Nicolas Carion et al.},
      year={2025},
      eprint={2511.16719},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.16719},
}

@article{ravi2024sam2,
  title={SAM 2: Segment Anything in Images and Videos},
  author={Ravi, Nikhila and Gabeur, Valentin and Hu, Yuan-Ting and Hu, Ronghang and Ryali, Chaitanya and Ma, Tengyu and Khedr, Haitham and R{\"a}dle, Roman and Rolland, Chloe and Gustafson, Laura and others},
  journal={arXiv preprint arXiv:2408.00714},
  year={2024}
}

@software{ultralytics,
  author = {Glenn Jocher and Ayush Chaurasia and Jing Qiu},
  title = {Ultralytics},
  year = {2023},
  url = {https://github.com/ultralytics/ultralytics},
  license = {AGPL-3.0}
}
```
