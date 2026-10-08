# Single-tree segmentation workflow

The labeler can show instance-mask candidates before manual annotation. The
human reviewer clicks the target tree mask, and the app saves the selected
mask's bounding box in the existing YOLO detection label format.

## Train from DatasetNinja / Supervisely export

Download and unpack the Urban Street: Tree dataset from DatasetNinja, then
convert its Supervisely instance masks to YOLO segmentation labels:

```powershell
cd D:\TREE
C:\Users\57680\.conda\envs\yolo\python.exe tools\export_supervisely_tree_seg_dataset.py `
  --input external_datasets\urban-street-tree `
  --output models\urban_tree_seg_dataset `
  --clean
```

Train a YOLO11 segmentation model:

```powershell
C:\Users\57680\.conda\envs\yolo\python.exe tools\train_yolo_seg_model.py `
  --data models\urban_tree_seg_dataset\data.yaml `
  --model yolo11n-seg.pt `
  --name urban_tree_seg_v1 `
  --epochs 60 `
  --image-size 640 `
  --batch 8
```

The backend looks for the trained weights here by default:

```text
D:\TREE\models\yolo_runs\urban_tree_seg_v1\weights\best.pt
```

You can override it with:

```powershell
$env:YOLO_TREE_SEG_MODEL="D:\TREE\models\yolo_runs\urban_tree_seg_v1\weights\best.pt"
$env:YOLO_TREE_SEG_CONF="0.12"
$env:YOLO_TREE_SEG_MAX_MASKS="8"
```

If the segmentation model is missing, the app falls back to the existing YOLO
box predictor so the labeling workflow still opens normally.

## Train from VOCdevkit export

If the dataset is unpacked as `VOCdevkit/VOC2012` with `JPEGImages`,
`SegmentationClass`, and `ImageSets/Segmentation`, convert it with:

```powershell
cd D:\TREE
C:\Users\57680\.conda\envs\yolo\python.exe tools\export_voc_tree_seg_dataset.py `
  --input dd\VOCdevkit\VOC2012 `
  --output models\urban_tree_voc_seg_dataset_v2
```

This follows the same VOC semantic-segmentation layout used by the Kaggle
notebooks for this dataset. The source masks keep species ids (`1..22`), but
the exporter merges all non-background tree pixels into one `tree` class and
extracts polygons from the semantic masks. That gives the labeler clickable
tree-mask candidates while keeping the training target aligned with the
dataset's semantic annotation format.

Then train the same YOLO11 segmentation model from the VOC-derived YAML:

```powershell
C:\Users\57680\.conda\envs\yolo\python.exe tools\train_yolo_seg_model.py `
  --data models\urban_tree_voc_seg_dataset_v2\data.yaml `
  --model D:\TREE\yolo11n-seg.pt `
  --name urban_tree_seg_v1 `
  --epochs 60 `
  --image-size 640 `
  --batch 8
```
