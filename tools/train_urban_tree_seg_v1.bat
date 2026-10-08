@echo off
setlocal
for %%I in ("%~dp0..") do set "TREE_ROOT=%%~fI"
set "PYTHON=%LOCALAPPDATA%\miniconda3\envs\yolo\python.exe"
cd /d "%TREE_ROOT%"
"%PYTHON%" tools\train_yolo_seg_model.py ^
  --data models\urban_tree_voc_seg_dataset_v2\data.yaml ^
  --model "%TREE_ROOT%\yolo11n-seg.pt" ^
  --name urban_tree_seg_v1 ^
  --epochs 60 ^
  --image-size 640 ^
  --batch 8 ^
  --workers 2 ^
  --device 0 ^
  > "%TREE_ROOT%\models\yolo_runs\urban_tree_seg_v1_train.log" ^
  2> "%TREE_ROOT%\models\yolo_runs\urban_tree_seg_v1_train.err.log"
