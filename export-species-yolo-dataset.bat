@echo off
setlocal
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -Command "& { conda run --no-capture-output -n yolo python .\export_species_yolo_dataset.py --species-root 'D:\TREE\external_datasets\inat_max150' --class-root 'D:\TREE\选取树种' --merge-map 'D:\TREE\tools\tree_species_label_merge_map.csv' --output 'D:\TREE\models\web_tree_species_seg_dataset_v1' --overwrite }"
if errorlevel 1 (
  echo.
  echo Export failed. Check the error message above.
  pause
)
