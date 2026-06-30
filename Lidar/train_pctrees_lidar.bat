@echo off
setlocal

set EXP_NAME=%~1
if "%EXP_NAME%"=="" (
  for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set EXP_NAME=tree_lidar_pct_%%i
)

set EPOCHS=%~2
if "%EPOCHS%"=="" set EPOCHS=1

set CACHE_POINTS=%~3
if "%CACHE_POINTS%"=="" set CACHE_POINTS=4096

set NUM_WORKERS=%~4
if "%NUM_WORKERS%"=="" set NUM_WORKERS=2

set CACHE_DIR=D:\TREE\lidar data\pctrees_cache_%CACHE_POINTS%

call D:\miniconda\Scripts\activate.bat yolo
set WANDB_MODE=offline
set PYTHONPATH=D:\TREE\Lidar\pctrees_github\PCT_Pytorch;%PYTHONPATH%

cd /d D:\TREE\Lidar\pctrees_github

if not exist "D:\TREE\lidar data\labels_pctrees_train.csv" (
  echo Preparing pctrees labels...
  python "D:\TREE\Lidar\prepare_pctrees_labels.py" ^
    --metadata "D:\TREE\lidar data\tree_metadata_dev.csv" ^
    --data-dir "D:\TREE\lidar data\train" ^
    --output "D:\TREE\lidar data\labels_pctrees_train.csv"
)

if not exist "%CACHE_DIR%\index.csv" (
  echo Building point cache at %CACHE_DIR% ...
  python "D:\TREE\Lidar\build_pctrees_point_cache.py" ^
    --labels "D:\TREE\lidar data\labels_pctrees_train.csv" ^
    --data-dir "D:\TREE\lidar data\train" ^
    --cache-dir "%CACHE_DIR%" ^
    --cache-points %CACHE_POINTS% ^
    --min-points 1024
)

echo Experiment: %EXP_NAME%
echo Epochs: %EPOCHS%
echo Cache points: %CACHE_POINTS%
echo Cache dir: %CACHE_DIR%
echo DataLoader workers: %NUM_WORKERS%
echo Progress: D:\TREE\Lidar\pctrees_github\checkpoints\%EXP_NAME%\progress.csv

python pct_main.py ^
  --exp_name %EXP_NAME% ^
  --epochs %EPOCHS% ^
  --batch_size 8 ^
  --test_batch_size 16 ^
  --num_workers %NUM_WORKERS% ^
  --num_points 1024 ^
  --min_points 1024 ^
  --train_split 0.8 ^
  --output_channels 33 ^
  --cache_dir "%CACHE_DIR%" ^
  --data_dir "D:\TREE\lidar data\train" ^
  --label_path "D:\TREE\lidar data\labels_pctrees_train.csv"

endlocal
