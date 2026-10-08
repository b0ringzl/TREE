@echo off
setlocal

for %%I in ("%~dp0..") do set "TREE_ROOT=%%~fI"
set "PYTHON=%LOCALAPPDATA%\miniconda3\envs\yolo\python.exe"

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

set "CACHE_DIR=%TREE_ROOT%\lidar data\pctrees_cache_%CACHE_POINTS%"

set "WANDB_MODE=offline"
set "PYTHONPATH=%TREE_ROOT%\Lidar\pctrees_github\PCT_Pytorch\pointnet2_ops_lib;%TREE_ROOT%\Lidar\pctrees_github\PCT_Pytorch;%PYTHONPATH%"

cd /d "%TREE_ROOT%\Lidar\pctrees_github"

if not exist "%TREE_ROOT%\lidar data\labels_pctrees_train.csv" (
  echo Preparing pctrees labels...
  "%PYTHON%" "%TREE_ROOT%\Lidar\prepare_pctrees_labels.py" ^
    --metadata "%TREE_ROOT%\lidar data\tree_metadata_dev.csv" ^
    --data-dir "%TREE_ROOT%\lidar data\train" ^
    --output "%TREE_ROOT%\lidar data\labels_pctrees_train.csv"
)

if not exist "%CACHE_DIR%\index.csv" (
  echo Building point cache at %CACHE_DIR% ...
  "%PYTHON%" "%TREE_ROOT%\Lidar\build_pctrees_point_cache.py" ^
    --labels "%TREE_ROOT%\lidar data\labels_pctrees_train.csv" ^
    --data-dir "%TREE_ROOT%\lidar data\train" ^
    --cache-dir "%CACHE_DIR%" ^
    --cache-points %CACHE_POINTS% ^
    --min-points 1024
)

echo Experiment: %EXP_NAME%
echo Epochs: %EPOCHS%
echo Cache points: %CACHE_POINTS%
echo Cache dir: %CACHE_DIR%
echo DataLoader workers: %NUM_WORKERS%
echo Progress: %TREE_ROOT%\Lidar\pctrees_github\checkpoints\%EXP_NAME%\progress.csv

"%PYTHON%" pct_main.py ^
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
  --data_dir "%TREE_ROOT%\lidar data\train" ^
  --label_path "%TREE_ROOT%\lidar data\labels_pctrees_train.csv"

endlocal
