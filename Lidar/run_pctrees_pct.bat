@echo off
setlocal
for %%I in ("%~dp0..") do set "TREE_ROOT=%%~fI"
set "PYTHON=%LOCALAPPDATA%\miniconda3\envs\yolo\python.exe"
set "WANDB_MODE=offline"
set "PYTHONPATH=%TREE_ROOT%\Lidar\pctrees_github\PCT_Pytorch\pointnet2_ops_lib;%TREE_ROOT%\Lidar\pctrees_github\PCT_Pytorch;%PYTHONPATH%"
cd /d "%TREE_ROOT%\Lidar\pctrees_github"
"%PYTHON%" pct_main.py %*
