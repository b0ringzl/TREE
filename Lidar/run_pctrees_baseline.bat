@echo off
setlocal
for %%I in ("%~dp0..") do set "TREE_ROOT=%%~fI"
set "PYTHON=%LOCALAPPDATA%\miniconda3\envs\yolo\python.exe"
set "WANDB_MODE=offline"
cd /d "%TREE_ROOT%\Lidar\pctrees_github"
"%PYTHON%" main.py %*
