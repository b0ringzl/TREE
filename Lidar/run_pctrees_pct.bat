@echo off
call D:\miniconda\Scripts\activate.bat yolo
set WANDB_MODE=offline
set PYTHONPATH=D:\TREE\Lidar\pctrees_github\PCT_Pytorch;%PYTHONPATH%
cd /d D:\TREE\Lidar\pctrees_github
python pct_main.py %*
