@echo off
call D:\miniconda\Scripts\activate.bat yolo
set WANDB_MODE=offline
cd /d D:\TREE\Lidar\pctrees_github
python main.py %*
