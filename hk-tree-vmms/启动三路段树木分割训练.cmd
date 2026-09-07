@echo off
chcp 65001 >nul
setlocal
title 三路段树木分割训练 - YOLO11m-seg
cd /d "%~dp0.."
"%USERPROFILE%\Desktop\receive\TREE\envs\yolo11-tree\Scripts\python.exe" "%~dp0tools\train_combined_tree_segmentation.py"
echo.
echo 训练程序已经结束。按任意键关闭窗口。
pause >nul
