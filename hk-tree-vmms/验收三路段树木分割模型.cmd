@echo off
chcp 65001 >nul
setlocal
title 三路段树木分割模型独立测试集验收
cd /d "%~dp0.."
"%USERPROFILE%\Desktop\receive\TREE\envs\yolo11-tree\Scripts\python.exe" "%~dp0tools\evaluate_combined_tree_segmentation.py"
echo.
echo 验收程序已经结束。按任意键关闭窗口。
pause >nul
