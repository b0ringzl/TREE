@echo off
chcp 65001 >nul
setlocal
title 三路段树木分割训练监控
cd /d "%~dp0.."
"%USERPROFILE%\Desktop\receive\TREE\envs\yolo11-tree\Scripts\python.exe" "%~dp0tools\monitor_yolo_training.py" --watch --interval 5
echo.
echo 监控已结束。按任意键关闭窗口。
pause >nul
