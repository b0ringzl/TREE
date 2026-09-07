@echo off
chcp 65001 >nul
setlocal
title 生成三路段合并训练数据

set "PROJECT=%~dp0"
set "PYTHON=%LOCALAPPDATA%\miniconda3\envs\yolo\python.exe"
if not exist "%PYTHON%" set "PYTHON=python.exe"

"%PYTHON%" "%PROJECT%tools\build_combined_training_datasets.py" --project-root "%PROJECT%"
if errorlevel 1 (
  echo.
  echo 数据生成失败，请查看上方错误信息。
  pause
  exit /b 1
)

echo.
echo 数据生成完成。结果位于 derived\combined_tree_segmentation 和 derived\common3_species_crops。
pause
endlocal
