@echo off
chcp 65001 >nul
setlocal
title 导出司徒拔道有效YOLO分割数据

set "PROJECT=%~dp0"
set "PYTHON=%LOCALAPPDATA%\miniconda3\envs\yolo\python.exe"
if not exist "%PYTHON%" set "PYTHON=python.exe"

"%PYTHON%" "%PROJECT%tools\export_effective_frame_yolo.py" --project-root "%PROJECT%" --labeler-name stubbs_road_frame_labeler --streams stubbs_pano_0,stubbs_pano_1
if errorlevel 1 (
  echo.
  echo 导出失败，请查看上方错误信息。
  pause
  exit /b 1
)

echo.
echo 导出完成。结果位于 derived\stubbs_road_frame_labeler\training_exports。
pause
endlocal
