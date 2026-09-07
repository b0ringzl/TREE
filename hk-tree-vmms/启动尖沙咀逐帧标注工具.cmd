@echo off
chcp 65001 >nul
setlocal
title 尖沙咀全景影像逐帧标注

set "PROJECT=%~dp0"
set "APP=%PROJECT%annotation_app"
set "BACKEND=%APP%\backend"
set "FRONTEND=%APP%\frontend"
set "LOGS=%APP%\logs\jianshazui"
set "PYTHON=%LOCALAPPDATA%\miniconda3\envs\yolo\python.exe"
set "BACKEND_LOG=%LOGS%\backend.log"
set "FRONTEND_LOG=%LOGS%\frontend.log"

if not defined VMMS_ROOT set "VMMS_ROOT=%~dp0..\..\vmms"
set "VMMS_TREE_STREAM_ID=jianshazui_pano_1"
set "VMMS_FRAME_STREAM_IDS=jianshazui_pano_1,jianshazui_pano_2"
set "VMMS_FRAME_LABELER_DATASET=jianshazui"
set "VMMS_LABELER_DERIVED_ROOT=%PROJECT%derived\jianshazui_labeler"
set "VMMS_FRAME_LABELER_ROOT=%PROJECT%derived\jianshazui_frame_labeler"
set "VMMS_FRAME_SEED_CLASSES=%PROJECT%derived\hewentian_frame_labeler\annotations\classes.json"
set "VITE_API_BASE=http://127.0.0.1:8023"

if not exist "%PYTHON%" (
  echo 找不到项目 Python 环境：%PYTHON%
  pause
  exit /b 1
)
where npm.cmd >nul 2>nul
if errorlevel 1 (
  echo 找不到 npm.cmd。
  pause
  exit /b 1
)
if not exist "%LOGS%" mkdir "%LOGS%"

curl.exe -fsS http://127.0.0.1:8023/api/health >nul 2>nul
if errorlevel 1 (
  start "尖沙咀标注后端" /min /D "%BACKEND%" cmd.exe /d /c "%PYTHON% -m uvicorn app.main:app --host 127.0.0.1 --port 8023 1>>%BACKEND_LOG% 2>>&1"
)

curl.exe -fsS http://127.0.0.1:5179/frame.html >nul 2>nul
if errorlevel 1 (
  start "尖沙咀标注前端" /min /D "%FRONTEND%" cmd.exe /d /c "npm.cmd run dev -- --port 5179 1>>%FRONTEND_LOG% 2>>&1"
)

for /L %%I in (1,1,60) do (
  curl.exe -fsS http://127.0.0.1:8023/api/health >nul 2>nul
  if not errorlevel 1 goto backend_ready
  ping 127.0.0.1 -n 2 >nul
)

echo 尖沙咀标注后端启动失败，请查看：%BACKEND_LOG%
pause
exit /b 1

:backend_ready
for /L %%I in (1,1,60) do (
  curl.exe -fsS http://127.0.0.1:5179/frame.html >nul 2>nul
  if not errorlevel 1 goto all_ready
  ping 127.0.0.1 -n 2 >nul
)

echo 尖沙咀标注前端启动失败，请查看：%FRONTEND_LOG%
pause
exit /b 1

:all_ready
start "" "http://127.0.0.1:5179/frame.html"
echo 尖沙咀逐帧标注工具已启动。
echo 后端：http://127.0.0.1:8023
echo 前端：http://127.0.0.1:5179/frame.html
endlocal
