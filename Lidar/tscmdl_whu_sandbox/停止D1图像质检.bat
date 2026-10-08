@echo off
setlocal
chcp 65001 >nul
for %%I in ("%~dp0..\..") do set "TREE_ROOT=%%~fI"
set "PID_FILE=%TREE_ROOT%\lidar data\whu\derived\tscmdl\d1_four_class_image_quality\review_server.pid"

if not exist "%PID_FILE%" (
  echo D1 质检服务当前没有 PID 记录，可能已经停止。
  pause
  exit /b 0
)

set /p PID=<"%PID_FILE%"
powershell -NoProfile -Command "$process = Get-Process -Id %PID% -ErrorAction SilentlyContinue; if ($process) { Stop-Process -Id %PID%; $process.WaitForExit(5000) | Out-Null }"
if exist "%PID_FILE%" del /q "%PID_FILE%"
echo D1 质检服务已停止。
pause
