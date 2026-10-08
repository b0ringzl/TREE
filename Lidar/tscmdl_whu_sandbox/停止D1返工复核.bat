@echo off
setlocal
chcp 65001 >nul
for %%I in ("%~dp0..\..") do set "TREE_ROOT=%%~fI"
set "BATCH_ROOT=%TREE_ROOT%\lidar data\whu\derived\tscmdl\d1_four_class_image_quality\rework_batches"

for /f "usebackq delims=" %%I in (`powershell -NoProfile -Command "$batch = Get-ChildItem -LiteralPath '%BATCH_ROOT%' -Directory ^| Where-Object { Test-Path -LiteralPath (Join-Path $_.FullName 'replacement_review_server.pid') } ^| Sort-Object Name -Descending ^| Select-Object -First 1; if ($batch) { Join-Path $batch.FullName 'replacement_review_server.pid' }"`) do set "PID_FILE=%%I"

if not defined PID_FILE (
  echo D1 返工复核服务当前没有 PID 记录，可能已经停止。
  pause
  exit /b 0
)

set /p PID=<"%PID_FILE%"
powershell -NoProfile -Command "$process = Get-Process -Id %PID% -ErrorAction SilentlyContinue; if ($process) { Stop-Process -Id %PID%; $process.WaitForExit(5000) ^| Out-Null }"
if exist "%PID_FILE%" del /q "%PID_FILE%"
echo D1 返工复核服务已停止。
pause
