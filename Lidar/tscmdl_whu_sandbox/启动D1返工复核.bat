@echo off
setlocal
chcp 65001 >nul
for %%I in ("%~dp0..\..") do set "TREE_ROOT=%%~fI"
set "PYTHON=%TREE_ROOT%\envs\whu-tscmdl\python.exe"
set "SCRIPT=%~dp0scripts\d1_image_review_server.py"
set "BATCH_ROOT=%TREE_ROOT%\lidar data\whu\derived\tscmdl\d1_four_class_image_quality\rework_batches"

if not exist "%PYTHON%" (
  echo [错误] 未找到 Python: %PYTHON%
  pause
  exit /b 1
)

for /f "usebackq delims=" %%I in (`powershell -NoProfile -Command "$batch = Get-ChildItem -LiteralPath '%BATCH_ROOT%' -Directory ^| Where-Object { Test-Path -LiteralPath (Join-Path $_.FullName 'replacement_review_manifest.json') } ^| Sort-Object Name -Descending ^| Select-Object -First 1; if ($batch) { $batch.FullName }"`) do set "LATEST_BATCH=%%I"

if not defined LATEST_BATCH (
  echo [错误] 未找到已生成的 D1 返工批次。
  pause
  exit /b 1
)

echo 正在启动 D1 返工复核：%LATEST_BATCH%
"%PYTHON%" "%SCRIPT%" --dataset-root "%LATEST_BATCH%" --manifest "%LATEST_BATCH%\replacement_review_manifest.json" --review "%LATEST_BATCH%\replacement_user_review.json" --pid-file "%LATEST_BATCH%\replacement_review_server.pid" --port 8767
if errorlevel 1 pause
