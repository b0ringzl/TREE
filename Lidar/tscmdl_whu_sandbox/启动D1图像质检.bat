@echo off
setlocal
chcp 65001 >nul
for %%I in ("%~dp0..\..") do set "TREE_ROOT=%%~fI"
set "PYTHON=%TREE_ROOT%\envs\whu-tscmdl\python.exe"
set "SCRIPT=%~dp0scripts\d1_image_review_server.py"

if not exist "%PYTHON%" (
  echo [错误] 未找到 Python: %PYTHON%
  pause
  exit /b 1
)

"%PYTHON%" "%SCRIPT%"
if errorlevel 1 pause
