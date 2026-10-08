@echo off
setlocal
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start-labelpaw-yolo.ps1"
if errorlevel 1 (
  echo.
  echo LabelPaw failed to start. Check the error message above.
  pause
)
