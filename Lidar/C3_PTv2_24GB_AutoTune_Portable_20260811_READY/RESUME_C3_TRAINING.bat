@echo off
title Portable C3 PTv2 - Resume Training
echo Resuming from the most recent complete last.pt checkpoint...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\start_c3_and_monitor.ps1"
if errorlevel 1 (
  echo.
  echo Resume failed. Review the message above and diagnostics folder.
  pause
)
