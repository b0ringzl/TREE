@echo off
title Portable C2 PTv2 Training
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\start_and_monitor.ps1"
if errorlevel 1 (
  echo.
  echo Startup failed. See the message above and diagnostics folder.
  pause
)
