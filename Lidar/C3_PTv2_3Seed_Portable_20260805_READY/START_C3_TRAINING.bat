@echo off
title Portable C3 PTv2 - Three-Seed Training
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\start_c3_and_monitor.ps1"
if errorlevel 1 (
  echo.
  echo Startup failed. Review the message above and diagnostics folder.
  pause
)
