@echo off
title C3 PTv2 Local Three-Seed Training
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start_c3_local.ps1"
if errorlevel 1 (
  echo.
  echo Local C3 startup failed. Review the message above.
  pause
)
