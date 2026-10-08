@echo off
title Portable C2 Device Check
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\prepare_runtime.ps1"
if errorlevel 1 (
  echo.
  echo Device check failed. See the message above.
)
pause
