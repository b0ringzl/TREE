@echo off
title Portable C3 24GB Device Check and Auto-Tune
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\prepare_runtime.ps1"
if errorlevel 1 (
  echo.
  echo Device check failed. Review the message above.
)
pause
