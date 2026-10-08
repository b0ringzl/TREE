@echo off
title Pack C3 Results
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\pack_results.ps1"
if errorlevel 1 (
  echo.
  echo Result packaging failed. Review the message above.
)
pause
