@echo off
title Portable C3 - Pause After Current Epoch
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\request_pause.ps1"
if errorlevel 1 (
  echo.
  echo Safe pause was not confirmed. Review the message above.
)
pause
