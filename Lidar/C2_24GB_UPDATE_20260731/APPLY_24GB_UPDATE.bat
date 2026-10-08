@echo off
title Apply C2 PTv2 24 GB Update
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0apply_update.ps1"
if errorlevel 1 (
  echo.
  echo Update failed. Read the message above.
  pause
)
