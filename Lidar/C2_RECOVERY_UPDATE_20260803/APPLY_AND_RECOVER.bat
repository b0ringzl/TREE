@echo off
title C2 PTv2 Recovery Update
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0apply_and_recover.ps1"
if errorlevel 1 (
  echo.
  echo Recovery failed. Read the message above and do not delete the checkpoint.
  pause
)
