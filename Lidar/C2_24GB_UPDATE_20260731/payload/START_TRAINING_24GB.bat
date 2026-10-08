@echo off
title Portable C2 PTv2 - 24 GB Mode
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\switch_to_high_vram.ps1"
if errorlevel 1 (
  echo.
  echo High-VRAM switch failed. The standard START_TRAINING.bat is still available.
  pause
)
