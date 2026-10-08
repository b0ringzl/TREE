@echo off
set "SCRIPT=%~dp0scripts\start_d1_resnet50_repeats.ps1"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT%" -Monitor
