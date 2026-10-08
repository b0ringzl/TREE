@echo off
setlocal
set "SCRIPT=%~dp0scripts\start_d1_ptv2_repeats.ps1"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT%"
if errorlevel 1 pause
