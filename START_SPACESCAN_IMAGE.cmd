@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0spacescan_trees_demo_v1\start-live.ps1" -Edition image-only -OpenBrowser
if errorlevel 1 pause
