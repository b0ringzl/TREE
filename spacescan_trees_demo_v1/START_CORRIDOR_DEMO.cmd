@echo off
setlocal
title SST Demo Launcher
echo Starting SST Demo. Please wait...
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start-live.ps1" -Edition image-only
if errorlevel 1 (
    echo.
    echo Demo could not start. Please keep this error message for troubleshooting.
    pause
    exit /b 1
)
if /i "%~1"=="--check" exit /b 0
start "" "http://127.0.0.1:8043/corridor"
if errorlevel 1 (
    echo Open http://127.0.0.1:8043/corridor in your browser.
    pause
    exit /b 1
)
exit /b 0
