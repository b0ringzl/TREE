@echo off
chcp 65001 >nul
setlocal
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0run-all-routes-labeler.ps1"
if errorlevel 1 (
    echo Startup failed. Logs: %~dp0annotation_app\logs
    pause
    exit /b 1
)
endlocal
