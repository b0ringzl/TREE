@echo off
chcp 65001 >nul
setlocal

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0run-hewentian-labeler.ps1"
if errorlevel 1 (
    echo.
    echo Startup failed. Logs: %~dp0annotation_app\logs
    echo Keep this window open and send the error message to Codex.
    pause
    exit /b 1
)

endlocal
