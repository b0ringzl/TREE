@echo off
setlocal
cd /d D:\TREE\hk-tree-labeler
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\run-session.ps1"
echo.
echo Press any key to close this window.
pause >nul
