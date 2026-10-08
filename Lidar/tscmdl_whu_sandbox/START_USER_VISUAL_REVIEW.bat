@echo off
setlocal
title WHU-STree User Visual Review
for %%I in ("%~dp0..\..") do set "TREE_ROOT=%%~fI"
set "PYTHON=%TREE_ROOT%\envs\whu-tscmdl\python.exe"
set "SCRIPT=%~dp0scripts\user_visual_review_server.py"
if not exist "%PYTHON%" (
  echo Python environment not found: %PYTHON%
  pause
  exit /b 1
)
"%PYTHON%" "%SCRIPT%" --port 8765
if errorlevel 1 (
  echo.
  echo User visual review failed. Review the message above.
  pause
)
