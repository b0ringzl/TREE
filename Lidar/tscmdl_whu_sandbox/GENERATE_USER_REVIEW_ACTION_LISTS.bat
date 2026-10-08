@echo off
setlocal
title WHU-STree User Review Action Lists
for %%I in ("%~dp0..\..") do set "TREE_ROOT=%%~fI"
set "PYTHON=%TREE_ROOT%\envs\whu-tscmdl\python.exe"
set "SCRIPT=%~dp0scripts\apply_user_visual_review.py"
if not exist "%PYTHON%" (
  echo Python environment not found: %PYTHON%
  pause
  exit /b 1
)
"%PYTHON%" "%SCRIPT%"
if errorlevel 1 (
  echo.
  echo Action-list export failed. Review the message above.
  pause
  exit /b 1
)
echo.
echo Action lists generated successfully.
pause
