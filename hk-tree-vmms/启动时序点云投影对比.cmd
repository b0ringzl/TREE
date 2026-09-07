@echo off
setlocal
cd /d "%~dp0"
set "PYTHON_EXE=%~dp0..\envs\whu-tscmdl\python.exe"
if not exist "%PYTHON_EXE%" (
  echo [ERROR] Python environment not found: %PYTHON_EXE%
  pause
  exit /b 1
)
"%PYTHON_EXE%" "%~dp0render_temporal_lidar_panorama.py" --frame-id 004338
if errorlevel 1 (
  echo.
  echo Projection failed. See the message above.
  pause
  exit /b 1
)
echo.
echo Projection completed. Opening comparison image...
start "" "%~dp0derived\temporal_lidar_projection\frame_004338\temporal_accumulation_comparison.jpg"
pause
