@echo off
set "PYTHON=%~dp0runtime\env\python.exe"
if not exist "%PYTHON%" (
  echo Runtime is not prepared. Run START_C3_TRAINING.bat first.
  pause
  exit /b 1
)
"%PYTHON%" "%~dp0app\scripts\monitor_c3.py" --root "%~dp0"
