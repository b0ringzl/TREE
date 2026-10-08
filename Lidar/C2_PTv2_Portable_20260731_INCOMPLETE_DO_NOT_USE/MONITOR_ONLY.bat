@echo off
set "PYTHON=%~dp0runtime\env\python.exe"
if not exist "%PYTHON%" (
  echo Runtime is not prepared. Run START_TRAINING.bat first.
  pause
  exit /b 1
)
"%PYTHON%" "%~dp0app\scripts\monitor_training.py" --run-dir "%~dp0output\run"
