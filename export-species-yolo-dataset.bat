@echo off
setlocal
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -Command "& { conda run --no-capture-output -n yolo python .\export_species_yolo_dataset.py --overwrite }"
if errorlevel 1 (
  echo.
  echo Export failed. Check the error message above.
  pause
)
