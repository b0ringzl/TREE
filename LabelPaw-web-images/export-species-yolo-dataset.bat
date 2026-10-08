@echo off
setlocal
cd /d "%~dp0"
for %%I in ("%~dp0..") do set "TREE_ROOT=%%~fI"
set "CONDA_EXE=%LOCALAPPDATA%\miniconda3\Scripts\conda.exe"
if not exist "%CONDA_EXE%" (
  echo Conda not found: %CONDA_EXE%
  exit /b 1
)
"%CONDA_EXE%" run --no-capture-output -n yolo python .\export_species_yolo_dataset.py --species-root "%TREE_ROOT%\external_datasets\inat_max150" --class-root "%TREE_ROOT%\选取树种" --merge-map "%TREE_ROOT%\tools\tree_species_label_merge_map.csv" --output "%TREE_ROOT%\models\web_tree_species_seg_dataset_v1" --overwrite
if errorlevel 1 (
  echo.
  echo Export failed. Check the error message above.
  pause
)
