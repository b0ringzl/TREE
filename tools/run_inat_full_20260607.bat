@echo off
setlocal
for %%I in ("%~dp0..") do set "TREE_ROOT=%%~fI"
set "PYTHON=%LOCALAPPDATA%\miniconda3\envs\yolo\python.exe"
cd /d "%TREE_ROOT%"
"%PYTHON%" tools\download_inaturalist_dated_dataset.py --batch-date 2026-06-07 --max-new-per-species 150 --output-root external_datasets\inaturalist_dated --exclude-root external_datasets\inat_max150 --request-delay 0.75 --max-network-wait-minutes 0 > "%TREE_ROOT%\external_datasets\inaturalist_dated\2026-06-07\download_full.stdout.log" 2> "%TREE_ROOT%\external_datasets\inaturalist_dated\2026-06-07\download_full.stderr.log"
