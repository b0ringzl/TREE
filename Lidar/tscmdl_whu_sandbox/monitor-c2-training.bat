@echo off
setlocal
for %%I in ("%~dp0..\..") do set "TREE_ROOT=%%~fI"
"%TREE_ROOT%\envs\whu-tscmdl\python.exe" "%~dp0scripts\monitor_training.py" --run-dir "%TREE_ROOT%\lidar data\whu\derived\tscmdl\c2_ptv2\run_batch16_seed20260731"
