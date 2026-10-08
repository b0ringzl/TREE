@echo off
setlocal
for %%I in ("%~dp0..\..") do set "TREE_ROOT=%%~fI"
"%TREE_ROOT%\envs\whu-tscmdl\python.exe" "%~dp0scripts\monitor_c3.py" --root "%TREE_ROOT%\lidar data\whu\derived\tscmdl\c3_ptv2_repeats"
