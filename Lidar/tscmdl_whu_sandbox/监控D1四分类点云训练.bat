@echo off
setlocal
set "ROOT=%~dp0..\.."
set "PYTHON=%ROOT%\envs\whu-tscmdl\python.exe"
set "MONITOR=%~dp0scripts\monitor_d1_ptv2_repeats.py"
set "SUITE=%ROOT%\lidar data\whu\derived\tscmdl\d1_ptv2_clean_repeats\20260818_protocol_v1"
"%PYTHON%" "%MONITOR%" --suite-root "%SUITE%" --no-exit-on-complete
pause
