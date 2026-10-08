@echo off
setlocal
set "ROOT=%~dp0..\.."
set "PYTHON=%ROOT%\envs\whu-tscmdl\python.exe"
set "REQUEST=%~dp0scripts\request_d1_ptv2_pause.py"
set "SUITE=%ROOT%\lidar data\whu\derived\tscmdl\d1_ptv2_clean_repeats\20260818_protocol_v1"
"%PYTHON%" "%REQUEST%" --suite-root "%SUITE%"
pause
