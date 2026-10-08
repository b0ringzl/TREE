@echo off
set "ROOT=%~dp0..\.."
set "PYTHON=%ROOT%\envs\whu-tscmdl\python.exe"
set "MONITOR=%~dp0scripts\monitor_d1_resnet50_repeats.py"
"%PYTHON%" "%MONITOR%"
pause
