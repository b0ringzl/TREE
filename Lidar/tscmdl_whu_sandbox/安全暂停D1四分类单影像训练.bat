@echo off
set "ROOT=%~dp0..\.."
set "PYTHON=%ROOT%\envs\whu-tscmdl\python.exe"
set "PAUSE_SCRIPT=%~dp0scripts\request_d1_resnet50_pause.py"
"%PYTHON%" "%PAUSE_SCRIPT%"
pause
