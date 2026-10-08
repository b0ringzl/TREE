@echo off
setlocal
set "ROOT=%~dp0"
set "ARCGIS_MCP_CONFIG=%ROOT%config.json"
"%ROOT%.venv\Scripts\python.exe" "%ROOT%server.py"

