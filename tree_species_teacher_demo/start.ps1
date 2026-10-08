$ErrorActionPreference = 'Stop'
$projectDirectory = Split-Path $PSScriptRoot -Parent
$pythonExecutable = Join-Path $env:USERPROFILE 'AppData/Local/miniconda3/envs/yolo/python.exe'
if (!(Test-Path -LiteralPath $pythonExecutable)) { throw "Python environment not found: $pythonExecutable" }
Set-Location -LiteralPath $projectDirectory
& $pythonExecutable -m uvicorn tree_species_teacher_demo.server:app --host 127.0.0.1 --port 8039
