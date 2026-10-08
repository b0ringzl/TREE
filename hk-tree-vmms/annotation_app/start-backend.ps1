$ErrorActionPreference = 'Stop'
$pythonExe = Join-Path $env:LOCALAPPDATA 'miniconda3\envs\yolo\python.exe'
if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw "YOLO Python environment not found: $pythonExe"
}
Set-Location (Join-Path $PSScriptRoot 'backend')
& $pythonExe -m uvicorn app.main:app --host 127.0.0.1 --port 8021

