$ErrorActionPreference = 'Stop'
$pythonExe = Join-Path $env:LOCALAPPDATA 'miniconda3\envs\yolo\python.exe'
if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw "YOLO Python environment not found: $pythonExe"
}
$env:VMMS_TREE_STREAM_IDS = 'hewentian_pano,jianshazui_pano_1,jianshazui_pano_2,stubbs_pano_0,stubbs_pano_1'
$env:VMMS_TREE_LABELER_TITLE = 'HK VMMS All Routes Coordinate Labeler'
$env:VMMS_LABELER_DERIVED_ROOT = Join-Path (Split-Path $PSScriptRoot -Parent) 'derived\all_routes_coordinate_labeler'
Set-Location (Join-Path $PSScriptRoot 'backend')
& $pythonExe -m uvicorn app.main:app --host 127.0.0.1 --port 8022
