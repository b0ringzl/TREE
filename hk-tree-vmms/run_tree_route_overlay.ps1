param(
    [double]$MaxDistanceM = 100.0,
    [double]$PrimaryDistanceM = 50.0
)

$ErrorActionPreference = 'Stop'
$pythonExe = Join-Path $env:LOCALAPPDATA 'miniconda3\envs\yolo\python.exe'
if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw "Python environment not found: $pythonExe"
}

$coordinateDir = Join-Path $PSScriptRoot 'outputs\coordinates'
$treeRoot = Split-Path $PSScriptRoot -Parent

& $pythonExe (Join-Path $PSScriptRoot 'overlay_tree_routes.py') `
    --route-geojson (Join-Path $coordinateDir 'route_trajectories.geojson') `
    --shapefile-dir (Join-Path $treeRoot 'arcgis_dd') `
    --output-dir $coordinateDir `
    --max-distance-m $MaxDistanceM `
    --primary-distance-m $PrimaryDistanceM

exit $LASTEXITCODE

