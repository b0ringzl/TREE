param(
    [string]$OutputRoot = "",
    [int]$ImageWorkers = 4
)

$ErrorActionPreference = "Stop"
$sandbox = Split-Path -Parent $PSScriptRoot
$projectRoot = Split-Path -Parent (Split-Path -Parent $sandbox)
$python = Join-Path $projectRoot "envs\whu-tscmdl\python.exe"
$whuRoot = Join-Path $projectRoot "lidar data\whu"
$derivedRoot = Join-Path $whuRoot "derived\tscmdl"
if (-not $OutputRoot) {
    $OutputRoot = Join-Path $derivedRoot "c1_full_shared_dataset"
}

Push-Location $sandbox
try {
    & $python ".\scripts\export_c1_full_shared_dataset.py" `
        --dataset-root $whuRoot `
        --inventory (Join-Path $derivedRoot "c1a_full_inventory\inventory.json") `
        --shared-assets (Join-Path $derivedRoot "c1b_dual_track_plan\shared_assets.csv") `
        --benchmark-manifest (Join-Path $derivedRoot "c1b_dual_track_plan\benchmark_19.csv") `
        --road-manifest (Join-Path $derivedRoot "c1b_dual_track_plan\road_domain_16_samples.csv") `
        --classes-19 (Join-Path $derivedRoot "c1b_dual_track_plan\classes_19.json") `
        --classes-16 (Join-Path $derivedRoot "c1b_dual_track_plan\classes_16.json") `
        --output-root $OutputRoot `
        --chunk-size 1000000 `
        --view-candidates 3 `
        --previews-per-class 2 `
        --image-workers $ImageWorkers
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
