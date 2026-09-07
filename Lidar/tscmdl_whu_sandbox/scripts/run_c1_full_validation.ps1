param(
    [string]$DatasetRoot = ""
)

$ErrorActionPreference = "Stop"
$sandbox = Split-Path -Parent $PSScriptRoot
$projectRoot = Split-Path -Parent (Split-Path -Parent $sandbox)
$python = Join-Path $projectRoot "envs\whu-tscmdl\python.exe"
if (-not $DatasetRoot) {
    $DatasetRoot = Join-Path $projectRoot "lidar data\whu\derived\tscmdl\c1_full_shared_dataset"
}

Push-Location $sandbox
try {
    & $python ".\scripts\validate_c1_full_dataset.py" `
        --dataset-root $DatasetRoot `
        --output (Join-Path $DatasetRoot "validation.json") `
        --progress-every 250
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
