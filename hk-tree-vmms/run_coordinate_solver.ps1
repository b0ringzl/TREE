param(
    [string]$VmmsRoot,
    [string]$OutputDir
)

$ErrorActionPreference = 'Stop'

if ([string]::IsNullOrWhiteSpace($VmmsRoot)) {
    $receiveRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
    $VmmsRoot = Join-Path $receiveRoot 'vmms'
}

if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = Join-Path $PSScriptRoot 'outputs\coordinates'
}

$pythonCandidates = @(
    (Join-Path $env:LOCALAPPDATA 'miniconda3\envs\yolo\python.exe'),
    (Join-Path $env:LOCALAPPDATA 'Programs\Python\Python310\python.exe')
)

$pythonExe = $null
foreach ($candidate in $pythonCandidates) {
    if (Test-Path -LiteralPath $candidate) {
        & $candidate -c "import pyproj" 2>$null
        if ($LASTEXITCODE -eq 0) {
            $pythonExe = $candidate
            break
        }
    }
}

if ($null -eq $pythonExe) {
    throw 'No Python environment with pyproj was found.'
}

& $pythonExe (Join-Path $PSScriptRoot 'coordinate_solver.py') `
    --vmms-root $VmmsRoot `
    --output-dir $OutputDir `
    --compute-downsample-bounds

exit $LASTEXITCODE

