$ErrorActionPreference = 'Stop'

$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..\..')).Path
$ReviewRoot = Join-Path $ProjectRoot 'lidar data\whu\derived\tscmdl\d2_exposure_stratified_evaluation\20260821_real_point_quality_review_v1'
$RuntimePath = Join-Path $ReviewRoot 'server_runtime.json'

if (-not (Test-Path -LiteralPath $RuntimePath)) {
    Write-Host 'No D2f runtime record was found.'
    exit 0
}

$Runtime = Get-Content -LiteralPath $RuntimePath -Raw | ConvertFrom-Json
$Process = Get-Process -Id ([int]$Runtime.pid) -ErrorAction SilentlyContinue
if ($null -ne $Process) {
    Stop-Process -Id $Process.Id
    Write-Host "Stopped D2f review server PID $($Process.Id)."
} else {
    Write-Host 'The recorded D2f server process is no longer running.'
}

$Runtime | Add-Member -NotePropertyName stopped_at -NotePropertyValue ((Get-Date).ToString('o')) -Force
$Runtime | ConvertTo-Json | Set-Content -LiteralPath $RuntimePath -Encoding utf8
