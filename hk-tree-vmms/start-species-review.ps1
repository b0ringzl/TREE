$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$appRoot = Join-Path $projectRoot "annotation_app"
$backendRoot = Join-Path $appRoot "backend"
$frontendRoot = Join-Path $appRoot "frontend"
$logRoot = Join-Path $appRoot "logs\exhaustive_species_review"
$pythonExe = Join-Path $env:LOCALAPPDATA "miniconda3\envs\yolo\python.exe"
$labelerRoot = Join-Path $projectRoot "derived\exhaustive_species_frame_labeler"
$importSummary = Join-Path $labelerRoot "IMPORT_SUMMARY.json"

if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw "Python environment not found: $pythonExe"
}
if (-not (Test-Path -LiteralPath $importSummary)) {
    throw "Imported review data not found: $importSummary"
}
$npmCommand = Get-Command npm.cmd -ErrorAction Stop
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null

if (-not $env:VMMS_ROOT) { $env:VMMS_ROOT = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\..\vmms")) }
$env:VMMS_TREE_STREAM_ID = "hewentian_pano"
$env:VMMS_FRAME_STREAM_IDS = "hewentian_pano,jianshazui_pano_1,jianshazui_pano_2,stubbs_pano_0,stubbs_pano_1"
$env:VMMS_FRAME_LABELER_DATASET = "exhaustive_species_review"
$env:VMMS_FRAME_LABELER_TITLE = "Hong Kong VMMS species review"
$env:VMMS_FRAME_MIN_ROUTE_DISTANCE_M_BY_STREAM = "hewentian_pano:1559.18"
$env:VMMS_LABELER_DERIVED_ROOT = Join-Path $projectRoot "derived\exhaustive_species_labeler"
$env:VMMS_FRAME_LABELER_ROOT = $labelerRoot
$env:VMMS_FRAME_SEED_CLASSES = Join-Path $labelerRoot "annotations\classes.json"
$env:VITE_API_BASE = "http://127.0.0.1:8025"

function Test-Url([string]$Url) {
    try {
        Invoke-WebRequest -UseBasicParsing -Uri $Url -TimeoutSec 2 | Out-Null
        return $true
    } catch {
        return $false
    }
}

if (-not (Test-Url "http://127.0.0.1:8025/api/health")) {
    Start-Process -FilePath $pythonExe `
        -ArgumentList @("-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8025") `
        -WorkingDirectory $backendRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $logRoot "backend.stdout.log") `
        -RedirectStandardError (Join-Path $logRoot "backend.stderr.log")
}

if (-not (Test-Url "http://127.0.0.1:5181/frame.html")) {
    Start-Process -FilePath $npmCommand.Source `
        -ArgumentList @("run", "dev", "--", "--port", "5181") `
        -WorkingDirectory $frontendRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $logRoot "frontend.stdout.log") `
        -RedirectStandardError (Join-Path $logRoot "frontend.stderr.log")
}

$backendReady = $false
for ($attempt = 0; $attempt -lt 60; $attempt++) {
    if (Test-Url "http://127.0.0.1:8025/api/health") {
        $backendReady = $true
        break
    }
    Start-Sleep -Milliseconds 500
}
if (-not $backendReady) {
    throw "Review backend failed to start. Check $logRoot\backend.stderr.log"
}

$frontendReady = $false
for ($attempt = 0; $attempt -lt 60; $attempt++) {
    if (Test-Url "http://127.0.0.1:5181/frame.html") {
        $frontendReady = $true
        break
    }
    Start-Sleep -Milliseconds 500
}
if (-not $frontendReady) {
    throw "Review frontend failed to start. Check $logRoot\frontend.stderr.log"
}

Start-Process "http://127.0.0.1:5181/frame.html"
Write-Host "VMMS species review tool is ready: http://127.0.0.1:5181/frame.html"
