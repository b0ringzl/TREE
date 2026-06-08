param(
    [switch]$CheckOnly
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$RuntimeRoot = Join-Path $ProjectRoot ".runtime"
$env:YOLO_CONFIG_DIR = Join-Path $RuntimeRoot "yolo-config"
$env:MPLCONFIGDIR = Join-Path $RuntimeRoot "matplotlib"
$env:HF_HOME = Join-Path $RuntimeRoot "huggingface"
$env:TORCH_HOME = Join-Path $RuntimeRoot "torch"
$env:QT_ENABLE_HIGHDPI_SCALING = "1"
$env:LABELPAW_DEFAULT_FORMAT = "yolo"
$env:LABELPAW_DEFAULT_MODE = "poly"

@(
    $RuntimeRoot,
    $env:YOLO_CONFIG_DIR,
    (Join-Path $env:YOLO_CONFIG_DIR "Ultralytics"),
    $env:MPLCONFIGDIR,
    $env:HF_HOME,
    $env:TORCH_HOME
) | ForEach-Object {
    New-Item -ItemType Directory -Force -Path $_ | Out-Null
}

$condaCandidates = @(
    "conda",
    "D:\miniconda\Scripts\conda.exe",
    "$env:USERPROFILE\miniconda3\Scripts\conda.exe",
    "$env:USERPROFILE\anaconda3\Scripts\conda.exe"
)

$condaExe = $null
foreach ($candidate in $condaCandidates) {
    try {
        $cmd = Get-Command $candidate -ErrorAction Stop
        $condaExe = $cmd.Source
        break
    } catch {
        if (Test-Path -LiteralPath $candidate) {
            $condaExe = $candidate
            break
        }
    }
}

if (-not $condaExe) {
    throw "Cannot find conda. Please install Miniconda/Anaconda or add conda to PATH."
}

Write-Host "Starting LabelPaw from $ProjectRoot"
Write-Host "Using conda executable: $condaExe"
Write-Host "Using conda environment: yolo"
Write-Host "Runtime/cache directory: $RuntimeRoot"

if ($CheckOnly) {
    & $condaExe run --no-capture-output -n yolo python -c "import torch; import main; print('LabelPaw environment check OK')"
    exit $LASTEXITCODE
}

& $condaExe run --no-capture-output -n yolo python main.py
