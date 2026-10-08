$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$Archive = Join-Path $Root "runtime\whu-tscmdl-windows-x64.tar.gz"
$Environment = Join-Path $Root "runtime\env"
$Python = Join-Path $Environment "python.exe"
$Ready = Join-Path $Root "runtime\runtime_ready.json"

if (-not [Environment]::Is64BitOperatingSystem) {
    throw "This bundle requires 64-bit Windows."
}
if (-not (Get-Command "nvidia-smi.exe" -ErrorAction SilentlyContinue)) {
    throw "NVIDIA driver not found. Install or update the NVIDIA driver first."
}
if (-not (Test-Path -LiteralPath $Archive)) {
    throw "Portable runtime archive is missing: $Archive"
}
if ((Test-Path -LiteralPath $Ready) -and (Test-Path -LiteralPath $Python)) {
    $ReadyState = Get-Content -LiteralPath $Ready -Raw | ConvertFrom-Json
    if ($ReadyState.root -ne $Root) {
        throw "This folder moved after setup. Delete runtime\env and runtime_ready.json, then start again."
    }
}

if (-not (Test-Path -LiteralPath $Python)) {
    $DriveName = [System.IO.Path]::GetPathRoot($Root).Substring(0, 1)
    $FreeBytes = (Get-PSDrive -Name $DriveName).Free
    if ($FreeBytes -lt 12GB) {
        throw "At least 12 GiB free space is required for runtime extraction and three runs."
    }
    New-Item -ItemType Directory -Force -Path $Environment | Out-Null
    Write-Host "Extracting the verified Python/PyTorch runtime..."
    & tar.exe -xzf $Archive -C $Environment
    if ($LASTEXITCODE -ne 0) {
        throw "Runtime extraction failed with code $LASTEXITCODE"
    }
    $CondaUnpack = Join-Path $Environment "Scripts\conda-unpack.exe"
    if (Test-Path -LiteralPath $CondaUnpack) {
        & $CondaUnpack
        if ($LASTEXITCODE -ne 0) {
            throw "conda-unpack failed with code $LASTEXITCODE"
        }
    }
    @{
        status = "ready"
        root = $Root
        prepared_at = [DateTimeOffset]::Now.ToString("yyyy-MM-ddTHH:mm:sszzz")
    } | ConvertTo-Json | Set-Content -LiteralPath $Ready -Encoding UTF8
}

$IntegrityReport = Join-Path $Root "diagnostics\integrity_report.json"
$NeedsIntegrityCheck = $true
if (Test-Path -LiteralPath $IntegrityReport) {
    try {
        $Integrity = Get-Content -LiteralPath $IntegrityReport -Raw | ConvertFrom-Json
        $NeedsIntegrityCheck = -not (
            $Integrity.status -eq "passed" -and
            $Integrity.computer_name -eq $env:COMPUTERNAME -and
            $Integrity.root -eq $Root
        )
    } catch {
        $NeedsIntegrityCheck = $true
    }
}
if ($NeedsIntegrityCheck) {
    Write-Host "Verifying files copied through USB..."
    & $Python (Join-Path $Root "tools\verify_bundle.py")
    if ($LASTEXITCODE -ne 0) {
        throw "Bundle integrity verification failed."
    }
}

Write-Host "Checking CUDA, 24 GB GPU, dataset, and kNN cache..."
& $Python (Join-Path $Root "tools\device_probe_c3.py")
if ($LASTEXITCODE -ne 0) {
    throw "C3 device compatibility check failed."
}

$BatchProbe = Join-Path $Root "diagnostics\batch48_smoke"
$BatchProbeFinal = Join-Path $BatchProbe "final_metrics.json"
if (-not (Test-Path -LiteralPath $BatchProbeFinal)) {
    Write-Host "Running one real batch-48 CUDA training probe..."
    & $Python (Join-Path $Root "app\scripts\train_b5_ptv2.py") `
        --dataset-root (Join-Path $Root "input\dataset") `
        --knn-cache (Join-Path $Root "input\knn_cache") `
        --output-dir $BatchProbe `
        --epochs 1 `
        --batch-size 48 `
        --eval-batch-size 64 `
        --learning-rate 0.001 `
        --min-learning-rate 0.00001 `
        --weight-decay 0.05 `
        --label-smoothing 0.1 `
        --patience 20 `
        --seed 20260728 `
        --workers 0 `
        --progress-every 1 `
        --smoke-steps 1 `
        --amp
    if ($LASTEXITCODE -ne 0) {
        throw "The fixed batch-48 training probe failed. Review diagnostics\batch48_smoke."
    }
}

Write-Host "Portable runtime and C3 device check passed."
