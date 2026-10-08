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
    throw "NVIDIA driver not found. Install the NVIDIA driver first."
}
if (-not (Test-Path -LiteralPath $Archive)) {
    throw "Portable runtime archive is missing: $Archive"
}
if ((Test-Path -LiteralPath $Ready) -and (Test-Path -LiteralPath $Python)) {
    $ReadyState = Get-Content -LiteralPath $Ready -Raw | ConvertFrom-Json
    if ($ReadyState.root -ne $Root) {
        throw "This folder was moved after runtime setup. Delete runtime\env and runtime_ready.json, then start again."
    }
}

if (-not (Test-Path -LiteralPath $Python)) {
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

if (-not (Test-Path -LiteralPath (Join-Path $Root "diagnostics\integrity_report.json"))) {
    Write-Host "Verifying files copied through USB..."
    & $Python (Join-Path $Root "tools\verify_bundle.py")
    if ($LASTEXITCODE -ne 0) {
        throw "Bundle integrity verification failed."
    }
}

Write-Host "Checking GPU, CUDA, dataset, cache, and checkpoint..."
& $Python (Join-Path $Root "tools\device_probe.py")
if ($LASTEXITCODE -ne 0) {
    throw "Device compatibility check failed."
}

Write-Host "Portable runtime and device check passed."
