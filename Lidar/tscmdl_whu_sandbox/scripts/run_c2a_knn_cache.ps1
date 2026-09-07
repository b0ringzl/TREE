param(
    [int]$MaxNewSamples = 0
)

$ErrorActionPreference = "Stop"
$Sandbox = Split-Path -Parent $PSScriptRoot
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $Sandbox)
$Python = Join-Path $ProjectRoot "envs\whu-tscmdl\python.exe"
$DerivedRoot = Join-Path $ProjectRoot "lidar data\whu\derived\tscmdl"
$Dataset = Join-Path $DerivedRoot "c1_full_shared_dataset"
$Output = Join-Path $DerivedRoot "c2a_ptv2\knn_memmap_k8"
$Log = Join-Path $Output "cache.log"

New-Item -ItemType Directory -Force -Path $Output | Out-Null
$Arguments = @(
    (Join-Path $Sandbox "scripts\cache_c2a_ptv2_knn_memmap.py"),
    "--dataset-root", $Dataset,
    "--output-dir", $Output,
    "--neighbours", "8",
    "--query-chunk-size", "2048",
    "--full-matrix-limit", "8192",
    "--checkpoint-every", "32",
    "--progress-every", "32"
)
if ($MaxNewSamples -gt 0) {
    $Arguments += @("--max-new-samples", "$MaxNewSamples")
}

& $Python @Arguments 2>&1 | Tee-Object -FilePath $Log -Append
exit $LASTEXITCODE
