param([switch]$InstallOnly)

$ErrorActionPreference = "Stop"
$UpdateRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$BundleRoot = Split-Path -Parent $UpdateRoot
if (-not (Test-Path -LiteralPath (Join-Path $BundleRoot "START_TRAINING.bat"))) {
    throw "Place this whole update folder inside the portable training folder."
}
Copy-Item -LiteralPath (Join-Path $UpdateRoot "payload\tools\run_training.ps1") `
    -Destination (Join-Path $BundleRoot "tools\run_training.ps1") -Force
Copy-Item -LiteralPath (Join-Path $UpdateRoot "payload\tools\switch_to_high_vram.ps1") `
    -Destination (Join-Path $BundleRoot "tools\switch_to_high_vram.ps1") -Force
Copy-Item -LiteralPath (Join-Path $UpdateRoot "payload\START_TRAINING_24GB.bat") `
    -Destination (Join-Path $BundleRoot "START_TRAINING_24GB.bat") -Force

$ManifestPath = Join-Path $BundleRoot "transfer_manifest.json"
if (Test-Path -LiteralPath $ManifestPath) {
    $ChangedPaths = @(
        "START_TRAINING_24GB.bat",
        "tools/run_training.ps1",
        "tools/switch_to_high_vram.ps1"
    )
    $Manifest = Get-Content -LiteralPath $ManifestPath -Raw | ConvertFrom-Json
    $Entries = @(
        $Manifest.files | Where-Object { $_.path -notin $ChangedPaths }
    )
    foreach ($RelativePath in $ChangedPaths) {
        $Path = Join-Path $BundleRoot ($RelativePath -replace "/", "\")
        $Entries += [pscustomobject]@{
            path = $RelativePath
            bytes = (Get-Item -LiteralPath $Path).Length
            sha256 = (
                Get-FileHash -LiteralPath $Path -Algorithm SHA256
            ).Hash.ToLowerInvariant()
        }
    }
    $Entries = @($Entries | Sort-Object path)
    $Manifest.files = $Entries
    $Manifest.file_count = $Entries.Count
    $Manifest.total_bytes = (
        $Entries | Measure-Object -Property bytes -Sum
    ).Sum
    $TemporaryManifest = "$ManifestPath.tmp"
    $Json = $Manifest | ConvertTo-Json -Depth 12
    [IO.File]::WriteAllText(
        $TemporaryManifest,
        $Json + [Environment]::NewLine,
        (New-Object Text.UTF8Encoding($false))
    )
    Move-Item -LiteralPath $TemporaryManifest -Destination $ManifestPath -Force
}

Write-Host "Update installed and the integrity manifest was refreshed."
if (-not $InstallOnly) {
    Write-Host "Starting the safe 24 GB switch..."
    & (Join-Path $BundleRoot "tools\switch_to_high_vram.ps1")
}
