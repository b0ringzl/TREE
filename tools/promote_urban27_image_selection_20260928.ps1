$ErrorActionPreference = 'Stop'
$workspace = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$current = Join-Path $workspace 'deliverables\香港市區常見27樹種_現有圖片副本_20260928_v2'
$curated = Join-Path $workspace 'deliverables\香港市區常見27樹種_現有圖片精簡版_20260928_v1'
$run = Join-Path $workspace 'outputs\urban27_image_quality_20260928_v1'
$archived = Join-Path $run 'original_path_before_curation'
$backup = Join-Path $run 'pre_curation_backup'
foreach ($path in @($current,$curated,$run,$backup,$archived)) {
    $full = [IO.Path]::GetFullPath($path)
    if (-not $full.StartsWith($workspace + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Path outside workspace: $path"
    }
}
if (-not (Test-Path -LiteralPath $current -PathType Container)) { throw 'Current folder missing' }
if (-not (Test-Path -LiteralPath $curated -PathType Container)) { throw 'Curated folder missing' }
if (Test-Path -LiteralPath $archived) { throw 'Archive destination already exists' }

$verification = Get-Content -LiteralPath (Join-Path $run 'pre_curation_backup_verification.json') -Raw | ConvertFrom-Json
if (-not $verification.sha256_verified -or $verification.backup -ne $backup) { throw 'Verified backup mismatch' }
$manifest = @(Import-Csv -LiteralPath (Join-Path $run 'pre_curation_backup_manifest.csv'))
$actual = @(Get-ChildItem -LiteralPath $current -Recurse -File)
if ($actual.Count -ne $manifest.Count) { throw 'The current folder changed during testing; no replacement performed' }
foreach ($item in $manifest) {
    $path = Join-Path $current $item.relative_path
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "The current folder changed: $($item.relative_path)" }
    $file = Get-Item -LiteralPath $path
    if ($file.Length -ne [long]$item.bytes -or
        (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $item.sha256) {
        throw "The current folder changed: $($item.relative_path)"
    }
}
$curatedFiles = @(Get-ChildItem -LiteralPath $curated -Recurse -File)
$curatedBytes = ($curatedFiles | Measure-Object Length -Sum).Sum
if ($curatedBytes -gt 1500000000) { throw 'Curated folder exceeds 1.5 GB' }
$selection = Get-Content -LiteralPath (Join-Path $run 'selection_summary.json') -Raw | ConvertFrom-Json
if ($curatedFiles.Count -lt $selection.image_count_after) { throw 'Curated image count inconsistent' }

Move-Item -LiteralPath $current -Destination $archived -ErrorAction Stop
try {
    Move-Item -LiteralPath $curated -Destination $current -ErrorAction Stop
} catch {
    if (-not (Test-Path -LiteralPath $current) -and (Test-Path -LiteralPath $archived)) {
        Move-Item -LiteralPath $archived -Destination $current -ErrorAction Stop
    }
    throw
}
$finalBytes = (Get-ChildItem -LiteralPath $current -Recurse -File | Measure-Object Length -Sum).Sum
if ($finalBytes -gt 1500000000) { throw 'Post-promotion size check failed' }
$record = [ordered]@{
    promoted_at=(Get-Date).ToString('o')
    current_folder=$current
    original_folder_archive=$archived
    verified_pre_curation_backup=$backup
    curated_folder_bytes=$finalBytes
    images=$selection.image_count_after
    deployed_files_changed=$false
}
$record | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $run 'promotion.json') -Encoding UTF8
Write-Output ($record | ConvertTo-Json -Compress)
