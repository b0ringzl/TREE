$ErrorActionPreference = 'Stop'
$workspace = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$source = Join-Path $workspace 'deliverables\香港市區常見27樹種_現有圖片副本_20260928_v2'
$run = Join-Path $workspace 'outputs\urban27_image_quality_20260928_v1'
$backup = Join-Path $run 'pre_curation_backup'
if (-not (Test-Path -LiteralPath $source -PathType Container) -or -not (Test-Path -LiteralPath $backup -PathType Container)) {
    throw 'Source or backup directory is missing'
}
$sourceFiles = @(Get-ChildItem -LiteralPath $source -Recurse -File)
$backupFiles = @(Get-ChildItem -LiteralPath $backup -Recurse -File)
if ($sourceFiles.Count -ne $backupFiles.Count) { throw 'File count mismatch' }
$rows = [System.Collections.Generic.List[object]]::new()
foreach ($file in $sourceFiles) {
    $relative = $file.FullName.Substring($source.Length+1)
    $copy = Join-Path $backup $relative
    if (-not (Test-Path -LiteralPath $copy -PathType Leaf)) { throw "Missing backup file: $relative" }
    $copyInfo = Get-Item -LiteralPath $copy
    if ($file.Length -ne $copyInfo.Length) { throw "Size mismatch: $relative" }
    $sourceHash = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    $backupHash = (Get-FileHash -LiteralPath $copy -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($sourceHash -ne $backupHash) { throw "SHA-256 mismatch: $relative" }
    $rows.Add([pscustomobject]@{
        relative_path=$relative
        original_path=$file.FullName
        backup_path=$copy
        bytes=$file.Length
        sha256=$sourceHash
    })
}
$manifest = Join-Path $run 'pre_curation_backup_manifest.csv'
$rows | Export-Csv -LiteralPath $manifest -NoTypeInformation -Encoding UTF8
$summary = [ordered]@{
    verified_at=(Get-Date).ToString('o')
    source=$source
    backup=$backup
    files=$rows.Count
    bytes=($rows | Measure-Object bytes -Sum).Sum
    sha256_verified=$true
    deployed_files_changed=$false
}
$summary | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $run 'pre_curation_backup_verification.json') -Encoding UTF8
Write-Output ($summary | ConvertTo-Json -Compress)
