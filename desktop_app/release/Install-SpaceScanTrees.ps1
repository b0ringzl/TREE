param(
    [Parameter(Mandatory=$true)][string]$BundleDir,
    [string]$InstallRoot = (Join-Path $env:LOCALAPPDATA 'Programs\SpaceScan Trees'),
    [switch]$VerifyOnly,
    [switch]$NoShortcuts,
    [switch]$SkipRuntimeCheck
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
Import-Module Microsoft.PowerShell.Utility -ErrorAction Stop

function Get-Sha256([string]$Path) {
    $stream = [IO.File]::OpenRead($Path)
    $algorithm = [Security.Cryptography.SHA256]::Create()
    try { return [BitConverter]::ToString($algorithm.ComputeHash($stream)).Replace('-', '') }
    finally { $algorithm.Dispose(); $stream.Dispose() }
}

function Assert-SafeRelative([string]$Value) {
    if ([string]::IsNullOrWhiteSpace($Value) -or
        $Value -match '(^[/\\]|^[A-Za-z]:|(^|[/\\])\.\.([/\\]|$)|:)' ) {
        throw "Unsafe release path: $Value"
    }
}

function Get-ManifestPath([string]$Directory) {
    $matches = @(Get-ChildItem -LiteralPath $Directory -File -Filter '*.release.json')
    if ($matches.Count -ne 1) {
        throw "Expected exactly one *.release.json in $Directory, found $($matches.Count)"
    }
    return $matches[0].FullName
}

$bundle = (Resolve-Path -LiteralPath $BundleDir).Path
$manifestPath = Get-ManifestPath $bundle
$manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ($manifest.format -ne 1 -or $manifest.version -notmatch '^v\d{4}\.\d{2}\.\d{2}\.\d+$') {
    throw 'Unsupported release manifest or version'
}
$version = [string]$manifest.version
$parts = @($manifest.parts)
if ($parts.Count -eq 0) { throw 'Release has no parts' }

Write-Host "Verifying SpaceScan Trees $version release parts..."
$compressedBytes = [long]0
foreach ($part in $parts) {
    Assert-SafeRelative ([string]$part.name)
    $path = Join-Path $bundle ([string]$part.name)
    $file = Get-Item -LiteralPath $path -ErrorAction Stop
    if ($file.Length -ne [long]$part.bytes) { throw "Part size mismatch: $($part.name)" }
    $actual = Get-Sha256 $path
    if ($actual -ine [string]$part.sha256) { throw "Part SHA-256 mismatch: $($part.name)" }
    $compressedBytes += $file.Length
}
if ($VerifyOnly) {
    Write-Host "RELEASE_PARTS_OK $version ($($parts.Count) parts)"
    return
}

$root = [IO.Path]::GetFullPath($InstallRoot)
if ([IO.Path]::GetPathRoot($root).TrimEnd('\') -eq $root.TrimEnd('\')) {
    throw 'InstallRoot cannot be a drive root'
}
New-Item -ItemType Directory -Path $root -Force | Out-Null
$destination = Join-Path $root $version
if (Test-Path -LiteralPath $destination) {
    throw "Version already installed; no overwrite: $destination"
}
$drive = [IO.DriveInfo]::new([IO.Path]::GetPathRoot($root))
$needed = [long]$manifest.uncompressed_bytes + $compressedBytes + 1GB
if ($drive.AvailableFreeSpace -lt $needed) {
    throw "Insufficient free space. Need at least $([math]::Round($needed/1GB,1)) GB free."
}
$nonce = [guid]::NewGuid().ToString('N')
$archive = Join-Path $root ".download-$version-$nonce.tar.gz"
$staging = Join-Path $root ".staging-$version-$nonce"
New-Item -ItemType Directory -Path $staging -ErrorAction Stop | Out-Null

Write-Host 'Reassembling verified archive...'
$output = [IO.File]::Create($archive)
try {
    foreach ($part in $parts) {
        $source = [IO.File]::OpenRead((Join-Path $bundle ([string]$part.name)))
        try { $source.CopyTo($output, 8MB) }
        finally { $source.Dispose() }
    }
} finally { $output.Dispose() }

Write-Host 'Checking archive paths...'
& tar.exe -tzf $archive | ForEach-Object { Assert-SafeRelative $_ }
if ($LASTEXITCODE -ne 0) { throw "Archive listing failed: $LASTEXITCODE" }
Write-Host 'Extracting versioned installation...'
& tar.exe -xzf $archive -C $staging
if ($LASTEXITCODE -ne 0) { throw "Archive extraction failed: $LASTEXITCODE" }

Write-Host 'Verifying extracted files...'
$records = @($manifest.files)
if ($records.Count -ne [int]$manifest.file_count) { throw 'Manifest file count mismatch' }
$stagingPrefix = [IO.Path]::GetFullPath($staging).TrimEnd('\') + '\'
foreach ($record in $records) {
    Assert-SafeRelative ([string]$record.path)
    $path = [IO.Path]::GetFullPath((Join-Path $staging ([string]$record.path)))
    if (-not $path.StartsWith($stagingPrefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Extracted path escaped staging: $($record.path)"
    }
    $file = Get-Item -LiteralPath $path -ErrorAction Stop
    if ($file.Length -ne [long]$record.bytes) { throw "File size mismatch: $($record.path)" }
    $actual = Get-Sha256 $path
    if ($actual -ine [string]$record.sha256) { throw "File SHA-256 mismatch: $($record.path)" }
}
if (-not (Test-Path -LiteralPath (Join-Path $staging 'SpaceScan.Trees.Desktop.exe'))) {
    throw 'Desktop executable missing after extraction'
}

# The version folder is never overwritten. An older version remains usable for
# rollback even if the new version fails its first-launch check.
Move-Item -LiteralPath $staging -Destination $destination -ErrorAction Stop
Remove-Item -LiteralPath $archive -Force
Copy-Item -LiteralPath $manifestPath -Destination (Join-Path $destination 'release_manifest.json')

if (-not $SkipRuntimeCheck) {
    Write-Host 'Running first-launch integrity and GPU check...'
    & (Join-Path $destination 'runtime\python.exe') -s (Join-Path $destination 'launch.py') --check-only
    if ($LASTEXITCODE -ne 0) {
        throw "Runtime check failed. Prior shortcuts were not changed. Inspect $destination"
    }
}

if (-not $NoShortcuts) {
    $shell = New-Object -ComObject WScript.Shell
    $desktopLink = Join-Path ([Environment]::GetFolderPath('DesktopDirectory')) 'SpaceScan Trees Desktop.lnk'
    $startDir = Join-Path $env:APPDATA 'Microsoft\Windows\Start Menu\Programs'
    $startLink = Join-Path $startDir 'SpaceScan Trees Desktop.lnk'
    $backupDir = Join-Path $root ("shortcut-backups\" + $version)
    New-Item -ItemType Directory -Path $backupDir -Force | Out-Null
    foreach ($item in @(@($desktopLink,'Desktop.lnk'),@($startLink,'StartMenu.lnk'))) {
        if (Test-Path -LiteralPath $item[0]) {
            Copy-Item -LiteralPath $item[0] -Destination (Join-Path $backupDir $item[1]) -Force
        }
        $shortcut = $shell.CreateShortcut($item[0])
        $shortcut.TargetPath = Join-Path $destination 'SpaceScan.Trees.Desktop.exe'
        $shortcut.WorkingDirectory = $destination
        $shortcut.IconLocation = Join-Path $destination 'SpaceScanTrees-white.ico'
        $shortcut.Save()
    }
}

$receipt = @{
    version = $version
    installed_utc = [DateTime]::UtcNow.ToString('o')
    source_commit = [string]$manifest.source_commit
    manifest_sha256 = Get-Sha256 $manifestPath
    runtime_check_passed = (-not $SkipRuntimeCheck)
    installation = $destination
} | ConvertTo-Json -Depth 3
Set-Content -LiteralPath (Join-Path $destination 'install_receipt.json') -Value $receipt -Encoding UTF8
Write-Host "INSTALL_OK $destination"
