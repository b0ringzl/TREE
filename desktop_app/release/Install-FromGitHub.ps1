param(
    [Parameter(Mandatory=$true)][string]$Version,
    [string]$Repository = 'b0ringzl/SpaceScan-Trees-Desktop-Private',
    [string]$DownloadRoot = (Join-Path ([Environment]::GetFolderPath('UserProfile')) 'Downloads\SpaceScan Trees Releases'),
    [string]$InstallRoot = (Join-Path $env:LOCALAPPDATA 'Programs\SpaceScan Trees'),
    [switch]$DownloadOnly,
    [switch]$NoShortcuts
)

$ErrorActionPreference = 'Stop'
if ($Version -notmatch '^v\d{4}\.\d{2}\.\d{2}\.\d+$') { throw 'Invalid version' }
if ($Repository -notmatch '^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$') { throw 'Invalid repository' }
$gh = Get-Command gh -ErrorAction SilentlyContinue
if (-not $gh) { throw 'Install GitHub CLI, then run gh auth login for an account with private-repository access.' }
& $gh.Source auth status *> $null
if ($LASTEXITCODE -ne 0) { throw 'GitHub CLI is not authenticated. Run gh auth login.' }

$destination = Join-Path ([IO.Path]::GetFullPath($DownloadRoot)) $Version
if (Test-Path -LiteralPath $destination) {
    throw "Download directory already exists; no overwrite: $destination"
}
New-Item -ItemType Directory -Path $destination -Force | Out-Null
& $gh.Source release download $Version --repo $Repository --dir $destination
if ($LASTEXITCODE -ne 0) { throw "GitHub release download failed; inspect $destination" }

$installer = Join-Path $destination 'Install-SpaceScanTrees.ps1'
if (-not (Test-Path -LiteralPath $installer)) {
    throw 'Release is missing Install-SpaceScanTrees.ps1'
}
if ($DownloadOnly) {
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $installer -BundleDir $destination -VerifyOnly
} else {
    $arguments = @('-NoProfile','-ExecutionPolicy','Bypass','-File',$installer,
                   '-BundleDir',$destination,'-InstallRoot',$InstallRoot)
    if ($NoShortcuts) { $arguments += '-NoShortcuts' }
    & powershell.exe @arguments
}
if ($LASTEXITCODE -ne 0) { throw "Verification or installation failed: $LASTEXITCODE" }
Write-Host "GITHUB_RELEASE_OK $destination"
