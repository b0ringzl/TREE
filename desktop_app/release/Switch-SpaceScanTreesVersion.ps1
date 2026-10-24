param(
    [Parameter(Mandatory=$true)][string]$Version,
    [string]$InstallRoot = (Join-Path $env:LOCALAPPDATA 'Programs\SpaceScan Trees')
)

$ErrorActionPreference = 'Stop'
if ($Version -notmatch '^v\d{4}\.\d{2}\.\d{2}\.\d+$') {
    throw 'Invalid version name'
}
$root = [IO.Path]::GetFullPath($InstallRoot)
$target = Join-Path $root $Version
$exe = Join-Path $target 'SpaceScan.Trees.Desktop.exe'
$receipt = Join-Path $target 'install_receipt.json'
if (-not (Test-Path -LiteralPath $exe) -or -not (Test-Path -LiteralPath $receipt)) {
    throw "Version is not a verified installation: $target"
}
$installed = Get-Content -LiteralPath $receipt -Raw -Encoding UTF8 | ConvertFrom-Json
if ($installed.version -ne $Version -or -not $installed.runtime_check_passed) {
    throw "Version has no successful runtime check: $target"
}
$backup = Join-Path $root ("shortcut-backups\switch-" + [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ'))
New-Item -ItemType Directory -Path $backup -Force | Out-Null
$desktopLink = Join-Path ([Environment]::GetFolderPath('DesktopDirectory')) 'SpaceScan Trees Desktop.lnk'
$startLink = Join-Path (Join-Path $env:APPDATA 'Microsoft\Windows\Start Menu\Programs') 'SpaceScan Trees Desktop.lnk'
$shell = New-Object -ComObject WScript.Shell
foreach ($item in @(@($desktopLink,'Desktop.lnk'),@($startLink,'StartMenu.lnk'))) {
    if (Test-Path -LiteralPath $item[0]) {
        Copy-Item -LiteralPath $item[0] -Destination (Join-Path $backup $item[1])
    }
    $shortcut = $shell.CreateShortcut($item[0])
    $shortcut.TargetPath = $exe
    $shortcut.WorkingDirectory = $target
    $shortcut.IconLocation = Join-Path $target 'SpaceScanTrees-white.ico'
    $shortcut.Save()
}
Write-Host "ACTIVE_VERSION=$Version"
Write-Host "PREVIOUS_SHORTCUTS=$backup"
