[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$PackageRoot,

    [string]$ProjectRoot = '',
    [string]$CodexHome = (Join-Path $env:USERPROFILE '.codex'),
    [ValidateSet('History', 'Exact')]
    [string]$CondaDefinition = 'History',
    [switch]$ApplyCodex,
    [switch]$InstallWingetPackages,
    [switch]$CreateCondaEnvironments,
    [switch]$InstallVSCodeExtensions,
    [switch]$ApplyGitConfig,
    [switch]$RestoreMemories,
    [switch]$IgnorePackageVersions,
    [switch]$Full
)

if (-not $ProjectRoot) {
    $ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
}

Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'Migration.Common.ps1')

$package = (Resolve-Path -LiteralPath $PackageRoot).Path
$manifestPath = Join-Path $package 'MIGRATION-MANIFEST.json'
if (-not (Test-Path -LiteralPath $manifestPath)) { throw "Not a migration package: $package" }
$manifest = Get-Content -Raw -LiteralPath $manifestPath | ConvertFrom-Json
if ($manifest.PackageKind -ne 'CodexEnvironmentOnly') { throw "Unsupported package kind: $($manifest.PackageKind)" }

if ($Full) {
    $ApplyCodex = $true
    $InstallWingetPackages = $true
    $CreateCondaEnvironments = $true
    $InstallVSCodeExtensions = $true
    $ApplyGitConfig = $true
}

if (-not ($ApplyCodex -or $InstallWingetPackages -or $CreateCondaEnvironments -or $InstallVSCodeExtensions -or $ApplyGitConfig -or $RestoreMemories)) {
    Write-Host 'Nothing selected. This is a safe preview.' -ForegroundColor Yellow
    Write-Host 'Choose individual switches or use -Full. Memories always require -RestoreMemories.'
    & (Join-Path $PSScriptRoot 'Test-TreeEnvironment.ps1') -PackageRoot $package
    exit $LASTEXITCODE
}

$timestamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$codexBackup = Join-Path $CodexHome "migration-backup-$timestamp"

if ($InstallWingetPackages) {
    Write-Section 'Install applications from Winget inventory'
    $wingetFile = Join-Path $package 'inventory\winget-packages.json'
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) { Write-Warning 'Winget is not available; skipping.' }
    elseif (-not (Test-Path -LiteralPath $wingetFile)) { Write-Warning 'No Winget export is present; skipping.' }
    else {
        $arguments = @('import', '--import-file', $wingetFile, '--ignore-unavailable', '--accept-package-agreements', '--accept-source-agreements', '--disable-interactivity')
        if ($IgnorePackageVersions) { $arguments += '--ignore-versions' }
        $result = Invoke-ExternalText -Command 'winget' -Arguments $arguments -AllowFailure
        $result.Output | ForEach-Object { Write-Host $_ }
        if ($result.ExitCode -ne 0) { Write-Warning "Winget import returned $($result.ExitCode). Continue with the remaining stages." }
    }
}

if ($ApplyCodex) {
    Write-Section 'Restore portable Codex configuration'
    Ensure-Directory $CodexHome | Out-Null
    foreach ($name in @('config.toml', 'AGENTS.md', 'rules')) { Backup-PathIfPresent -Path (Join-Path $CodexHome $name) -BackupRoot $codexBackup }

    $portableRoot = Join-Path $package 'codex-portable'
    $configTemplate = Join-Path $portableRoot 'config.portable.toml'
    if (Test-Path -LiteralPath $configTemplate) {
        $configText = Get-Content -Raw -LiteralPath $configTemplate
        $newHome = [System.IO.Path]::GetFullPath($env:USERPROFILE).TrimEnd('\')
        $newProject = [System.IO.Path]::GetFullPath($ProjectRoot).TrimEnd('\')
        $configText = $configText.Replace('__USER_HOME_ESCAPED__', $newHome.Replace('\', '\\'))
        $configText = $configText.Replace('__PROJECT_ROOT_ESCAPED__', $newProject.Replace('\', '\\'))
        $configText = $configText.Replace('__USER_HOME__', $newHome)
        $configText = $configText.Replace('__PROJECT_ROOT_LOWER__', $newProject.ToLowerInvariant())
        $configText = $configText.Replace('__PROJECT_ROOT__', $newProject)
        if ($configText -match '__REQUIRED_SECRET__') {
            $candidate = Join-Path $CodexHome 'config.toml.migration-candidate'
            Write-Utf8File -Path $candidate -Content $configText
            Write-Warning "Secrets must be re-entered. A candidate was written instead of replacing config.toml: $candidate"
        }
        else {
            Write-Utf8File -Path (Join-Path $CodexHome 'config.toml') -Content $configText
        }
    }
    if (Test-Path -LiteralPath (Join-Path $portableRoot 'AGENTS.md')) {
        Copy-Item -LiteralPath (Join-Path $portableRoot 'AGENTS.md') -Destination (Join-Path $CodexHome 'AGENTS.md') -Force
    }
    if (Test-Path -LiteralPath (Join-Path $portableRoot 'rules')) {
        Copy-DirectoryPortable -Source (Join-Path $portableRoot 'rules') -Destination (Join-Path $CodexHome 'rules')
    }
    if (Test-Path -LiteralPath (Join-Path $portableRoot 'skills')) {
        $targetSkills = Ensure-Directory (Join-Path $CodexHome 'skills')
        foreach ($skill in Get-ChildItem -LiteralPath (Join-Path $portableRoot 'skills') -Directory) {
            Backup-PathIfPresent -Path (Join-Path $targetSkills $skill.Name) -BackupRoot (Join-Path $codexBackup 'skills')
            Copy-DirectoryPortable -Source $skill.FullName -Destination (Join-Path $targetSkills $skill.Name)
        }
    }
    Write-Host "Existing Codex files were backed up to: $codexBackup"
    Write-Host 'Restart the Codex desktop app after this script finishes.'
}

if ($RestoreMemories) {
    Write-Section 'Restore explicitly requested memories'
    $memorySource = Join-Path $package 'codex-portable\memories'
    if (-not (Test-Path -LiteralPath $memorySource)) { Write-Warning 'This package does not contain memories.' }
    else {
        $memoryTarget = Join-Path $CodexHome 'memories'
        Backup-PathIfPresent -Path $memoryTarget -BackupRoot $codexBackup
        Copy-DirectoryPortable -Source $memorySource -Destination $memoryTarget
    }
}

if ($ApplyGitConfig) {
    Write-Section 'Restore selected global Git settings'
    $gitFile = Join-Path $package 'inventory\git-global-portable.json'
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) { Write-Warning 'Git is not installed; skipping.' }
    elseif (Test-Path -LiteralPath $gitFile) {
        foreach ($setting in @(Read-JsonArrayFile -Path $gitFile)) {
            & git config --global $setting.Key $setting.Value
            if ($LASTEXITCODE -ne 0) { Write-Warning "Could not set Git key: $($setting.Key)" }
        }
    }
}

if ($InstallVSCodeExtensions) {
    Write-Section 'Install VS Code extensions'
    $extensionFile = Join-Path $package 'inventory\vscode-extensions.txt'
    if (-not (Get-Command code -ErrorAction SilentlyContinue)) { Write-Warning 'VS Code command line is not available; skipping.' }
    elseif (Test-Path -LiteralPath $extensionFile) {
        foreach ($entry in Get-Content -LiteralPath $extensionFile | Where-Object { $_.Trim() }) {
            $extensionId = ($entry -split '@')[0]
            & code --install-extension $extensionId --force
            if ($LASTEXITCODE -ne 0) { Write-Warning "Could not install VS Code extension: $extensionId" }
        }
    }
}

if ($CreateCondaEnvironments) {
    Write-Section "Create Conda environments from $CondaDefinition definitions"
    $condaIndex = Join-Path $package 'conda\conda-environments.json'
    if (-not (Get-Command conda -ErrorAction SilentlyContinue)) { Write-Warning 'Conda is not installed; install Miniconda first and rerun this stage.' }
    elseif (Test-Path -LiteralPath $condaIndex) {
        $existingResult = Invoke-ExternalText -Command 'conda' -Arguments @('env', 'list', '--json')
        $existingInfo = ($existingResult.Output -join "`n") | ConvertFrom-Json
        $existing = @($existingInfo.envs | ForEach-Object { [System.IO.Path]::GetFullPath($_).TrimEnd('\') })
        foreach ($environment in @(Read-JsonArrayFile -Path $condaIndex)) {
            if ($environment.IsBase) { continue }
            $definitionFile = if ($CondaDefinition -eq 'Exact') { $environment.ExactFile } else { $environment.HistoryFile }
            $definitionPath = Join-Path (Join-Path $package 'conda') $definitionFile
            if (-not (Test-Path -LiteralPath $definitionPath)) { Write-Warning "Missing Conda definition: $definitionPath"; continue }
            if ($environment.RelativeToProjectRoot) {
                $targetPrefix = [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot $environment.RelativeToProjectRoot)).TrimEnd('\')
                if ($existing -contains $targetPrefix) { Write-Host "Conda environment already exists: $targetPrefix"; continue }
                Ensure-Directory (Split-Path -Parent $targetPrefix) | Out-Null
                $result = Invoke-ExternalText -Command 'conda' -Arguments @('env', 'create', '--prefix', $targetPrefix, '--file', $definitionPath, '--yes') -AllowFailure
            }
            else {
                $existingNames = @($existing | ForEach-Object { Split-Path -Leaf $_ })
                if ($existingNames -contains $environment.Name) { Write-Host "Conda environment already exists: $($environment.Name)"; continue }
                $result = Invoke-ExternalText -Command 'conda' -Arguments @('env', 'create', '--name', $environment.Name, '--file', $definitionPath, '--yes') -AllowFailure
            }
            $result.Output | ForEach-Object { Write-Host $_ }
            if ($result.ExitCode -ne 0) { Write-Warning "Conda creation failed for $($environment.Name). Try -CondaDefinition Exact or inspect the YAML." }
        }
    }
}

Write-Section 'Post-restore verification'
& (Join-Path $PSScriptRoot 'Test-TreeEnvironment.ps1') -PackageRoot $package -ProjectRoot $ProjectRoot
exit $LASTEXITCODE
