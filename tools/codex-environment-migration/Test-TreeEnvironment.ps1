[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$PackageRoot,
    [string]$ProjectRoot = '',
    [string]$CodexHome = (Join-Path $env:USERPROFILE '.codex'),
    [switch]$PackageAuditOnly
)

if (-not $ProjectRoot) {
    $ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
}

Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'Migration.Common.ps1')

$package = (Resolve-Path -LiteralPath $PackageRoot).Path
$failures = New-Object System.Collections.Generic.List[string]
$warnings = New-Object System.Collections.Generic.List[string]

Write-Section 'Migration package safety audit'
$manifestFile = Join-Path $package 'MIGRATION-MANIFEST.json'
if (-not (Test-Path -LiteralPath $manifestFile)) { $failures.Add('MIGRATION-MANIFEST.json is missing.') }
else {
    $manifest = Get-Content -Raw -LiteralPath $manifestFile | ConvertFrom-Json
    if ($manifest.PackageKind -ne 'CodexEnvironmentOnly') { $failures.Add('PackageKind is not CodexEnvironmentOnly.') }
    if ($manifest.IncludesProjectFiles) { $failures.Add('Manifest unexpectedly says that project files are included.') }
    if ($manifest.IncludesAuthentication) { $failures.Add('Manifest unexpectedly says that authentication is included.') }
}

$forbiddenNames = @('auth.json', 'cap_sid', 'installation_id', 'session_index.jsonl', 'state_5.sqlite', 'logs_2.sqlite', 'goals_1.sqlite', 'memories_1.sqlite')
foreach ($file in Get-ChildItem -LiteralPath $package -File -Recurse -Force) {
    if ($forbiddenNames -contains $file.Name) { $failures.Add("Forbidden file found: $($file.FullName)") }
    if ($file.Extension -like '.sqlite*') { $failures.Add("SQLite state found: $($file.FullName)") }
    if ($file.Length -gt 100MB) { $warnings.Add("Unexpectedly large file (>100 MB): $($file.FullName)") }
    if ($file.Length -lt 10MB -and $file.Extension -in @('.toml', '.yml', '.yaml', '.json', '.txt', '.md')) {
        $text = Get-Content -Raw -LiteralPath $file.FullName -ErrorAction SilentlyContinue
        if ($text -match 'sk-[A-Za-z0-9_-]{16,}' -or $text -match 'gh[pousr]_[A-Za-z0-9]{16,}') {
            $failures.Add("Possible token found in: $($file.FullName)")
        }
    }
}
foreach ($directory in Get-ChildItem -LiteralPath $package -Directory -Recurse -Force) {
    if ($directory.Name -in @('.sandbox-secrets', 'sessions', 'browser', 'plugins', '.git')) {
        $failures.Add("Forbidden state directory found: $($directory.FullName)")
    }
}

$portableConfig = Join-Path $package 'codex-portable\config.portable.toml'
if (Test-Path -LiteralPath $portableConfig) {
    $configText = Get-Content -Raw -LiteralPath $portableConfig
    if ($configText -match 'sk-[A-Za-z0-9_-]{16,}' -or $configText -match 'gh[pousr]_[A-Za-z0-9]{16,}') {
        $failures.Add('Portable Codex config appears to contain a token.')
    }
    if ($configText -match '\[mcp_servers\.node_repl') { $failures.Add('Machine-generated node_repl MCP config was not removed.') }
    if ($configText -match '(?im)^\s*notify\s*=') { $failures.Add('Machine-specific notify command was not removed.') }
}

if ($failures.Count -eq 0) { Write-Host 'Package audit passed.' -ForegroundColor Green }
else { $failures | ForEach-Object { Write-Host "FAIL: $_" -ForegroundColor Red } }
$warnings | ForEach-Object { Write-Host "WARN: $_" -ForegroundColor Yellow }

if ($PackageAuditOnly) {
    if ($failures.Count -gt 0) { exit 1 } else { exit 0 }
}

Write-Section 'Tool comparison'
$expectedFile = Join-Path $package 'inventory\tool-versions.json'
$comparison = @()
if (Test-Path -LiteralPath $expectedFile) {
    foreach ($expected in @(Read-JsonArrayFile -Path $expectedFile)) {
        if (-not $expected.Found) { continue }
        $toolName = [string]$expected.Name
        $versionArguments = if ($toolName -eq 'nvidia-smi') { @('--query-gpu=name,driver_version,memory.total', '--format=csv,noheader') } else { @('--version') }
        $actual = Get-ToolVersionRecord -Name $toolName -VersionArguments $versionArguments
        $status = if (-not $actual.Found) { 'Missing' } elseif ($actual.Version -eq $expected.Version) { 'Match' } else { 'Different' }
        $comparison += [PSCustomObject]@{ Tool = $expected.Name; Status = $status; Expected = $expected.Version; Actual = $actual.Version }
        if ($status -eq 'Missing') { $warnings.Add("Expected tool is missing: $($expected.Name)") }
    }
}
$comparison | Format-Table -AutoSize -Wrap

Write-Section 'Codex portable files'
$codexChecks = @(
    [PSCustomObject]@{ Item = 'config.toml'; Present = (Test-Path -LiteralPath (Join-Path $CodexHome 'config.toml')) },
    [PSCustomObject]@{ Item = 'AGENTS.md'; Present = (Test-Path -LiteralPath (Join-Path $CodexHome 'AGENTS.md')) },
    [PSCustomObject]@{ Item = 'rules'; Present = (Test-Path -LiteralPath (Join-Path $CodexHome 'rules')) }
)
$codexChecks | Format-Table -AutoSize

Write-Section 'Conda environment presence'
$condaComparison = @()
$condaIndex = Join-Path $package 'conda\conda-environments.json'
if ((Get-Command conda -ErrorAction SilentlyContinue) -and (Test-Path -LiteralPath $condaIndex)) {
    $actualResult = Invoke-ExternalText -Command 'conda' -Arguments @('env', 'list', '--json')
    $actualInfo = ($actualResult.Output -join "`n") | ConvertFrom-Json
    $actualPrefixes = @($actualInfo.envs | ForEach-Object { [System.IO.Path]::GetFullPath($_).TrimEnd('\') })
    foreach ($expected in @(Read-JsonArrayFile -Path $condaIndex)) {
        if ($expected.IsBase) { continue }
        $expectedTarget = if ($expected.RelativeToProjectRoot) {
            [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot $expected.RelativeToProjectRoot)).TrimEnd('\')
        } else { $expected.Name }
        $present = if ($expected.RelativeToProjectRoot) { $actualPrefixes -contains $expectedTarget } else { @($actualPrefixes | ForEach-Object { Split-Path -Leaf $_ }) -contains $expected.Name }
        $condaComparison += [PSCustomObject]@{ Environment = $expected.Name; Target = $expectedTarget; Present = $present }
    }
}
$condaComparison | Format-Table -AutoSize

$report = [ordered]@{
    CheckedAt = (Get-Date).ToString('o')
    Computer = $env:COMPUTERNAME
    Failures = @($failures)
    Warnings = @($warnings)
    ToolComparison = @($comparison)
    CodexChecks = @($codexChecks)
    CondaComparison = @($condaComparison)
}
$reportPath = Join-Path $package ("verification-$($env:COMPUTERNAME)-" + (Get-Date -Format 'yyyyMMdd-HHmmss') + '.json')
Write-JsonFile -Path $reportPath -InputObject $report
Write-Host "Verification report: $reportPath"

if ($failures.Count -gt 0) { exit 1 }
exit 0
