[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$DestinationRoot,

    [string]$ProjectRoot = '',
    [string]$CodexHome = (Join-Path $env:USERPROFILE '.codex'),
    [string]$PackageName,
    [switch]$SkipWingetExport,
    [switch]$IncludeMemories
)

if (-not $ProjectRoot) {
    $ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
}

Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'Migration.Common.ps1')

$project = (Resolve-Path -LiteralPath $ProjectRoot).Path.TrimEnd('\')
$destination = Ensure-Directory $DestinationRoot
if (Test-IsChildPath -Parent $project -Child $destination) {
    throw 'DestinationRoot cannot be inside the project. Use a USB drive or another directory.'
}

if (-not $PackageName) { $PackageName = 'TREE-Codex-Environment-' + (Get-Date -Format 'yyyyMMdd-HHmmss') }
$packageRoot = Join-Path $destination $PackageName
if (Test-Path -LiteralPath $packageRoot) { throw "Package already exists: $packageRoot" }

$inventoryRoot = Ensure-Directory (Join-Path $packageRoot 'inventory')
$codexRoot = Ensure-Directory (Join-Path $packageRoot 'codex-portable')
$condaRoot = Ensure-Directory (Join-Path $packageRoot 'conda')
$projectManifestRoot = Ensure-Directory (Join-Path $packageRoot 'project-manifests')
$scriptRoot = Ensure-Directory (Join-Path $packageRoot 'scripts')
$logRoot = Ensure-Directory (Join-Path $packageRoot 'logs')
$secretPlaceholders = @()

Write-Section 'System and tool inventory'
$computerSystem = Get-CimInstance Win32_ComputerSystem | Select-Object Manufacturer, Model, TotalPhysicalMemory
$operatingSystem = Get-CimInstance Win32_OperatingSystem | Select-Object Caption, Version, BuildNumber, OSArchitecture
$processors = @(Get-CimInstance Win32_Processor | Select-Object Name, NumberOfCores, NumberOfLogicalProcessors)
$video = @(Get-CimInstance Win32_VideoController | Select-Object Name, DriverVersion, AdapterRAM)
Write-JsonFile -Path (Join-Path $inventoryRoot 'hardware.json') -InputObject ([ordered]@{
    Computer = $computerSystem
    OperatingSystem = $operatingSystem
    Processors = $processors
    VideoControllers = $video
})

$toolSpecs = @(
    @{ Name = 'git'; Args = @('--version') },
    @{ Name = 'codex'; Args = @('--version') },
    @{ Name = 'python'; Args = @('--version') },
    @{ Name = 'py'; Args = @('--version') },
    @{ Name = 'conda'; Args = @('--version') },
    @{ Name = 'uv'; Args = @('--version') },
    @{ Name = 'node'; Args = @('--version') },
    @{ Name = 'npm'; Args = @('--version') },
    @{ Name = 'pnpm'; Args = @('--version') },
    @{ Name = 'yarn'; Args = @('--version') },
    @{ Name = 'code'; Args = @('--version') },
    @{ Name = 'winget'; Args = @('--version') },
    @{ Name = 'docker'; Args = @('--version') },
    @{ Name = 'nvidia-smi'; Args = @('--query-gpu=name,driver_version,memory.total', '--format=csv,noheader') }
)
$tools = foreach ($spec in $toolSpecs) { Get-ToolVersionRecord -Name $spec.Name -VersionArguments $spec.Args }
Write-JsonFile -Path (Join-Path $inventoryRoot 'tool-versions.json') -InputObject @($tools)
$tools | Format-Table -AutoSize

if (Get-Command python -ErrorAction SilentlyContinue) {
    $pipFreeze = Invoke-ExternalText -Command 'python' -Arguments @('-m', 'pip', 'freeze') -AllowFailure
    Write-Utf8File -Path (Join-Path $inventoryRoot 'system-python-pip-freeze.txt') -Content $pipFreeze.Output
}

if (Get-Command code -ErrorAction SilentlyContinue) {
    $extensions = Invoke-ExternalText -Command 'code' -Arguments @('--list-extensions', '--show-versions') -AllowFailure
    Write-Utf8File -Path (Join-Path $inventoryRoot 'vscode-extensions.txt') -Content (($extensions.Output | Where-Object { $_ -match '^[A-Za-z0-9_.-]+@[0-9]' }) | Sort-Object -Unique)
}

if (Get-Command npm -ErrorAction SilentlyContinue) {
    $npmRootResult = Invoke-ExternalText -Command 'npm' -Arguments @('root', '-g') -AllowFailure
    $npmRoot = $npmRootResult.Output | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -Last 1
    $npmPackages = @()
    if ($npmRoot) {
        $packageJsonFiles = Get-ChildItem -LiteralPath $npmRoot -Directory -Force | ForEach-Object {
            if ($_.Name.StartsWith('@')) { Get-ChildItem -LiteralPath $_.FullName -Directory -Force | ForEach-Object { Join-Path $_.FullName 'package.json' } }
            else { Join-Path $_.FullName 'package.json' }
        }
        foreach ($packageFile in $packageJsonFiles) {
            if (Test-Path -LiteralPath $packageFile) {
                try {
                    $package = Get-Content -Raw -LiteralPath $packageFile | ConvertFrom-Json
                    $npmPackages += [PSCustomObject]@{ Name = $package.name; Version = $package.version }
                }
                catch { }
            }
        }
    }
    Write-JsonFile -Path (Join-Path $inventoryRoot 'npm-global-packages.json') -InputObject @($npmPackages | Sort-Object Name)
}

if (Get-Command git -ErrorAction SilentlyContinue) {
    $gitKeys = @('user.name', 'user.email', 'core.autocrlf', 'core.longpaths', 'init.defaultBranch', 'pull.rebase', 'push.autoSetupRemote', 'credential.helper')
    $gitSettings = foreach ($key in $gitKeys) {
        $result = Invoke-ExternalText -Command 'git' -Arguments @('config', '--global', '--get', $key) -AllowFailure
        if ($result.ExitCode -eq 0 -and $result.Output.Count -gt 0) {
            [PSCustomObject]@{ Key = $key; Value = ($result.Output -join "`n") }
        }
    }
    Write-JsonFile -Path (Join-Path $inventoryRoot 'git-global-portable.json') -InputObject @($gitSettings)
}

Write-Section 'Winget application inventory'
if ((Get-Command winget -ErrorAction SilentlyContinue) -and (-not $SkipWingetExport)) {
    $wingetPath = Join-Path $inventoryRoot 'winget-packages.json'
    $wingetResult = Invoke-ExternalText -Command 'winget' -Arguments @('export', '--output', $wingetPath, '--include-versions', '--accept-source-agreements') -AllowFailure
    Write-Utf8File -Path (Join-Path $logRoot 'winget-export.log') -Content $wingetResult.Output
    if ($wingetResult.ExitCode -ne 0) { Write-Warning 'Winget export failed; see logs\winget-export.log.' }
}

Write-Section 'Conda environment definitions'
$condaEnvironments = @()
if (Get-Command conda -ErrorAction SilentlyContinue) {
    $condaList = Invoke-ExternalText -Command 'conda' -Arguments @('env', 'list', '--json')
    $condaInfo = ($condaList.Output -join "`n") | ConvertFrom-Json
    $condaBaseResult = Invoke-ExternalText -Command 'conda' -Arguments @('info', '--base')
    $condaBasePrefix = [System.IO.Path]::GetFullPath(($condaBaseResult.Output | Where-Object { $_.Trim() } | Select-Object -Last 1)).TrimEnd('\')
    foreach ($prefix in @($condaInfo.envs)) {
        $prefixPath = [System.IO.Path]::GetFullPath($prefix).TrimEnd('\')
        $leaf = Split-Path -Leaf $prefixPath
        $isBase = $prefixPath -ieq $condaBasePrefix
        $name = if ($isBase) { 'base' } else { $leaf }
        $safeName = Get-SafeFileName $name
        $historyName = "$safeName.history.yml"
        $exactName = "$safeName.exact.yml"
        $historyResult = Invoke-ExternalText -Command 'conda' -Arguments @('env', 'export', '--prefix', $prefixPath, '--from-history') -AllowFailure
        $exactResult = Invoke-ExternalText -Command 'conda' -Arguments @('env', 'export', '--prefix', $prefixPath) -AllowFailure
        $historyLines = @($historyResult.Output | Where-Object { $_ -notmatch '^prefix:\s*' })
        $exactLines = @($exactResult.Output | Where-Object { $_ -notmatch '^prefix:\s*' })
        for ($index = 0; $index -lt $historyLines.Count; $index++) {
            if ($historyLines[$index] -match '^name:\s*') { $historyLines[$index] = "name: $name" }
            if ($historyLines[$index] -match '^\s{2}([A-Za-z0-9_]*(TOKEN|SECRET|PASSWORD|API_KEY|CREDENTIAL)[A-Za-z0-9_]*):') {
                $historyLines[$index] = "  $($matches[1]): __REQUIRED_SECRET__"
                $secretPlaceholders += "conda:${name}:$($matches[1])"
            }
        }
        for ($index = 0; $index -lt $exactLines.Count; $index++) {
            if ($exactLines[$index] -match '^name:\s*') { $exactLines[$index] = "name: $name" }
            if ($exactLines[$index] -match '^\s{2}([A-Za-z0-9_]*(TOKEN|SECRET|PASSWORD|API_KEY|CREDENTIAL)[A-Za-z0-9_]*):') {
                $exactLines[$index] = "  $($matches[1]): __REQUIRED_SECRET__"
                $secretPlaceholders += "conda:${name}:$($matches[1])"
            }
        }
        Write-Utf8File -Path (Join-Path $condaRoot $historyName) -Content $historyLines
        Write-Utf8File -Path (Join-Path $condaRoot $exactName) -Content $exactLines
        $insideProject = Test-IsChildPath -Parent $project -Child $prefixPath
        $relativePrefix = if ($insideProject) { Get-RelativePathPortable -BasePath $project -Path $prefixPath } else { $null }
        $condaEnvironments += [PSCustomObject]@{
            Name = $name
            OriginalPrefix = $prefixPath
            IsBase = $isBase
            RelativeToProjectRoot = $relativePrefix
            HistoryFile = $historyName
            ExactFile = $exactName
            HistoryExportSucceeded = ($historyResult.ExitCode -eq 0)
            ExactExportSucceeded = ($exactResult.ExitCode -eq 0)
        }
    }
}
Write-JsonFile -Path (Join-Path $condaRoot 'conda-environments.json') -InputObject @($condaEnvironments)

Write-Section 'Small project dependency manifests'
$manifestNames = @(
    'requirements*.txt', 'environment*.yml', 'environment*.yaml', 'pyproject.toml', 'uv.lock',
    'Pipfile', 'Pipfile.lock', 'poetry.lock', 'package.json', 'package-lock.json', 'pnpm-lock.yaml',
    'yarn.lock', 'Dockerfile', 'docker-compose*.yml', 'docker-compose*.yaml', '.python-version', '.nvmrc'
)
$projectManifests = @()
$candidateFiles = & rg --files $project
foreach ($file in $candidateFiles) {
    $leaf = Split-Path -Leaf $file
    $matched = $false
    foreach ($pattern in $manifestNames) { if ($leaf -like $pattern) { $matched = $true; break } }
    if (-not $matched) { continue }
    $relative = Get-RelativePathPortable -BasePath $project -Path $file
    $destinationFile = Join-Path $projectManifestRoot $relative
    Ensure-Directory (Split-Path -Parent $destinationFile) | Out-Null
    Copy-Item -LiteralPath $file -Destination $destinationFile -Force
    $projectManifests += [PSCustomObject]@{ RelativePath = $relative; FileName = $leaf }
}
Write-JsonFile -Path (Join-Path $projectManifestRoot 'project-manifests.json') -InputObject @($projectManifests)

Write-Section 'Portable Codex configuration'
$configSource = Join-Path $CodexHome 'config.toml'
if (Test-Path -LiteralPath $configSource) {
    $sourceHome = [System.IO.Path]::GetFullPath($env:USERPROFILE).TrimEnd('\')
    $currentSection = ''
    $skipSection = $false
    $portableLines = New-Object System.Collections.Generic.List[string]
    foreach ($line in Get-Content -LiteralPath $configSource) {
        if ($line -match '^\s*\[(.+)\]\s*$') {
            $currentSection = $matches[1]
            $skipSection = ($currentSection -like 'marketplaces.*') -or
                ($currentSection -like 'mcp_servers.node_repl*') -or
                ($currentSection -eq 'shell_environment_policy.set')
            if ($currentSection -like 'projects.*' -and $currentSection -notmatch [regex]::Escape($project.ToLowerInvariant())) {
                $skipSection = $true
            }
        }
        if ($skipSection) { continue }
        if ($line -match '^\s*notify\s*=') { continue }

        $portable = $line
        $portable = $portable.Replace($sourceHome.Replace('\', '\\'), '__USER_HOME_ESCAPED__')
        $portable = $portable.Replace($project.Replace('\', '\\'), '__PROJECT_ROOT_ESCAPED__')
        $portable = $portable.Replace($sourceHome, '__USER_HOME__')
        $portable = $portable.Replace($project, '__PROJECT_ROOT__')
        $portable = $portable.Replace($project.ToLowerInvariant(), '__PROJECT_ROOT_LOWER__')

        if ($portable -match '^\s*([A-Za-z0-9_.-]*(token|secret|password|api[_-]?key|credential)[A-Za-z0-9_.-]*)\s*=') {
            $key = $matches[1]
            $indent = $portable.Substring(0, $portable.IndexOf($key))
            $portable = "$indent$key = '__REQUIRED_SECRET__'"
            $secretPlaceholders += "$currentSection.$key"
        }
        $portable = $portable -replace 'sk-[A-Za-z0-9_-]{16,}', '__REQUIRED_SECRET__'
        $portable = $portable -replace 'gh[pousr]_[A-Za-z0-9]{16,}', '__REQUIRED_SECRET__'
        $portableLines.Add($portable)
    }
    Write-Utf8File -Path (Join-Path $codexRoot 'config.portable.toml') -Content $portableLines
}

foreach ($name in @('AGENTS.md')) {
    $source = Join-Path $CodexHome $name
    if (Test-Path -LiteralPath $source) { Copy-Item -LiteralPath $source -Destination (Join-Path $codexRoot $name) -Force }
}
foreach ($name in @('rules')) {
    $source = Join-Path $CodexHome $name
    if (Test-Path -LiteralPath $source) { Copy-DirectoryPortable -Source $source -Destination (Join-Path $codexRoot $name) }
}
$skillsSource = Join-Path $CodexHome 'skills'
if (Test-Path -LiteralPath $skillsSource) {
    $skillsDestination = Ensure-Directory (Join-Path $codexRoot 'skills')
    foreach ($skill in Get-ChildItem -LiteralPath $skillsSource -Directory -Force | Where-Object { $_.Name -ne '.system' }) {
        Copy-DirectoryPortable -Source $skill.FullName -Destination (Join-Path $skillsDestination $skill.Name)
    }
}
if ($IncludeMemories) {
    $memoriesSource = Join-Path $CodexHome 'memories'
    if (Test-Path -LiteralPath $memoriesSource) {
        Copy-DirectoryPortable -Source $memoriesSource -Destination (Join-Path $codexRoot 'memories') -ExcludeDirectories @('.git')
    }
}
Write-Utf8File -Path (Join-Path $codexRoot 'required-secrets.txt') -Content (@($secretPlaceholders | Sort-Object -Unique) + @('', 'Values are intentionally not exported. Re-enter them manually on the new computer.'))

Write-Section 'Package scripts and manifest'
foreach ($scriptName in @('Migration.Common.ps1', 'Restore-TreeEnvironment.ps1', 'Test-TreeEnvironment.ps1')) {
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot $scriptName) -Destination (Join-Path $scriptRoot $scriptName) -Force
}
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'README.md') -Destination (Join-Path $packageRoot 'README.md') -Force

$manifest = [ordered]@{
    SchemaVersion = 1
    PackageKind = 'CodexEnvironmentOnly'
    CreatedAt = (Get-Date).ToString('o')
    SourceComputer = $env:COMPUTERNAME
    SourceUser = $env:USERNAME
    SourceProjectRoot = $project
    SourceCodexHome = $CodexHome
    IncludesProjectFiles = $false
    IncludesAuthentication = $false
    IncludesMemories = [bool]$IncludeMemories
    Contents = @('tool inventory', 'Winget export when available', 'Conda YAML definitions', 'small dependency manifests', 'portable Codex config', 'custom rules and skills')
    ExplicitlyExcluded = @('project code and data', 'auth.json', 'cap_sid', '.sandbox-secrets', 'installation_id', 'sessions', 'SQLite databases', 'logs', 'browser state', 'plugin caches')
}
Write-JsonFile -Path (Join-Path $packageRoot 'MIGRATION-MANIFEST.json') -InputObject $manifest

& (Join-Path $scriptRoot 'Test-TreeEnvironment.ps1') -PackageRoot $packageRoot -PackageAuditOnly
if ($LASTEXITCODE -ne 0) { throw 'Package audit failed. The package was left in place for inspection.' }

Write-Host "`nEnvironment package created successfully:" -ForegroundColor Green
Write-Host $packageRoot
Write-Host 'No project code, project data, Codex authentication, chats, or SQLite state was copied.'
