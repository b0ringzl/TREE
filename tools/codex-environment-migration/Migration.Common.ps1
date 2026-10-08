Set-StrictMode -Version 2.0

function Write-Section {
    param([Parameter(Mandatory = $true)][string]$Title)
    Write-Host "`n=== $Title ===" -ForegroundColor Cyan
}

function Ensure-Directory {
    param([Parameter(Mandatory = $true)][string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        New-Item -ItemType Directory -Path $Path -Force | Out-Null
    }
    return (Resolve-Path -LiteralPath $Path).Path
}

function Write-Utf8File {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [AllowNull()][object]$Content
    )
    $parent = Split-Path -Parent $Path
    if ($parent) { Ensure-Directory $parent | Out-Null }
    $text = if ($null -eq $Content) { '' } elseif ($Content -is [string]) { $Content } else { ($Content -join [Environment]::NewLine) }
    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($Path, $text, $utf8NoBom)
}

function Write-JsonFile {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][object]$InputObject,
        [int]$Depth = 8
    )
    Write-Utf8File -Path $Path -Content ($InputObject | ConvertTo-Json -Depth $Depth)
}

function Read-JsonArrayFile {
    param([Parameter(Mandatory = $true)][string]$Path)
    $data = Get-Content -Raw -LiteralPath $Path | ConvertFrom-Json
    if ($null -eq $data) { return }
    if (($data -is [System.Collections.IEnumerable]) -and (-not ($data -is [string]))) {
        foreach ($item in $data) { Write-Output $item }
    }
    else {
        Write-Output $data
    }
}

function Get-SafeFileName {
    param([Parameter(Mandatory = $true)][string]$Name)
    $safe = $Name
    foreach ($char in [System.IO.Path]::GetInvalidFileNameChars()) {
        $safe = $safe.Replace([string]$char, '_')
    }
    return $safe.Trim().TrimEnd('.')
}

function Invoke-ExternalText {
    param(
        [Parameter(Mandatory = $true)][string]$Command,
        [string[]]$Arguments = @(),
        [switch]$AllowFailure
    )
    $oldErrorAction = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        try {
            $output = & $Command @Arguments 2>&1 | ForEach-Object { $_.ToString() }
            $code = $LASTEXITCODE
        }
        catch {
            if (-not $AllowFailure) { throw }
            $output = @($_.Exception.Message)
            $code = 9009
        }
        if (($code -ne 0) -and (-not $AllowFailure)) {
            throw "Command failed ($code): $Command $($Arguments -join ' ')`n$($output -join [Environment]::NewLine)"
        }
        return [PSCustomObject]@{ ExitCode = $code; Output = @($output) }
    }
    finally {
        $ErrorActionPreference = $oldErrorAction
    }
}

function Copy-DirectoryPortable {
    param(
        [Parameter(Mandatory = $true)][string]$Source,
        [Parameter(Mandatory = $true)][string]$Destination,
        [string[]]$ExcludeDirectories = @()
    )
    if (-not (Test-Path -LiteralPath $Source)) { return }
    Ensure-Directory $Destination | Out-Null
    $arguments = @($Source, $Destination, '/E', '/COPY:DAT', '/DCOPY:DAT', '/R:1', '/W:1', '/XJ', '/NFL', '/NDL', '/NJH', '/NJS')
    if ($ExcludeDirectories.Count -gt 0) {
        $arguments += '/XD'
        $arguments += $ExcludeDirectories
    }
    & robocopy @arguments | Out-Null
    $code = $LASTEXITCODE
    if ($code -ge 8) { throw "Robocopy failed ($code): $Source -> $Destination" }
}

function Get-RelativePathPortable {
    param(
        [Parameter(Mandatory = $true)][string]$BasePath,
        [Parameter(Mandatory = $true)][string]$Path
    )
    $base = [System.IO.Path]::GetFullPath($BasePath).TrimEnd('\') + '\'
    $full = [System.IO.Path]::GetFullPath($Path)
    $baseUri = New-Object System.Uri($base)
    $fullUri = New-Object System.Uri($full)
    return [System.Uri]::UnescapeDataString($baseUri.MakeRelativeUri($fullUri).ToString()).Replace('/', '\')
}

function Backup-PathIfPresent {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$BackupRoot
    )
    if (-not (Test-Path -LiteralPath $Path)) { return }
    Ensure-Directory $BackupRoot | Out-Null
    $name = Split-Path -Leaf $Path
    $destination = Join-Path $BackupRoot $name
    if ((Get-Item -LiteralPath $Path).PSIsContainer) {
        Copy-DirectoryPortable -Source $Path -Destination $destination
    }
    else {
        Copy-Item -LiteralPath $Path -Destination $destination -Force
    }
}

function Get-ToolVersionRecord {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [string[]]$VersionArguments = @('--version')
    )
    $command = Get-Command $Name -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $command) {
        return [PSCustomObject]@{ Name = $Name; Found = $false; Path = $null; Version = $null }
    }
    if ($Name -eq 'codex') {
        $package = Get-AppxPackage OpenAI.Codex -ErrorAction SilentlyContinue | Select-Object -First 1
        $version = if ($package) { "AppX $($package.Version)" } else { 'Installed; version unavailable' }
        return [PSCustomObject]@{ Name = $Name; Found = $true; Path = $command.Source; Version = $version }
    }
    # Invoke by command name so Windows App Execution Aliases (such as Codex)
    # are resolved by PowerShell instead of executing a protected package path.
    $result = Invoke-ExternalText -Command $Name -Arguments $VersionArguments -AllowFailure
    $version = ($result.Output | Where-Object { $_.Trim() } | Select-Object -First 1)
    return [PSCustomObject]@{ Name = $Name; Found = $true; Path = $command.Source; Version = $version }
}

function Test-IsChildPath {
    param(
        [Parameter(Mandatory = $true)][string]$Parent,
        [Parameter(Mandatory = $true)][string]$Child
    )
    $parentFull = [System.IO.Path]::GetFullPath($Parent).TrimEnd('\') + '\'
    $childFull = [System.IO.Path]::GetFullPath($Child).TrimEnd('\') + '\'
    return $childFull.StartsWith($parentFull, [System.StringComparison]::OrdinalIgnoreCase)
}
