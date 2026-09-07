param(
    [switch]$OpenBrowser
)

$ErrorActionPreference = 'Stop'

$SandboxRoot = Split-Path -Parent $PSScriptRoot
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $SandboxRoot)
$Python = Join-Path $ProjectRoot 'envs\whu-tscmdl\python.exe'
$Server = Join-Path $PSScriptRoot 'tree_species_application_server.py'
$RuntimeRoot = Join-Path $SandboxRoot 'runtime\tree_species_application'
$Stdout = Join-Path $RuntimeRoot 'server_stdout.log'
$Stderr = Join-Path $RuntimeRoot 'server_stderr.log'
$Port = 8771

New-Item -ItemType Directory -Force -Path $RuntimeRoot | Out-Null
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python environment not found: $Python"
}

$existing = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "Application already running: http://127.0.0.1:$Port/"
    exit 0
}

$process = Start-Process -FilePath $Python `
    -ArgumentList @($Server, '--host', '127.0.0.1', '--port', "$Port", '--no-browser') `
    -WorkingDirectory $SandboxRoot `
    -RedirectStandardOutput $Stdout `
    -RedirectStandardError $Stderr `
    -WindowStyle Hidden `
    -PassThru

$deadline = (Get-Date).AddSeconds(45)
do {
    Start-Sleep -Milliseconds 500
    if ($process.HasExited) {
        throw "Application exited early. See $Stderr"
    }
    try {
        $summary = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/summary" -TimeoutSec 2
        break
    } catch {
        $summary = $null
    }
} while ((Get-Date) -lt $deadline)

if (-not $summary) {
    throw "Application did not become ready. See $Stderr"
}

Write-Host "Application ready: http://127.0.0.1:$Port/"
Write-Host "D1 samples: $($summary.datasets.d1.sample_count); Shenyang samples: $($summary.datasets.shenyang.sample_count); mode: $($summary.mode)"
if ($OpenBrowser) {
    Start-Process "http://127.0.0.1:$Port/"
}
