param([switch]$NoBrowser)

$ErrorActionPreference = 'Stop'
$demoRoot = $PSScriptRoot
$demoPython = Join-Path $env:LOCALAPPDATA 'miniconda3\envs\yolo\python.exe'
$demoUrl = 'http://127.0.0.1:8037'
$demoReady = $false
try {
    $demoHealth = Invoke-RestMethod "$demoUrl/api/health" -TimeoutSec 2
    $demoReady = $demoHealth.status -eq 'ok'
} catch {}
if (-not $demoReady) {
    if (-not (Test-Path -LiteralPath $demoPython)) {
        throw "Python environment not found: $demoPython"
    }
    Start-Process -FilePath $demoPython -ArgumentList @('server.py') -WorkingDirectory $demoRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $demoRoot 'server.stdout.log') -RedirectStandardError (Join-Path $demoRoot 'server.stderr.log')
    for ($demoAttempt=0; $demoAttempt -lt 40; $demoAttempt++) {
        try {
            $demoHealth = Invoke-RestMethod "$demoUrl/api/health" -TimeoutSec 2
            if ($demoHealth.status -eq 'ok') { $demoReady = $true; break }
        } catch {}
        Start-Sleep -Milliseconds 500
    }
}
if (-not $demoReady) { throw 'Demo failed to start. See server.stderr.log in this folder.' }
Write-Host "Tree Species Demo is ready: $demoUrl"
if (-not $NoBrowser) { Start-Process $demoUrl }
