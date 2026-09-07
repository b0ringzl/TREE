param(
    [ValidateSet('tree', 'frame')]
    [string]$Mode = 'tree'
)

$ErrorActionPreference = 'Stop'
$targetUrl = if ($Mode -eq 'frame') {
    'http://127.0.0.1:5177/frame.html'
} else {
    'http://127.0.0.1:5177/'
}

$appRoot = Join-Path $PSScriptRoot 'annotation_app'
$backend = Join-Path $appRoot 'start-backend.ps1'
$frontend = Join-Path $appRoot 'start-frontend.ps1'
$logRoot = Join-Path $appRoot 'logs'
$powershellExe = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'

New-Item -ItemType Directory -Path $logRoot -Force | Out-Null

function Test-Backend {
    try {
        $health = Invoke-RestMethod -Uri 'http://127.0.0.1:8021/api/health' -TimeoutSec 2
        return $health.status -eq 'ok'
    } catch {
        return $false
    }
}

function Test-Frontend {
    try {
        $response = Invoke-WebRequest -Uri 'http://127.0.0.1:5177/' -UseBasicParsing -TimeoutSec 2
        return $response.StatusCode -eq 200
    } catch {
        return $false
    }
}

if (-not (Test-Backend)) {
    Start-Process -FilePath $powershellExe -WindowStyle Hidden -ArgumentList @(
        '-NoLogo', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', ('"' + $backend + '"')
    ) -RedirectStandardOutput (Join-Path $logRoot 'backend.stdout.log') `
      -RedirectStandardError (Join-Path $logRoot 'backend.stderr.log')
}

if (-not (Test-Frontend)) {
    Start-Process -FilePath $powershellExe -WindowStyle Hidden -ArgumentList @(
        '-NoLogo', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', ('"' + $frontend + '"')
    ) -RedirectStandardOutput (Join-Path $logRoot 'frontend.stdout.log') `
      -RedirectStandardError (Join-Path $logRoot 'frontend.stderr.log')
}

$ready = $false
for ($attempt = 0; $attempt -lt 60; $attempt++) {
    if ((Test-Backend) -and (Test-Frontend)) {
        $ready = $true
        break
    }
    Start-Sleep -Milliseconds 500
}

if (-not $ready) {
    throw "何文田标注工具未能启动。请查看日志：$logRoot"
}

Start-Process $targetUrl
Write-Host "何文田标注工具已启动：$targetUrl" -ForegroundColor Green
