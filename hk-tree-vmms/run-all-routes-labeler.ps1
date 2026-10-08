$ErrorActionPreference = 'Stop'
$targetUrl = 'http://127.0.0.1:5178/'
$appRoot = Join-Path $PSScriptRoot 'annotation_app'
$logRoot = Join-Path $appRoot 'logs'
$powershellExe = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
New-Item -ItemType Directory -Path $logRoot -Force | Out-Null

function Test-Backend {
    try {
        $health = Invoke-RestMethod -Uri 'http://127.0.0.1:8022/api/health' -TimeoutSec 3
        return $health.status -eq 'ok' -and @($health.tree_routes).Count -eq 6
    } catch { return $false }
}
function Test-Frontend {
    try {
        $response = Invoke-WebRequest -Uri $targetUrl -UseBasicParsing -TimeoutSec 3
        return $response.StatusCode -eq 200
    } catch { return $false }
}

if (-not (Test-Backend)) {
    Start-Process -FilePath $powershellExe -WindowStyle Hidden -ArgumentList @(
        '-NoLogo', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
        ('"' + (Join-Path $appRoot 'start-backend-all-routes.ps1') + '"')
    ) -RedirectStandardOutput (Join-Path $logRoot 'all-routes-backend.stdout.log') `
      -RedirectStandardError (Join-Path $logRoot 'all-routes-backend.stderr.log')
}
if (-not (Test-Frontend)) {
    Start-Process -FilePath $powershellExe -WindowStyle Hidden -ArgumentList @(
        '-NoLogo', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
        ('"' + (Join-Path $appRoot 'start-frontend-all-routes.ps1') + '"')
    ) -RedirectStandardOutput (Join-Path $logRoot 'all-routes-frontend.stdout.log') `
      -RedirectStandardError (Join-Path $logRoot 'all-routes-frontend.stderr.log')
}

$ready = $false
for ($attempt = 0; $attempt -lt 60; $attempt++) {
    if ((Test-Backend) -and (Test-Frontend)) { $ready = $true; break }
    Start-Sleep -Milliseconds 500
}
if (-not $ready) { throw "All-routes labeler failed to start; check $logRoot" }
Start-Process $targetUrl
Write-Host "All-routes coordinate labeler ready: $targetUrl" -ForegroundColor Green
