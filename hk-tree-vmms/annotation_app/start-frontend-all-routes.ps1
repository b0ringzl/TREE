$ErrorActionPreference = 'Stop'
$env:VITE_API_BASE = 'http://127.0.0.1:8022'
Set-Location (Join-Path $PSScriptRoot 'frontend')
$npm = Get-Command npm.cmd -ErrorAction SilentlyContinue
if (-not $npm) {
    throw 'npm.cmd was not found.'
}
& $npm.Source run dev -- --host 127.0.0.1 --port 5178 --strictPort
