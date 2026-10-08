$ErrorActionPreference = 'Stop'
$env:VITE_API_BASE = 'http://127.0.0.1:8021'
Set-Location (Join-Path $PSScriptRoot 'frontend')
$npm = Get-Command npm.cmd -ErrorAction SilentlyContinue
if (-not $npm) {
    throw 'npm.cmd was not found.'
}
& $npm.Source run dev -- --port 5177

