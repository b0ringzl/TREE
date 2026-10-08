$ErrorActionPreference = "Stop"
$Frontend = Join-Path $PSScriptRoot "frontend"
Set-Location $Frontend
try {
    npm run dev -- --host 127.0.0.1 --port 5173 *>> (Join-Path $Frontend "vite.log")
}
catch {
    $_ | Out-String | Out-File (Join-Path $Frontend "vite.err.log") -Append
    throw
}
