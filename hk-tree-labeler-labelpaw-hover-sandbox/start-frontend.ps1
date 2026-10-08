$ErrorActionPreference = "Stop"
$env:VITE_API_BASE = "http://127.0.0.1:8010"
$Project = $PSScriptRoot
$Root = Split-Path -Parent $Project
$Frontend = Join-Path $Project "frontend"
Set-Location $Frontend
try {
    $Vite = Join-Path $Frontend "node_modules\vite\bin\vite.js"
    if (-not (Test-Path $Vite)) {
        $Vite = Join-Path $Root "hk-tree-labeler\frontend\node_modules\vite\bin\vite.js"
    }
    node $Vite --host 127.0.0.1 --port 5175 *>> (Join-Path $Frontend "vite.log")
}
catch {
    $_ | Out-String | Out-File (Join-Path $Frontend "vite.err.log") -Append
    throw
}
