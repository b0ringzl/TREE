param([switch]$NoBrowser)
$ErrorActionPreference = 'Stop'
$reviewPython = Join-Path $env:LOCALAPPDATA 'miniconda3\envs\yolo\python.exe'
$reviewUrl = 'http://127.0.0.1:8038/herbarium-review'
$reviewReady = $false
try { $reviewResult = Invoke-RestMethod 'http://127.0.0.1:8038/api/herbarium-review' -TimeoutSec 3; $reviewReady = $null -ne $reviewResult.rows } catch {}
if (-not $reviewReady) {
    Start-Process -FilePath $reviewPython -ArgumentList @('herbarium_review_api.py') -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $PSScriptRoot 'herbarium-review.stdout.log') -RedirectStandardError (Join-Path $PSScriptRoot 'herbarium-review.stderr.log')
    for ($reviewAttempt = 0; $reviewAttempt -lt 20; $reviewAttempt++) {
        Start-Sleep -Milliseconds 500
        try { $reviewResult = Invoke-RestMethod 'http://127.0.0.1:8038/api/herbarium-review' -TimeoutSec 3; if ($null -ne $reviewResult.rows) { $reviewReady = $true; break } } catch {}
    }
}
if (-not $reviewReady) { throw 'Review service failed to start; see herbarium-review.stderr.log.' }
Write-Host $reviewUrl
if (-not $NoBrowser) { Start-Process $reviewUrl }
