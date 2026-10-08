$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
& (Join-Path $PSScriptRoot "prepare_runtime.ps1")

$Python = Join-Path $Root "runtime\env\python.exe"
$Summary = Join-Path $Root "output\c3_summary.json"
$Runner = Join-Path $PSScriptRoot "run_c3.ps1"

New-Item -ItemType Directory -Force -Path (Join-Path $Root "output") | Out-Null
$Existing = Get-CimInstance Win32_Process | Where-Object {
    $_.ProcessId -ne $PID -and
    $_.CommandLine -and
    $_.CommandLine.Contains($Root) -and
    (
        $_.CommandLine.Contains("run_c3.ps1") -or
        $_.CommandLine.Contains("train_b5_ptv2.py")
    )
}

$AllComplete = $false
if (Test-Path -LiteralPath $Summary) {
    try {
        $SummaryState = Get-Content -LiteralPath $Summary -Raw | ConvertFrom-Json
        $AllComplete = $SummaryState.status -eq "complete"
    } catch {
        $AllComplete = $false
    }
}

if (-not $Existing -and -not $AllComplete) {
    $Process = Start-Process `
        -FilePath "powershell.exe" `
        -ArgumentList @(
            "-NoProfile",
            "-ExecutionPolicy", "Bypass",
            "-File", "`"$Runner`""
        ) `
        -WindowStyle Hidden `
        -PassThru
    Write-Host "C3 runner launched. Wrapper PID: $($Process.Id)"
} elseif ($Existing) {
    Write-Host "C3 training is already running; opening the monitor."
} else {
    Write-Host "All C3 seeds are complete; opening the final result."
}

& $Python (Join-Path $Root "app\scripts\monitor_c3.py") --root $Root
