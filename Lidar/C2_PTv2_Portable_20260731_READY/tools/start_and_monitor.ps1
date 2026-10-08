$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
& (Join-Path $PSScriptRoot "prepare_runtime.ps1")

$Python = Join-Path $Root "runtime\env\python.exe"
$RunDir = Join-Path $Root "output\run"
$CheckpointSource = Join-Path $Root "input\checkpoint"
$FinalMetrics = Join-Path $RunDir "final_metrics.json"

New-Item -ItemType Directory -Force -Path $RunDir | Out-Null
if (-not (Test-Path -LiteralPath (Join-Path $RunDir "last.pt"))) {
    Copy-Item -Path (Join-Path $CheckpointSource "*") `
        -Destination $RunDir -Force
}

$Existing = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -in @("powershell.exe", "python.exe") -and
    $_.CommandLine -and
    $_.CommandLine.Contains($Root) -and
    (
        $_.CommandLine.Contains("run_training.ps1") -or
        $_.CommandLine.Contains("train_b5_ptv2.py")
    )
}

if (-not $Existing -and -not (Test-Path -LiteralPath $FinalMetrics)) {
    $Runner = Join-Path $PSScriptRoot "run_training.ps1"
    $Process = Start-Process `
        -FilePath "powershell.exe" `
        -ArgumentList @(
            "-NoProfile",
            "-ExecutionPolicy", "Bypass",
            "-File", "`"$Runner`""
        ) `
        -WindowStyle Hidden `
        -PassThru
    Write-Host "Training launched. Wrapper PID: $($Process.Id)"
} elseif ($Existing) {
    Write-Host "Training is already running; opening the monitor."
} else {
    Write-Host "Training is already complete; opening the final result."
}

& $Python (Join-Path $Root "app\scripts\monitor_training.py") `
    --run-dir $RunDir
