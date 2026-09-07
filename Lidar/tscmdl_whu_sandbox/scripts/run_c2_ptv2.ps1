param(
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"

$Sandbox = Split-Path -Parent $PSScriptRoot
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $Sandbox)
$Python = Join-Path $ProjectRoot "envs\whu-tscmdl\python.exe"
$DerivedRoot = Join-Path $ProjectRoot "lidar data\whu\derived\tscmdl"
$Dataset = Join-Path $DerivedRoot "c1_full_shared_dataset"
$Cache = Join-Path $DerivedRoot "c2a_ptv2\knn_memmap_k8"
$RunDir = Join-Path $DerivedRoot "c2_ptv2\run_batch16_seed20260731"
$StdoutLog = Join-Path $RunDir "console.out.log"
$StderrLog = Join-Path $RunDir "console.err.log"
$LauncherState = Join-Path $RunDir "launcher_state.json"

New-Item -ItemType Directory -Force -Path $RunDir | Out-Null
$Arguments = @(
    (Join-Path $Sandbox "scripts\train_b5_ptv2.py"),
    "--dataset-root", $Dataset,
    "--knn-cache", $Cache,
    "--output-dir", $RunDir,
    "--epochs", "120",
    "--batch-size", "16",
    "--eval-batch-size", "16",
    "--learning-rate", "0.001",
    "--min-learning-rate", "0.00001",
    "--weight-decay", "0.05",
    "--label-smoothing", "0.1",
    "--patience", "20",
    "--seed", "20260731",
    "--workers", "0",
    "--progress-every", "20",
    "--amp"
)
if ($Resume) {
    $Arguments += "--resume"
}

@{
    status = "running"
    updated_at = [DateTimeOffset]::Now.ToString("yyyy-MM-ddTHH:mm:sszzz")
    wrapper_pid = $PID
    resume = [bool]$Resume
    command = "$Python $($Arguments -join ' ')"
} | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $LauncherState -Encoding UTF8

try {
    & $Python @Arguments 1>> $StdoutLog 2>> $StderrLog
    $ExitCode = $LASTEXITCODE
    $Status = if ($ExitCode -eq 0) { "complete" } else { "failed" }
    $Message = if ($ExitCode -eq 0) {
        "Training process exited normally"
    } else {
        "Training process exited with code $ExitCode"
    }
} catch {
    $ExitCode = 1
    $Status = "failed"
    $Message = $_.Exception.Message
}

@{
    status = $Status
    updated_at = [DateTimeOffset]::Now.ToString("yyyy-MM-ddTHH:mm:sszzz")
    wrapper_pid = $PID
    exit_code = $ExitCode
    resume = [bool]$Resume
    message = $Message
} | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $LauncherState -Encoding UTF8

exit $ExitCode
