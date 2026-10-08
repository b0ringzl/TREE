$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"

$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root "runtime\env\python.exe"
$Trainer = Join-Path $Root "app\scripts\train_b5_ptv2.py"
$Dataset = Join-Path $Root "input\dataset"
$Cache = Join-Path $Root "input\knn_cache"
$RunDir = Join-Path $Root "output\run"
$LauncherState = Join-Path $RunDir "launcher_state.json"
$StdoutLog = Join-Path $RunDir "console.out.log"
$StderrLog = Join-Path $RunDir "console.err.log"

$Arguments = @(
    $Trainer,
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
    "--amp",
    "--resume"
)

@{
    status = "running"
    updated_at = [DateTimeOffset]::Now.ToString("yyyy-MM-ddTHH:mm:sszzz")
    wrapper_pid = $PID
    resume = $true
} | ConvertTo-Json | Set-Content -LiteralPath $LauncherState -Encoding UTF8

try {
    & $Python @Arguments 1>> $StdoutLog 2>> $StderrLog
    $ExitCode = $LASTEXITCODE
    $Status = if ($ExitCode -eq 0) { "complete" } else { "failed" }
    $Message = "Training process exited with code $ExitCode"
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
    message = $Message
} | ConvertTo-Json | Set-Content -LiteralPath $LauncherState -Encoding UTF8

exit $ExitCode
