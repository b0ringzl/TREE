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
$ProfilePath = Join-Path $Root "training_profile.json"

$ProfileName = "standard-8gb"
$BatchSize = 16
$EvalBatchSize = 16
$Workers = 0
$ProgressEvery = 20

if (Test-Path -LiteralPath $ProfilePath) {
    $Profile = Get-Content -LiteralPath $ProfilePath -Raw | ConvertFrom-Json
    if ($Profile.name) {
        $ProfileName = [string]$Profile.name
    }
    if ($null -ne $Profile.batch_size) {
        $BatchSize = [int]$Profile.batch_size
    }
    if ($null -ne $Profile.eval_batch_size) {
        $EvalBatchSize = [int]$Profile.eval_batch_size
    }
    if ($null -ne $Profile.workers) {
        $Workers = [int]$Profile.workers
    }
    if ($null -ne $Profile.progress_every) {
        $ProgressEvery = [int]$Profile.progress_every
    }
}

if ($BatchSize -lt 2 -or $EvalBatchSize -lt 1 -or $Workers -lt 0) {
    throw "Invalid values in training_profile.json"
}

$Arguments = @(
    $Trainer,
    "--dataset-root", $Dataset,
    "--knn-cache", $Cache,
    "--output-dir", $RunDir,
    "--epochs", "120",
    "--batch-size", [string]$BatchSize,
    "--eval-batch-size", [string]$EvalBatchSize,
    "--learning-rate", "0.001",
    "--min-learning-rate", "0.00001",
    "--weight-decay", "0.05",
    "--label-smoothing", "0.1",
    "--patience", "20",
    "--seed", "20260731",
    "--workers", [string]$Workers,
    "--progress-every", [string]$ProgressEvery,
    "--amp",
    "--resume"
)

@{
    status = "running"
    updated_at = [DateTimeOffset]::Now.ToString("yyyy-MM-ddTHH:mm:sszzz")
    wrapper_pid = $PID
    resume = $true
    profile = $ProfileName
    batch_size = $BatchSize
    eval_batch_size = $EvalBatchSize
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
