$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"

$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root "runtime\env\python.exe"
$Trainer = Join-Path $Root "app\scripts\train_b5_ptv2.py"
$Summarizer = Join-Path $Root "app\scripts\summarize_c3_results.py"
$Dataset = Join-Path $Root "input\dataset"
$Cache = Join-Path $Root "input\knn_cache"
$Output = Join-Path $Root "output"
$ConfigPath = Join-Path $Root "c3_config.json"
$LauncherState = Join-Path $Output "c3_launcher_state.json"
$Config = Get-Content -LiteralPath $ConfigPath -Raw | ConvertFrom-Json

function Write-LauncherState {
    param(
        [string]$Status,
        [int]$Seed,
        [int]$SeedIndex,
        [int]$ExitCode,
        [string]$Message
    )
    @{
        status = $Status
        updated_at = [DateTimeOffset]::Now.ToString("yyyy-MM-ddTHH:mm:sszzz")
        wrapper_pid = $PID
        active_seed = $Seed
        seed_index = $SeedIndex
        seed_count = @($Config.seeds).Count
        exit_code = $ExitCode
        message = $Message
    } | ConvertTo-Json | Set-Content -LiteralPath $LauncherState -Encoding UTF8
}

New-Item -ItemType Directory -Force -Path $Output | Out-Null
$Seeds = @($Config.seeds | ForEach-Object { [int]$_ })
$Completed = 0

for ($Index = 0; $Index -lt $Seeds.Count; $Index++) {
    $Seed = $Seeds[$Index]
    $RunDir = Join-Path $Output ("runs\seed_{0}" -f $Seed)
    $FinalMetrics = Join-Path $RunDir "final_metrics.json"
    $LastCheckpoint = Join-Path $RunDir "last.pt"
    $StdoutLog = Join-Path $RunDir "console.out.log"
    $StderrLog = Join-Path $RunDir "console.err.log"
    New-Item -ItemType Directory -Force -Path $RunDir | Out-Null

    if (Test-Path -LiteralPath $FinalMetrics) {
        try {
            $Final = Get-Content -LiteralPath $FinalMetrics -Raw | ConvertFrom-Json
            if ($Final.status -eq "complete") {
                $Completed++
                & $Python $Summarizer --root $Root
                continue
            }
        } catch {
        }
    }

    $Arguments = @(
        $Trainer,
        "--dataset-root", $Dataset,
        "--knn-cache", $Cache,
        "--output-dir", $RunDir,
        "--epochs", [string]$Config.epochs,
        "--batch-size", [string]$Config.batch_size,
        "--eval-batch-size", [string]$Config.eval_batch_size,
        "--learning-rate", [string]$Config.learning_rate,
        "--min-learning-rate", [string]$Config.min_learning_rate,
        "--weight-decay", [string]$Config.weight_decay,
        "--label-smoothing", [string]$Config.label_smoothing,
        "--patience", [string]$Config.patience,
        "--seed", [string]$Seed,
        "--workers", [string]$Config.workers,
        "--progress-every", [string]$Config.progress_every,
        "--amp"
    )
    if (Test-Path -LiteralPath $LastCheckpoint) {
        $Arguments += "--resume"
    }

    Write-LauncherState `
        -Status "running" `
        -Seed $Seed `
        -SeedIndex ($Index + 1) `
        -ExitCode 0 `
        -Message ("Training seed {0}" -f $Seed)

    $PreviousErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        & $Python @Arguments 1>> $StdoutLog 2>> $StderrLog
        $ExitCode = $LASTEXITCODE
    } catch {
        $ExitCode = 1
        $_ | Out-String | Add-Content -LiteralPath $StderrLog -Encoding UTF8
    } finally {
        $ErrorActionPreference = $PreviousErrorActionPreference
    }

    if ($ExitCode -ne 0) {
        $ErrorLines = @()
        if (Test-Path -LiteralPath $StderrLog) {
            $ErrorLines = @(
                Get-Content -LiteralPath $StderrLog -Tail 80 |
                    Where-Object { $_.Trim() }
            )
        }
        $Message = if ($ErrorLines.Count -gt 0) {
            [string]$ErrorLines[-1]
        } else {
            "Training process exited with code $ExitCode"
        }
        Write-LauncherState `
            -Status "failed" `
            -Seed $Seed `
            -SeedIndex ($Index + 1) `
            -ExitCode $ExitCode `
            -Message $Message
        & $Python $Summarizer --root $Root
        exit $ExitCode
    }

    $Completed++
    & $Python $Summarizer --root $Root
}

Write-LauncherState `
    -Status "complete" `
    -Seed 0 `
    -SeedIndex $Seeds.Count `
    -ExitCode 0 `
    -Message ("All {0} seeds completed successfully" -f $Completed)
& $Python $Summarizer --root $Root
exit 0
