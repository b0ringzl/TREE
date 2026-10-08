param(
    [switch]$SkipEpochWait,
    [switch]$NoLaunch
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root "runtime\env\python.exe"
$Trainer = Join-Path $Root "app\scripts\train_b5_ptv2.py"
$Dataset = Join-Path $Root "input\dataset"
$Cache = Join-Path $Root "input\knn_cache"
$RunDir = Join-Path $Root "output\run"
$LastCheckpoint = Join-Path $RunDir "last.pt"
$FinalMetrics = Join-Path $RunDir "final_metrics.json"
$ProfilePath = Join-Path $Root "training_profile.json"

function Get-RootProcesses {
    Get-CimInstance Win32_Process | Where-Object {
        $_.ProcessId -ne $PID -and
        $_.CommandLine -and
        $_.CommandLine.Contains($Root)
    }
}

function Get-TrainingProcesses {
    Get-RootProcesses | Where-Object {
        $_.Name -eq "python.exe" -and
        $_.CommandLine.Contains("train_b5_ptv2.py")
    }
}

function Get-MonitorProcesses {
    Get-RootProcesses | Where-Object {
        $_.Name -eq "python.exe" -and
        $_.CommandLine.Contains("monitor_training.py")
    }
}

function Stop-OldTrainingStack {
    $Processes = Get-RootProcesses | Where-Object {
        $_.CommandLine.Contains("train_b5_ptv2.py") -or
        $_.CommandLine.Contains("run_training.ps1")
    }
    foreach ($Process in $Processes) {
        Stop-Process -Id $Process.ProcessId -Force -ErrorAction SilentlyContinue
    }
    Start-Sleep -Seconds 3
}

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Portable runtime is not ready. Run START_TRAINING.bat once first."
}
if (Test-Path -LiteralPath $FinalMetrics) {
    Write-Host "Training is already complete. No profile switch is needed."
    exit 0
}

$MonitorWasRunning = @(Get-MonitorProcesses).Count -gt 0
$Training = @(Get-TrainingProcesses)
if ($Training.Count -gt 0 -and -not $SkipEpochWait) {
    if (Test-Path -LiteralPath $LastCheckpoint) {
        $InitialCheckpointTime = (Get-Item -LiteralPath $LastCheckpoint).LastWriteTimeUtc
        Write-Host "The current epoch is still running."
        Write-Host "Waiting for its checkpoint before switching to 24 GB mode..."
        while (@(Get-TrainingProcesses).Count -gt 0) {
            Start-Sleep -Seconds 10
            if (Test-Path -LiteralPath $FinalMetrics) {
                Write-Host "Training completed while waiting. No switch is needed."
                exit 0
            }
            if (
                (Test-Path -LiteralPath $LastCheckpoint) -and
                (Get-Item -LiteralPath $LastCheckpoint).LastWriteTimeUtc -gt
                    $InitialCheckpointTime
            ) {
                Write-Host "A new complete checkpoint is available."
                Start-Sleep -Seconds 5
                break
            }
        }
    } else {
        Write-Host "No checkpoint exists yet; stopping the initial partial epoch."
    }
}

if (@(Get-TrainingProcesses).Count -gt 0) {
    Write-Host "Stopping the old training process..."
    Stop-OldTrainingStack
}

$MemoryValues = @(
    & nvidia-smi `
        --query-gpu=memory.total `
        --format=csv,noheader,nounits 2>$null
)
if ($LASTEXITCODE -ne 0 -or $MemoryValues.Count -eq 0) {
    throw "nvidia-smi could not read GPU memory."
}
$TotalMemoryMiB = [int](($MemoryValues[0] -replace "\s", ""))
if ($TotalMemoryMiB -lt 18000) {
    throw "24 GB mode requires at least 18,000 MiB detected VRAM."
}

$Candidates = if ($TotalMemoryMiB -ge 22000) {
    @(48, 40, 32)
} else {
    @(32, 24)
}
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$SelectedBatch = $null

foreach ($Candidate in $Candidates) {
    $EvalBatch = [Math]::Min(64, $Candidate * 2)
    $ProbeDir = Join-Path $Root (
        "diagnostics\high_vram_probe_{0}_batch{1}" -f $Timestamp, $Candidate
    )
    Write-Host "Testing batch $Candidate (evaluation batch $EvalBatch)..."
    & $Python $Trainer `
        --dataset-root $Dataset `
        --knn-cache $Cache `
        --output-dir $ProbeDir `
        --epochs 1 `
        --batch-size $Candidate `
        --eval-batch-size $EvalBatch `
        --learning-rate 0.001 `
        --min-learning-rate 0.00001 `
        --weight-decay 0.05 `
        --label-smoothing 0.1 `
        --patience 20 `
        --seed 20260731 `
        --workers 0 `
        --progress-every 1 `
        --smoke-steps 1 `
        --amp
    if ($LASTEXITCODE -eq 0) {
        $SelectedBatch = $Candidate
        $SelectedEvalBatch = $EvalBatch
        break
    }
    Write-Host "Batch $Candidate did not pass; trying a smaller setting."
    Start-Sleep -Seconds 3
}

if ($null -eq $SelectedBatch) {
    throw "No high-VRAM batch passed the real-data probe. Standard mode remains available."
}

@{
    schema_version = 1
    name = "high-vram-auto"
    created_at = [DateTimeOffset]::Now.ToString("yyyy-MM-ddTHH:mm:sszzz")
    detected_vram_mib = $TotalMemoryMiB
    batch_size = $SelectedBatch
    eval_batch_size = $SelectedEvalBatch
    workers = 0
    progress_every = 10
    note = "Resume profile selected by a one-step real-data CUDA probe."
} | ConvertTo-Json | Set-Content -LiteralPath $ProfilePath -Encoding UTF8

Write-Host ""
Write-Host "24 GB mode is ready."
Write-Host "Training batch: $SelectedBatch"
Write-Host "Evaluation batch: $SelectedEvalBatch"
Write-Host "The optimizer, learning-rate scheduler, and checkpoint are preserved."

if (-not $NoLaunch) {
    if ($MonitorWasRunning -and @(Get-MonitorProcesses).Count -gt 0) {
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
        Write-Host "High-VRAM training launched. Wrapper PID: $($Process.Id)"
        Write-Host "The existing monitor will continue to show this run."
    } else {
        & (Join-Path $PSScriptRoot "start_and_monitor.ps1")
    }
}
