param(
    [string]$PidFile = "",
    [int]$ProcessId = 91716,
    [string]$RunDir = "",
    [string]$LogPath = "",
    [int]$IntervalSeconds = 1
)

$ErrorActionPreference = "SilentlyContinue"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
if (-not $RunDir) { $RunDir = Join-Path $ProjectRoot "models\yolo_runs\urban_tree_seg_v1" }
if (-not $LogPath) { $LogPath = Join-Path $ProjectRoot "models\yolo_runs\urban_tree_seg_v1_resume.log" }
$resultsPath = Join-Path $RunDir "results.csv"
$weightsDir = Join-Path $RunDir "weights"
$bestPath = Join-Path $weightsDir "best.pt"
$lastPath = Join-Path $weightsDir "last.pt"
$argsPath = Join-Path $RunDir "args.yaml"

function Convert-ToDouble {
    param([string]$Value)
    $culture = [System.Globalization.CultureInfo]::InvariantCulture
    $result = 0.0
    if ([double]::TryParse($Value, [System.Globalization.NumberStyles]::Float, $culture, [ref]$result)) {
        return $result
    }
    return 0.0
}

function Get-TargetEpoch {
    if (Test-Path $argsPath) {
        $line = Get-Content $argsPath | Where-Object { $_ -match "^epochs:\s*\d+" } | Select-Object -First 1
        if ($line -match "(\d+)") {
            return [int]$Matches[1]
        }
    }
    return $null
}

function Write-TrainingSummary {
    param([string]$Status)

    Write-Host ""
    Write-Host "Training summary - $Status" -ForegroundColor Cyan

    if (!(Test-Path $resultsPath)) {
        Write-Host "results.csv not found: $resultsPath" -ForegroundColor Yellow
        return
    }

    $rows = @(Import-Csv $resultsPath)
    if ($rows.Count -eq 0) {
        Write-Host "results.csv is empty: $resultsPath" -ForegroundColor Yellow
        return
    }

    $final = $rows[-1]
    $bestMask = $rows | Sort-Object { Convert-ToDouble $_.'metrics/mAP50-95(M)' } -Descending | Select-Object -First 1
    $bestBox = $rows | Sort-Object { Convert-ToDouble $_.'metrics/mAP50-95(B)' } -Descending | Select-Object -First 1

    $targetEpoch = Get-TargetEpoch
    $finalEpoch = [int]$final.epoch
    if ($targetEpoch) {
        Write-Host "Epochs: $finalEpoch/$targetEpoch"
    } else {
        Write-Host "Epochs: $finalEpoch"
    }
    Write-Host "Final mask: P=$($final.'metrics/precision(M)') R=$($final.'metrics/recall(M)') mAP50=$($final.'metrics/mAP50(M)') mAP50-95=$($final.'metrics/mAP50-95(M)')"
    Write-Host "Final box : P=$($final.'metrics/precision(B)') R=$($final.'metrics/recall(B)') mAP50=$($final.'metrics/mAP50(B)') mAP50-95=$($final.'metrics/mAP50-95(B)')"
    Write-Host "Best mask mAP50-95: $($bestMask.'metrics/mAP50-95(M)') at epoch $($bestMask.epoch)" -ForegroundColor Green
    Write-Host "Best box  mAP50-95: $($bestBox.'metrics/mAP50-95(B)') at epoch $($bestBox.epoch)" -ForegroundColor Green
    Write-Host "Final train loss: box=$($final.'train/box_loss') seg=$($final.'train/seg_loss') cls=$($final.'train/cls_loss') dfl=$($final.'train/dfl_loss')"

    if (Test-Path $bestPath) {
        $best = Get-Item $bestPath
        Write-Host "best.pt: $bestPath ($([math]::Round($best.Length / 1MB, 1)) MB, $($best.LastWriteTime))"
    }
    if (Test-Path $lastPath) {
        $last = Get-Item $lastPath
        Write-Host "last.pt: $lastPath ($([math]::Round($last.Length / 1MB, 1)) MB, $($last.LastWriteTime))"
    }
}

$esc = [char]27
Write-Host "Streaming YOLO training progress every $IntervalSeconds second(s). Press Ctrl+C to stop."

while ($true) {
    $now = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

    $proc = Get-Process -Id $ProcessId
    if ($proc) {
        $procText = "PID=$ProcessId CPU=$([math]::Round($proc.CPU, 1))s"
    } else {
        $procText = "PID=$ProcessId stopped"
    }

    $gpuRaw = nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits
    $gpuParts = @($gpuRaw -split ",").ForEach({ $_.Trim() })
    if ($gpuParts.Count -ge 5) {
        $gpuText = "GPU=$($gpuParts[0])% MEM=$($gpuParts[1])/$($gpuParts[2])MiB TEMP=$($gpuParts[3])C PWR=$($gpuParts[4])W"
    } else {
        $gpuText = "GPU=unknown"
    }

    $epochText = "epoch=unknown"
    if (Test-Path $resultsPath) {
        $lastResult = Get-Content $resultsPath -Tail 1
        $completedEpoch = ($lastResult -split ",")[0].Trim()
        if ($completedEpoch) {
            $epochText = "completed_epoch=$completedEpoch"
        }
    }

    $checkpointText = "checkpoint=unknown"
    if (Test-Path $lastPath) {
        $last = Get-Item $lastPath
        $checkpointText = "last.pt=$($last.LastWriteTime.ToString('HH:mm:ss'))"
    }

    $progressText = "progress=waiting for log"
    if (Test-Path $LogPath) {
        $progressLine = Get-Content $LogPath -Tail 80 |
            ForEach-Object { $_ -replace "$esc\[[0-9;?]*[ -/]*[@-~]", "" } |
            ForEach-Object { $_ -replace "[^\x20-\x7E]", " " } |
            Where-Object { $_ -match "\d+/\d+.*\d+/\d+" } |
            Select-Object -Last 1
        if ($progressLine) {
            $progressText = ($progressLine -replace "\s+", " ").Trim()
        }
    }

    Write-Host "$now | $procText | $gpuText | $epochText | $checkpointText | " -NoNewline
    Write-Host $progressText -ForegroundColor Red

    if (!$proc) {
        $targetEpoch = Get-TargetEpoch
        $completedEpoch = $null
        if (Test-Path $resultsPath) {
            $lastResult = Get-Content $resultsPath -Tail 1
            $completedEpoch = [int](($lastResult -split ",")[0].Trim())
        }

        if ($targetEpoch -and $completedEpoch -ge $targetEpoch) {
            Write-TrainingSummary "completed"
        } else {
            Write-TrainingSummary "process stopped before target epoch"
        }
        break
    }

    Start-Sleep -Seconds $IntervalSeconds
}
