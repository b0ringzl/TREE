param(
    [string]$Experiment = "",
    [string]$Root = "D:\TREE\Lidar\pctrees_github",
    [int]$IntervalSec = 2
)

$ErrorActionPreference = "SilentlyContinue"

function Get-ProgressPath {
    param([string]$Root, [string]$Experiment)
    if ($Experiment) {
        return Join-Path $Root "checkpoints\$Experiment\progress.csv"
    }
    $latest = Get-ChildItem -LiteralPath (Join-Path $Root "checkpoints") -Recurse -Filter progress.csv |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if ($latest) { return $latest.FullName }
    return $null
}

function Format-Bar {
    param([double]$Percent)
    $width = 34
    $filled = [Math]::Max(0, [Math]::Min($width, [Math]::Floor($Percent * $width / 100)))
    return ("#" * $filled) + ("-" * ($width - $filled))
}

while ($true) {
    $progressPath = Get-ProgressPath -Root $Root -Experiment $Experiment
    Clear-Host

    if (-not $progressPath -or -not (Test-Path -LiteralPath $progressPath)) {
        Write-Host "Waiting for pctrees progress.csv..."
        Write-Host "Root: $Root"
        if ($Experiment) { Write-Host "Experiment: $Experiment" }
        Start-Sleep -Seconds $IntervalSec
        continue
    }

    $rows = Import-Csv -LiteralPath $progressPath
    if (-not $rows -or $rows.Count -eq 0) {
        Write-Host "Progress file exists but has no rows yet:"
        Write-Host $progressPath
        Start-Sleep -Seconds $IntervalSec
        continue
    }

    $last = @($rows)[-1]
    $percent = 0.0
    [double]::TryParse($last.percent, [ref]$percent) | Out-Null
    $bar = Format-Bar -Percent $percent

    Write-Host "pctrees LiDAR Training Monitor"
    Write-Host "Progress file: $progressPath"
    Write-Host ""
    Write-Host ("Experiment : {0}" -f $last.experiment)
    Write-Host ("Stage      : {0}" -f $last.stage)
    Write-Host ("Epoch      : {0}/{1}" -f $last.epoch, $last.total_epochs)
    Write-Host ("Batch      : {0}/{1}" -f $last.batch, $last.total_batches)
    Write-Host ("Progress   : [{0}] {1}%" -f $bar, $last.percent)
    Write-Host ("Loss       : {0}" -f $last.loss)
    Write-Host ("Accuracy   : {0}" -f $last.accuracy)
    Write-Host ("Samples    : {0}/{1}" -f $last.samples_seen, $last.total_samples)
    Write-Host ("Elapsed    : {0}s" -f $last.elapsed_sec)
    Write-Host ("ETA        : {0}s" -f $last.eta_sec)
    Write-Host ("Updated    : {0}" -f $last.timestamp)
    Write-Host ""
    Write-Host "Press Ctrl+C to stop watching. Training will keep running in its own window."

    Start-Sleep -Seconds $IntervalSec
}
