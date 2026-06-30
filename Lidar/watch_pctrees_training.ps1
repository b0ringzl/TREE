param(
    [string]$Experiment = "",
    [string]$Root = "D:\TREE\Lidar\pctrees_github",
    [string]$CacheDir = "D:\TREE\lidar data\pctrees_cache_4096",
    [string]$LabelPath = "D:\TREE\lidar data\labels_pctrees_train.csv",
    [int]$MinPoints = 1024,
    [ValidateSet("Auto", "Training", "Cache")]
    [string]$Mode = "Auto",
    [int]$IntervalSec = 2,
    [switch]$Once
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

function Get-TrainingRows {
    param([string]$Root, [string]$Experiment)
    $progressPath = Get-ProgressPath -Root $Root -Experiment $Experiment
    if (-not $progressPath -or -not (Test-Path -LiteralPath $progressPath)) {
        return $null
    }
    $rows = Import-Csv -LiteralPath $progressPath
    if (-not $rows -or $rows.Count -eq 0) {
        return $null
    }
    return @{
        Path = $progressPath
        Rows = @($rows)
    }
}

function Get-CacheStatus {
    param([string]$CacheDir, [string]$LabelPath, [int]$MinPoints)
    if (-not (Test-Path -LiteralPath $CacheDir)) {
        return @{
            Exists = $false
            Complete = $false
            Total = 0
            Done = 0
            Percent = 0.0
        }
    }

    $total = 0
    if (Test-Path -LiteralPath $LabelPath) {
        $labels = Import-Csv -LiteralPath $LabelPath
        $total = @($labels | Where-Object { [int]$_.num_points -ge $MinPoints }).Count
    }
    $files = @(Get-ChildItem -LiteralPath $CacheDir -Filter "*.pt" -File)
    $done = $files.Count
    $indexPath = Join-Path $CacheDir "index.csv"
    $percent = if ($total -gt 0) { [Math]::Round(100.0 * $done / $total, 2) } else { 0.0 }
    $latest = $files | Sort-Object LastWriteTime | Select-Object -Last 1
    $processes = @(Get-CimInstance Win32_Process -Filter "name = 'python.exe' or name = 'pythonw.exe'" |
        Where-Object { $_.CommandLine -match "build_pctrees_point_cache" })

    return @{
        Exists = $true
        Complete = ((Test-Path -LiteralPath $indexPath) -and ($total -eq 0 -or $done -ge $total))
        Total = $total
        Done = $done
        Percent = $percent
        LatestFile = if ($latest) { $latest.Name } else { "" }
        LatestWrite = if ($latest) { $latest.LastWriteTime.ToString("yyyy-MM-dd HH:mm:ss") } else { "" }
        IndexExists = Test-Path -LiteralPath $indexPath
        Running = $processes.Count -gt 0
        ProcessCount = $processes.Count
    }
}

function Show-CacheStatus {
    param($Status, [string]$CacheDir, [string]$LabelPath, [int]$MinPoints)
    $bar = Format-Bar -Percent $Status.Percent
    Write-Host "pctrees LiDAR Point Cache Monitor"
    Write-Host "Cache dir : $CacheDir"
    Write-Host "Label file: $LabelPath"
    Write-Host ""
    if (-not $Status.Exists) {
        Write-Host "Cache directory does not exist yet."
        return
    }
    Write-Host ("Progress  : [{0}] {1}%" -f $bar, $Status.Percent)
    Write-Host ("Files     : {0}/{1}" -f $Status.Done, $Status.Total)
    Write-Host ("Min points: {0}" -f $MinPoints)
    Write-Host ("Latest    : {0}" -f $Status.LatestFile)
    Write-Host ("Updated   : {0}" -f $Status.LatestWrite)
    Write-Host ("Index.csv : {0}" -f $Status.IndexExists)
    Write-Host ("Builder   : {0} process(es)" -f $Status.ProcessCount)
    if (-not $Status.Running -and -not $Status.Complete) {
        Write-Host ""
        Write-Host "Cache build is not running. Re-run train_pctrees_lidar.bat to resume."
    }
}

function Show-TrainingStatus {
    param($Training)
    if (-not $Training) {
        Write-Host "Waiting for pctrees progress.csv..."
        Write-Host "Root: $Root"
        if ($Experiment) { Write-Host "Experiment: $Experiment" }
        return
    }

    $last = $Training.Rows[-1]
    $percent = 0.0
    [double]::TryParse($last.percent, [ref]$percent) | Out-Null
    $bar = Format-Bar -Percent $percent

    Write-Host "pctrees LiDAR Training Monitor"
    Write-Host "Progress file: $($Training.Path)"
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
}

while ($true) {
    Clear-Host
    $cacheStatus = Get-CacheStatus -CacheDir $CacheDir -LabelPath $LabelPath -MinPoints $MinPoints
    $showCache = $Mode -eq "Cache" -or ($Mode -eq "Auto" -and $cacheStatus.Exists -and -not $cacheStatus.Complete)
    if ($showCache) {
        Show-CacheStatus -Status $cacheStatus -CacheDir $CacheDir -LabelPath $LabelPath -MinPoints $MinPoints
    } else {
        $training = Get-TrainingRows -Root $Root -Experiment $Experiment
        Show-TrainingStatus -Training $training
    }
    Write-Host ""
    Write-Host "Press Ctrl+C to stop watching. The underlying job will keep running in its own window."

    if ($Once) { break }
    Start-Sleep -Seconds $IntervalSec
}
