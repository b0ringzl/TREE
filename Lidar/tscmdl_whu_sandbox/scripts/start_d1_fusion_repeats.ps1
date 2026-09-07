param(
    [string]$WorkspaceRoot = ([IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\..\.."))),
    [string]$SuiteRoot = (Join-Path $WorkspaceRoot "lidar data\whu\derived\tscmdl\d1_fusion_clean_repeats\20260818_protocol_v1")
)

$ErrorActionPreference = "Stop"
$python = Join-Path $WorkspaceRoot "envs\whu-tscmdl\python.exe"
$runner = Join-Path $WorkspaceRoot "Lidar\tscmdl_whu_sandbox\scripts\run_d1_fusion_repeats.py"
$dataset = Join-Path $WorkspaceRoot "lidar data\whu\derived\tscmdl\d1_four_class_training_package\20260817_165127"
$knn = Join-Path $WorkspaceRoot "lidar data\whu\derived\tscmdl\d1_ptv2_knn_cache\20260818_d1_four_class_k8"
$imageSuite = Join-Path $WorkspaceRoot "lidar data\whu\derived\tscmdl\d1_resnet50_clean_repeats\20260817_protocol_v1"
$pointSuite = Join-Path $WorkspaceRoot "lidar data\whu\derived\tscmdl\d1_ptv2_clean_repeats\20260818_protocol_v1"
$qualityReview = Join-Path $dataset "quality_reviews_for_training.json"
$stdout = Join-Path $SuiteRoot "runner_stdout.log"
$stderr = Join-Path $SuiteRoot "runner_stderr.log"
$lock = Join-Path $SuiteRoot "runner.lock.json"

foreach ($path in @($python, $runner, $dataset, $knn, $imageSuite, $pointSuite, $qualityReview, (Join-Path $SuiteRoot "protocol.json"))) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Required path is missing: $path"
    }
}

if (Test-Path -LiteralPath $lock) {
    $old = Get-Content -LiteralPath $lock -Raw | ConvertFrom-Json
    if ($old.pid -and (Get-Process -Id $old.pid -ErrorAction SilentlyContinue)) {
        throw "D1 fusion runner is already active with PID $($old.pid)"
    }
}

$arguments = @(
    ('"{0}"' -f $runner),
    "--dataset-root", ('"{0}"' -f $dataset),
    "--knn-cache", ('"{0}"' -f $knn),
    "--image-suite", ('"{0}"' -f $imageSuite),
    "--point-suite", ('"{0}"' -f $pointSuite),
    "--output-dir", ('"{0}"' -f $SuiteRoot),
    "--quality-review", ('"{0}"' -f $qualityReview),
    "--seeds", "20260728", "20260729", "20260730",
    "--cache-batch-size", "56",
    "--epochs", "100",
    "--batch-size", "256",
    "--eval-batch-size", "256",
    "--patience", "20",
    "--workers", "0"
)
$process = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $WorkspaceRoot -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru
@{
    pid = $process.Id
    started_at = (Get-Date).ToString("o")
    python = $python
    runner = $runner
    suite_root = $SuiteRoot
} | ConvertTo-Json | Set-Content -LiteralPath $lock -Encoding UTF8
Write-Output "D1 fusion runner started: PID=$($process.Id)"
Write-Output "Monitor: & '$python' '$(Join-Path $WorkspaceRoot "Lidar\tscmdl_whu_sandbox\scripts\monitor_d1_fusion_repeats.py")' --suite-root '$SuiteRoot'"
