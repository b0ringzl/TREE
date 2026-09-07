param(
    [switch]$Monitor
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$Python = Join-Path $ProjectRoot "envs\whu-tscmdl\python.exe"
$Runner = Join-Path $PSScriptRoot "run_d1_resnet50_parallel.py"
$MonitorScript = Join-Path $PSScriptRoot "monitor_d1_resnet50_repeats.py"
$SuiteRoot = Join-Path $ProjectRoot "lidar data\whu\derived\tscmdl\d1_resnet50_clean_repeats\20260817_protocol_v1"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "D1 Python environment not found: $Python"
}

$Existing = Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -and
    (
        $_.CommandLine.Contains("run_d1_resnet50_repeats.py") -or
        $_.CommandLine.Contains("run_d1_resnet50_parallel.py")
    ) -and
    $_.CommandLine.Contains($ProjectRoot)
}

if (-not $Existing) {
    $LogStamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $StdoutLog = Join-Path $SuiteRoot "parallel_runner_stdout_$LogStamp.log"
    $StderrLog = Join-Path $SuiteRoot "parallel_runner_stderr_$LogStamp.log"
    $Process = Start-Process `
        -FilePath $Python `
        -ArgumentList @($Runner) `
        -WorkingDirectory $ProjectRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput $StdoutLog `
        -RedirectStandardError $StderrLog `
        -PassThru
    Write-Host "D1 three-seed runner started. PID=$($Process.Id)"
    Write-Host "Runner stdout: $StdoutLog"
    Write-Host "Runner stderr: $StderrLog"
} else {
    Write-Host "D1 three-seed runner is already active."
}

if ($Monitor) {
    & $Python $MonitorScript --suite-root $SuiteRoot
}
