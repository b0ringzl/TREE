param(
    [switch]$Monitor
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$Python = Join-Path $ProjectRoot "envs\whu-tscmdl\python.exe"
$Runner = Join-Path $PSScriptRoot "run_d1_ptv2_repeats.py"
$MonitorScript = Join-Path $PSScriptRoot "monitor_d1_ptv2_repeats.py"
$SuiteRoot = Join-Path $ProjectRoot "lidar data\whu\derived\tscmdl\d1_ptv2_clean_repeats\20260818_protocol_v1"

foreach ($Path in @($Python, $Runner, $MonitorScript, (Join-Path $SuiteRoot "protocol.json"))) {
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Required D1 PTv2 path is missing: $Path"
    }
}

$Existing = Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -and
    $_.CommandLine.Contains("run_d1_ptv2_repeats.py") -and
    $_.CommandLine.Contains($SuiteRoot)
}
if ($Existing) {
    throw "A D1 PTv2 runner is already active for this suite."
}

$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$Stdout = Join-Path $SuiteRoot ("runner_stdout_{0}.log" -f $Timestamp)
$Stderr = Join-Path $SuiteRoot ("runner_stderr_{0}.log" -f $Timestamp)
$RunnerArguments = @(
    ('"{0}"' -f $Runner),
    "--suite-root",
    ('"{0}"' -f $SuiteRoot)
)
$Process = Start-Process `
    -FilePath $Python `
    -ArgumentList $RunnerArguments `
    -WorkingDirectory $ProjectRoot `
    -RedirectStandardOutput $Stdout `
    -RedirectStandardError $Stderr `
    -WindowStyle Hidden `
    -PassThru

@{
    status = "launched"
    launched_at = [DateTimeOffset]::Now.ToString("yyyy-MM-ddTHH:mm:sszzz")
    process_id = $Process.Id
    suite_root = $SuiteRoot
    stdout_log = $Stdout
    stderr_log = $Stderr
} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $SuiteRoot "launcher_info.json") -Encoding UTF8

Write-Host "D1 PTv2 runner launched. PID=$($Process.Id)"
Write-Host "Suite: $SuiteRoot"
if ($Monitor) {
    & $Python $MonitorScript --suite-root $SuiteRoot --no-exit-on-complete
}
