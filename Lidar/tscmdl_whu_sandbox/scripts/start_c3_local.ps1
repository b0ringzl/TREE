$ErrorActionPreference = "Stop"

$Sandbox = Split-Path -Parent $PSScriptRoot
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $Sandbox)
$Root = Join-Path $ProjectRoot "lidar data\whu\derived\tscmdl\c3_ptv2_repeats"
$Python = Join-Path $ProjectRoot "envs\whu-tscmdl\python.exe"
$Runner = Join-Path $Sandbox "scripts\run_c3_local.ps1"
$Monitor = Join-Path $Sandbox "scripts\monitor_c3.py"
$ConfigPath = Join-Path $Root "c3_config.json"
$SummaryPath = Join-Path $Root "output\c3_summary.json"

foreach ($Path in @($Python, $Runner, $Monitor, $ConfigPath)) {
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Required C3 path is missing: $Path"
    }
}
$Config = Get-Content -LiteralPath $ConfigPath -Raw | ConvertFrom-Json
if ([int]$Config.batch_size -ne 20 -or [int]$Config.eval_batch_size -ne 64) {
    throw "Local C3 fixed batch configuration changed unexpectedly."
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
if ($TotalMemoryMiB -lt 7800) {
    throw "Local C3 requires the validated 8 GB-class GPU profile."
}

$Existing = Get-CimInstance Win32_Process | Where-Object {
    $_.ProcessId -ne $PID -and
    $_.CommandLine -and
    $_.CommandLine.Contains($Root) -and
    (
        $_.CommandLine.Contains("run_c3_local.ps1") -or
        $_.CommandLine.Contains("train_b5_ptv2.py")
    )
}
$AllComplete = $false
if (Test-Path -LiteralPath $SummaryPath) {
    try {
        $Summary = Get-Content -LiteralPath $SummaryPath -Raw | ConvertFrom-Json
        $AllComplete = $Summary.status -eq "complete"
    } catch {
        $AllComplete = $false
    }
}

if (-not $Existing -and -not $AllComplete) {
    $Process = Start-Process `
        -FilePath "powershell.exe" `
        -ArgumentList @(
            "-NoProfile",
            "-ExecutionPolicy", "Bypass",
            "-File", "`"$Runner`""
        ) `
        -WindowStyle Hidden `
        -PassThru
    Write-Host "Local C3 runner launched. Wrapper PID: $($Process.Id)"
} elseif ($Existing) {
    Write-Host "Local C3 training is already running; opening the monitor."
} else {
    Write-Host "All local C3 seeds are complete; opening the final result."
}

& $Python $Monitor --root $Root
